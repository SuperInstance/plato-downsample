//! # plato-downsample
//!
//! Intelligent downsampling for PLATO tile streams with anomaly preservation.
//! Reduces data volume while preserving important features (anomalies, transitions, periodicity).
//! Critical for ESP32→coordinator bandwidth.

use serde::{Deserialize, Serialize};

/// Downsampling method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownsampleMethod {
    /// Largest-Triangle-Three-Buckets — gold standard for visual downsampling.
    LTTB,
    /// Preserve peaks and valleys.
    MinMax,
    /// Simple bucket averaging.
    Average,
    /// Alias for LTTB (Largest Triangle).
    LargestTriangle,
    /// Random sampling with reproducible seed.
    Random,
}

/// Configuration for downsampling operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampleConfig {
    pub method: DownsampleMethod,
    pub target_rate: f64,
    pub preserve_anomalies: bool,
}

/// Anomaly detection parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPreserver {
    /// Number of standard deviations beyond which a point is considered anomalous.
    pub threshold_std: f64,
    /// Minimum gap between preserved anomaly indices.
    pub min_gap: usize,
}

/// Result of a downsampling operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampledSeries {
    pub original_len: usize,
    pub downsampled_len: usize,
    pub indices: Vec<usize>,
    pub method: DownsampleMethod,
}

impl DownsampledSeries {
    /// Compute MSE between original data and linearly-interpolated reconstruction
    /// from the downsampled points.
    pub fn reconstruction_error(&self, original: &[(f64, f64)]) -> f64 {
        if self.indices.is_empty() || original.len() < 2 {
            return 0.0;
        }

        let downsampled: Vec<(f64, f64)> = self.indices.iter().map(|&i| original[i]).collect();
        let mut sum_sq = 0.0;

        for (i, (x, y)) in original.iter().enumerate() {
            let reconstructed = interpolate(&downsampled, *x);
            let diff = y - reconstructed;
            sum_sq += diff * diff;
        }

        sum_sq / original.len() as f64
    }
}

/// Linear interpolation of `target_x` within the downsampled series.
fn interpolate(downsampled: &[(f64, f64)], target_x: f64) -> f64 {
    if downsampled.is_empty() {
        return 0.0;
    }
    if downsampled.len() == 1 {
        return downsampled[0].1;
    }

    // Find the two surrounding points
    if target_x <= downsampled[0].0 {
        return downsampled[0].1;
    }
    if target_x >= downsampled.last().unwrap().0 {
        return downsampled.last().unwrap().1;
    }

    for i in 0..downsampled.len() - 1 {
        if downsampled[i].0 <= target_x && target_x <= downsampled[i + 1].0 {
            let (x0, y0) = downsampled[i];
            let (x1, y1) = downsampled[i + 1];
            let t = (target_x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    downsampled.last().unwrap().1
}

/// Largest-Triangle-Three-Buckets downsampling.
///
/// The gold standard for visual downsampling — preserves the visual shape of the data.
pub fn lttb(data: &[(f64, f64)], threshold: usize) -> Vec<(f64, f64)> {
    if data.len() <= threshold || threshold < 3 {
        return data.to_vec();
    }

    let mut sampled = Vec::with_capacity(threshold);
    sampled.push(data[0]);

    // Bucket size for the middle buckets (first and last are always included)
    let bucket_size = (data.len() - 2) as f64 / (threshold - 2) as f64;

    let mut prev_selected = 0;

    for i in 0..(threshold - 2) {
        let avg_start = ((i as f64 * bucket_size).floor() as usize) + 1;
        let avg_end = (((i + 1) as f64 * bucket_size).floor() as usize + 1).min(data.len() - 1);

        // Calculate average point of next bucket
        let mut avg_x = 0.0;
        let mut avg_y = 0.0;
        let avg_count = (avg_end - avg_start) as f64;
        if avg_count > 0.0 {
            for j in avg_start..avg_end {
                avg_x += data[j].0;
                avg_y += data[j].1;
            }
            avg_x /= avg_count;
            avg_y /= avg_count;
        } else if avg_start < data.len() {
            avg_x = data[avg_start].0;
            avg_y = data[avg_start].1;
        }

        // Current bucket range
        let cur_start = if i == 0 { 1 } else { (i as f64 * bucket_size).floor() as usize + 1 };
        let cur_end = ((i + 1) as f64 * bucket_size).floor() as usize + 1;
        let cur_end = cur_end.min(data.len() - 1);

        // Select point in current bucket with largest triangle area
        let (ax, ay) = data[prev_selected];
        let mut max_area = -1.0;
        let mut best_idx = cur_start;

        for j in cur_start..cur_end {
            let area = ((ax - avg_x) * (data[j].1 - ay)
                - (ax - data[j].0) * (avg_y - ay))
            .abs();
            if area > max_area {
                max_area = area;
                best_idx = j;
            }
        }

        sampled.push(data[best_idx]);
        prev_selected = best_idx;
    }

    sampled.push(*data.last().unwrap());
    sampled
}

/// Min-Max downsampling — preserves peaks and valleys.
///
/// Divides data into buckets and selects both the minimum and maximum Y value from each.
pub fn min_max_downsample(data: &[(f64, f64)], threshold: usize) -> Vec<(f64, f64)> {
    if data.len() <= threshold || threshold < 2 {
        return data.to_vec();
    }

    // Each bucket contributes 2 points (min and max), so bucket count = threshold / 2
    let num_buckets = (threshold / 2).max(1);
    let bucket_size = data.len() as f64 / num_buckets as f64;

    let mut sampled = Vec::with_capacity(threshold);

    for i in 0..num_buckets {
        let start = (i as f64 * bucket_size).floor() as usize;
        let end = ((i + 1) as f64 * bucket_size).floor() as usize;
        let end = end.max(start + 1).min(data.len());

        let mut min_idx = start;
        let mut max_idx = start;

        for j in start..end {
            if data[j].1 < data[min_idx].1 {
                min_idx = j;
            }
            if data[j].1 > data[max_idx].1 {
                max_idx = j;
            }
        }

        if min_idx == max_idx {
            sampled.push(data[min_idx]);
        } else if min_idx < max_idx {
            sampled.push(data[min_idx]);
            sampled.push(data[max_idx]);
        } else {
            sampled.push(data[max_idx]);
            sampled.push(data[min_idx]);
        }
    }

    // Trim to threshold if we went over
    sampled.truncate(threshold);
    sampled
}

/// Average downsampling — simple bucket averaging.
///
/// Smooths noise by replacing each bucket with its centroid.
pub fn average_downsample(data: &[(f64, f64)], bucket_size: usize) -> Vec<(f64, f64)> {
    if bucket_size <= 1 || data.len() <= bucket_size {
        return data.to_vec();
    }

    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let end = (i + bucket_size).min(data.len());
        let count = end - i;

        let sum_x: f64 = data[i..end].iter().map(|p| p.0).sum();
        let sum_y: f64 = data[i..end].iter().map(|p| p.1).sum();

        result.push((sum_x / count as f64, sum_y / count as f64));
        i = end;
    }

    result
}

/// Random downsampling with reproducible seed.
///
/// Uses a simple linear congruential generator for deterministic results.
pub fn random_downsample(data: &[(f64, f64)], threshold: usize, seed: u64) -> Vec<(f64, f64)> {
    if data.len() <= threshold || threshold == 0 {
        return data.to_vec();
    }

    // Always include first and last
    if threshold < 2 {
        return vec![data[0]];
    }

    let mut rng_state = seed;
    let mut next_random = || -> u64 {
        // LCG: Numerical Recipes
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state
    };

    // Generate threshold-2 random indices from 1..len-1
    let mut indices: Vec<usize> = (1..data.len() - 1).collect();

    // Fisher-Yates shuffle with our PRNG
    for i in (1..indices.len()).rev() {
        let j = (next_random() as usize) % (i + 1);
        indices.swap(i, j);
    }

    let mut selected: Vec<usize> = indices[..(threshold - 2).min(indices.len())].to_vec();
    selected.push(0);
    selected.push(data.len() - 1);
    selected.sort();
    selected.dedup();

    selected.into_iter().map(|i| data[i]).collect()
}

/// Ensure anomaly points survive downsampling by merging them into the index set.
///
/// An anomaly is a point whose Y value exceeds `threshold_std` standard deviations
/// from the mean. Returns the union of `indices` and detected anomaly indices.
pub fn preserve_anomalies(
    data: &[(f64, f64)],
    indices: &[usize],
    threshold_std: f64,
) -> Vec<usize> {
    if data.is_empty() {
        return indices.to_vec();
    }

    // Compute mean and stddev of Y values
    let n = data.len() as f64;
    let mean: f64 = data.iter().map(|p| p.1).sum::<f64>() / n;
    let variance: f64 = data.iter().map(|p| (p.1 - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev < f64::EPSILON {
        return indices.to_vec();
    }

    let threshold = threshold_std * std_dev;

    let mut result: Vec<usize> = indices.to_vec();
    let mut last_anomaly = 0usize;

    for (i, (_x, y)) in data.iter().enumerate() {
        if (y - mean).abs() > threshold {
            // Respect min_gap — handled by caller if needed; here we just add
            if i == 0 || i - last_anomaly >= 1 {
                result.push(i);
                last_anomaly = i;
            }
        }
    }

    result.sort();
    result.dedup();
    result
}

/// Main downsampling entry point.
pub fn downsample(data: &[(f64, f64)], config: &DownsampleConfig) -> DownsampledSeries {
    let original_len = data.len();
    let threshold = if config.target_rate > 0.0 && config.target_rate < 1.0 {
        (original_len as f64 * config.target_rate).max(3.0) as usize
    } else {
        config.target_rate as usize
    };
    let threshold = threshold.max(3);

    let downsampled = match config.method {
        DownsampleMethod::LTTB | DownsampleMethod::LargestTriangle => lttb(data, threshold),
        DownsampleMethod::MinMax => min_max_downsample(data, threshold),
        DownsampleMethod::Average => {
            let bucket_size = (original_len as f64 / threshold as f64).ceil() as usize;
            average_downsample(data, bucket_size.max(1))
        }
        DownsampleMethod::Random => random_downsample(data, threshold, 42),
    };

    // Find indices of downsampled points in original data
    let indices = find_indices(data, &downsampled);

    let final_indices = if config.preserve_anomalies {
        preserve_anomalies(data, &indices, 2.0)
    } else {
        indices
    };

    DownsampledSeries {
        original_len,
        downsampled_len: final_indices.len(),
        indices: final_indices,
        method: config.method,
    }
}

/// Find the indices in `original` closest to each point in `sampled` (nearest X match).
fn find_indices(original: &[(f64, f64)], sampled: &[(f64, f64)]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(sampled.len());
    let mut search_start = 0;

    for point in sampled {
        let mut best_idx = search_start;
        let mut best_dist = f64::INFINITY;
        for j in search_start..original.len() {
            let dist = (original[j].0 - point.0).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = j;
            }
            // Once x starts moving past target, no need to continue
            if original[j].0 > point.0 && dist > best_dist {
                break;
            }
        }
        indices.push(best_idx);
        search_start = best_idx + 1;
    }

    indices
}

/// Compute compression ratio (original / downsampled).
pub fn compression_ratio(original_len: usize, downsampled_len: usize) -> f64 {
    if downsampled_len == 0 {
        return 0.0;
    }
    original_len as f64 / downsampled_len as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                (x, (x * 0.02).sin())
            })
            .collect()
    }

    fn sine_with_outliers(n: usize) -> Vec<(f64, f64)> {
        let mut data = sine_wave(n);
        // Add outliers
        if data.len() > 50 {
            data[25].1 = 10.0;
            data[75].1 = -10.0;
        }
        data
    }

    fn constant_data(n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|i| (i as f64, 5.0)).collect()
    }

    // === LTTB Tests ===

    #[test]
    fn test_lttb_preserves_sine_shape() {
        let data = sine_wave(1000);
        let downsampled = lttb(&data, 50);
        assert_eq!(downsampled.len(), 50);

        // First and last points preserved
        assert!((downsampled.first().unwrap().1 - data[0].1).abs() < 1e-10);
        assert!((downsampled.last().unwrap().1 - data.last().unwrap().1).abs() < 1e-10);

        // Peak and trough should be approximately preserved
        let max_y = downsampled.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let min_y = downsampled.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        assert!(max_y > 0.95); // sine peak ≈ 1.0
        assert!(min_y < -0.95); // sine trough ≈ -1.0
    }

    #[test]
    fn test_lttb_short_data() {
        let data = sine_wave(10);
        let downsampled = lttb(&data, 50);
        assert_eq!(downsampled.len(), 10); // Returns all data
    }

    #[test]
    fn test_lttb_single_point() {
        let data = vec![(0.0, 1.0)];
        let downsampled = lttb(&data, 10);
        assert_eq!(downsampled.len(), 1);
    }

    #[test]
    fn test_lttb_constant_data() {
        let data = constant_data(100);
        let downsampled = lttb(&data, 10);
        assert_eq!(downsampled.len(), 10);
        for p in &downsampled {
            assert!((p.1 - 5.0).abs() < 1e-10);
        }
    }

    // === MinMax Tests ===

    #[test]
    fn test_minmax_preserves_peaks_and_valleys() {
        let data = sine_wave(1000);
        let downsampled = min_max_downsample(&data, 50);
        assert!(downsampled.len() <= 50);
        assert!(downsampled.len() >= 2);

        let max_y = downsampled.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let min_y = downsampled.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        assert!(max_y > 0.95);
        assert!(min_y < -0.95);
    }

    #[test]
    fn test_minmax_short_data() {
        let data = vec![(0.0, 1.0), (1.0, 2.0)];
        let downsampled = min_max_downsample(&data, 100);
        assert_eq!(downsampled.len(), 2);
    }

    // === Average Tests ===

    #[test]
    fn test_average_smooths_noise() {
        // Noisy data around 0
        let data: Vec<(f64, f64)> = (0..1000)
            .map(|i| {
                let x = i as f64;
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                (x, noise)
            })
            .collect();

        let downsampled = average_downsample(&data, 100);
        assert_eq!(downsampled.len(), 10);

        // Averaged values should be close to 0
        for p in &downsampled {
            assert!(p.1.abs() < 0.01);
        }
    }

    #[test]
    fn test_average_short_data() {
        let data = vec![(0.0, 1.0), (1.0, 2.0)];
        let downsampled = average_downsample(&data, 5);
        assert_eq!(downsampled.len(), 2);
    }

    // === Random Tests ===

    #[test]
    fn test_random_reproducible_with_seed() {
        let data = sine_wave(1000);
        let d1 = random_downsample(&data, 50, 42);
        let d2 = random_downsample(&data, 50, 42);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_random_different_seeds() {
        let data = sine_wave(1000);
        let d1 = random_downsample(&data, 50, 42);
        let d2 = random_downsample(&data, 50, 99);
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_random_includes_endpoints() {
        let data = sine_wave(1000);
        let downsampled = random_downsample(&data, 50, 42);
        assert!((downsampled.first().unwrap().0 - data[0].0).abs() < 1e-10);
        assert!((downsampled.last().unwrap().0 - data.last().unwrap().0).abs() < 1e-10);
    }

    // === Anomaly Preservation ===

    #[test]
    fn test_preserve_anomalies_catches_outliers() {
        let data = sine_with_outliers(100);
        let downsampled = lttb(&data, 20);
        let base_indices = find_indices(&data, &downsampled);
        let anomaly_indices = preserve_anomalies(&data, &base_indices, 2.0);

        // The outlier at index 25 (y=10) and 75 (y=-10) should be included
        assert!(anomaly_indices.contains(&25));
        assert!(anomaly_indices.contains(&75));
    }

    #[test]
    fn test_preserve_anomalies_no_false_positives() {
        let data = constant_data(100);
        let indices = vec![0, 50, 99];
        let result = preserve_anomalies(&data, &indices, 2.0);
        // Constant data — no anomalies
        assert_eq!(result, indices);
    }

    // === Reconstruction Error ===

    #[test]
    fn test_reconstruction_error_low_for_smooth() {
        let data = sine_wave(1000);
        let series = downsample(&data, &DownsampleConfig {
            method: DownsampleMethod::LTTB,
            target_rate: 0.1,
            preserve_anomalies: false,
        });
        let error = series.reconstruction_error(&data);
        // LTTB should give low reconstruction error for smooth sine wave
        assert!(error < 0.01, "Reconstruction error too high: {}", error);
    }

    // === Compression Ratio ===

    #[test]
    fn test_compression_ratio() {
        assert!((compression_ratio(1000, 100) - 10.0).abs() < 1e-10);
        assert!((compression_ratio(1000, 500) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compression_ratio_zero() {
        assert_eq!(compression_ratio(100, 0), 0.0);
    }

    // === Comparison Tests ===

    #[test]
    fn test_lttb_vs_minmax_vs_average_error() {
        let data = sine_wave(1000);
        let threshold = 50;
        let bucket_size = 20;

        let lttb_data = lttb(&data, threshold);
        let minmax_data = min_max_downsample(&data, threshold);
        let avg_data = average_downsample(&data, bucket_size);

        // All should produce fewer points than original
        assert!(lttb_data.len() <= threshold + 1);
        assert!(minmax_data.len() <= threshold + 1);
        assert!(avg_data.len() < data.len());

        // All should preserve general range
        for ds in &[&lttb_data, &minmax_data, &avg_data] {
            let max_y = ds.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let min_y = ds.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            assert!(max_y > 0.8, "max_y too low: {}", max_y);
            assert!(min_y < -0.8, "min_y too high: {}", min_y);
        }
    }

    // === Large Dataset Performance ===

    #[test]
    fn test_large_dataset_performance() {
        let data: Vec<(f64, f64)> = (0..10_000)
            .map(|i| {
                let x = i as f64;
                (x, (x * 0.001).sin() + 0.5 * (x * 0.01).cos())
            })
            .collect();

        let start = std::time::Instant::now();
        let downsampled = lttb(&data, 100);
        let elapsed = start.elapsed();

        assert_eq!(downsampled.len(), 100);
        // Should complete in well under 1 second for 10K points
        assert!(elapsed.as_millis() < 1000, "LTTB too slow: {:?}", elapsed);
    }

    // === Downsample with config ===

    #[test]
    fn test_downsample_all_methods() {
        let data = sine_wave(1000);

        for method in &[
            DownsampleMethod::LTTB,
            DownsampleMethod::MinMax,
            DownsampleMethod::Average,
            DownsampleMethod::LargestTriangle,
            DownsampleMethod::Random,
        ] {
            let series = downsample(&data, &DownsampleConfig {
                method: *method,
                target_rate: 0.05,
                preserve_anomalies: false,
            });
            assert!(series.downsampled_len > 0);
            assert_eq!(series.original_len, 1000);
            assert_eq!(series.method, *method);
        }
    }
}
