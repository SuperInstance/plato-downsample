# plato-downsample

> Intelligent downsampling for PLATO tile streams with anomaly preservation

## What This Does

plato-downsample reduces data volume while preserving the shape of your signal. It implements multiple methods — LTTB (Largest-Triangle-Three-Buckets), MinMax, Average, and Random — and can optionally preserve anomalous data points that would otherwise be lost.

## The Key Idea

An ESP32 sensor producing 100 readings/second can't send them all to the cloud. But naive downsampling (take every Nth point) destroys the interesting parts — the spikes, dips, and anomalies. LTTB solves this by dividing data into buckets and selecting the point in each bucket that creates the largest triangle with its neighbors, preserving the visual shape of the signal. MinMax preserves peaks and valleys. Anomaly preservation flags points that deviate from the mean by N standard deviations and forces them into the output.

## Install

```bash
cargo add plato-downsample
```

## Quick Start

```rust
use plato_downsample::*;

let data: Vec<(f64, f64)> = (0..1000)
    .map(|i| (i as f64, (i as f64 * 0.01).sin()))
    .collect();

// Downsample to 50 points using LTTB
let result = downsample(&data, 50, DownsampleMethod::LTTB);
println!("{} → {} points", result.original_len, result.downsampled_len);

// Check reconstruction quality
let error = result.reconstruction_error(&data);
println!("MSE: {:.6}", error);
```

## API Reference

| Type | Description |
|---|---|
| `DownsampleMethod` | `LTTB` / `MinMax` / `Average` / `LargestTriangle` / `Random` |
| `DownsampleConfig { method, target_rate, preserve_anomalies }` | Full configuration |
| `AnomalyPreserver { threshold_std, min_gap }` | Preserve points > N std devs from mean |
| `DownsampledSeries { original_len, downsampled_len, indices, method }` | Result with `reconstruction_error()` |

## Testing

19 tests: LTTB accuracy, MinMax peak preservation, average smoothing, anomaly preservation, edge cases (empty, single point, target > source), reconstruction error.

## License

Apache-2.0
