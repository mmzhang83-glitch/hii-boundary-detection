# Test Suite Documentation

## Test Contents

### Synthetic Model Tests (run_test_plan.py)

Runs 4 standard test types on 4 synthetic HII region models.

| Model | Radial Profile | Tunable Parameters |
|-------|---------------|-------------------|
| Sharp Step | Step function: A (r≤R₀), B (r>R₀) | `crater_r0`, `crater_a`, `crater_b` |
| Linear Ramp | Linear transition | `ramp_half_widths`: [5, 10] |
| Sigmoid | Smooth S-curve transition | `sigmoid_ks`: [1, 5] |
| Gaussian Ring | Gaussian peak at R₀ | `ring_sigmas`: [5, 10] |

| Test Type | Purpose | Bootstrap | Pass Criteria |
|-----------|---------|:---------:|---------------|
| baseline | Clean image, no noise | disabled | MRE < 2.0 px |
| noise | Noise sweep (5–100% of signal amplitude) | N iterations | MRE < 5.0 px |
| center | Center offset sensitivity (10–50%) | N iterations | report only |
| rmax | Search radius sensitivity (150 px) | N iterations | report only |

### Real Data Tests (run_test_plan_real.py)

Runs the full pipeline (catalog → download → preprocess → detect → report) on GLIMPSE 8μm images of 5 bubbles from Churchwell+ (2006).

## Code Structure

```
├── run_test_plan.py              # Synthetic model test orchestrator
├── run_test_plan_elliptical.py   # Elliptical prior test orchestrator
├── run_test_plan_real.py         # Real data full-pipeline orchestrator
├── quick_test.py                 # Quick smoke test (Sigmoid + Gaussian + Real)
├── test_models.py                # Synthetic model definitions
├── test_generators.py            # Radial profiles + 2D image generation
├── test_runner.py                # baseline/noise/center/rmax tests
├── test_runner_elliptical.py     # Elliptical tests
├── test_real_bubbles.py          # Real image boundary detection
├── test_analysis.py              # Expected boundary + MRE/RMS metrics
├── test_diagnostics.py           # Diagnostic plots (overlay/pipeline/error)
├── test_report.py                # MD + HTML report generator
├── fetch_bubble_catalog.py       # Vizier catalog query
├── download_glimpse_images.py    # IRSA SIA download
├── preprocess_glimpse.py         # à trous wavelet point source removal
├── test_config.yaml              # Synthetic test parameters
├── test_config_elliptical.yaml   # Elliptical test parameters
├── test_config_real.yaml         # Real data pipeline parameters
└── logging_setup.py              # Logging configuration
```

## Usage

```bash
# Synthetic models
python run_test_plan.py                          # All tests
python run_test_plan.py --model "Sigmoid"        # Specific model
python run_test_plan.py --bootstrap-n 0          # Skip bootstrap (fastest)

# Real data
python run_test_plan_real.py                     # Full pipeline
python run_test_plan_real.py --only-plot         # Re-plot only

# Quick verification
python quick_test.py                             # All
python quick_test.py --skip-real                 # Skip real data
```

### CLI Parameters (run_test_plan.py)

| Parameter | Description |
|-----------|-------------|
| `--model MODEL` | Run only models matching this substring |
| `--bootstrap-n N` | Bootstrap iterations |
| `--n-workers N` | Parallel processes |
| `--skip-noise` | Skip noise sweep |
| `--no-sensitivity` | Skip center offset + rmax tests |
| `--seed N` | Random seed |
| `--config PATH` | Path to config file |

### Diagnostic Plots

| Plot | Content |
|------|---------|
| overlay.png | Full image + zoom + error band + polar uncertainty panel |
| pipeline.png | 6-panel: polar → smoothed → gradient → score → cost → cross-section |
| error.png | Detected − expected vs angle |
| algo_diag.png | rmin scan Δdiff curve + candidate boundaries |
