# boundary_detection

Automatic HII region boundary detection with bootstrap uncertainty estimation.

A full-pipeline algorithm based on **polar transform + Sobel gradient + Viterbi dynamic programming + stable boundary scanning + bootstrap resampling**. Supports both circular and elliptical prior boundary detection, with a complete real-data pipeline (catalog query → image download → point-source removal → detection → HTML report).

---

## Installation

```bash
cd package
pip install -e .
```

Dependencies: `numpy>=1.24,<2` `scipy` `matplotlib` `astropy` `pyyaml` `scikit-image` `astroquery` `watroo`
Python ≥ 3.10

---

## Quick Start

```python
from boundary_detection import detect_hii_boundary

result = detect_hii_boundary(data, xc=150, yc=150, rmax=80)
print(result["boundary_radii"])        # (360,) boundary radii per angle
print(result["boundary_uncertainty"])   # (360,) 1σ uncertainty per angle
```

---

## Algorithm Overview

### Pipeline

```
Input: 2D image data + center (xc,yc) + search radius rmax
  │
  ▼
[1] Polar transform ─── warp_polar / manual_polar_transform
  │                      Cartesian → (r,θ) polar grid
  │                      skimage.transform.warp_polar
  ▼
[2] Preprocessing ───── Radial Gaussian smoothing (scipy.ndimage.gaussian_filter1d)
  │                      Angular smoothing (same)
  ▼
[3] Gradient score map ─ Sobel radial gradient (scipy.ndimage.sobel)
  │                       Error weighting + angular SNR + gradient coherence penalty
  │                       score = G² / pixel_dr
  │                       cost = -log(score/Σscore + ε)
  ▼
[4] DP optimal path ──── Viterbi dynamic programming + L-curve auto penalty
  │                       Find 360° closed path with minimum cost
  ▼
[5] Stable boundary scan ─ Sweep rmin → compute boundary diff → find stable region
  │                          Multi-zone scoring: mean_cost + length + gradient consistency
  ▼
[6] Bootstrap ────────── N noise resamples → re-detect → per-angle std = 1σ
  │                       multiprocessing parallel
  ▼
Output: boundary_radii, boundary_uncertainty, boundary_x/y
```

### Step 1: Polar Transform

**Mathematical Principle**

With `(xc, yc)` as origin, transform the circular ROI `[rmin, rmax]` to an `(r, θ)` polar grid:

```
x = xc + r · cos(θ)
y = yc + r · sin(θ)

where r ∈ [rmin, rmax], θ ∈ [0, 2π)
```

**Implementation**

Uses `skimage.transform.warp_polar()` with bilinear interpolation:

- Radial samples: `Nr = ⌊rmax⌋ + 1` (accommodates boundary at rmax)
- Angular samples: `Nθ = 360` (one per degree, matching 360° cyclic boundary)
- Output shape: `(Nr, 360)` 2D float array
- Interpolation: bilinear (`order=1`), sub-pixel accuracy

The alternative `manual_polar_transform()` uses direct trigonometric mapping + `scipy.ndimage.map_coordinates` for scenarios requiring precise sampling control (e.g., polar background in uncertainty panels).

**Python packages**: `skimage.transform.warp_polar`, `scipy.ndimage.map_coordinates`

### Step 2: Preprocessing Smoothing

**Mathematical Principle**

Two independent Gaussian smoothing steps:

1. **Radial smoothing** (along axis=0): suppresses pixel noise on individual rays
2. **Angular smoothing** (along axis=1): removes discontinuities between adjacent angles

```
I_smooth(r, θ) = G_σ_radial(r) ∗ I_raw(r, θ)   (per column)
I_smooth(r, θ) = G_σ_angular(θ) ∗ I_smooth(r, θ) (per row)
```

Where `G_σ` is a 1D Gaussian kernel with σ = FWHM / 2.355.

**Implementation**

`scipy.ndimage.gaussian_filter1d` applied per axis via the `axis` parameter. Radial and angular σ are controlled by `smoothing_fwhm` and `cost_map_smoothing_sigma` respectively; set to 0 to skip.

**Python packages**: `scipy.ndimage.gaussian_filter1d`

### Step 3: Gradient Score Map and Cost Map

#### 3a. Sobel Gradient

**Mathematical Principle**

Apply Sobel operator along the radial axis to compute the gradient `G = ∂I/∂r`:

```
Sobel kernel (radial):
    [-1, 0, 1]
    [-2, 0, 2]  · 1/8
    [-1, 0, 1]

G(r, θ) = Sobel(I_smooth)[r, θ]
```

**Implementation**

`scipy.ndimage.sobel(polar_smooth, axis=0, mode='mirror')`.

`mode='mirror'` handles boundaries by mirror-padding at rmin and rmax, avoiding edge artifacts.

**Python packages**: `scipy.ndimage.sobel`

#### 3b. Error Weighting

When an instrument error map is available (per-pixel 1σ measurement uncertainty), it is mapped through the same polar transform and used to suppress noisy pixels via `1/σ²` weighting:

```
G_weighted(r, θ) = G(r, θ) / σ²(r, θ)
```

Skipped when no error_map is provided (all weights = 1).

#### 3c. Gradient Angular Smoothing (Optional)

Post-Sobel angular Gaussian smoothing (parameter `gradient_smoothing_sigma`) to reduce gradient jumps between adjacent angles.

#### 3d. Angular SNR Weighting (Optional)

For spatially varying noise (e.g., GLIMPSE mosaic coverage variations), enable `angular_snr_weighting`:

1. Smooth signal and noise along the angular direction
2. SNR(θ) = signal_smooth(θ) / noise_smooth(θ)
3. Multiply score by angle-dependent SNR weight

**Python packages**: `scipy.ndimage.gaussian_filter1d`

#### 3e. Gradient Sign Coherence Penalty

**Mathematical Principle**

Real boundaries have consistent gradient signs across adjacent angles (all positive), while noise regions have random signs. The coherence penalty is defined as:

```
sign(r, θ) = sign(-G(r, θ))  # flipped so positive indicates boundary
coherence(r, θ) = local consistency of sign along angular direction
penalty(r, θ) = β · (1 - coherence(r, θ))
```

**Implementation**

A sliding window along the angular direction computes the standard deviation of gradient signs as a measure of inconsistency. `coherence_penalty_weight` (β) controls penalty strength; 0 disables it.

**Python packages**: `scipy.ndimage.generic_filter` (sliding window), `numpy.std`

#### 3f. Score and Cost

**Mathematical Principle**

```
pixel_dr = 2.0              # Sobel kernel effective radial distance (pixels)

score(r, θ) = G(r, θ)² / pixel_dr    (G > 0)
score(r, θ) = 0                       (G ≤ 0)

cost(r, θ) = -log( score(r,θ) / Σscore + ε )
```

- `G²` amplifies strong gradients and suppresses weak ones
- `/pixel_dr` normalizes to unit pixel distance
- Negative gradients (bright→dark) are zeroed out, as HII boundaries are dark→bright rising edges
- `-log` transforms score to cost: strong gradient → low cost
- `ε = 1e-10` prevents log(0)

**Python packages**: `numpy`

### Step 4: DP Optimal Path Search

**Mathematical Principle**

Find a closed path `p(θ)` of length 360 on the cost map `C(r, θ)` (size Nr × 360) minimizing total cost:

```
minimize  Σ_θ C(p(θ), θ) + α · Σ_θ (p(θ) - p(θ-1))²

where p(359) wraps back to p(0), forming a closed boundary
```

- First term: data fidelity (sum of boundary costs)
- Second term: smoothness penalty (squared radial jumps between adjacent angles)

**Implementation: Viterbi Algorithm**

Each angle θ is treated as a "stage" in dynamic programming, with Nr candidate states (radius index i). The transition cost between states is:

```
transition(i, k) = α · (i - k)²
```

Recurrence relation:

```
dp[θ][i] = C(i, θ) + min_k { dp[θ-1][k] + α · (i - k)² }
```

The final step (θ = 359) must return to the θ = 0 state, forming a closed loop. The full path is recovered via backtracking.

**Python packages**: `numpy` (pure NumPy implementation)

#### 4a. L-curve Auto Penalty

The smoothness penalty α cannot be determined a priori. The L-curve method selects it automatically:

1. Sample 30 α values uniformly in log space (`np.logspace(-6, 2, 30)`)
2. Run DP for each α, recording data fidelity f(α) and path roughness r(α)
3. Plot (log f, log r) in log-log coordinates; find the point of maximum curvature
4. The maximum curvature point represents the optimal trade-off between fidelity and smoothness

Curvature formula (discrete three-point method):

```
κ(i) = 2 · (x'y'' - y'x'') / (x'² + y'²)^(3/2)
```

where x = log f, y = log r, derivatives approximated by central differences.

**Python packages**: `numpy` (log sampling, curvature computation)

### Step 5: Stable Boundary Scan

**Mathematical Principle and Motivation**

DP requires an inner mask radius rmin (radius inside which gradients near the center are ignored), but rmin cannot be known a priori. Too small → noisy center interferes; too large → the boundary gets masked. The algorithm exploits a useful phenomenon: there exists a range of rmin values over which the detected boundary remains **essentially unchanged**.

**Implementation**

1. Sample `n_steps` rmin values uniformly in `[rmin_start, rmax_limit]`
   - `rmin_start = max(rmax × rmin_start_ratio, rmin_min_pixels)`
   - `rmax_limit = rmax × rmax_limit_ratio`
2. Run DP for each rmin, extract the boundary
3. Using the first rmin's boundary as reference, compute differences for all boundaries:
   ```
   diff[i] = mean(|boundary_i - boundary_ref|)
   Δdiff[i] = |diff[i] - diff[i-1]|
   ```
4. Find stable region: `stable_window` consecutive rmin points with `Δdiff < stable_threshold`
5. When multiple stable regions exist, weighted scoring selects the best:
   - `mean_score` (gradient score): 0.4 weight
   - `length` (stable region width): 0.3 weight
   - `std` (gradient direction consistency): 0.3 weight
6. Per-angle average of all boundaries within the stable region is the final output

**Python packages**: `numpy`, `scipy.ndimage.gaussian_filter1d`

### Step 6: Bootstrap Uncertainty Estimation

**Mathematical Principle**

Estimate per-angle standard deviation of boundary detection via parametric bootstrap:

1. Assume measurement errors follow a Gaussian distribution `N(0, σ²(x,y))`
2. Generate N noise realizations, each adding independent noise to the original image
3. Run full boundary detection on each realization
4. Per-angle standard deviation of N boundaries → 1σ uncertainty

```
uncertainty(θ) = std({ boundary_i(θ) | i = 1..N })
```

**Three Scenarios**

| Scenario | clean_image | error_map | Method |
|----------|:-----------:|:---------:|--------|
| A | ✓ | ✓ | `clean_image + N(0, σ²)` → detect on each noise-added image |
| B | ✗ | ✓ | `noisy_image + N(0, σ²)` → detect, √2 correction |
| C | ✗ | ✗ | Skip, return None |

Scenario B's √2 correction: the image already contains noise; adding more noise doubles the variance, so `σ_boundary = σ_bootstrap / √2`.

**Implementation**

- `multiprocessing.Pool` parallel execution (`n_workers` controls process count)
- MAD outlier rejection: iterations with `|r_i - median| > k · MAD` are flagged and removed (`k=3.0`)
- Unstable boundary filtering: unreasonable boundaries from bootstrap are rejected

**Python packages**: `numpy`, `multiprocessing`

---

## Elliptical Prior Detection

For bubbles with significant ellipticity, the elliptical prior constrains boundary search:

```python
from boundary_detection import detect_hii_boundary_elliptical

result = detect_hii_boundary_elliptical(
    data, xc=150, yc=150, a=80, b=60, phi=np.pi/6
)
```

Parameters: semi-major axis `a`, semi-minor axis `b`, position angle `phi` (radians). The algorithm is similar to circular detection but performs the polar transform in elliptical coordinates.

---

## Real Data Pipeline

For Spitzer GLIMPSE 8μm images of the Churchwell+ (2006) bubble catalog.

### End-to-End Flow

```
[1] fetch_bubble_catalog.py    → Vizier query Churchwell+ 2006 catalog → CSV
[2] download_glimpse_images.py → IRSA SIA query → download 8μm mosaic FITS
[3] preprocess_glimpse.py      → Bright source masking + à trous wavelet inpainting → cleaned.fits, sigma.fits
[4] test_real_bubbles.py       → Boundary detection + Bootstrap → result.json, arrays.npz
[5] test_report.py             → HTML report
```

Unified entry point: `python run_test_plan_real.py`

### Preprocessing: Point Source Removal

**Mathematical Principle: à trous Wavelet Transform**

B3 spline wavelet à trous (with holes) algorithm for 5-scale decomposition:

```
I = w1 + w2 + w3 + w4 + w5 + c5

where w_j = c_{j-1} - c_j  (wavelet planes)
      c_j = c_{j-1} ∗ h_j   (convolution with B3 spline kernel, dilated by 2^j)
      c_0 = I
```

Point sources are concentrated in w1 (smallest scale). Iterative inpainting replaces point-source pixels in the w1 plane with local median values.

**Implementation Steps**

1. **Bright source masking**: `DAOStarFinder` (`photutils`) detection with peak-based radius tiers
   - peak < 1000 → mask radius 5 px
   - 1000 ≤ peak < 2000 → mask radius 10 px
   - peak ≥ 2000 → mask radius 40 px
2. **Bright source filling**: Biharmonic interpolation (`scipy.interpolate`) or bg+noise filling
3. **Wavelet inpainting**: `watroo.AtrousTransform` (B3spline) 5-scale decomposition, `inpaint_iters` iterations
4. **Refine**: Residual image re-detection with DAOStarFinder + refill (`n_refine_iters` iterations)
5. **Sigma map**: Sliding MAD window (`mad_window` size) on w1 plane, sparse grid (`sigma_stride` step), `scipy.interpolate.griddata` interpolation to full image

**Python packages**: `watroo` (AtrousTransform/B3spline), `photutils` (DAOStarFinder), `astropy.io.fits`, `scipy.ndimage`, `scipy.interpolate.griddata`, `skimage.filters`, `skimage.morphology`

### Detection: Downsampling for Speed

The `cleanmap_downsamp_scale` parameter controls the downsampling factor:

```python
# Original image
I_down = gaussian_filter(cleaned, sigma=downsamp/2)[::downsamp, ::downsamp]
```

Gaussian anti-aliasing (σ = downsamp/2) followed by uniform sampling. Detection runs on the thumbnail; results are scaled back to original resolution by the `downsamp` factor.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `smoothing_fwhm` | 2.0 px | Radial Gaussian smoothing FWHM, 0=off |
| `cost_map_smoothing_sigma` | 5.0 px | Angular smoothing σ, 0=off |
| `gradient_smoothing_sigma` | 0.0 px | Post-Sobel angular smoothing σ, 0=off |
| `boundary_smoothing_sigma` | 0.0 px | Final boundary smoothing σ, 0=off |
| `rmin_start_ratio` | 0.05 | rmin start = max(rmax × ratio, rmin_min_pixels) |
| `rmin_min_pixels` | 5.0 px | Hard rmin lower bound |
| `rmax_limit_ratio` | 0.7 | rmin upper bound = rmax × ratio |
| `n_steps` | 0 | Number of rmin scan points (0=auto, step ≈ 2 px) |
| `stable_window` | 3 | Consecutive points for stable region |
| `stable_threshold` | 1 | Δdiff stability threshold |
| `contrast_min` | 0.05 | Boundary contrast threshold, 0=off |
| `coherence_penalty_weight` | 0.5 | Gradient sign coherence penalty β, 0=off |
| `angular_snr_weighting` | false | Angular SNR weighting |
| `n_bootstrap` | 100 | Bootstrap iterations, 0=skip |
| `n_workers` | 1 | Number of parallel processes |
| `detect_rising_edge` | true | true=dark→bright (rising edge) |

Full parameter documentation in `hii_detection_config.yaml`. Priority: **explicit argument > config file > built-in defaults**.

---

## Anti-Noise Mechanisms (Application Order)

1. `error_map` weighting — `1/σ²` suppresses high-noise pixels
2. `angular_snr_weighting` — angular structure SNR modulates score
3. `gradient_smoothing_sigma` — angular gradient smoothing
4. `coherence_penalty_weight` — gradient sign coherence penalty
5. `rmin_min_pixels` — hard lower bound prevents DP shortcut through noisy center
6. `boundary_smoothing_sigma` — final boundary smoothing
7. `contrast_min` — boundary contrast filter

---

## Running Tests

```bash
# Synthetic model tests
python run_test_plan.py                          # All 4 models × 4 test types
python run_test_plan.py --model "Sigmoid"        # Single model
python run_test_plan.py --bootstrap-n 0          # Skip bootstrap (fastest)

# Real data
python run_test_plan_real.py                     # Full pipeline
python run_test_plan_real.py --only-plot         # Re-plot only

# Quick verification after packaging
python quick_test.py                             # Sigmoid + Gaussian + Real
```

---

## Package Structure

```
boundary_detection/                # Core package
├── detect_hii_boundary.py         # Unified entry point + parameter resolution
├── detect_hii_boundary_elliptical.py  # Elliptical prior
├── find_stable_boundary.py        # Stable boundary dispatcher
├── find_stable_boundary_by_scan.py  # rmin scan
├── extract_circle_boundary.py     # Polar → gradient → cost → DP
├── find_boundary_dp.py            # Viterbi DP + L-curve
├── find_global_optimal_boundary.py  # Polar transform
├── bootstrap_boundary.py          # Bootstrap uncertainty
├── hii_detection_config.yaml      # Algorithm parameters
└── data/                          # Bundled data (catalog + FITS)

test/                              # Test suite
├── test_models.py                 # 4 synthetic model generators
├── test_runner.py                 # baseline/noise/center/rmax
├── test_diagnostics.py            # Diagnostic plots (overlay/pipeline/error)
├── test_report.py                 # MD + HTML report
├── run_test_plan.py               # Synthetic test orchestrator
├── run_test_plan_real.py          # Real data pipeline orchestrator
├── preprocess_glimpse.py          # à trous wavelet point source removal
├── fetch_bubble_catalog.py        # Vizier catalog query
├── download_glimpse_images.py     # IRSA SIA download
├── test_real_bubbles.py           # Real image detection
└── quick_test.py                  # Quick smoke test
```
