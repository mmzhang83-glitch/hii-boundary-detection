"""诊断测试：对比度与梯度集中度对噪声路径的判别能力

Gaussian Ring sigma=10, rmax=150, bootstrap=100 iterations.
使用干净图像计算对比度，排除噪声干扰。
"""
import numpy as np
import yaml, sys
from pathlib import Path
from scipy.ndimage import sobel

sys.path.insert(0, str(Path(__file__).parent))

from boundary_detection.detect_hii_boundary import detect_hii_boundary
from test_models import make_gaussian_ring
from test_runner import _shift_image
from boundary_detection.find_global_optimal_boundary import manual_polar_transform


def run_test():
    print("=" * 70)
    print("Contrast & Gradient Ratio Diagnostic")
    print("Model: Gaussian Ring sigma=10, rmax=150")
    print("=" * 70)

    with open('test_config.yaml') as f:
        config = yaml.safe_load(f)
    config['shape'] = tuple(config['shape'])
    shape = tuple(config['shape'])

    m = make_gaussian_ring(shape, config, sigma=10)
    rmax = 150
    noise_level = 0.30
    noise_sigma = noise_level * config.get('ring_b_max', 1.0)
    expected_r = config.get('ring_r0', 60.0)
    n_angles = 360
    n_radii = int(np.ceil(rmax)) + 1

    # Precompute clean image polar (once)
    print("\nPrecomputing clean polar image ...")
    polar_clean = manual_polar_transform(
        m.clean_image, (m.xc, m.yc),
        (n_radii, n_angles), (0, rmax),
    )
    dv_clean = sobel(polar_clean, axis=0, mode='mirror')
    dv_abs = np.abs(dv_clean)
    global_grad_mean = float(np.mean(dv_abs))
    signal_amp = config.get('ring_b_max', 1.0)
    print(f"Global mean gradient: {global_grad_mean:.6f}")

    # Run bootstrap
    rng = np.random.default_rng(42)
    noisy = m.clean_image + rng.normal(0, noise_sigma, shape)
    error_map = np.full(shape, noise_sigma)

    print("\nRunning bootstrap x 100 ...")
    result = detect_hii_boundary(
        data=_shift_image(noisy), xc=m.xc, yc=m.yc, rmax=rmax,
        error_map=error_map, clean_image=m.clean_image,
        n_bootstrap=100, seed=42,
    )

    bb = result.get('bootstrap_boundaries')
    if bb is None:
        print("ERROR: no bootstrap results")
        return

    n_iter = bb.shape[0]
    exp_radii = np.full(n_angles, expected_r, dtype=float)
    r_grid = np.linspace(0, rmax, n_radii)

    # Compute metrics per iteration
    mre_all = np.zeros(n_iter)
    contrast_all = np.zeros(n_iter)
    grad_ratio_all = np.zeros(n_iter)
    strip = 5

    print(f"Computing metrics for {n_iter} iterations ...")
    for i in range(n_iter):
        radii = bb[i]
        mre_all[i] = float(np.mean(np.abs(radii - exp_radii)))

        idx_per_angle = np.clip(
            np.interp(radii, r_grid, np.arange(n_radii)).astype(int),
            0, n_radii - 1,
        )

        # Metric 1: Contrast = outer - inner (per angle, median)
        diffs = np.zeros(n_angles)
        for j in range(n_angles):
            bi = idx_per_angle[j]
            s_start = max(0, bi - strip)
            s_end = min(n_radii, bi + strip + 1)
            if s_end - s_start < 2:
                continue
            mid = (s_start + s_end) // 2
            inner = np.mean(polar_clean[s_start:mid, j]) if mid > s_start else 0
            outer = np.mean(polar_clean[mid:s_end, j]) if s_end > mid else 0
            diffs[j] = outer - inner
        contrast_all[i] = float(np.median(np.abs(diffs))) / signal_amp

        # Metric 2: Gradient ratio = boundary |dv| / global |dv|
        grads = np.zeros(n_angles)
        for j in range(n_angles):
            bi = idx_per_angle[j]
            g_start = max(0, bi - strip)
            g_end = min(n_radii, bi + strip + 1)
            if g_end > g_start:
                grads[j] = float(np.mean(dv_abs[g_start:g_end, j]))
        grad_ratio_all[i] = float(np.median(grads) / global_grad_mean)

    # Analysis
    good = mre_all < 10
    bad = ~good

    print(f"\n{'='*70}")
    print(f"Classification: GOOD(MRE<10):{good.sum()}  BAD(MRE>=10):{bad.sum()}")
    print(f"  GOOD mean MRE = {mre_all[good].mean():.2f}")
    print(f"  BAD  mean MRE = {mre_all[bad].mean():.1f}")

    # Top 25 / Bottom 25
    order = np.argsort(mre_all)
    print(f"\n{'Rk':>3} {'Iter':>5} {'MRE':>8} {'Contrast':>10} {'GradRatio':>11} {'Class':>6}")
    print("-" * 50)
    for rank in range(25):
        idx = order[rank]
        cls = "GOOD" if good[idx] else "BAD"
        print(f"{rank+1:>3} {idx:>5} {mre_all[idx]:>8.2f} {contrast_all[idx]:>10.5f} "
              f"{grad_ratio_all[idx]:>11.4f} {cls:>6}")
    print("  ...")
    for rank in range(max(0, n_iter-25), n_iter):
        idx = order[rank]
        cls = "GOOD" if good[idx] else "BAD"
        print(f"{rank+1:>3} {idx:>5} {mre_all[idx]:>8.2f} {contrast_all[idx]:>10.5f} "
              f"{grad_ratio_all[idx]:>11.4f} {cls:>6}")

    # Stats
    print(f"\n{'='*70}")
    print("Mean metrics:")
    cg, cb = contrast_all[good].mean(), contrast_all[bad].mean()
    gg, gb = grad_ratio_all[good].mean(), grad_ratio_all[bad].mean()
    print(f"  Contrast:  GOOD={cg:.5f}  BAD={cb:.5f}  ratio=GOOD/BAD={cg/cb:.1f}x" if cb > 0 else f"  Contrast:  GOOD={cg:.5f}  BAD={cb:.5f}")
    print(f"  GradRatio: GOOD={gg:.4f}  BAD={gb:.4f}  ratio=GOOD/BAD={gg/gb:.1f}x" if gb > 0 else f"  GradRatio: GOOD={gg:.4f}  BAD={gb:.4f}")

    # Binning
    print("\nContrast bins:")
    for lo, hi, label in [(0, 0.02, '~0'), (0.02, 0.05, '.02-.05'), (0.05, 0.1, '.05-.1'),
                           (0.1, 0.5, '.1-.5'), (0.5, 100, '>.5')]:
        mask = (contrast_all > lo) & (contrast_all <= hi)
        if mask.sum() > 0:
            print(f"  [{label:>10}]: {mask.sum():3d}  MRE={mre_all[mask].mean():.1f}")

    print("\nGradRatio bins:")
    for lo, hi, label in [(0, 0.5, '<0.5'), (0.5, 1, '0.5-1'), (1, 2, '1-2'),
                           (2, 5, '2-5'), (5, 100, '>5')]:
        mask = (grad_ratio_all > lo) & (grad_ratio_all <= hi)
        if mask.sum() > 0:
            print(f"  [{label:>8}]: {mask.sum():3d}  MRE={mre_all[mask].mean():.1f}")

    # Best thresholds
    best_sep, best_c = 0, 0
    for t in np.linspace(0, np.percentile(contrast_all, 95), 50):
        pred = contrast_all >= t
        if pred.sum() > 0 and (~pred).sum() > 0:
            sep = mre_all[~pred].mean() - mre_all[pred].mean()
            if sep > best_sep:
                best_sep, best_c = sep, t
    print(f"\nBest Contrast threshold: {best_c:.5f}")
    pred = contrast_all >= best_c
    print(f"  >= {best_c:.5f}: {pred.sum()} iters, MRE={mre_all[pred].mean():.1f}")
    print(f"  <  {best_c:.5f}: {(~pred).sum()} iters, MRE={mre_all[~pred].mean():.1f}")

    best_sep, best_g = 0, 0
    for t in np.linspace(0.5, np.percentile(grad_ratio_all, 95), 50):
        pred = grad_ratio_all >= t
        if pred.sum() > 0 and (~pred).sum() > 0:
            sep = mre_all[~pred].mean() - mre_all[pred].mean()
            if sep > best_sep:
                best_sep, best_g = sep, t
    print(f"\nBest GradRatio threshold: {best_g:.4f}")
    pred = grad_ratio_all >= best_g
    print(f"  >= {best_g:.4f}: {pred.sum()} iters, MRE={mre_all[pred].mean():.1f}")
    print(f"  <  {best_g:.4f}: {(~pred).sum()} iters, MRE={mre_all[~pred].mean():.1f}")

    print("\nDone.")


if __name__ == '__main__':
    run_test()
