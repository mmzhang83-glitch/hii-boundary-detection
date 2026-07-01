"""诊断 GR sigma=10 rmax=150 bootstrap 边界和 contrast 效果

用法: python test_contrast_deep_diagnostic.py
"""
import numpy as np, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from boundary_detection.detect_hii_boundary import detect_hii_boundary
from test_models import make_gaussian_ring
from test_runner import _shift_image, _compute_expected_boundary
from test_generators import gaussian_ring
from boundary_detection.find_global_optimal_boundary import manual_polar_transform
from scipy.ndimage import gaussian_filter1d


def compute_contrast_on_image(polar_img, path_r_indices, strip=4, rising=True):
    """复现 extract_circle_boundary 的 contrast 计算"""
    n_angles = len(path_r_indices)
    per = np.zeros(n_angles)
    for j in range(n_angles):
        bi = int(path_r_indices[j])
        ss = max(0, bi - strip)
        se = min(polar_img.shape[0], bi + strip + 1)
        if se - ss < 2:
            continue
        im = np.mean(polar_img[ss:bi, j]) if bi > ss else 0.0
        om = np.mean(polar_img[bi:se, j]) if se > bi else 0.0
        per[j] = om - im if rising else im - om
    return float(np.median(per))


def run():
    with open('test_config.yaml') as f:
        config = yaml.safe_load(f)
    shape = tuple(config['shape'])

    m = make_gaussian_ring(shape, config, sigma=10)
    rmax = 150
    noise_level = 0.30
    noise_sigma = noise_level * config.get('ring_b_max', 1.0)
    n_angles = 360
    n_radii = int(np.ceil(rmax)) + 1
    r_grid = np.linspace(0, rmax, n_radii)
    sigma_smooth = 2.0 / 2.355

    # 正确的期望边界
    exp_b = _compute_expected_boundary(m, config)
    expected_r = exp_b.radius
    print("=" * 70)
    print(f"GR sigma=10, rmax=150, noise=30%")
    print(f"Expected boundary (max rising gradient): r = {expected_r:.1f}")
    print(f"Model.expected_radius (ring peak): r = {m.expected_radius:.1f}")
    print(f"  → Δ = {m.expected_radius - expected_r:.1f} px (rising edge vs peak)")
    print("=" * 70)

    # 干净 polar
    polar_clean = manual_polar_transform(m.clean_image, (m.xc, m.yc), (n_radii, n_angles), (0, rmax))
    polar_clean_s = gaussian_filter1d(polar_clean, sigma=sigma_smooth, axis=0, mode='reflect')

    # Bootstrap
    rng = np.random.default_rng(43)
    noisy = m.clean_image + rng.normal(0, noise_sigma, shape)
    error_map = np.full(shape, noise_sigma)

    result = detect_hii_boundary(
        data=_shift_image(noisy), xc=m.xc, yc=m.yc, rmax=rmax,
        error_map=error_map, clean_image=m.clean_image,
        n_bootstrap=50, seed=43,
    )

    bb = result.get('bootstrap_boundaries')
    n_iter = bb.shape[0]

    # 初始 noisy 图（bootstrap 前）的 polar → 模拟管线实际用的 contrast
    polar_noisy = manual_polar_transform(noisy, (m.xc, m.yc), (n_radii, n_angles), (0, rmax))
    polar_noisy_s = gaussian_filter1d(polar_noisy, sigma=sigma_smooth, axis=0, mode='reflect')

    # 每个迭代的指标
    mre_vs_expected = np.zeros(n_iter)
    mre_vs_peak = np.zeros(n_iter)
    mean_r = np.zeros(n_iter)
    c_noisy = np.zeros(n_iter)
    c_clean = np.zeros(n_iter)
    exp_arr = np.full(n_angles, expected_r)
    peak_arr = np.full(n_angles, m.expected_radius)

    for i in range(n_iter):
        radii = bb[i]
        mean_r[i] = float(np.mean(radii))
        mre_vs_expected[i] = float(np.mean(np.abs(radii - exp_arr)))
        mre_vs_peak[i] = float(np.mean(np.abs(radii - peak_arr)))
        idx = np.clip(np.interp(radii, r_grid, np.arange(n_radii)).astype(int), 0, n_radii - 1)
        # noisy contrast：初始 noisy 图 polar（近似管线 bootstrap 中的实际对比度）
        c_noisy[i] = compute_contrast_on_image(polar_noisy_s, idx, strip=4, rising=True)
        # clean contrast：干净图 polar（理想情况）
        c_clean[i] = compute_contrast_on_image(polar_clean_s, idx, strip=4, rising=True)

    good = mre_vs_expected < 5.0
    bad = ~good
    catastrophic = mean_r > 80

    print(f"\n--- 分类 (MRE vs expected_r={expected_r:.1f}) ---")
    print(f"  GOOD (MRE<5): {good.sum():3d}   mean MRE = {mre_vs_expected[good].mean():.2f}")
    print(f"  BAD  (MRE>=5): {bad.sum():3d}   mean MRE = {mre_vs_expected[bad].mean():.2f}")
    print(f"  Catastrophic (mean_r>80): {catastrophic.sum()}")

    print(f"\n--- Contrast 统计 ---")
    print(f"  GOOD: noisy_contrast mean={c_noisy[good].mean():.4f}  "
          f"min={c_noisy[good].min():.4f}  max={c_noisy[good].max():.4f}")
    print(f"  BAD:  noisy_contrast mean={c_noisy[bad].mean():.4f}  "
          f"min={c_noisy[bad].min():.4f}  max={c_noisy[bad].max():.4f}")

    print(f"\n--- 最差迭代 ---")
    order = np.argsort(mre_vs_expected)[::-1]
    for rank in range(min(15, n_iter)):
        idx = order[rank]
        tag = "CATASTROPHIC" if mean_r[idx] > 80 else ("GOOD" if good[idx] else "BAD ")
        print(f"  #{rank+1:2d} iter={idx:3d} MRE={mre_vs_expected[idx]:6.1f} "
              f"mean_r={mean_r[idx]:6.1f} contrast={c_noisy[idx]:.5f} [{tag}]")

    print(f"\n--- 最好迭代 ---")
    for rank in range(min(10, n_iter)):
        idx = order[n_iter - 1 - rank]
        print(f"  #{rank+1:2d} iter={idx:3d} MRE={mre_vs_expected[idx]:6.1f} "
              f"mean_r={mean_r[idx]:6.1f} contrast={c_noisy[idx]:.5f} [GOOD]")

    # Contrast 阈值分析
    print(f"\n--- contrast_min 阈值效果 ---")
    for thresh in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        keep = c_noisy >= thresh
        n_good_keep = (keep & good).sum()
        n_bad_keep = (keep & bad).sum()
        n_bad_rej = ((~keep) & bad).sum()
        n_good_rej = ((~keep) & good).sum()
        print(f"  thresh={thresh:.3f}: keep={keep.sum():3d} "
              f"(good_keep={n_good_keep:2d} bad_keep={n_bad_keep:2d}) "
              f"rej_bad={n_bad_rej:2d} rej_good={n_good_rej:2d}")

    # 管线输出
    print(f"\n--- 管线最终结果 ---")
    print(f"  boundary mean_r = {float(np.mean(result['boundary_radii'])):.1f}")
    print(f"  MRE vs expected={expected_r:.1f}: {float(np.mean(np.abs(result['boundary_radii'] - exp_arr))):.2f}")
    print(f"  uncertainty = {result.get('mean_uncertainty', 'N/A')}")
    print(f"  scenario = {result.get('scenario')}")

    sri = result.get('stable_regions_info', [])
    for s in sri[:3]:
        print(f"  stable_region: rmin={s.get('rmin','?'):.1f} "
              f"mean_r={s.get('mean_radius','?')} contrast={s.get('boundary_contrast','?')}")

    print("\nDone.")


if __name__ == '__main__':
    run()
