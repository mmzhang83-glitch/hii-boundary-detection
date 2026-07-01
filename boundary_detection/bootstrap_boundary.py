"""Bootstrap resampling for boundary uncertainty estimation.

Provides per-angle 1σ uncertainty by generating N noise realizations,
re-running boundary detection on each, and computing the standard
deviation of detected radii across realizations.
"""

import numpy as np
from typing import Optional, Dict
import logging
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger("hii_boundary.bootstrap")


def remove_outliers_mad(data: np.ndarray, k: float = 3.0) -> np.ndarray:
    """使用 MAD 方法去除 outlier，返回逐行布尔掩码。

    对每行计算其与整体中位数的偏差，使用 MAD 检测 outlier 行。

    Parameters
    ----------
    data : np.ndarray
        2D 数组，形状 (n_iterations, n_angles)
    k : float
        MAD 阈值，默认 3.0

    Returns
    -------
    np.ndarray
        1D 布尔掩码，形状 (n_iterations,)，True 表示保留，False 表示 outlier
    """
    # 计算每行的中位数（聚合为行级别指标）
    row_medians = np.median(data, axis=1)

    # 计算整体中位数和 MAD
    global_median = np.median(row_medians)
    mad = np.median(np.abs(row_medians - global_median))

    # 计算修正 z-score
    epsilon = 1e-10  # 避免除零
    modified_z = 0.6745 * (row_medians - global_median) / (mad + epsilon)

    # 返回掩码：|z| <= k 为保留
    return np.abs(modified_z) <= k


# 子进程全局变量，由 _init_pool 设置
_shared_data = {}


def _init_pool(data: np.ndarray, error_map: np.ndarray, algo_params: dict,
               clean_image: Optional[np.ndarray] = None):
    """子进程初始化：接收大数组作为全局变量。

    在 Pool 创建时调用一次，大数组通过此函数传递给子进程，
    避免每个任务都序列化大数组。
    """
    global _shared_data
    _shared_data = {
        'data': data,
        'error_map': error_map,
        'algo_params': algo_params,
    }
    if clean_image is not None:
        _shared_data['clean_image'] = clean_image


def _run_one_bootstrap(args: tuple) -> tuple:
    """单次 bootstrap 迭代（顶层函数，支持 pickle）。

    从 _shared_data 全局变量获取大数组，只接收小参数。

    Parameters
    ----------
    args : tuple
        (i, xc, yc, rmax, base_image_type, seed)
        - i: 迭代索引
        - xc, yc: 中心坐标
        - rmax: 最大搜索半径
        - base_image_type: 'clean' 或 'noisy'
        - seed: 随机种子

    Returns
    -------
    tuple : (i, boundary_radii)
    """
    from .find_stable_boundary import find_stable_boundary

    i, xc, yc, rmax, base_image_type, seed = args

    # 从全局变量获取大数组
    data = _shared_data['data']
    error_map = _shared_data['error_map']
    algo_params = _shared_data['algo_params']

    # 根据场景选择基础图像
    if base_image_type == 'clean':
        base_image = _shared_data.get('clean_image', data)
    else:
        base_image = data

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, error_map)
    boot_image = base_image + noise

    result = find_stable_boundary(
        data=boot_image, xc=xc, yc=yc, rmax=rmax,
        error_map=error_map, **algo_params,
    )
    return i, result['boundary_radii'], result.get('has_stable_region', True)


def bootstrap_boundary_uncertainty(
    data: np.ndarray,
    xc: float,
    yc: float,
    rmax: float,
    *,
    n_bootstrap: int = 100,
    n_workers: int = 1,  # 进程数，1=单进程（默认），>1=多进程并行
    error_map: Optional[np.ndarray] = None,
    clean_image: Optional[np.ndarray] = None,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """Estimate per-angle boundary uncertainty via parametric bootstrap.

    Parameters
    ----------
    data : np.ndarray
        The current image (may be noisy or clean).
    xc, yc : float
        Center coordinates (pixels).
    rmax : float
        Maximum search radius.
    n_bootstrap : int
        Number of bootstrap iterations (default 100).
    n_workers : int
        Number of parallel worker processes.  1 (default) = sequential;
        >1 = multiprocessing.Pool with that many workers.
    error_map : np.ndarray or None
        Per-pixel 1σ measurement uncertainty.  If None, scenario C
        (no uncertainty estimation possible).
    clean_image : np.ndarray or None
        Noise-free reference image.  If provided, scenario A:
        bootstrap adds fresh noise to the clean image directly.
        If None and error_map is not None, scenario B:
        bootstrap adds noise on top of the already-noisy data,
        and a √2 correction is applied.
    seed : int
        Random seed for reproducibility.
    **kwargs
        Forwarded to ``find_stable_boundary``.

    Returns
    -------
    dict with keys:
        boundary_radii : (360,) ndarray — boundary from original image
        boundary_angles : (360,) ndarray — angle array in radians
        boundary_uncertainty : (360,) ndarray or None — 1σ per angle
        bootstrap_boundaries : (N, 360) ndarray — all N boundaries
        scenario : str — 'A', 'B', or 'C'
        correction_applied : bool
        mean_uncertainty : float or None
        max_uncertainty : float or None
    """
    from .find_stable_boundary import find_stable_boundary

    # Detect scenario
    if error_map is None:
        scenario = 'C'
        correction = 1.0
    elif clean_image is not None:
        scenario = 'A'
        correction = 1.0
    else:
        scenario = 'B'
        correction = 1.0 / np.sqrt(2.0)

    # Detect boundary from the original image
    orig_result = find_stable_boundary(
        data=data, xc=xc, yc=yc, rmax=rmax,
        error_map=error_map, **kwargs,
    )
    boundary_radii = orig_result['boundary_radii']
    boundary_angles = orig_result['boundary_angles']

    if scenario == 'C':
        logger.info("Bootstrap: scenario C (no error_map), skipping")
        return {
            'boundary_radii': boundary_radii,
            'boundary_angles': boundary_angles,
            'boundary_uncertainty': None,
            'bootstrap_boundaries': np.empty((0, len(boundary_radii))),
            'scenario': 'C',
            'correction_applied': False,
            'mean_uncertainty': None,
            'max_uncertainty': None,
        }

    # Run bootstrap
    n_angles = len(boundary_radii)
    all_boundaries = np.empty((n_bootstrap, n_angles))
    all_has_stable = np.ones(n_bootstrap, dtype=bool)

    base_image_type = 'clean' if scenario == 'A' else 'noisy'

    # 准备算法参数（只包含 kwargs 中实际传入的参数，None 不覆盖默认值）
    # 注意：outlier_removal 和 outlier_k 不在此列表中，它们只在 detect_hii_boundary.py 中使用
    _algo_keys = [
        'method', 'smoothing_fwhm', 'cost_map_smoothing_sigma',
        'gradient_smoothing_sigma', 'boundary_smoothing_sigma',
        'rmin_start_ratio', 'rmin_min_pixels', 'rmax_limit_ratio',
        'angular_snr_weighting', 'angular_snr_sigma',
        'coherence_penalty_weight', 'coherence_sigma',
        'detect_rising_edge',
        'n_steps', 'stable_window', 'stable_threshold', 'fallback',
    ]
    algo_params = {k: v for k in _algo_keys
                   if (v := kwargs.get(k)) is not None}

    # 构建任务参数（只包含小参数，不包含大数组）
    task_args = [
        (i, xc, yc, rmax, base_image_type, seed + i)
        for i in range(n_bootstrap)
    ]

    logger.info("Bootstrap: %d iterations, scenario %s, n_workers=%d",
                n_bootstrap, scenario, n_workers)

    if n_workers <= 1:
        # 单进程：直接循环（保持向后兼容）
        _init_pool(data, error_map, algo_params, clean_image=clean_image)
        try:
            for args in tqdm(task_args, desc="Bootstrap"):
                i, boundary, has_stable = _run_one_bootstrap(args)
                all_boundaries[i] = boundary
                all_has_stable[i] = has_stable
        finally:
            global _shared_data
            _shared_data = {}
    else:
        try:
            with mp.Pool(
                processes=n_workers,
                initializer=_init_pool,
                initargs=(data, error_map, algo_params, clean_image)
            ) as pool:
                for i, boundary, has_stable in tqdm(
                    pool.imap(_run_one_bootstrap, task_args),
                    total=n_bootstrap,
                    desc="Bootstrap"
                ):
                    all_boundaries[i] = boundary
                    all_has_stable[i] = has_stable
        except Exception as e:
            logger.error("Bootstrap parallel failed: %s", e)
            raise

    # 筛选有稳定区域的 bootstrap 结果
    n_valid = int(np.sum(all_has_stable))
    n_filtered = n_bootstrap - n_valid
    valid_boundaries = all_boundaries[all_has_stable]

    if n_filtered > 0:
        logger.info("Bootstrap: filtered %d/%d iterations without stable region",
                    n_filtered, n_bootstrap)

    if n_valid == 0:
        # 所有迭代都没有稳定区域，使用全部结果（降级处理）
        logger.warning("Bootstrap: no iteration has stable region, using all results")
        valid_boundaries = all_boundaries
        n_valid = n_bootstrap

    sigma_raw = np.std(valid_boundaries, axis=0)
    sigma = sigma_raw * correction

    mean_sigma = float(np.mean(sigma))
    max_sigma = float(np.max(sigma))

    logger.info("Bootstrap: done — %d valid iterations, mean σ=%.2f px, max σ=%.2f px (scenario %s)",
                  n_valid, mean_sigma, max_sigma, scenario)

    return {
        'boundary_radii': boundary_radii,
        'boundary_angles': boundary_angles,
        'boundary_uncertainty': sigma,
        'bootstrap_boundaries': all_boundaries,
        'bootstrap_has_stable': all_has_stable,
        'n_valid_bootstrap': n_valid,
        'n_filtered_bootstrap': n_filtered,
        'scenario': scenario,
        'correction_applied': correction != 1.0,
        'mean_uncertainty': mean_sigma,
        'max_uncertainty': max_sigma,
    }
