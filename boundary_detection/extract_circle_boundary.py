"""
从FITS图像提取圆内像素边界模块

功能：输入一张图像，找到指定圆内像素值分布的边界，生成诊断图
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy.ndimage import gaussian_filter1d, sobel
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as pixel_scale
from astropy.visualization import AsymmetricPercentileInterval
from matplotlib.colors import PowerNorm

from .find_global_optimal_boundary import manual_polar_transform
import logging

logger = logging.getLogger("hii_boundary.extract")


def generate_test_ring_image(
    shape: Tuple[int, int],
    r_inner: float,
    r_outer: float,
    xc: float,
    yc: float,
    inner_value: float = 0.0,
    ring_value: float = 1.0,
    outer_value: float = 0.0,
    add_noise: float = 0.0
) -> np.ndarray:
    """
    生成环形测试图像
    
    参数:
        shape: 图像形状 (height, width)
        r_inner: 内半径
        r_outer: 外半径  
        xc, yc: 圆心坐标
        inner_value: 内圆区域的值
        ring_value: 圆环区域的值
        outer_value: 外圆区域的值
        add_noise: 高斯噪声标准差
    
    返回:
        2D numpy数组
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    
    img = np.zeros(shape)
    img[r <= r_inner] = inner_value
    img[(r > r_inner) & (r <= r_outer)] = ring_value
    img[r > r_outer] = outer_value
    
    if add_noise > 0:
        img = img + np.random.normal(0, add_noise, shape)
    
    return img


def generate_gaussian_ring_image(
    shape: Tuple[int, int],
    r_center: float,
    sigma: float,
    xc: float,
    yc: float,
    max_value: float = 1.0,
    add_noise: float = 0.0
) -> np.ndarray:
    """
    生成高斯圆环测试图像（沿径向呈高斯分布）
    
    参数:
        shape: 图像形状 (height, width)
        r_center: 高斯分布的中心半径
        sigma: 高斯分布的标准差
        xc, yc: 圆心坐标
        max_value: 高斯峰值
        add_noise: 高斯噪声标准差
    
    返回:
        2D numpy数组
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    
    # 高斯分布
    img = max_value * np.exp(-((r - r_center)**2) / (2 * sigma**2))
    
    if add_noise > 0:
        img = img + np.random.normal(0, add_noise, shape)
    
    return img


def extract_circle_boundary(
    data: np.ndarray,
    xc: float,
    yc: float,
    rmax: float,
    rmin_scale: float = 2.0,
    angular_snr_weighting: bool = False,
    angular_snr_sigma: float = 3.0,
    coherence_penalty_weight: float = 0.0,
    coherence_sigma: float = 3.0,
    smoothing_fwhm: float = 2.0,
    cost_map_smoothing_sigma: float = 5.0,
    gradient_smoothing_sigma: float = 0.0,
    error_map: Optional[np.ndarray] = None,
    error_weighting: bool = True,
    showfile: Optional[Path] = None,
    cached_polar: Optional[np.ndarray] = None,  # 新增：缓存的极坐标图像
    detect_rising_edge: bool = True,
    ellipse: Optional[tuple] = None,
    contrast_strip_width: int = 4,
) -> Dict:
    """
    从图像提取圆内像素值分布的边界
    
    参数:
        data: 2D图像数据
        xc, yc: 圆心坐标（像素）
        rmax: 最大搜索半径
        rmin_scale: rmin = rmax / rmin_scale
        angular_snr_weighting: 是否启用角度方向结构SNR加权抑制噪声，默认为False
        angular_snr_sigma: 角度SNR加权的平滑sigma（像素），默认为3.0
        coherence_penalty_weight: 梯度符号一致性惩罚强度 β。0=禁用，建议0.3-1.0
        coherence_sigma: 符号一致性的角度平滑sigma（度），默认为3.0
        smoothing_fwhm: 径向平滑FWHM，设为0则不做平滑
        cost_map_smoothing_sigma: 角度平滑sigma，设为0则不做平滑
        gradient_smoothing_sigma: Sobel梯度后平滑sigma，设为0则不做平滑
        error_map: 误差图（与data同形状），用于误差加权梯度计算
        error_weighting: 是否启用误差加权，默认为True
        showfile: 诊断图保存路径
        cached_polar: 缓存的极坐标图像（shape: (num_radii_samples, 360)），
                      非None时跳过极坐标变换直接使用缓存结果
        detect_rising_edge: 是否检测上升沿（True=正梯度，False=负梯度），默认True
    
    返回:
        包含边界信息的字典
    """
    from .find_boundary_dp import find_boundary_dp
    
    # 预处理：零值转为NaN
    data = data.copy()
    data[data == 0] = np.nan
    
    # 极坐标变换
    rmin = rmax / rmin_scale
    _radius_range = (0.0, 1.0) if ellipse is not None else (0, rmax)

    if cached_polar is not None:
        expected_samples = int(np.ceil(rmax)) + 1
        if cached_polar.shape[0] != expected_samples:
            raise ValueError(
                f"cached_polar shape[0]={cached_polar.shape[0]} does not match "
                f"expected {expected_samples} for rmax={rmax}"
            )
        polar_image = cached_polar
        num_radii_samples = polar_image.shape[0]
    else:
        num_radii_samples = int(np.ceil(rmax)) + 1
        polar_image = manual_polar_transform(
            data=data,
            center=(xc, yc),
            output_shape=(num_radii_samples, 360),
            radius_range=_radius_range,
            ellipse=ellipse,
        )

    # ROI提取
    if ellipse is not None:
        f_min = 1.0 / rmin_scale if rmin_scale > 0 else 0.0
        r_min_idx = int(np.floor(f_min * num_radii_samples))
        rmin = f_min * rmax  # approximate for display
    else:
        rmin = rmax / rmin_scale
        r_min_idx = int(np.floor(rmin))
    polar_roi = polar_image[r_min_idx:, :]
    rr_roi = np.arange(r_min_idx, num_radii_samples)
    
    # 平滑处理
    if smoothing_fwhm > 0:
        sigma_radial = smoothing_fwhm / 2.355
        polar_smooth_r = gaussian_filter1d(
            polar_roi.astype(float),
            sigma=sigma_radial,
            axis=0,
            mode='mirror'
        )
    else:
        polar_smooth_r = polar_roi.astype(float)
    
    if cost_map_smoothing_sigma > 0:
        sigma_angular = cost_map_smoothing_sigma / 2.355
        polar_smooth = gaussian_filter1d(
            polar_smooth_r,
            sigma=sigma_angular,
            axis=1,
            mode='wrap'
        )
    else:
        polar_smooth = polar_smooth_r
    
    # 计算梯度分数图（支持误差加权）
    from scipy.ndimage import sobel
    
    # 计算原始梯度
    dv_map_raw = sobel(polar_smooth, axis=0, mode='mirror')
    
    # 误差加权处理
    if error_map is not None and error_weighting:
        # 将误差图也转换到极坐标
        polar_error = manual_polar_transform(
            data=error_map,
            center=(xc, yc),
            output_shape=(num_radii_samples, 360),
            radius_range=_radius_range,
            ellipse=ellipse,
        )
        # 提取ROI
        polar_error_roi = polar_error[r_min_idx:, :]
        
        # 误差加权：误差越大，权重越低
        # 权重 = 1 / (error^2 + epsilon)，避免除零
        epsilon = 1e-8
        error_weight = 1.0 / (polar_error_roi**2 + epsilon)
        
        # 误差传播：对梯度乘以权重
        # 高误差区域会被抑制
        dv_map = dv_map_raw * error_weight
        
        # 记录最大权重位置用于诊断
        max_weight_loc = np.unravel_index(np.argmax(error_weight), error_weight.shape)
        max_weight_value = error_weight[max_weight_loc]
    else:
        # 无误差图时使用原始梯度
        dv_map = dv_map_raw
        error_weight = None

    # 角度方向结构 SNR 加权：抑制纯噪声区域，保留结构化边界
    if angular_snr_weighting:
        angular_mean = gaussian_filter1d(
            polar_smooth, sigma=angular_snr_sigma, axis=1, mode='wrap',
        )
        noise_est = np.abs(polar_smooth - angular_mean)
        noise_est = gaussian_filter1d(
            noise_est, sigma=angular_snr_sigma, axis=1, mode='wrap',
        )
        snr = np.abs(dv_map) / (noise_est + 1e-10)
        w = snr ** 2 / (1.0 + snr ** 2)
        dv_map = dv_map * w

    # 梯度后处理平滑（抗噪声增强）
    if gradient_smoothing_sigma > 0:
        sigma_grad = gradient_smoothing_sigma / 2.355
        dv_map = gaussian_filter1d(dv_map, sigma=sigma_grad, axis=1, mode='wrap')
    
    pixel_dr = 2.0  # Sobel核有效距离
    score_map = np.zeros_like(dv_map, dtype=float)
    if detect_rising_edge:
        growth_mask = dv_map > 0
    else:
        growth_mask = dv_map < 0
    
    if np.any(growth_mask):
        score2 = dv_map[growth_mask]
        score_map[growth_mask] = (dv_map[growth_mask] * score2) / pixel_dr
    
    # 构建成本地图
    valid_scores = score_map[growth_mask]
    
    if len(valid_scores) == 0:
        cost_map = np.ones_like(score_map) * 10.0
    else:
        epsilon = 1e-10
        log_scores = np.log(valid_scores + epsilon)
        
        min_log = np.min(log_scores)
        max_log = np.max(log_scores)
        
        normalized_score_map = np.zeros_like(score_map)
        if (max_log - min_log) > 0:
            normalized_log = (np.log(score_map[growth_mask] + epsilon) - min_log) / (max_log - min_log)
            normalized_score_map[growth_mask] = normalized_log
        
        base_cost = 1e-6
        cost_map = (1.0 - normalized_score_map) + base_cost
        cost_map[~growth_mask] = 1.1

    # 梯度符号一致性惩罚：噪声区域梯度符号随机 → coherence低 → 加惩罚
    # 真实边界梯度符号一致（均为正）→ coherence高 → 惩罚≈0
    if coherence_penalty_weight > 0:
        if detect_rising_edge:
            sign_map = np.sign(dv_map)
        else:
            sign_map = np.sign(-dv_map)  # flip so negative gradients become positive for coherence check
        # NaN 防护：data 中的零值像素会被替换为 NaN，可能通过
        # polar_transform → sobel 传播到 dv_map。np.sign(NaN)=NaN 会通过
        # gaussian_filter1d 污染周围区域。将 NaN 替换为 0 避免此问题。
        sign_map = np.nan_to_num(sign_map, nan=0.0)
        coherence = np.abs(gaussian_filter1d(
            sign_map.astype(float),
            sigma=coherence_sigma,
            axis=1,
            mode='wrap',
        ))
        # penalty = β × (1 − coherence)，加到cost_map
        coherence_penalty = coherence_penalty_weight * (1.0 - coherence)
        cost_map += coherence_penalty

    # 极坐标网格的物理半径坐标
    full_r_grid = np.linspace(0, rmax, num_radii_samples)

    cost_map_radii = (rr_roi[:-1] + rr_roi[1:]) / 2.0

    # 使用DP寻找最优路径
    cost_map_roi = cost_map[:len(rr_roi)-1, :]
    cost_map_radii_roi = cost_map_radii[:len(rr_roi)-1]

    # 使用 L-曲线方法自动寻找最优平滑惩罚参数
    from .find_boundary_dp import find_boundary_dp, find_best_penalty_l_curve
    penalty_lcurve_file = None
    if showfile is not None:
        penalty_lcurve_file = showfile.with_name(
            showfile.stem + '_lcurve' + showfile.suffix
        )
    smoothness_penalty = find_best_penalty_l_curve(
        cost_map_roi,
        cost_map_radii_roi,
        penalty_range=np.logspace(-6, 0, 30),
        showfile=penalty_lcurve_file
    )

    path_r_indices = find_boundary_dp(cost_map_roi, smoothness_penalty)

    # 将行索引映射回物理半径
    path_row_idx = rr_roi[:len(rr_roi)-1][path_r_indices]
    final_radii = full_r_grid[path_row_idx.astype(int)]
    final_angles_rad = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    # 椭圆模式：将索引半径转换为每角度的物理像素距离
    if ellipse is not None:
        a, b, phi = ellipse
        cos_t = np.cos(final_angles_rad - phi)
        sin_t = np.sin(final_angles_rad - phi)
        r_ell = a * b / np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
        # final_radii 是 full_r_grid[row] = row * rmax/(n-1) = fraction * rmax
        # 物理半径 = fraction * r_ell = final_radii * r_ell / rmax
        final_radii = final_radii * r_ell / rmax
    
    # 计算边界路径的 cost 统计
    if len(path_r_indices) > 0:
        # 获取边界上每一点对应的角度索引
        n_angles = cost_map_roi.shape[1]
        angle_indices = np.arange(n_angles)
        path_costs = cost_map_roi[path_r_indices, angle_indices]
    else:
        path_costs = np.array([])
    
    mean_cost = np.mean(path_costs) if len(path_costs) > 0 else np.inf
    total_cost = np.sum(path_costs) if len(path_costs) > 0 else np.inf

    # 边界路径的原始 score（未经过 log 变换，区分度更高）
    # score_map 比 cost_map_roi 多一行，path_r_indices 在有效范围内
    if len(path_r_indices) > 0:
        path_scores = score_map[path_r_indices, angle_indices]
        mean_score = float(np.mean(path_scores)) if len(path_scores) > 0 else 0.0
        total_score = float(np.sum(path_scores)) if len(path_scores) > 0 else 0.0
    else:
        mean_score = 0.0
        total_score = 0.0

    # === 边界对比度（边界两侧像素值差异，噪声路径 contrast≈0）===
    # detect_rising_edge=True: 内暗外亮，contrast = outer - inner > 0
    # detect_rising_edge=False: 内亮外暗，contrast = inner - outer > 0
    per_angle_contrast = np.zeros(n_angles)
    for j in range(n_angles):
        bi = int(path_r_indices[j])
        s_start = max(0, bi - contrast_strip_width)
        s_end = min(polar_smooth.shape[0], bi + contrast_strip_width + 1)
        if s_end - s_start < 2:
            continue
        inner_mean = np.mean(polar_smooth[s_start:bi, j]) if bi > s_start else 0.0
        outer_mean = np.mean(polar_smooth[bi:s_end, j]) if s_end > bi else 0.0
        if detect_rising_edge:
            per_angle_contrast[j] = outer_mean - inner_mean
        else:
            per_angle_contrast[j] = inner_mean - outer_mean
    boundary_contrast = float(np.median(per_angle_contrast))

    # === 边界可靠性估计 ===
    # 基于梯度强度和梯度一致性的组合
    n_angles = cost_map_roi.shape[1]
    angle_indices = np.arange(n_angles)

    # 梯度强度：边界位置处的 |dv_map|
    g_raw = np.abs(dv_map[path_r_indices, angle_indices])

    # 百分位归一化到 [0, 1]（用 5%/95% 避免极端值压缩）
    p5, p95 = np.percentile(g_raw, [5, 95])
    if p95 > p5:
        g_norm = np.clip((g_raw - p5) / (p95 - p5), 0, 1)
    else:
        g_norm = np.ones_like(g_raw)

    # 梯度一致性：边界位置处梯度符号的角度平滑
    boundary_sign = np.sign(dv_map[path_r_indices, angle_indices])
    boundary_sign = np.nan_to_num(boundary_sign, nan=0.0)
    coherence = np.abs(gaussian_filter1d(
        boundary_sign.astype(float), sigma=5.0, mode='wrap'))

    # 组合：梯度强度 × 一致性
    boundary_confidence = g_norm * coherence

    # 阈值：Otsu 自适应，失败则降级为固定阈值 0.3
    try:
        from skimage.filters import threshold_otsu
        confidence_threshold = float(threshold_otsu(boundary_confidence))
    except Exception:
        confidence_threshold = 0.3
    boundary_valid = boundary_confidence > confidence_threshold

    logger.debug("polar %.0f×%d grid, r=[%.1f, %.0f], α=%.2e, cost=%.2f",
                360, cost_map_roi.shape[0], rmin, rmax,
                smoothness_penalty, mean_cost)

    # 返回结果（包含中间产物供诊断可视化使用）
    result = {
        'boundary_radii': final_radii,
        'boundary_angles': final_angles_rad,
        'xc': xc,
        'yc': yc,
        'smoothness_penalty': smoothness_penalty,
        'cost_map': cost_map_roi,
        'cost_map_radii': cost_map_radii_roi,
        'rmin': rmin,
        'rmax': rmax,
        'error_weight_used': error_weight is not None,
        'path_costs': path_costs,
        'mean_cost': mean_cost,
        'total_cost': total_cost,
        'mean_score': mean_score,
        'total_score': total_score,
        'boundary_contrast': boundary_contrast,
        'boundary_confidence': boundary_confidence,
        'boundary_valid': boundary_valid,
        'confidence_threshold': confidence_threshold,
        # Intermediate products for diagnostic visualization
        'polar_image_full': polar_image,
        'polar_roi': polar_roi,
        'polar_smooth': polar_smooth,
        'rr_roi': rr_roi,
        'score_map': score_map,
        'dv_map': dv_map,
        'dv_map_raw': dv_map_raw,   # 原始Sobel梯度（误差加权/SNR/后平滑之前）
        'full_r_grid': full_r_grid,
    }
    result['_cached_polar'] = polar_image  # 缓存供下次使用
    
    # 生成诊断图
    if showfile is not None:
        fig = plt.figure(figsize=(10, 8))
        
        # 原始图像
        ax1 = fig.add_subplot(2, 2, 1)
        interval = AsymmetricPercentileInterval(0.01, 99.5)
        vmin, vmax = interval.get_limits(data[~np.isnan(data)])
        ax1.imshow(data, origin='lower', cmap='viridis',
                   norm=PowerNorm(gamma=1, vmin=vmin, vmax=vmax))
        ax1.plot(xc, yc, 'r+', markersize=10)
        circle = plt.Circle((xc, yc), rmax, fill=False, color='white', linestyle='--')
        ax1.add_patch(circle)
        ax1.set_title('Input Image')
        
        # 成本地图
        ax2 = fig.add_subplot(2, 2, 2)
        extent = [0, 360, rmin, rmax]
        ax2.imshow(cost_map[r_min_idx:, :], origin='lower', cmap='magma_r',
                   extent=extent, aspect='auto')
        ax2.plot(final_angles_rad / np.pi * 180, final_radii, 'r-', linewidth=2)
        ax2.set_title('Cost Map')
        ax2.set_xlabel('Angle (deg)')
        ax2.set_ylabel('Radius')
        
        # 检测到的边界
        ax3 = fig.add_subplot(2, 2, 3)
        x_boundary = xc + final_radii * np.cos(final_angles_rad)
        y_boundary = yc + final_radii * np.sin(final_angles_rad)
        ax3.imshow(data, origin='lower', cmap='viridis',
                   norm=PowerNorm(gamma=1, vmin=vmin, vmax=vmax))
        ax3.plot(x_boundary, y_boundary, 'r-', linewidth=2, label='Detected Boundary')
        ax3.plot(xc, yc, 'r+', markersize=10)
        ax3.set_title('Detected Boundary')
        ax3.legend()
        
        # 边界半径分布
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(final_angles_rad / np.pi * 180, final_radii, 'b-', linewidth=1)
        ax4.set_xlabel('Angle (deg)')
        ax4.set_ylabel('Radius (pixels)')
        ax4.set_title('Boundary Radius Profile')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(showfile, dpi=150, bbox_inches='tight')
        plt.close()
    
    return result


if __name__ == '__main__':
    # 测试代码
    shape = (200, 200)
    r_inner = 30
    r_outer = 60
    xc, yc = 100, 100
    
    img = generate_test_ring_image(shape, r_inner, r_outer, xc, yc)
    
    result = extract_circle_boundary(img, xc, yc, r_outer * 1.2, showfile=None)
    
    print(f"Boundary radii: min={result['boundary_radii'].min():.1f}, "
          f"max={result['boundary_radii'].max():.1f}, "
          f"mean={result['boundary_radii'].mean():.1f}")
