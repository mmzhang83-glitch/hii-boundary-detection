"""
稳定边界检测算法

支持两种方法：
1. iterative: 迭代法 - 从较小的 rmin 开始，逐步增大 rmin，直到边界稳定
2. scan: 扫描法 - 遍历 rmin_range 内的所有值，找到边界变化的稳定区
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal
from scipy.spatial.distance import cdist

from .extract_circle_boundary import extract_circle_boundary
from .find_stable_boundary_by_scan import find_stable_boundary_by_scan as find_stable_boundary_scan


def compare_boundaries(boundary1: np.ndarray, boundary2: np.ndarray) -> float:
    """
    比较两个边界的差异
    
    返回边界之间的平均距离（归一化到半径）
    """
    if len(boundary1) != len(boundary2):
        # 插值到相同点数
        n = max(len(boundary1), len(boundary2))
        boundary1_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(boundary1)),
            boundary1
        )
        boundary2_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(boundary2)),
            boundary2
        )
    else:
        boundary1_interp = boundary1
        boundary2_interp = boundary2
    
    return np.mean(np.abs(boundary1_interp - boundary2_interp))


def find_stable_boundary(
    data: np.ndarray,
    xc: float,
    yc: float,
    rmax: float,
    method: Literal['scan', 'iterative'] = 'iterative',
    smoothing_fwhm: float = 2.0,
    cost_map_smoothing_sigma: float = 5.0,
    gradient_smoothing_sigma: float = 0.0,
    boundary_smoothing_sigma: float = 0.0,
    error_map: Optional[np.ndarray] = None,
    error_weighting: bool = True,
    show_diagnostics: bool = False,
    showfile: Optional[Path] = None,
    rmin_start_ratio: float = 0.05,
    rmin_min_pixels: float = 5.0,
    rmax_limit_ratio: float = 0.7,
    angular_snr_weighting: bool = False,
    angular_snr_sigma: float = 3.0,
    coherence_penalty_weight: float = 0.0,
    coherence_sigma: float = 3.0,
    stability_threshold: float = 0.02,
    max_iterations: int = 20,
    consecutive_stable: int = 3,
    n_steps: int = 50,
    stable_window: int = 5,
    stable_threshold: float = 0.02,
    fallback: str = 'min_adjacent_change',
    detect_rising_edge: bool = True,
    gradient_strip_width: int = 3,
    selection_weight_cost: float = 0.4,
    selection_weight_length: float = 0.3,
    selection_weight_std: float = 0.3,
    use_fractional_scan: bool = False,
    fractional_rmax: Optional[float] = None,
    ellipse: Optional[tuple] = None,
    contrast_min: float = 0.01,
    contrast_strip_width: int = 4,
) -> Dict:
    """
    稳定边界检测
    
    参数:
        data: 2D图像数据
        xc, yc: 圆心坐标（像素）
        rmax: 最大搜索半径
        method: 'scan' (扫描法) 或 'iterative' (迭代法)
        smoothing_fwhm: 径向平滑FWHM
        cost_map_smoothing_sigma: 角度平滑sigma
        gradient_smoothing_sigma: Sobel梯度后平滑sigma
        boundary_smoothing_sigma: 边界后处理平滑sigma
        error_map: 误差图（与data同形状），用于误差加权梯度计算
        error_weighting: 是否启用误差加权，默认为True
        show_diagnostics: 是否显示诊断图（scan法）或迭代图（iterative法）
        showfile: 诊断图保存路径
        
        迭代法专用参数 (method='iterative'):
            rmin_start_ratio: 初始 rmin = rmax * rmin_start_ratio
            rmax_limit_ratio: rmin 上限 = rmax * rmax_limit_ratio
            stability_threshold: 稳定性阈值
            max_iterations: 最大迭代次数
            consecutive_stable: 连续多少次稳定才判断收敛
        
        扫描法专用参数 (method='scan'):
            n_steps: 扫描点数
            stable_window: 稳定区连续点数
            stable_threshold: 稳定区变化阈值
            fallback: 无稳定区时的策略
    
    返回:
        包含边界信息和诊断数据的字典
    """
    # Floor rmin_start_ratio so rmin >= rmin_min_pixels
    if rmax * rmin_start_ratio < rmin_min_pixels:
        rmin_start_ratio = rmin_min_pixels / rmax

    if method == 'scan':
        return find_stable_boundary_scan(
            data=data,
            xc=xc,
            yc=yc,
            rmax=rmax,
            rmin_start_ratio=rmin_start_ratio,
            rmax_limit_ratio=rmax_limit_ratio,
            smoothing_fwhm=smoothing_fwhm,
            cost_map_smoothing_sigma=cost_map_smoothing_sigma,
            gradient_smoothing_sigma=gradient_smoothing_sigma,
            n_steps=n_steps,
            stable_window=stable_window,
            stable_threshold=stable_threshold,
            fallback=fallback,
            boundary_smoothing_sigma=boundary_smoothing_sigma,
            error_map=error_map,
            error_weighting=error_weighting,
            show_diagnostics=show_diagnostics,
            showfile=showfile,
            angular_snr_weighting=angular_snr_weighting,
            angular_snr_sigma=angular_snr_sigma,
            coherence_penalty_weight=coherence_penalty_weight,
            coherence_sigma=coherence_sigma,
            detect_rising_edge=detect_rising_edge,
            gradient_strip_width=gradient_strip_width,
            selection_weight_cost=selection_weight_cost,
            selection_weight_length=selection_weight_length,
            selection_weight_std=selection_weight_std,
            use_fractional_scan=use_fractional_scan,
            fractional_rmax=fractional_rmax,
            ellipse=ellipse,
            contrast_min=contrast_min,
            contrast_strip_width=contrast_strip_width,
        )
    # 边界历史记录
    boundary_history = []
    rmin_history = []
    extract_history = []  # 完整 extract_circle_boundary 结果
    stable_count = 0  # 连续稳定计数
    cached_polar = None  # 缓存的极坐标图像，避免重复计算

    # 迭代寻找边界
    current_rmin_ratio = rmin_start_ratio

    for iteration in range(max_iterations):
        current_rmin = rmax * current_rmin_ratio
        rmin_scale = rmax / current_rmin

        # 提取边界（传递平滑参数和误差图）
        result = extract_circle_boundary(
            data, xc, yc, rmax,
            rmin_scale=rmin_scale,
            smoothing_fwhm=smoothing_fwhm,
            cost_map_smoothing_sigma=cost_map_smoothing_sigma,
            gradient_smoothing_sigma=gradient_smoothing_sigma,
            error_map=error_map,
            error_weighting=error_weighting,
            angular_snr_weighting=angular_snr_weighting,
            angular_snr_sigma=angular_snr_sigma,
            coherence_penalty_weight=coherence_penalty_weight,
            coherence_sigma=coherence_sigma,
            showfile=None,
            cached_polar=cached_polar,  # 传入缓存，跳过极坐标变换
            detect_rising_edge=detect_rising_edge,
            ellipse=ellipse,
            contrast_strip_width=contrast_strip_width,
        )

        # 保存缓存供下次使用（仅首次需要从结果中提取）
        if cached_polar is None and '_cached_polar' in result:
            cached_polar = result['_cached_polar']

        boundary_radii = result['boundary_radii']

        # 记录历史
        boundary_history.append(boundary_radii.copy())
        rmin_history.append(current_rmin)
        extract_history.append(result)

        # 检查稳定性（至少需要2次迭代）
        if len(boundary_history) >= 2:
            prev_boundary = boundary_history[-2]
            curr_boundary = boundary_history[-1]

            # 计算边界差异
            diff = compare_boundaries(prev_boundary, curr_boundary)

            # 归一化差异
            normalized_diff = diff / rmax

            if normalized_diff < stability_threshold:
                # 边界稳定，连续计数+1
                stable_count += 1

                # 连续多次稳定才判断收敛（避免局部极小）
                if stable_count >= consecutive_stable:
                    final_boundary = curr_boundary
                    final_rmin = current_rmin
                    final_extract = result
                    converged = True
                    break
            else:
                # 不稳定，重置计数
                stable_count = 0
        else:
            final_boundary = boundary_radii
            final_rmin = current_rmin
            final_extract = result
            converged = False
            stable_count = 0

        # 增加 rmin 比例
        current_rmin_ratio += (rmax_limit_ratio - rmin_start_ratio) / max_iterations

        # 检查是否超过上限
        if current_rmin_ratio >= rmax_limit_ratio:
            converged = False
            break
    else:
        # 达到最大迭代次数
        final_boundary = boundary_history[-1]
        final_rmin = current_rmin
        final_extract = extract_history[-1]
        converged = False
    
    # 边界后处理平滑（抗噪声增强）
    if boundary_smoothing_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma = boundary_smoothing_sigma / 2.355
        final_boundary = gaussian_filter1d(final_boundary, sigma=sigma, mode='wrap')
    
    # 确保 final_extract 在异常情况下也有值（rmin_start_ratio > rmax_limit_ratio）
    if 'final_extract' not in dir():
        final_extract = extract_history[-1] if extract_history else None

    # 构建最终结果
    n_angles = len(final_boundary)
    final_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # 笛卡尔坐标下的边界点
    final_boundary_x = xc + final_boundary * np.cos(final_angles)
    final_boundary_y = yc + final_boundary * np.sin(final_angles)

    result = {
        'boundary_radii': final_boundary,
        'boundary_angles': final_angles,
        'boundary_x': final_boundary_x,
        'boundary_y': final_boundary_y,
        'xc': xc,
        'yc': yc,
        'n_iterations': len(boundary_history),
        'rmin_final': final_rmin,
        'converged': converged,
        'boundary_history': boundary_history,
        'rmin_history': rmin_history,
        '_extract_result': final_extract,  # 最终迭代的完整 extract 结果
    }
    # 透传边界可靠性字段
    if final_extract is not None:
        result['boundary_confidence'] = final_extract.get('boundary_confidence')
        result['boundary_valid'] = final_extract.get('boundary_valid')
        result['confidence_threshold'] = final_extract.get('confidence_threshold')
    
    # 生成诊断图
    if showfile is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 图1: 边界演化
        ax1 = axes[0, 0]
        for i, (r, rm) in enumerate(zip(boundary_history, rmin_history)):
            ax1.plot(np.linspace(0, 360, len(r)), r, 'o-', 
                     alpha=0.5, label=f'iter {i+1}: rmin={rm:.0f}')
        ax1.set_xlabel('Angle (deg)')
        ax1.set_ylabel('Radius (pixels)')
        ax1.set_title('Boundary Evolution')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 图2: 最终边界在图像上
        ax2 = axes[0, 1]
        # 显示图像
        from astropy.visualization import AsymmetricPercentileInterval
        from matplotlib.colors import PowerNorm
        interval = AsymmetricPercentileInterval(0.01, 99.5)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin, vmax = interval.get_limits(valid_data)
        else:
            vmin, vmax = 0, 1
        ax2.imshow(data, origin='lower', cmap='viridis',
                   norm=PowerNorm(gamma=1, vmin=vmin, vmax=vmax))
        
        # 绘制边界
        x_boundary = xc + final_boundary * np.cos(final_angles)
        y_boundary = yc + final_boundary * np.sin(final_angles)
        ax2.plot(x_boundary, y_boundary, 'r-', linewidth=2, label='Stable Boundary')
        ax2.plot(xc, yc, 'r+', markersize=10)
        ax2.set_title('Final Stable Boundary')
        ax2.legend()
        
        # 图3: 迭代历史
        ax3 = axes[1, 0]
        if len(boundary_history) >= 2:
            diffs = []
            for i in range(1, len(boundary_history)):
                d = compare_boundaries(boundary_history[i-1], boundary_history[i])
                diffs.append(d / rmax)
            ax3.plot(range(2, len(diffs)+2), diffs, 'o-')
            ax3.axhline(stability_threshold, color='r', linestyle='--', 
                        label=f'Threshold={stability_threshold}')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Normalized Difference')
            ax3.set_title('Boundary Stability')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 图4: 文本总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"""
        Stable Boundary Detection Results
        ==================================
        
        Converged: {converged}
        Iterations: {len(boundary_history)}
        
        Final rmin: {final_rmin:.1f} pixels
        rmin/rmax ratio: {final_rmin/rmax:.2f}
        
        Boundary radius:
          min = {final_boundary.min():.1f}
          max = {final_boundary.max():.1f}
          mean = {final_boundary.mean():.1f}
        """
        ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, 
                 fontsize=10, verticalalignment='top',
                 family='monospace')
        
        plt.tight_layout()
        plt.savefig(showfile, dpi=150, bbox_inches='tight')
        plt.close()
    
    return result


if __name__ == '__main__':
    from .extract_circle_boundary import generate_test_ring_image
    
    # 测试
    shape = (300, 300)
    r_inner = 50
    r_outer = 80
    xc, yc = 150, 150
    
    img = generate_test_ring_image(shape, r_inner, r_outer, xc, yc)
    
    result = find_stable_boundary(img, xc, yc, r_outer * 1.1)
    
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['n_iterations']}")
    print(f"Boundary radius: mean={result['boundary_radii'].mean():.1f}")
