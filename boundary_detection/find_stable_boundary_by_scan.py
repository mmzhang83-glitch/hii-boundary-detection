"""
全扫描稳定边界检测算法

核心思路：
1. 在 [rmin_start_ratio, rmax_limit_ratio] 之间均匀取 n_steps 个点
2. 对每个 rmin_ratio 提取边界和 cost
3. 以 rmin_start_ratio 的边界为参考，计算所有边界的差异
4. 找到 diff 曲线的稳定区（连续 stable_window 个点变化 < stable_threshold）
5. 如果有稳定区，按 mean_cost（每角度平均代价）选最佳；如果无稳定区，使用 fallback 策略
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy.ndimage import gaussian_filter1d

from .extract_circle_boundary import extract_circle_boundary


def compare_boundaries(boundary1: np.ndarray, boundary2: np.ndarray) -> float:
    """
    比较两个边界的差异
    
    返回边界之间的平均距离（归一化到半径）
    """
    if len(boundary1) != len(boundary2):
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


def find_all_stable_regions(
    diffs: np.ndarray, 
    window: int, 
    threshold: float
) -> List[Tuple[int, int]]:
    """
    找所有连续 window 个点 diff 变化 < threshold 的稳定区
    
    参数:
        diffs: 差异数组
        window: 连续多少个点
        threshold: 变化阈值
    
    返回:
        List of (start_idx, end_idx)，stable 区间的起始和结束索引
    """
    if len(diffs) < window:
        return []
    
    delta = np.abs(np.diff(diffs))
    n = len(delta)
    
    stable_regions = []
    i = 0
    while i <= n - window:
        if np.all(delta[i:i+window] < threshold):
            start = i
            end = i + window - 1
            # 向右扩展：检查 end+1 是否也满足条件再纳入
            while end + 1 < n and delta[end + 1] < threshold:
                end += 1
            stable_regions.append((start, end))
            i = end + 1
        else:
            i += 1
    
    return stable_regions


def find_min_adjacent_change_boundary(
    boundaries: List[np.ndarray],
    diffs: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    找到相邻边界变化最小的边界

    返回: (best_idx, best_boundary)
    """
    if len(boundaries) < 2:
        return 0, boundaries[0]

    adjacent_changes = []
    for i in range(len(boundaries) - 1):
        change = compare_boundaries(boundaries[i], boundaries[i+1])
        adjacent_changes.append((change, i, i+1))

    min_change = min(adjacent_changes, key=lambda x: x[0])
    best_idx = min_change[2]

    return best_idx, boundaries[best_idx]


def _compute_gradient_direction_std(
    boundary: np.ndarray,
    dv_map: np.ndarray,
    rmax: float,
    strip_width: int,
) -> float:
    """
    计算边界条带内的梯度方向标准差。

    对每个角度，取边界 ±strip_width 范围内的梯度值，
    统计正梯度占比，然后计算所有角度正梯度占比的 std。

    参数:
        boundary: 边界半径数组，长度 = n_angles
        dv_map: 极坐标梯度图，形状 (n_radii, n_angles)
        rmax: 最大搜索半径（像素），用于构建径向网格
        strip_width: 条带半宽（像素）

    返回:
        float: 正梯度占比跨角度的标准差，范围 [0, 1]
               std 越小表示梯度方向越一致，边界精度越高
    """
    n_radii, n_angles = dv_map.shape

    # 构建径向网格索引
    r_grid = np.linspace(0, rmax, n_radii)
    boundary_idx = np.interp(boundary, r_grid, np.arange(n_radii))

    positive_ratios = []
    for j in range(n_angles):
        center = int(np.round(boundary_idx[j]))
        start = max(0, center - strip_width)
        end = min(n_radii, center + strip_width + 1)
        strip_gradient = dv_map[start:end, j]
        strip_signs = np.sign(strip_gradient)
        strip_signs = strip_signs[strip_signs != 0]
        if len(strip_signs) > 0:
            positive_ratios.append(np.mean(strip_signs > 0))

    if not positive_ratios:
        return 1.0

    return float(np.std(positive_ratios))


def _composite_score(
    stable_info: list,
    weights: dict,
) -> list:
    """
    对多个稳定区计算综合评分，选最高分。

    每项指标 min-max 归一化到 [0, 1] 后加权求和：
      score = w_score × norm(score)       # mean_score 越大越好（原始梯度分）
            + w_len   × norm(length)       # length 越大越好
            + w_std   × (1 - norm(std))    # std 越小越好

    归一化分母为零时该项权重均分到剩余项。

    参数:
        stable_info: 稳定区信息列表，每项需含 mean_score, indices, grad_std
        weights: dict 含 cost(兼容)/score, length, std 三个权重

    返回:
        stable_info 列表，每项附加 'score' 和 'score_detail' 字段
    """
    # 兼容旧权重键名 'cost'，新代码使用 'score'
    w_score = weights.get('score', weights.get('cost', 0.4))
    w_len = weights['length']
    w_std = weights['std']

    scores = np.array([s.get('mean_score', 0.0) for s in stable_info])
    lengths = np.array([s['indices'][1] - s['indices'][0] + 1 for s in stable_info])
    stds = np.array([s.get('grad_std', 1.0) for s in stable_info])

    def _normalize_minmax(values, invert=False):
        """min-max normalize to [0, 1], optionally invert (1 - norm)."""
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-12:
            return np.ones_like(values) * (0.0 if invert else 1.0), 0.0
        norm = (values - vmin) / (vmax - vmin)
        return (1.0 - norm) if invert else norm, 1.0

    score_norm, score_ok = _normalize_minmax(scores, invert=False)  # score越大越好
    len_norm, len_ok = _normalize_minmax(lengths, invert=False)     # length越大越好
    std_norm, std_ok = _normalize_minmax(stds, invert=True)         # std越小越好

    # 处理归一化无效的情况（所有值相同 → 权重均分）
    ok_flags = {'score': score_ok, 'length': len_ok, 'std': std_ok}
    w_eff = {'score': w_score, 'length': w_len, 'std': w_std}

    for key, ok in ok_flags.items():
        if ok < 0.5:
            redistribute = w_eff[key] / max(len([v for v in ok_flags.values() if v > 0.5]), 1)
            for other_key in ok_flags:
                if ok_flags[other_key] > 0.5:
                    w_eff[other_key] += redistribute
            w_eff[key] = 0.0

    for i, info in enumerate(stable_info):
        info['score'] = float(
            w_eff['score'] * score_norm[i] +
            w_eff['length'] * len_norm[i] +
            w_eff['std'] * std_norm[i]
        )
        info['score_detail'] = {
            'score_raw': float(scores[i]),
            'score_norm': float(score_norm[i]),
            'cost_raw': float(stable_info[i].get('mean_cost', 0)),
            'length_raw': int(lengths[i]),
            'length_norm': float(len_norm[i]),
            'std_raw': float(stds[i]),
            'std_norm': float(std_norm[i]),
            'weights_effective': {k: float(v) for k, v in w_eff.items()},
        }

    return stable_info


def find_stable_boundary_by_scan(
    data: np.ndarray,
    xc: float,
    yc: float,
    rmax: float,
    rmin_start_ratio: float = 0.05,
    rmax_limit_ratio: float = 0.70,
    smoothing_fwhm: float = 2.0,
    cost_map_smoothing_sigma: float = 5.0,
    gradient_smoothing_sigma: float = 0.0,
    n_steps: int = 50,
    stable_window: int = 5,
    stable_threshold: float = 0.02,
    fallback: str = 'min_adjacent_change',
    boundary_smoothing_sigma: float = 0.0,
    error_map: Optional[np.ndarray] = None,
    error_weighting: bool = True,
    show_diagnostics: bool = False,
    showfile: Optional[Path] = None,
    angular_snr_weighting: bool = False,
    angular_snr_sigma: float = 3.0,
    coherence_penalty_weight: float = 0.0,
    coherence_sigma: float = 3.0,
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
    全扫描找稳定边界

    参数:
        data: 2D图像数据
        xc, yc: 圆心坐标（像素）
        rmax: 最大搜索半径
        rmin_start_ratio: rmin 起始比例
        rmax_limit_ratio: rmin 上限比例
        n_steps: 扫描点数
        stable_window: 稳定区连续点数
        stable_threshold: 稳定区变化阈值
        fallback: 无稳定区时的策略，'min_adjacent_change' 或 'return_none'
        boundary_smoothing_sigma: 边界后处理平滑 sigma
        error_map: 误差图
        error_weighting: 是否启用误差加权
        show_diagnostics: 是否显示诊断图
        showfile: 诊断图保存路径
        gradient_strip_width: 梯度方向条带半宽（像素），0=禁用
        selection_weight_cost: mean_cost 评分权重
        selection_weight_length: 稳定区长度评分权重
        selection_weight_std: 梯度方向 std 评分权重
        use_fractional_scan: 是否使用分数扫描模式（椭圆先验）。True 时 rmin_start_ratio / rmax_limit_ratio 视为分数 (0~1)，转换为 rmin_scale = 1/fraction
        fractional_rmax: 分数模式下用于自动 n_steps 计算的参考 rmax。为 None 时使用 rmax

    返回:
        包含边界信息和诊断数据的字典
    """
    # Auto-calculate n_steps if set to 0: scan every 2 pixels between start and end
    if n_steps == 0:
        if use_fractional_scan:
            ref = fractional_rmax if fractional_rmax is not None else rmax
            interval = max(1.0, ref * 0.02)
            n_steps = max(2, int(np.ceil(
                (rmax_limit_ratio - rmin_start_ratio) * ref / interval
            )) + 1)
        else:
            rmin_start = rmax * rmin_start_ratio
            rmin_end = rmax * rmax_limit_ratio
            pixel_interval = 2.0
            n_steps = max(2, int(np.ceil((rmin_end - rmin_start) / pixel_interval)) + 1)

    rmin_ratios = np.linspace(rmin_start_ratio, rmax_limit_ratio, n_steps)

    boundaries = []
    costs = []
    mean_costs = []
    mean_scores = []
    extract_results = []  # full extract_circle_boundary results
    cached_polar = None  # 缓存的极坐标图像

    for ratio in rmin_ratios:
        if use_fractional_scan:
            rmin_scale = 1.0 / ratio if ratio > 0 else 100.0
        else:
            rmin = rmax * ratio
            rmin_scale = rmax / rmin

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
            cached_polar=cached_polar,  # 传入缓存
            detect_rising_edge=detect_rising_edge,
            ellipse=ellipse,
            contrast_strip_width=contrast_strip_width,
        )

        # 保存缓存供下次使用
        if cached_polar is None and '_cached_polar' in result:
            cached_polar = result['_cached_polar']

        boundaries.append(result['boundary_radii'])
        costs.append(result['total_cost'])
        mean_costs.append(result['mean_cost'])
        mean_scores.append(result.get('mean_score', 0.0))
        extract_results.append(result)
    
    ref_boundary = boundaries[0]
    diffs = np.array([0.0] + [compare_boundaries(ref_boundary, b) for b in boundaries[1:]])
    
    stable_regions = find_all_stable_regions(diffs, stable_window, stable_threshold)
    
    has_stable_region = len(stable_regions) > 0
    boundary_type = None
    best_stable_region = None
    fallback_used = False
    final_boundary = None
    final_rmin_ratio = None
    final_extract_idx = None  # index into extract_results for _extract_result

    if has_stable_region:
        if len(stable_regions) == 1:
            boundary_type = 'single_stable'
        else:
            boundary_type = 'multiple_stable'

        stable_info = []
        for (start, end) in stable_regions:
            avg_boundary = np.mean(boundaries[start:end+1], axis=0)
            mid_idx = (start + end) // 2
            total_cost = costs[mid_idx]
            mean_cost = mean_costs[mid_idx]
            mean_rmin_ratio = np.mean(rmin_ratios[start:end+1])
            ext = extract_results[mid_idx]
            stable_info.append({
                'indices': (start, end),
                'mid_idx': mid_idx,
                'boundary': avg_boundary,
                'total_cost': total_cost,
                'mean_cost': mean_cost,
                'mean_score': mean_scores[mid_idx],
                'boundary_contrast': ext.get('boundary_contrast', 0.0),
                'rmin_ratio': mean_rmin_ratio
            })

        # 过滤噪声路径：contrast 低于阈值说明边界两侧无物理信号差异
        # 保留 contrast >= contrast_threshold 的（按 detect_rising_edge 方向定义）
        valid_stable = [s for s in stable_info
                        if s.get('boundary_contrast', 0) >= contrast_min]
        if valid_stable:
            stable_info = valid_stable
        # 如果全部被过滤，保持原样（退化场景：所有区都是噪声？理论上不存在）

        # 多稳定区选择：计算梯度方向 std + 综合评分（使用 mean_score 替代 mean_cost）
        if gradient_strip_width > 0:
            for info in stable_info:
                ext = extract_results[info['mid_idx']]
                dv_map = ext.get('dv_map')
                if dv_map is not None:
                    info['grad_std'] = _compute_gradient_direction_std(
                        info['boundary'], dv_map, rmax, gradient_strip_width,
                    )
                else:
                    info['grad_std'] = 1.0

        if len(stable_info) >= 2 and gradient_strip_width > 0:
            # 使用综合评分：cost + length + gradient_std
            weights = {
                'score': selection_weight_cost,
                'length': selection_weight_length,
                'std': selection_weight_std,
            }
            stable_scored = _composite_score(stable_info, weights)
            best_stable_region = max(stable_scored, key=lambda x: x['score'])
        elif len(stable_info) >= 2:
            # 降级: score + length 二项评分（无 gradient_std）
            # mean_score 越大越好
            stable_info_sorted = sorted(stable_info, key=lambda x: x.get('mean_score', 0), reverse=True)
            score_best = stable_info_sorted[0].get('mean_score', 0)
            score_second = stable_info_sorted[1].get('mean_score', 0)
            score_diff_ratio = (score_best - score_second) / score_best if score_best > 0 else 0
            if score_diff_ratio < 0.15:
                len_best = stable_info_sorted[0]['indices'][1] - stable_info_sorted[0]['indices'][0] + 1
                len_second = stable_info_sorted[1]['indices'][1] - stable_info_sorted[1]['indices'][0] + 1
                if len_second > len_best:
                    best_stable_region = stable_info_sorted[1]
                else:
                    best_stable_region = stable_info_sorted[0]
            else:
                best_stable_region = stable_info_sorted[0]
        else:
            best_stable_region = stable_info[0]

        final_boundary = best_stable_region['boundary']
        final_rmin_ratio = best_stable_region['rmin_ratio']
        final_extract_idx = best_stable_region['mid_idx']
    else:
        boundary_type = 'no_stable'

        if fallback == 'min_adjacent_change':
            best_idx, fallback_boundary = find_min_adjacent_change_boundary(boundaries, diffs)
            final_boundary = fallback_boundary
            final_rmin_ratio = rmin_ratios[best_idx]
            final_extract_idx = best_idx
            fallback_used = True
            best_stable_region = None
        else:
            final_boundary = boundaries[-1]
            final_rmin_ratio = rmin_ratios[-1]
            final_extract_idx = -1
            fallback_used = True
            best_stable_region = None
    
    if boundary_smoothing_sigma > 0:
        sigma = boundary_smoothing_sigma / 2.355
        final_boundary = gaussian_filter1d(final_boundary, sigma=sigma, mode='wrap')
    
    n_angles = len(final_boundary)
    final_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # 笛卡尔坐标下的边界点
    final_boundary_x = xc + final_boundary * np.cos(final_angles)
    final_boundary_y = yc + final_boundary * np.sin(final_angles)

    final_rmin = rmax * final_rmin_ratio
    final_extract = extract_results[final_extract_idx]

    result = {
        'boundary_radii': final_boundary,
        'boundary_angles': final_angles,
        'boundary_x': final_boundary_x,
        'boundary_y': final_boundary_y,
        'xc': xc,
        'yc': yc,
        'rmin_ratio': final_rmin_ratio,
        'rmin_final': final_rmin,
        '_extract_result': final_extract,
        'boundary_type': boundary_type,
        'stable_regions': stable_regions,
        'stable_regions_info': stable_info if has_stable_region else [],
        'best_stable_region': best_stable_region,
        'has_stable_region': has_stable_region,
        'fallback_used': fallback_used,
        'rmin_ratios': rmin_ratios,
        'boundaries': boundaries,
        'costs': costs,
        'mean_costs': mean_costs,
        'mean_scores': mean_scores,
        'diffs': diffs,
        'n_iterations': n_steps,
        'scan_mode': 'fractional' if use_fractional_scan else 'pixel',
    }
    if use_fractional_scan:
        result['f_min_final'] = final_rmin_ratio
    # 透传边界可靠性字段
    if final_extract is not None:
        result['boundary_confidence'] = final_extract.get('boundary_confidence')
        result['boundary_valid'] = final_extract.get('boundary_valid')
        result['confidence_threshold'] = final_extract.get('confidence_threshold')
    
    if show_diagnostics:
        _plot_diagnostics(data, xc, yc, rmax, final_boundary, final_angles,
                         diffs, stable_regions, boundaries, rmin_ratios,
                         boundary_type, costs, showfile, 
                         stable_threshold=stable_threshold,
                         fallback_used=fallback_used)
    
    return result


def _plot_diagnostics(
    data, xc, yc, rmax, final_boundary, final_angles,
    diffs, stable_regions, boundaries, rmin_ratios,
    boundary_type, costs, showfile,
    stable_threshold=0.02,
    fallback_used=False
):
    """绘制诊断图"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: delta curve (change in boundary diff between adjacent rmin steps).
    # Stable regions are where |delta| < threshold for consecutive steps.
    delta = np.abs(np.diff(diffs))
    delta_ratios = (rmin_ratios[:-1] + rmin_ratios[1:]) / 2  # midpoints
    ax1 = axes[0, 0]
    ax1.plot(delta_ratios, delta, 'b-o', markersize=3, linewidth=1)
    for (start, end) in stable_regions:
        # stable_regions indices are based on delta (length n-1),
        # but rmin_ratios has length n, so extend end by 1 to cover full range
        ax1.axvspan(rmin_ratios[start], rmin_ratios[min(end + 1, len(rmin_ratios) - 1)],
                    alpha=0.3, color='green', label='Stable Region' if start == stable_regions[0][0] else '')
    ax1.axhline(stable_threshold, color='r', linestyle='--', label=f'Threshold={stable_threshold}')
    ax1.set_xlabel('rmin_ratio')
    ax1.set_ylabel('|Δ(Boundary Diff)| (stability)')
    ax1.set_title(f'Stability Curve - {boundary_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    # Determine which rmin ratios fall within stable regions
    stable_ratios = set()
    for (start, end) in stable_regions:
        for idx in range(start, min(end + 2, len(rmin_ratios))):
            stable_ratios.add(idx)
    # Plot all boundaries: red for stable, gray for non-stable
    has_stable_label = False
    has_nonstable_label = False
    for i, (b, ratio) in enumerate(zip(boundaries, rmin_ratios)):
        if i in stable_ratios:
            color = 'red'
            label = 'Stable' if not has_stable_label else ''
            has_stable_label = True
        else:
            color = 'gray'
            label = 'Non-stable' if not has_nonstable_label else ''
            has_nonstable_label = True
        ax2.plot(np.linspace(0, 360, len(b)), b, '-', color=color, alpha=0.5, linewidth=0.8, label=label)
    ax2.set_xlabel('Angle (deg)')
    ax2.set_ylabel('Radius (pixels)')
    ax2.set_title('All Boundaries')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    from astropy.visualization import AsymmetricPercentileInterval
    from matplotlib.colors import PowerNorm
    interval = AsymmetricPercentileInterval(0.01, 99.5)
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        vmin, vmax = interval.get_limits(valid_data)
    else:
        vmin, vmax = 0, 1
    ax3.imshow(data, origin='lower', cmap='viridis',
               norm=PowerNorm(gamma=1, vmin=vmin, vmax=vmax))
    x_boundary = xc + final_boundary * np.cos(final_angles)
    y_boundary = yc + final_boundary * np.sin(final_angles)
    ax3.plot(x_boundary, y_boundary, 'r-', linewidth=2, label='Stable Boundary')
    ax3.plot(xc, yc, 'r+', markersize=10)
    circle = plt.Circle((xc, yc), rmax, fill=False, color='white', linestyle='--', linewidth=1,
                        label=f'Search Region (rmax={rmax:.0f} px)')
    ax3.add_patch(circle)
    ax3.set_title('Final Stable Boundary on Image')
    ax3.legend()
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    region_info = ""
    if stable_regions:
        for i, (start, end) in enumerate(stable_regions):
            region_info += f"Region {i+1}: [{rmin_ratios[start]:.3f}, {rmin_ratios[end]:.3f}]\n"

    best_idx = np.argmin([compare_boundaries(boundaries[0], b) for b in boundaries])
    summary = (
        "Stable Boundary Detection Results (Scan Method)\n"
        "=================================================\n"
        "\n"
        f"Boundary Type: {boundary_type}\n"
        f"Stable Regions: {len(stable_regions)}\n"
        f"{region_info if region_info else 'No stable regions found'}"
        f"Fallback Used: {fallback_used}\n"
        "\n"
        f"Final rmin_ratio: {rmin_ratios[best_idx]:.3f}\n"
        "\n"
        "Boundary Statistics:\n"
        f"  min = {final_boundary.min():.1f}\n"
        f"  max = {final_boundary.max():.1f}\n"
        f"  mean = {final_boundary.mean():.1f}\n"
        "\n"
        f"Cost: {costs[best_idx]:.4f}\n"
        f"Mean Cost: {costs[best_idx] / 360:.4f}\n"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             family='monospace')
    
    plt.tight_layout()
    if showfile is not None:
        plt.savefig(showfile, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    from .extract_circle_boundary import generate_test_ring_image
    
    shape = (300, 300)
    r_inner = 50
    r_outer = 80
    xc, yc = 150, 150
    
    img = generate_test_ring_image(shape, r_inner, r_outer, xc, yc, add_noise=0.05)
    
    result = find_stable_boundary_by_scan(
        img, xc, yc, r_outer * 1.1,
        show_diagnostics=True
    )
    
    print(f"Boundary type: {result['boundary_type']}")
    print(f"Has stable region: {result['has_stable_region']}")
    print(f"Stable regions: {result['stable_regions']}")
    print(f"Fallback used: {result['fallback_used']}")
    print(f"Boundary radius: mean={result['boundary_radii'].mean():.1f}")
