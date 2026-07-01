import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import logging

logger = logging.getLogger("hii_boundary.extract.dp")

def find_boundary_dp(cost_map: np.ndarray, smoothness_penalty: float) -> np.ndarray:
    """
    使用动态规划（维特比算法）寻找成本地图中的最优周期性路径。

    参数:
    ----------
    cost_map : np.ndarray
        成本地图，维度为 (半径, 角度)。值越小越好。
    smoothness_penalty : float
        平滑惩罚因子 (alpha)。值越大，路径越平滑。
        它控制了半径跳变的成本：cost = alpha * (delta_r)^2。

    返回:
    -------
    np.ndarray
        一个一维数组，包含了每个角度上最优路径的半径索引。
    """
    num_radii, num_angles = cost_map.shape

    # 初始化累积成本矩阵 D 和回溯指针矩阵 P
    D = np.zeros_like(cost_map)
    P = np.zeros_like(cost_map, dtype=int)

    # --- 1. 周期性处理：扩展成本地图 ---
    # 为了处理 0/360 度的连接，我们将地图在角度方向上“卷起来”
    # 我们将地图的后半部分拼到前面，前半部分拼到后面
    # 这样，在中心区域的路径寻找就自然地包含了周期性
    
    extended_map = np.concatenate(
        [cost_map[:, num_angles//2:], cost_map, cost_map[:, :num_angles//2]], 
        axis=1
    )
    extended_num_angles = extended_map.shape[1]
    
    # 在扩展后的地图上进行计算
    D_ext = np.zeros_like(extended_map)
    P_ext = np.zeros_like(extended_map, dtype=int)
    D_ext[:, 0] = extended_map[:, 0]

    # --- 2. 递推计算 ---
    # 生成一个 (num_radii, num_radii) 的矩阵，预计算所有可能的跳变成本
    # transition_costs[i, k] = cost of jumping from row k to row i
    k = np.arange(num_radii)
    i = k[:, np.newaxis]
    transition_costs = smoothness_penalty * (i - k)**2

    # 从左到右，逐列填充
    for j in range(1, extended_num_angles):
        # 对于当前列的每一个点 D_ext[i, j]
        # 我们需要找到前一列 D_ext[:, j-1] 中哪个点能以最低成本到达它
        
        # total_costs_to_reach_j[i, k] = D_ext[k, j-1] + transition_costs[i, k]
        # 这是从前一列的第 k 行到达当前列的第 i 行的总成本
        total_costs_to_reach_j = D_ext[:, j-1] + transition_costs
        
        # 找到每个 i 的最小成本和对应的 k
        # np.min(..., axis=1) 会找到每一行的最小值
        min_costs = np.min(total_costs_to_reach_j, axis=1)
        best_prev_indices_k = np.argmin(total_costs_to_reach_j, axis=1)
        
        # 更新 D 和 P 矩阵
        D_ext[:, j] = extended_map[:, j] + min_costs
        P_ext[:, j] = best_prev_indices_k
        
    # --- 3. 回溯寻找路径 ---
    # 在扩展地图的中心区域找到最优路径
    start_trace_j = num_angles + (num_angles//2) - 1
    
    # 找到在这一列的路径终点
    path = np.zeros(num_angles, dtype=int)
    path[-1] = np.argmin(D_ext[:, start_trace_j])

    # 从右到左回溯
    for j in range(start_trace_j, num_angles//2, -1):
        prev_path_point_idx = path[j - num_angles//2]
        path[j - num_angles//2 - 1] = P_ext[prev_path_point_idx, j]

    return path


def find_best_penalty_l_curve(
    cost_map: np.ndarray, 
    rr_coords: np.ndarray,
    penalty_range: np.ndarray = None,
    showfile: Optional[Union[Path, str]] = None
) -> float:
    """
    使用 L-曲线方法为动态规划寻找最佳的 smoothness_penalty。

    参数:
    ----------
    cost_map : np.ndarray
        成本地图，维度为 (半径, 角度)。
    rr_coords : np.ndarray
        与 cost_map 的行对应的半径值数组。
    penalty_range : np.ndarray, optional
        要探索的 penalty 值的对数序列。如果为 None，则使用默认范围。

    返回:
    -------
    float
        推荐的最佳 smoothness_penalty 值。
    """
    if penalty_range is None:
        # 默认探索范围从 1e-5 到 1e2
        penalty_range = np.logspace(-6, 2, 30)

    fidelities = []
    roughnesses = []
    
    logger.debug("L-curve: exploring %d points from %.0e to %.0e",
                 len(penalty_range), penalty_range[0], penalty_range[-1])
    for i, penalty in enumerate(penalty_range):
        logger.debug("penalty %d/%d: %.2e", i + 1, len(penalty_range), penalty)
        
        # 1. 计算路径
        path_r_indices = find_boundary_dp(cost_map, penalty)
        path_radii = rr_coords[path_r_indices]
        
        # 2. 计算度量
        # 数据保真度项: 路径上的平均成本
        path_costs = cost_map[path_r_indices, np.arange(cost_map.shape[1])]
        fidelities.append(np.mean(path_costs))
        
        # 平滑度/粗糙度项: 半径差的平方均值 (周期性)
        # 使用 np.roll 来计算周期性差分
        diffs = path_radii - np.roll(path_radii, 1)
        roughnesses.append(np.mean(diffs**2))
        
    logger.debug("L-curve scan complete — %d valid points", len(fidelities))
    
    fidelities = np.array(fidelities)
    roughnesses = np.array(roughnesses)

    cond=roughnesses>0.
    fidelities=fidelities[cond]
    roughnesses=roughnesses[cond]

    # Fallback: if all roughnesses are 0 (perfectly flat boundaries),
    # return the smallest penalty in the range
    if len(roughnesses) == 0:
        logger.debug("L-curve: all penalties produce zero roughness, using smallest penalty")
        valid_penalties = penalty_range[cond]
        return valid_penalties[0] if len(valid_penalties) > 0 else 1e-5
    
    # 1. (尾部截断) 移除饱和的平坦部分 (逻辑不变，但更安全)
    # 如果 roughnesses 只有一个值，则不截断
    if len(np.unique(roughnesses)) > 1:
        # 使用百分比变化来定义“平坦”，而不是绝对值
        # 从后向前，找到第一个 roughness 变化超过 1% 的点
        relative_change = np.abs(np.diff(roughnesses)) / (roughnesses[:-1] + 1e-9)
        last_significant_idx = len(roughnesses) - 1
        for i in range(len(relative_change) - 1, -1, -1):
            if relative_change[i] > 0.01: # 变化超过 1%
                last_significant_idx = i + 1
                break
            if i == 0: # 如果整个尾部都太平坦
                last_significant_idx = 1
    else:
        last_significant_idx = len(roughnesses) - 1

    # 2. (头部截断) 移除极度粗糙的“随机游走”部分
    # 找到 roughness 的最小值（最平滑的情况）
    min_roughness = np.nanmin(roughnesses)
    # 定义一个阈值：任何比最小值大超过1000倍的 roughness 都被认为是无效的
    # 这个 1000 是一个经验值，可以调整
    #roughness_threshold = min_roughness * 100000
    
    first_significant_idx = 0
    # 从头开始，找到第一个 roughness 低于阈值的点
    relative_change = np.abs(np.diff(roughnesses)) / (roughnesses[0] + 1e-9)
    #pdb.set_trace()
    for i in range(len(relative_change)):
        if relative_change[i] > 0.01:
            first_significant_idx = i
            break
            
    # 确保区间有效
    if first_significant_idx >= last_significant_idx:
        logger.debug("L-curve: invalid range after truncation, using all points")
        first_significant_idx = 0
        last_significant_idx = len(roughnesses) - 1

    logger.debug("L-curve: explored %d points, range indices [%d:%d] (%d valid)",
                 len(penalty_range), first_significant_idx, last_significant_idx,
                 last_significant_idx - first_significant_idx + 1)

    indices_to_process = np.arange(first_significant_idx, 
                                   last_significant_idx + 1)
    if len(indices_to_process) < 3:
        logger.debug("L-curve: valid segment too short (%d points)", len(indices_to_process))
        return penalty_range[len(penalty_range) // 2]
        
    fidelities_processed = fidelities[indices_to_process]
    roughnesses_processed = roughnesses[indices_to_process]
    penalties_processed = penalty_range[indices_to_process]
    
    # --- 后续所有计算都使用处理后的数据 ---

    # 归一化（log 空间，避免量纲差异导致一方主导拐点选择）
    log_f = np.log(fidelities_processed + 1e-12)
    log_r = np.log(roughnesses_processed + 1e-12)
    norm_fidelities = (log_f - np.min(log_f)) / (np.max(log_f) - np.min(log_f) + 1e-9)
    norm_roughnesses = (log_r - np.min(log_r)) / (np.max(log_r) - np.min(log_r) + 1e-9)

    # (寻找拐点的代码与之前相同，但现在作用于更干净的数据)
    p1 = np.array([norm_fidelities[0], norm_roughnesses[0]])
    p2 = np.array([norm_fidelities[-1], norm_roughnesses[-1]])
    
    line_vec = p2 - p1
    line_len_sq = np.sum(line_vec**2)
    
    if line_len_sq == 0:
        best_index_processed = 0
    else:
        points_vec = np.vstack([norm_fidelities, norm_roughnesses]).T - p1
        cross_product = np.cross(points_vec, line_vec)
        distances = np.abs(cross_product) / np.sqrt(line_len_sq)
        best_index_processed = np.argmax(distances)

    best_penalty = penalties_processed[best_index_processed]

    if showfile is not None:
        Path(showfile).parent.mkdir(parents=True, exist_ok=True)
        # 4. 可视化
        fig=plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.25,hspace=0.15)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("L-Curve")
        symsize=3
        # 绘制 L-曲线
        ax1.plot(fidelities, roughnesses, 'x', c='gray', alpha=0.5, 
                label='All Explored Points',ms=symsize)
        ax1.plot(fidelities_processed, roughnesses_processed, 'o-', 
                c='blue', label='Processed L-Curve',ms=symsize)
        
        # 标记最佳点
        ax1.plot(fidelities_processed[best_index_processed], 
                roughnesses_processed[best_index_processed], 
                 'r*', markersize=symsize*5, 
                 label=f'Best Point (Penalty = {best_penalty:.2e})')
        
        ax1.plot([fidelities_processed[0], fidelities_processed[-1]], 
                 [roughnesses_processed[0], roughnesses_processed[-1]], 
                 'g--', alpha=0.7, label='Line between endpoints')
        
        # 添加每个点的 penalty 值标签，以便调试
        #for i, penalty in enumerate(penalty_range):
        #    if i % 4 == 0: # 每隔几个点标记一次，避免拥挤
        #        ax.text(fidelities[i], roughnesses[i], f' {penalty:.1e}', fontsize=8, verticalalignment='top')
                
        ax1.set_xlabel("Fidelity")# (Average Cost on Path) -> More Jagged")
        ax1.set_ylabel("Roughness")# (Sum of Squared Diffs) -> Smoother")
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        from auto_adjust_log_ticks import auto_adjust_log_ticks

        auto_adjust_log_ticks(ax1, 'x')
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Normalized L-Curve")
        
        # 绘制归一化后的曲线
        ax2.plot(norm_fidelities, norm_roughnesses, 'o-', c='blue', 
                 label='Normalized L-Curve',ms=symsize)
        
        # (新增) 绘制端点连线，即“对角线”
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', 
                 label='Line between endpoints')
        
        # 标记找到的最佳点
        best_point_norm = (norm_fidelities[best_index_processed], 
                           norm_roughnesses[best_index_processed])
        ax2.plot(best_point_norm[0], best_point_norm[1], 
                 'r*', markersize=symsize*5, label='Elbow Point (Max Distance)')
        

        ax2.set_xlabel("Normalized Fidelity")
        ax2.set_ylabel("Normalized Roughness")
        ax2.grid(True)
        #ax2.legend()
        # 设置坐标轴范围为 [0, 1] 并保持正方形，以获得最佳几何视图
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        #ax2.set_aspect('equal', adjustable='box')

        plt.savefig(showfile,bbox_inches = "tight",dpi=300)
        plt.close()
        #pdb.set_trace()
    
    return best_penalty
