import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from skimage.transform import warp_polar,warp
from skimage.graph import route_through_array
from pathlib import Path
from typing import Union, Optional
import pdb
from astropy.io import fits

from astropy.visualization import PercentileInterval, AsymmetricPercentileInterval,hist,ImageNormalize,AsinhStretch,PowerStretch,ZScaleInterval,LinearStretch
from astropy.wcs.utils import proj_plane_pixel_scales as pixel_scale
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.image as mpimg
#rcParams['text.usetex'] = True
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging

from scipy.ndimage import map_coordinates
from scipy.ndimage import sobel
from .find_boundary_dp import find_boundary_dp,find_best_penalty_l_curve

logger = logging.getLogger("hii_boundary.extract.polar")

def manual_polar_transform(
    data: np.ndarray,
    center: tuple,
    output_shape: tuple,
    radius_range: tuple,
    ellipse: Optional[tuple] = None,
) -> np.ndarray:
    """
    手动实现极坐标变换，作为 warp_polar 的替代方案。
    输出数组的维度固定为 (半径, 角度)。

    参数:
    ----------
    data : np.ndarray
        输入的笛卡尔坐标图像 (y, x 索引)。
    center : tuple
        中心点坐标 (xc, yc)。
    output_shape : tuple
        期望的输出形状 (num_radii, num_angles)。
    radius_range : tuple
        半径范围 (r_min, r_max)。
    ellipse : Optional[tuple], default=None
        椭圆参数 (a, b, phi)，其中 a 和 b 是半轴长度，phi 是方位角（弧度）。
        当提供时，径向网格随角度变化，沿椭圆模式采样。
        当为 None（默认）时，保持原有的圆形行为。

    返回:
    -------
    np.ndarray
        一个形状为 output_shape 的极坐标图像。
    """
    num_radii, num_angles = output_shape
    r_min, r_max = radius_range
    xc, yc = center

    # --- 1. 创建输出网格的坐标 ---
    # `theta_grid` 和 `r_grid` 的形状都是 (num_radii, num_angles)
    
    # 角度网格：沿 axis=1 (列) 变化，从 0 到 2*pi
    theta_coords = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    theta_grid = np.tile(theta_coords, (num_radii, 1))

    if ellipse is not None:
        a, b, phi = ellipse
        cos_t = np.cos(theta_coords - phi)
        sin_t = np.sin(theta_coords - phi)
        r_ell = a * b / np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
        r_ell_grid = np.tile(r_ell, (num_radii, 1))
        r_frac = np.linspace(r_min, r_max, num_radii)
        r_grid = np.tile(r_frac, (num_angles, 1)).T * r_ell_grid
    else:
        r_coords = np.linspace(r_min, r_max, num_radii)
        r_grid = np.tile(r_coords, (num_angles, 1)).T
    
    # --- 2. 将极坐标网格映射回笛卡尔坐标 ---
    # x = xc + r * cos(theta)
    # y = yc + r * sin(theta)
    x_cart = xc + r_grid * np.cos(theta_grid)
    y_cart = yc + r_grid * np.sin(theta_grid)

    # --- 3. 使用 map_coordinates 进行插值 ---
    # `map_coordinates` 需要的坐标格式是 [[y1, y2, ...], [x1, x2, ...]]
    # 我们的 y_cart 和 x_cart 已经是2D网格，需要先将它们展平 (flatten)
    coords = np.vstack((y_cart.ravel(), x_cart.ravel()))

    # `order=1` 是线性插值，`order=3` 是三次样条插值（更平滑但稍慢）
    # `cval=np.nan` 表示如果坐标超出了图像边界，则用 NaN 填充
    polar_values_flat = map_coordinates(
        data, 
        coordinates=coords, 
        order=1, 
        cval=np.nan,
        prefilter=True # 推荐为 True 以获得更好的插值质量
    )
    
    # --- 4. 将一维结果重新塑形为输出图像 ---
    polar_image = polar_values_flat.reshape(output_shape)
    
    return polar_image

def manual_polar_to_cartesian(
    polar_image: np.ndarray, 
    center: tuple, 
    output_shape: tuple,
    radius_range: tuple
) -> np.ndarray:
    """
    手动实现极坐标到笛卡尔坐标的逆变换。

    参数:
    ----------
    polar_image : np.ndarray
        输入的极坐标图像，维度为 (半径, 角度)。
    center : tuple
        原始笛卡尔图像的中心点坐标 (xc, yc)。
    output_shape : tuple
        期望输出的笛卡尔图像的形状 (height, width)。
    radius_range : tuple
        输入极坐标图像所代表的半径范围 (r_min, r_max)。

    返回:
    -------
    np.ndarray
        一个形状为 output_shape 的笛卡尔坐标图像。
    """
    height, width = output_shape
    num_radii, num_angles = polar_image.shape
    r_min, r_max = radius_range
    xc, yc = center

    # --- 1. 创建输出笛卡尔网格的坐标 ---
    # `x_grid` 和 `y_grid` 的形状都是 (height, width)
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # --- 2. 将笛卡尔网格映射回极坐标 ---
    # 计算相对坐标
    dx = x_grid - xc
    dy = y_grid - yc

    # 计算极坐标 (物理单位)
    r_physical = np.hypot(dx, dy)
    # 将 theta 范围从 [-pi, pi] 转换到 [0, 2*pi)
    theta_physical = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

    # --- 3. 将物理极坐标转换为极坐标图像的像素索引 ---
    # 角度索引
    # 假设极坐标图像的角度范围是 [0, 2*pi)
    theta_idx = (theta_physical / (2 * np.pi)) * num_angles

    # 半径索引
    # 假设半径是线性映射的
    # r_idx = (r_physical - r_min) * (num_radii - 1) / (r_max - r_min)
    # 使用 np.interp 更为稳健，它可以自动处理边界情况
    r_idx = np.interp(r_physical, [r_min, r_max], [0, num_radii - 1])
    
    # --- 4. 使用 map_coordinates 进行插值 ---
    # `map_coordinates` 需要的坐标格式是 [[row_coords], [col_coords]]
    # 在我们的极坐标图中, row 是半径, col 是角度
    # 展平网格以进行插值
    coords = np.vstack((r_idx.ravel(), theta_idx.ravel()))

    # `cval=0` 表示如果坐标超出了极坐标图像的边界
    # (例如，半径小于r_min或大于r_max)，则用0填充
    cartesian_values_flat = map_coordinates(
        polar_image,
        coordinates=coords,
        order=1,
        cval=0.0, 
        prefilter=True
    )
    
    # --- 5. 将一维结果重新塑形为输出图像 ---
    cartesian_image = cartesian_values_flat.reshape(output_shape)
    
    return cartesian_image

def find_local_minimum_boundary(cost_map: np.ndarray, rr_coords: np.ndarray) -> tuple:
    """
    在成本地图的每个角度（列）上独立地寻找成本最低的点。
    这会生成一个可能不连续的“贪心”边界。

    参数:
    ----------
    cost_map : np.ndarray
        成本地图，维度为 (半径, 角度)。
    rr_coords : np.ndarray
        与 cost_map 的行对应的半径值数组。

    返回:
    -------
    tuple[np.ndarray, np.ndarray]
        一个包含两个一维数组的元组: (local_min_radii, angles_rad)。
        - local_min_radii: 在每个角度上成本最低点对应的半径值。
        - angles_rad: 对应的角度数组 (0 到 2*pi)。
    """
    
    # 1. 沿半径方向 (axis=0) 找到每列的最小值的索引
    # np.argmin 会返回一个一维数组，其长度等于列数
    min_cost_indices_r = np.argmin(cost_map, axis=0)
    
    # 2. 将这些半径索引转换回真实的半径值
    local_min_radii = rr_coords[min_cost_indices_r]
    
    # 3. 创建对应的角度数组
    num_angles = cost_map.shape[1]
    angles_rad = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    return local_min_radii, angles_rad

def find_global_optimal_boundary(
    data: np.ndarray, 
    xc: float, 
    yc: float, 
    rlim: list,
    num_angles: int = 360,
    smoothing_fwhm: float = 2.0,
    cost_map_smoothing_sigma: float = 5.0,
    showfile: Optional[Union[Path, str]] = None,
    method='dp'
) -> tuple:
    """
    使用全局图搜索算法寻找一条最佳的闭合边界。

    (函数文档字符串保持不变...)
    """
    # --- 输入验证 ---
    r_min, r_max = rlim
    if r_min >= r_max:
        raise ValueError("rlim 无效，r_min 必须小于 r_max。")

    # --- 步骤 1: 极坐标变换 ---
    num_radii_samples_full = int(np.ceil(r_max))
    
    #polar_image_full = warp_polar(
    #    data, 
    #    center=(xc, yc), 
    #    radius=r_max, 
    #    output_shape=(num_radii_samples_full, num_angles),
    #    scaling='linear'
    #)

    polar_image_full = manual_polar_transform(
        data=data,
        center=(xc, yc),
        output_shape=(num_radii_samples_full, num_angles),
        radius_range=(0, r_max)
    )
    #fits.writeto(str(showfile).replace('.png','.fits'),polar,overwrite=True)
    #databack = manual_polar_to_cartesian(polar, center=(xc, yc), 
    #                                     radius_range=(0,r_max),
    #                                     output_shape=data.shape)
    
    #fits.writeto(str(showfile).replace('.png','_back.fits'),databack,overwrite=True)
    if showfile is not None:
        fig=plt.figure(figsize=(8,8))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.55,hspace=0.35)
        #pdb.set_trace()
        axformatter='d.d'
        ccmap='viridis'#'RdBu_r'
        gamma=1
        barwidth=5
        interval= AsymmetricPercentileInterval(0.01,99.5)
        vmin,vmax=interval.get_limits(polar_image_full)
        #pdb.set_trace()
        ax1 = fig.add_subplot(3, 1, 1)
        extent = [0, 360, 0, r_max]
        im1=ax1.imshow(polar_image_full, origin='lower', cmap=ccmap,
                        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
                        extent=extent,aspect='auto')
        ax1.set_autoscale_on(True)
        ax1.set_ylabel("Radius (pixels)")
        ax1.axhline(r_min,color='red')
        fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.02,
                     label='Av')
    
        

    #pdb.set_trace()
    r_min_idx = int(np.floor(r_min))
    polar_image_roi = polar_image_full[r_min_idx:, :]
    
    rr_roi = np.arange(r_min_idx, num_radii_samples_full)
    
    if rr_roi.shape[0] < 2:
        logger.error("insufficient radial samples in rlim range [%.1f, %.1f]", r_min, r_max)
        return None, None, None, None
    #pdb.set_trace()
    # --- 步骤 2: 构建 dv²/dr 分数地图 ---
    # 2a. 沿半径方向平滑
    # 2a-1. 沿半径方向 (axis=0) 平滑
    sigma_radial = smoothing_fwhm / 2.355 if smoothing_fwhm > 0 else 0
    # 使用 'mirror' 或 'reflect' 处理顶部和底部的物理边界
    polar_image_smoothed_r = gaussian_filter1d(
        polar_image_roi.astype(float), 
        sigma=sigma_radial, 
        axis=0, 
        mode='mirror'
    )

    # 2a-2. 沿角度方向 (axis=1) 平滑
    # 使用 'wrap' 处理 0-360 度的周期性边界
    cost_map_smoothing_sigma = cost_map_smoothing_sigma / 2.355
    polar_image_smooth = gaussian_filter1d(
        polar_image_smoothed_r,
        sigma=cost_map_smoothing_sigma,
        axis=1,
        mode='wrap'
    )

    if showfile is not None:
        ax2 = fig.add_subplot(3, 1, 2)
        extent = [0, 360, r_min, r_max]
        vmin,vmax=interval.get_limits(polar_image_smooth)
        im2=ax2.imshow(polar_image_smooth, origin='lower', cmap=ccmap,
                        norm=PowerNorm(gamma=gamma-0.5, vmin=vmin, vmax=vmax),
                        extent=extent,aspect='auto')
        ax2.set_autoscale_on(True)
        #ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("Radius (pixels)")
        fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02,
                     label='Av')


    #pdb.set_trace()
    # 2b. (已修改) 使用 Sobel 算子计算梯度 dv
    # `sobel` 会返回一个与输入形状相同的数组
    # 它在内部处理了多个像素的考虑
    dv_map = sobel(polar_image_smooth, axis=0, mode='mirror') 
    # mode='mirror' 确保在图像顶部和底部有合理的梯度估计

    # 2c. (已修改) 计算 dr
    # 计算单个像素间隔的半径变化
    # rr_roi 是我们的半径坐标轴
    if len(rr_roi) > 1:
        pixel_dr = rr_roi[1] - rr_roi[0]
    else:
        pixel_dr = 1.0
    # Sobel 核的有效距离是2个像素
    dr = 2.0 * pixel_dr 
    # dr 现在是一个标量，我们可以在计算中直接使用它

    # 2d. (已修改) 计算分数地图
    # 现在 dv_map 和 polar_image_roi 形状相同
    score_map = np.zeros_like(dv_map, dtype=float)
    growth_mask = dv_map > 0

    if not np.any(growth_mask):
        logger.warning("no growth points found in the specified annular region")
        return None, None, None, None

    # 直接用标量 dr 进行计算，无需广播
    score2=dv_map[growth_mask]#
    #score2=polar_image_smooth[growth_mask]
    score_map[growth_mask] = (dv_map[growth_mask]*score2) / dr
    # **********************************************

    # --- 步骤 3: 从分数地图创建成本地图 ---
    valid_scores = score_map[growth_mask]
    
    if len(valid_scores) == 0:
        # 这个检查虽然前面有，但在这里再次确认更安全
        # 如果没有有效分数，创建一个均匀的高成本地图
        cost_map = np.ones_like(score_map) * 10.0 # 任意高成本
    else:
        # --- 对数变换 ---
        # 加一个很小的数 epsilon 防止 log(0)
        epsilon = 1e-10
        log_scores = np.log(valid_scores + epsilon)
        
        # 对 log_scores 进行归一化，使其范围在 [0, 1]
        # 这种归一化更稳健
        min_log_score = np.min(log_scores)
        max_log_score = np.max(log_scores)
        
        # 创建一个与 score_map 同样大小的归一化分数图
        # 先用0填充，然后只在增长点处填入归一化后的对数分数
        normalized_score_map = np.zeros_like(score_map)
        if (max_log_score - min_log_score) > 0:
            normalized_log_scores = (np.log(score_map[growth_mask] + epsilon) - min_log_score) / (max_log_score - min_log_score)
            normalized_score_map[growth_mask] = normalized_log_scores
        
        # --- 计算成本 ---
        # 成本 = 1 - 归一化对数分数
        # 这样，原始分数最高的点，其成本最低 (接近0)
        # 原始分数最低的点，其成本最高 (接近1)
        base_cost = 1e-6 # 避免0成本
        cost_map = (1.0 - normalized_score_map) + base_cost
        
        # 对没有增长的点 (dv <= 0) 赋予一个较高的惩罚成本
        # 这会强制路径避开这些区域
        cost_map[~growth_mask] = 1.1 # 比最高成本(1.0+base_cost)稍高一点
    
    # --- 步骤 4: 寻找最短路径 ---
    final_cost_map=cost_map
    fits.writeto(str(showfile).replace('.png','_scoremap.fits'),
                 score_map,overwrite=True)
    fits.writeto(str(showfile).replace('.png','_costmap.fits'),
                 cost_map,overwrite=True)
    if showfile is not None:
        ax3 = fig.add_subplot(3, 1, 3)
        extent = [0, 360, r_min, r_max]
        vmin,vmax=interval.get_limits(final_cost_map)
        im3=ax3.imshow(final_cost_map, origin='lower', cmap=ccmap,
                        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
                        extent=extent,aspect='auto')
        ax3.set_autoscale_on(True)
        ax3.set_xlabel("Angle (degrees)")
        ax3.set_ylabel("Radius (pixels)")
        fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.02,
                     label='Costmap')
        

    cost_weight = 1  # <--- 这是一个可以调整的关键参数！

    weighted_cost_map = final_cost_map * cost_weight
    cost_map_radii = (rr_roi[:-1] + rr_roi[1:]) / 2.0

    if method == 'dp':
        penalty_range = np.logspace(-6, 0, 30)
        best_penalty=find_best_penalty_l_curve(weighted_cost_map,
                                               cost_map_radii,
                                               penalty_range=penalty_range,
                                               showfile=showfile.with_name('find_best_penalty.png'))
        
        #pdb.set_trace()
        smoothness_penalty=best_penalty#0.00005
        path_r_indices = find_boundary_dp(weighted_cost_map, smoothness_penalty)
        
        # --- 步骤 5: 提取结果 ---
        final_radii = rr_roi[path_r_indices]
        num_angles = weighted_cost_map.shape[1]
        final_angles_rad = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    else:    
    # 4b. 寻找路径
        try:
            start_point_row = final_cost_map.shape[0] // 2
            indices, _ = route_through_array(
                weighted_cost_map, 
                start=(start_point_row, 0),
                end=(start_point_row, final_cost_map.shape[1]-1),
                fully_connected=True, 
                geometric=False
            )
        except Exception as e:
            logger.error("route_through_array failed: %s", e)
            return None, None, final_cost_map, rr_roi

        if not indices:
            logger.error("no path found via route_through_array")
            return None, None, final_cost_map, rr_roi
            
        # 4c. 提取结果
        path_r_indices = np.array([i[0] for i in indices])
        path_theta_indices = np.array([i[1] for i in indices])
        
        final_radii = (rr_roi[path_r_indices] + rr_roi[path_r_indices + 1]) / 2.0
        final_angles_rad = (path_theta_indices / num_angles) * 2 * np.pi
    
    # 因为 cost_map 的行数比 rr_roi 少1，所以其半径坐标轴也需要调整
    

    if final_cost_map is not None:
        local_min_radii, local_min_angles = find_local_minimum_boundary(final_cost_map, cost_map_radii)


    if showfile is not None:
        ax3.plot(final_angles_rad/np.pi*180,final_radii,'-',color='red')
        #ax3.plot(local_min_angles/np.pi*180,local_min_radii,'-',color='cyan')
        ax2.plot(final_angles_rad/np.pi*180,final_radii,'-',color='red',
                 alpha=0.5)
        plt.savefig(showfile,bbox_inches = "tight",dpi=300)
        plt.close()

    #pdb.set_trace()

    results={'final_radii':final_radii,
             'final_angles_rad':final_angles_rad,
             'final_cost_map':final_cost_map,
             'cost_map_radii':cost_map_radii,
             'local_min_radii':local_min_radii,
             'local_min_angles':local_min_angles}
    
    return results

# --- 使用示例 (与之前相同，但需要调整返回值接收) ---

if __name__ == '__main__':
    # (创建虚拟图像的代码保持不变)
    shape = (300, 300)
    yc, xc = shape[0] / 2.0, shape[1] / 2.0
    y, x = np.indices(shape)
    r_map = np.hypot(x - xc, y - yc)
    theta_map = np.arctan2(y - yc, x - xc)
    
    ring_radius = 80
    ring_width = 15
    radius_variation = 10 * np.cos(3 * theta_map + np.pi/2)
    ring_mask = (r_map > (ring_radius + radius_variation - ring_width/2)) & \
                (r_map < (ring_radius + radius_variation + ring_width/2))
    
    image = np.zeros(shape)
    image[ring_mask] = 10.0
    
    image = gaussian_filter1d(image, sigma=3)
    image += np.random.normal(0, 0.5, shape)

    search_rlim = [50, 110]
    # 接收更新后的返回值
    final_radii, final_angles, dbg_cost_map, dbg_cost_map_radii = find_global_optimal_boundary(
        image, xc, yc, rlim=search_rlim, 
        smoothing_fwhm=1, 
        cost_map_smoothing_sigma=1.0
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.2]})

    # (左图 ax1 的绘图代码保持不变)
    ax1.set_title("Image with Optimal Boundary")
    ax1.imshow(image, cmap='viridis', origin='lower')
    circle_min = plt.Circle((xc, yc), search_rlim[0], color='white', fill=False, linestyle='--', alpha=0.7)
    circle_max = plt.Circle((xc, yc), search_rlim[1], color='white', fill=False, linestyle='--', alpha=0.7)
    ax1.add_artist(circle_min)
    ax1.add_artist(circle_max)

    if final_radii is not None:
        x_coords = xc + final_radii * np.cos(final_angles)
        y_coords = yc + final_radii * np.sin(final_angles)
        x_plot = np.append(x_coords, x_coords[0])
        y_plot = np.append(y_coords, y_coords[0])
        ax1.plot(x_plot, y_plot, 'r-', linewidth=2.5, label='Optimal Boundary')

    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(0, shape[1]-1)
    ax1.set_ylim(0, shape[0]-1)

    # (右图 ax2 的绘图代码更新以使用新的返回值)
    if dbg_cost_map is not None:
        ax2.set_title("Final Cost Map (Angle vs. Radius)")
        im = ax2.imshow(dbg_cost_map, aspect='auto', cmap='magma_r', origin='lower',
                        extent=[0, 360, dbg_cost_map_radii[0], dbg_cost_map_radii[-1]]) # 使用 dbg_cost_map_radii
        fig.colorbar(im, ax=ax2, label="Cost (lower is better)")
        
        if final_radii is not None:
            ax2.plot(np.rad2deg(final_angles), final_radii, 'w-', linewidth=2, label='Shortest Path')
        ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("Radius (pixels)")
        ax2.legend()

    plt.tight_layout()
    plt.show()
