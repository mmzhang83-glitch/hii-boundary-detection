# boundary_detection

HII 区（电离氢区）边界自动检测与不确定性估计。

基于 **极坐标变换 + Sobel 梯度 + Viterbi 动态规划 + 稳定边界扫描 + Bootstrap 重采样** 的全流程算法，支持圆形和椭圆先验边界检测，包含完整的真实数据 pipeline（星表查询 → 图像下载 → 点源去除 → 检测 → HTML 报告）。

---

## 安装

```bash
cd package
pip install -e .
```

### 从 GitHub 安装

```bash
# 方式一：直接 pip 安装
pip install git+https://github.com/mmzhang83-glitch/hii-boundary-detection.git

# 方式二：克隆后安装
git clone https://github.com/mmzhang83-glitch/hii-boundary-detection.git
cd hii-boundary-detection
pip install -e .
```

#### 验证安装

```bash
python -c "from boundary_detection import detect_hii_boundary; print('导入成功')"
```

#### 运行测试

```bash
# 合成模型测试（Sharp Step 模型，跳过 bootstrap 快速验证）
python test/run_test_plan.py --model "Sharp Step" --bootstrap-n 0

# 全部合成模型测试
python test/run_test_plan.py

# 快速验证
python test/quick_test.py --skip-real
```

依赖：`numpy>=1.24,<2` `scipy` `matplotlib` `astropy` `pyyaml` `scikit-image` `astroquery` `watroo`
Python ≥ 3.10

---

## 快速开始

```python
from boundary_detection import detect_hii_boundary

result = detect_hii_boundary(data, xc=150, yc=150, rmax=80)
print(result["boundary_radii"])       # (360,) 边界半径
print(result["boundary_uncertainty"])  # (360,) 1σ 误差
```

---

## 算法详解

### 整体流程

```
输入: 2D图像 data + 中心 (xc,yc) + 搜索半径 rmax
  │
  ▼
[1] 极坐标变换 ─── warp_polar / manual_polar_transform
  │               直角坐标 → (r,θ) 极坐标网格
  │               skimage.transform.warp_polar
  ▼
[2] 预处理 ────── 径向高斯平滑 (scipy.ndimage.gaussian_filter1d)
  │               角度方向平滑 (同上)
  ▼
[3] 梯度分数图 ─── Sobel 径向梯度 (scipy.ndimage.sobel)
  │               误差加权 + 角度SNR + 梯度一致性惩罚
  │               score = G² / pixel_dr
  │               cost = -log(score/Σscore + ε)
  ▼
[4] DP 最优路径 ── Viterbi 动态规划 + L-curve 自动 penalty
  │               在 cost map 上找 360° 最低 cost 闭合路径
  ▼
[5] 稳定边界扫描 ── 遍历 rmin → 边界差 diff 计算 → 寻找稳定区
  │               多稳定区评分：mean_cost + 长度 + 梯度方向一致性
  ▼
[6] Bootstrap ─── N 次噪声重采样 → 重新检测 → 逐角度 std = 1σ
  │               multiprocessing 并行
  ▼
输出: boundary_radii, boundary_uncertainty, boundary_x/y
```

### 第一步：极坐标变换

**数学原理**

以 `(xc, yc)` 为原点，将圆形 ROI `[rmin, rmax]` 变换到 `(r, θ)` 极坐标网格：

```
x = xc + r · cos(θ)
y = yc + r · sin(θ)

其中 r ∈ [rmin, rmax], θ ∈ [0, 2π)
```

**实现方法**

使用 `skimage.transform.warp_polar()` 进行双线性插值变换：

- 径向采样点数：`Nr = ⌊rmax⌋ + 1`（可以容纳 rmax 处的边界）
- 角度采样点数：`Nθ = 360`（每度一个值，匹配 360° 周期性边界）
- 输出形状：`(Nr, 360)` 的二维浮点数组
- 插值方法：双线性（`order=1`），保证亚像素精度

备选方案 `manual_polar_transform()` 用于需要精确控制采样的场景（如 uncertainty 面板的极坐标背景），直接使用三角函数映射 + `scipy.ndimage.map_coordinates`。

**Python 包**：`skimage.transform.warp_polar`、`scipy.ndimage.map_coordinates`

### 第二步：预处理平滑

**数学原理**

两个独立的高斯平滑步骤：

1. **径向平滑**（沿 axis=0）：抑制单条射线上的像素噪声
2. **角度平滑**（沿 axis=1）：消除相邻角度间的不连续性

```
I_smooth(r, θ) = G_σ_radial(r) ∗ I_raw(r, θ)   (每列独立)
I_smooth(r, θ) = G_σ_angular(θ) ∗ I_smooth(r, θ) (每行独立)
```

其中 `G_σ` 是一维高斯核，标准差 σ = FWHM / 2.355。

**实现方法**

`scipy.ndimage.gaussian_filter1d` 逐轴应用，通过 `axis` 参数控制方向。径向和角度平滑的 σ 分别由 `smoothing_fwhm` 和 `cost_map_smoothing_sigma` 控制，设为 0 则跳过。

**Python 包**：`scipy.ndimage.gaussian_filter1d`

### 第三步：梯度分数图与 Cost Map

#### 3a. Sobel 梯度

**数学原理**

沿径向（axis=0）应用 Sobel 算子计算梯度 `G = ∂I/∂r`：

```
Sobel 核（径向）:
    [-1, 0, 1]
    [-2, 0, 2]  · 1/8
    [-1, 0, 1]

G(r, θ) = Sobel(I_smooth)[r, θ]
```

**实现方法**

`scipy.ndimage.sobel(polar_smooth, axis=0, mode='mirror')`。

`mode='mirror'` 处理边界：在 rmin 和 rmax 处镜像填充，避免边界伪影。

**Python 包**：`scipy.ndimage.sobel`

#### 3b. 误差加权

当有仪器误差图 `error_map`（每个像素的 1σ 测量不确定度）时，通过极坐标变换同步映射，然后用 `1/σ²` 权重抑制高噪声像素：

```
G_weighted(r, θ) = G(r, θ) / σ²(r, θ)
```

无 error_map 时跳过此步骤（所有权重为 1）。

#### 3c. 梯度角度平滑（可选）

Sobel 梯度后沿角度方向再做一次高斯平滑（参数 `gradient_smoothing_sigma`），减小相邻角度之间的梯度跳变。

#### 3d. 角度 SNR 加权（可选）

对于空间变化噪声（如 GLIMPSE mosaic 的覆盖率差异），启用 `angular_snr_weighting`：

1. 沿角度方向高斯平滑信号和噪声
2. SNR(θ) = signal_smooth(θ) / noise_smooth(θ)
3. score 乘以角度相关的 SNR 权重

**Python 包**：`scipy.ndimage.gaussian_filter1d`

#### 3e. 梯度符号一致性惩罚

**数学原理**

真实边界的梯度符号在相邻角度之间一致（均为正梯度），而噪声区域的符号随机。定义一致性惩罚：

```
sign(r, θ) = sign(-G(r, θ))  # 翻转为正表示边界特征
coherence(r, θ) = 角度方向上 sign 的局部一致性
penalty(r, θ) = β · (1 - coherence(r, θ))
```

**实现方法**

沿角度方向（axis=1）滑动窗口计算梯度符号的标准差，作为不一致性的度量。`coherence_penalty_weight`（β）控制惩罚强度，0 为禁用。

**Python 包**：`scipy.ndimage.generic_filter`（滑动窗口）、`numpy.std`

#### 3f. Score 与 Cost

**数学原理**

```
pixel_dr = 2.0              # Sobel 核有效径向距离（像素）

score(r, θ) = G(r, θ)² / pixel_dr    (G > 0)
score(r, θ) = 0                       (G ≤ 0)

cost(r, θ) = -log( score(r,θ) / Σscore + ε )
```

- `G²` 强化强梯度，抑制弱梯度
- `/pixel_dr` 归一化到单位像素距离
- 负梯度（亮→暗）置零，因为 HII 区边界是从暗到亮的正梯度
- `-log` 将 score 转换为 cost：强梯度 → 低 cost
- `ε = 1e-10` 防止 log(0)

**Python 包**：`numpy`

### 第四步：DP 最优路径搜索

**数学原理**

在 cost map `C(r, θ)`（尺寸 Nr × 360）上，寻找一条周长为 360 的闭合路径 `p(θ)`（θ = 0, 1, ..., 359），使得总 cost 最小：

```
minimize  Σ_θ C(p(θ), θ) + α · Σ_θ (p(θ) - p(θ-1))²

其中 p(359) 回到 p(0)，形成闭合边界
```

- 第一项：数据保真度（边界的 cost 总和）
- 第二项：平滑惩罚（相邻角度半径跳变的平方）

**实现方法：Viterbi 算法**

将每个角度 θ 视为动态规划的一个"阶段"，每个阶段有 Nr 个候选状态（半径索引 i），状态转移代价为：

```
transition(i, k) = α · (i - k)²
```

递推公式：

```
dp[θ][i] = C(i, θ) + min_k { dp[θ-1][k] + α · (i - k)² }
```

最后一步（θ = 359）必须回到 θ = 0 的状态，形成闭合路径。通过回溯（backtracking）恢复完整路径。

**Python 包**：`numpy`（纯 NumPy 实现，非 scipy 的 Viterbi）

#### 4a. L-curve 自动 Penalty

平滑惩罚 α 无法先验确定。使用 L-curve 方法自动选择：

1. 在 log 空间均匀采样 30 个 α 值（`np.logspace(-6, 2, 30)`）
2. 对每个 α 跑一次 DP，记录数据保真度 f(α) 和路径粗糙度 r(α)
3. 在对数坐标下画出 (log f, log r) 曲线，找曲率最大点
4. 曲率最大点对应保真度与平滑度的最优 trade-off

曲率公式（离散三点法）：

```
κ(i) = 2 · (x'y'' - y'x'') / (x'² + y'²)^(3/2)
```

其中 x = log f, y = log r，导数用中心差分近似。

**Python 包**：`numpy`（对数采样、曲率计算）

### 第五步：稳定边界扫描

**数学原理与动机**

DP 需要指定 rmin（内部屏蔽半径：忽略中心附近区域的梯度），但 rmin 无法先验确定。rmin 太小 → 噪声中心干扰，rmin 太大 → 边界被屏蔽。算法利用一个现象：存在一个 rmin 的范围使得检测到的边界**基本不变**。

**实现方法**

1. 在 `[rmin_start, rmax_limit]` 内均匀采样 `n_steps` 个 rmin 值
   - `rmin_start = max(rmax × rmin_start_ratio, rmin_min_pixels)`
   - `rmax_limit = rmax × rmax_limit_ratio`
2. 每个 rmin 跑一次 DP，提取边界
3. 以第一个 rmin 的边界为参考，计算所有边界的差异：
   ```
   diff[i] = mean(|boundary_i - boundary_ref|)
   Δdiff[i] = |diff[i] - diff[i-1]|
   ```
4. 寻找稳定区：连续 `stable_window` 个 rmin 满足 `Δdiff < stable_threshold`
5. 有多个稳定区时，加权评分选择最优：
   - `mean_score`（梯度分）：0.4 权重
   - `length`（稳定区宽度）：0.3 权重
   - `std`（梯度方向一致性）：0.3 权重
6. 稳定区内所有边界的逐角度平均值作为最终输出

**Python 包**：`numpy`、`scipy.ndimage.gaussian_filter1d`

### 第六步：Bootstrap 不确定性估计

**数学原理**

通过参数化 Bootstrap 估计边界检测的逐角度标准差：

1. 假设测量误差服从高斯分布 `N(0, σ²(x,y))`
2. 生成 N 组噪声实现，每组在原始图像上叠加独立噪声
3. 每组跑一次完整的边界检测
4. 逐角度计算 N 个边界的标准差 → 1σ 不确定度

```
uncertainty(θ) = std({ boundary_i(θ) | i = 1..N })
```

**三种场景**

| 场景 | clean_image | error_map | 方法 |
|------|:-----------:|:---------:|------|
| A | ✓ | ✓ | `clean_image + N(0, σ²)` → 每次加独立噪声检测 |
| B | ✗ | ✓ | `noisy_image + N(0, σ²)` → 检测，√2 修正 |
| C | ✗ | ✗ | 跳过，返回 None |

场景 B 的 √2 修正是因为图像本身已含噪声，叠加噪声使总方差加倍，因此 `σ_boundary = σ_bootstrap / √2`。

**实现方法**

- `multiprocessing.Pool` 并行执行（`n_workers` 控制进程数）
- MAD outlier 剔除：`|r_i - median| > k · MAD` 的迭代被标记并移除（`k=3.0`）
- 稳定边界过滤：Bootstrap 中产生的不合理边界被剔除

**Python 包**：`numpy`、`multiprocessing`

---

## 椭圆先验检测

当气泡具有显著椭率时，可以用椭圆先验约束边界搜索：

```python
from boundary_detection import detect_hii_boundary_elliptical

result = detect_hii_boundary_elliptical(
    data, xc=150, yc=150, a=80, b=60, phi=np.pi/6
)
```

参数：半长轴 `a`、半短轴 `b`、位置角 `phi`（弧度）。算法类似圆形检测但在椭圆坐标系中进行极坐标变换。

---

## 真实数据 Pipeline

针对 Spitzer GLIMPSE 8μm 图像中 Churchwell+ (2006) 的 bubble 目录。

### 整体流程

```
[1] fetch_bubble_catalog.py    → Vizier 查询 Churchwell+ 2006 星表 → CSV
[2] download_glimpse_images.py → IRSA SIA 查询 → 下载 8μm mosaic FITS
[3] preprocess_glimpse.py      → 亮源掩膜 + à trous 小波 inpainting → cleaned.fits, sigma.fits
[4] test_real_bubbles.py       → 边界检测 + Bootstrap → result.json, arrays.npz
[5] test_report.py             → HTML 报告
```

统一入口：`python run_test_plan_real.py`

### 预处理：点源去除

**数学原理：à trous 小波变换**

B3 spline 小波基的 à trous（带孔）算法进行 5 尺度分解：

```
I = w1 + w2 + w3 + w4 + w5 + c5

其中 w_j = c_{j-1} - c_j  (小波平面)
     c_j = c_{j-1} ∗ h_j   (卷积 B3 spline 核，核膨胀 2^j)
     c_0 = I
```

点源主要集中在 w1（最小尺度），通过迭代 inpainting 在 w1 平面上替换点源像素为周围中值。

**实现步骤**

1. **亮源掩膜**：`DAOStarFinder`（`photutils`）检测 + 峰值分档
   - peak < 1000 → 掩膜半径 5 px
   - 1000 ≤ peak < 2000 → 掩膜半径 10 px
   - peak ≥ 2000 → 掩膜半径 40 px
2. **亮源回填**：biharmonic 插值（`scipy.interpolate`）或 bg+噪声回填
3. **小波 inpainting**：`watroo.AtrousTransform`（B3spline）5 尺度分解，`inpaint_iters` 次迭代
4. **Refine**：残差图二次 DAOStarFinder 检测 + 精填（`n_refine_iters` 次）
5. **Sigma map**：在 w1 平面上滑动 MAD 窗口（`mad_window` 大小），稀疏网格（`sigma_stride` 步长），`scipy.interpolate.griddata` 插值到全图

**Python 包**：`watroo`（AtrousTransform/B3spline）、`photutils`（DAOStarFinder）、`astropy.io.fits`、`scipy.ndimage`、`scipy.interpolate.griddata`、`skimage.filters`、`skimage.morphology`

### 检测：下采样加速

`cleanmap_downsamp_scale` 参数控制下采样因子：

```python
# 原图
I_down = gaussian_filter(cleaned, sigma=downsamp/2)[::downsamp, ::downsamp]
```

先用高斯抗锯齿（σ = downsamp/2），再等距采样。检测在缩略图上进行，结果按 `downsamp` 倍数缩放回原始分辨率。

---

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `smoothing_fwhm` | 2.0 px | 径向高斯平滑 FWHM，0=禁用 |
| `cost_map_smoothing_sigma` | 5.0 px | 角度方向平滑 σ，0=禁用 |
| `gradient_smoothing_sigma` | 0.0 px | Sobel 后角度平滑 σ，0=禁用 |
| `boundary_smoothing_sigma` | 0.0 px | 最终边界平滑 σ，0=禁用 |
| `rmin_start_ratio` | 0.05 | rmin 起始 = max(rmax × ratio, rmin_min_pixels) |
| `rmin_min_pixels` | 5.0 px | rmin 硬下限 |
| `rmax_limit_ratio` | 0.7 | rmin 上限 = rmax × ratio |
| `n_steps` | 0 | rmin 扫描点数（0=auto，步长约 2 px） |
| `stable_window` | 3 | 稳定区连续点数 |
| `stable_threshold` | 1 | Δdiff 稳定性阈值 |
| `contrast_min` | 0.05 | 边界对比度阈值，0=禁用 |
| `coherence_penalty_weight` | 0.5 | 梯度符号一致性惩罚 β，0=禁用 |
| `angular_snr_weighting` | false | 角度 SNR 加权 |
| `n_bootstrap` | 100 | Bootstrap 迭代次数，0=跳过 |
| `n_workers` | 1 | 并行进程数 |
| `detect_rising_edge` | true | true=暗→亮（rising edge） |

完整参数见 `hii_detection_config.yaml`。优先级：**显式传参 > 配置文件 > 内置默认值**。

---

## 对抗噪声机制（按应用顺序）

1. `error_map` 加权 — `1/σ²` 抑制高噪声像素
2. `angular_snr_weighting` — 角度结构 SNR 调制 score
3. `gradient_smoothing_sigma` — 角度方向梯度平滑
4. `coherence_penalty_weight` — 梯度符号一致性惩罚
5. `rmin_min_pixels` — 硬下限防止 DP 穿越噪声中心
6. `boundary_smoothing_sigma` — 最终边界平滑
7. `contrast_min` — 边界对比度过滤

---

## 运行测试

```bash
# 合成模型测试
python run_test_plan.py                          # 全部 4 模型 × 4 测试类型
python run_test_plan.py --model "Sigmoid"        # 仅 Sigmoid
python run_test_plan.py --bootstrap-n 0          # 跳过 Bootstrap（最快）

# 真实数据
python run_test_plan_real.py                     # 全流程
python run_test_plan_real.py --only-plot         # 仅重绘图

# 打包后快速验证
python quick_test.py                             # Sigmoid + Gaussian + Real
```

---

## 包结构

```
boundary_detection/                # 核心包
├── detect_hii_boundary.py         # 统一入口 + 参数解析
├── detect_hii_boundary_elliptical.py  # 椭圆先验
├── find_stable_boundary.py        # 稳定边界调度
├── find_stable_boundary_by_scan.py  # rmin 扫描
├── extract_circle_boundary.py     # 极坐标 → 梯度 → cost → DP
├── find_boundary_dp.py            # Viterbi DP + L-curve
├── find_global_optimal_boundary.py  # 极坐标变换
├── bootstrap_boundary.py          # Bootstrap 误差
├── hii_detection_config.yaml      # 算法参数
└── data/                          # 捆绑数据（catalog + FITS）

test/                              # 测试套件
├── test_models.py                 # 4 种合成模型生成器
├── test_runner.py                 # baseline/noise/center/rmax
├── test_diagnostics.py            # 诊断图（overlay/pipeline/error）
├── test_report.py                 # MD + HTML 报告
├── run_test_plan.py               # 合成测试编排
├── run_test_plan_real.py          # 真实数据全流程编排
├── preprocess_glimpse.py          # à trous 小波点源去除
├── fetch_bubble_catalog.py        # Vizier 星表查询
├── download_glimpse_images.py     # IRSA SIA 下载
├── test_real_bubbles.py           # 真实图像检测
└── quick_test.py                  # 快速流程测试
```
