# 测试套件说明

## 测试内容

### 合成模型测试（run_test_plan.py）

对 4 种合成 HII 区模型执行 4 类标准测试。

| 模型 | 径向轮廓 | 可调参数 |
|------|----------|----------|
| Sharp Step | 阶跃函数：A (r≤R₀)，B (r>R₀) | `crater_r0`, `crater_a`, `crater_b` |
| Linear Ramp | 线性过渡区 | `ramp_half_widths`: [5, 10] |
| Sigmoid | 平滑 S 形过渡 | `sigmoid_ks`: [1, 5] |
| Gaussian Ring | 高斯峰在 R₀ 处 | `ring_sigmas`: [5, 10] |

| 测试类型 | 目的 | Bootstrap | 通过标准 |
|----------|------|:---------:|----------|
| baseline | 干净图像，无噪声 | 禁用 | MRE < 2.0 px |
| noise | 噪声扫描（5–100% 信号幅值） | N 次 | MRE < 5.0 px |
| center | 中心偏移敏感度（10–50%） | N 次 | 仅报告 |
| rmax | 搜索半径敏感度（150 px） | N 次 | 仅报告 |

### 真实数据测试（run_test_plan_real.py）

对 Churchwell+ (2006) 5 个 bubble 的 GLIMPSE 8μm 图像执行完整 pipeline（catalog → download → preprocess → detect → report）。

## 代码结构

```
├── run_test_plan.py              # 合成模型测试编排
├── run_test_plan_elliptical.py   # 椭圆先验测试编排
├── run_test_plan_real.py         # 真实数据全流程编排
├── quick_test.py                 # 快速验证（Sigmoid + Gaussian + Real）
├── test_models.py                # 合成模型定义
├── test_generators.py            # 径向轮廓 + 2D 图像生成
├── test_runner.py                # baseline/noise/center/rmax 测试
├── test_runner_elliptical.py     # 椭圆测试
├── test_real_bubbles.py          # 真实图像边界检测
├── test_analysis.py              # 期望边界 + MRE/RMS 指标
├── test_diagnostics.py           # 诊断图（overlay/pipeline/error）
├── test_report.py                # MD + HTML 报告
├── fetch_bubble_catalog.py       # Vizier 星表查询
├── download_glimpse_images.py    # IRSA SIA 下载
├── preprocess_glimpse.py         # à trous 小波点源去除
├── test_config.yaml              # 合成测试参数
├── test_config_elliptical.yaml   # 椭圆测试参数
├── test_config_real.yaml         # 真实数据 pipeline 参数
└── logging_setup.py              # 日志配置
```

## 使用方法

```bash
# 合成模型
python run_test_plan.py                          # 全部测试
python run_test_plan.py --model "Sigmoid"        # 指定模型
python run_test_plan.py --bootstrap-n 0          # 跳过 Bootstrap（最快）

# 真实数据
python run_test_plan_real.py                     # 全流程
python run_test_plan_real.py --only-plot         # 仅重绘图

# 快速验证
python quick_test.py                             # 全部
python quick_test.py --skip-real                 # 跳过真实数据
```

### CLI 参数（run_test_plan.py）

| 参数 | 说明 |
|------|------|
| `--model MODEL` | 仅运行匹配此子串的模型 |
| `--bootstrap-n N` | Bootstrap 迭代次数 |
| `--n-workers N` | 并行进程数 |
| `--skip-noise` | 跳过噪声扫描 |
| `--no-sensitivity` | 跳过中心偏移 + rmax 测试 |
| `--seed N` | 随机种子 |
| `--config PATH` | 配置文件路径 |

### 诊断图

| 图 | 内容 |
|----|------|
| overlay.png | 全图 + 缩放 + 误差带 + 极坐标 uncertainty 面板 |
| pipeline.png | 6-panel：极坐标 → 平滑 → 梯度 → score → cost → 截面 |
| error.png | 检测 − 期望 vs 角度 |
| algo_diag.png | rmin 扫描 Δdiff 曲线 + 候选边界 |
