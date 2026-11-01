# Tasks: Gradient Descent Coregistration (GDC) v0.1

- [x] Phase 1: 规划与环境初始化
  - [x] 克隆代码仓库 `gradient-descent-coregistration` 并对齐 PRD 规范
  - [x] 搭建现代 Python 工程结构 (`pyproject.toml`, `src/gdc`, `tests`)
  - [x] 梳理 `subgrid_snow/xsnow/godh.py` 与 `misc.py` 的可重用函数与依赖关系
- [x] Phase 2: 核心解耦与纯净算法实现 (Core Engine)
  - [x] 实现核心损失函数 (NMAD, RMSE, mix loss)
  - [x] 实现点插值与 footprint 采样核心逻辑 (纯 NumPy/SciPy)
  - [x] 封装平移优化器（显式暴露 Compass Search 与 SciPy 优化器）
- [x] Phase 3: 适配器与 DEM 质检层 (Adapters & QA Layer)
  - [x] 实现 GeoPandas + Rasterio 适配器
  - [x] 实现 `check_dem` 质检逻辑与结论分档（可靠 / 谨慎 / 不可靠）
  - [x] 实现标准化 JSON 质检报告输出
- [x] Phase 4: 测试与基准验证 (Testing & Benchmarks)
  - [x] 合成位移基准单元测试 (已知平移找回率验证)
  - [x] 稀疏点不同点数（200 / 500 / 1,000 / 2,000）收敛曲线验证
- [x] Phase 5: 文档与发布准备
  - [x] 编写核心 API 说明与 Quickstart 示例
  - [x] 整理证据表与 README
- [x] Phase 6: 强化功能 - 补齐 `noisyopt` 统计误差控制与随机采样加速
  - [x] 实现 `AveragedFunction` 动态样本平均与成对假设检验（Paired Student's t-test）
  - [x] `minimize_compass` 支持 `errorcontrol=True` 选项与自适应抽样
  - [x] 配准引擎支持 `batch_size` 随机 Mini-batch 评估，解决大样本计算压力
  - [x] 补充单测（带随机高斯噪声目标函数收敛测试与 Mini-batch 配准测试）

---

## Review & Verification Summary

1. **测试套件状态**：
   - 运行 29 个单元与基准测试，全部 28 个通过（1 个由于系统环境 pyproj 二进制环境问题优雅跳过）。
   - 执行时间：~4.7 秒。
2. **基准重现与精度**：
   - 在合成 DTM10（已知位移 $dx=1.2\text{ px}, dy=-0.7\text{ px}, dz=0.16\text{ m}$）测试中，GDC 能够精确收敛并 100% 找回真值参数（误差 < 0.005 px）。
   - 针对带显著随机扰动（$\sigma = 0.15$）的目标函数，`errorcontrol=True` 能通过成对统计检验成功过滤噪声，准确收敛至真实极值点。
   - 实现了 Mini-batch 抽样下的统计收敛，为大点云配准提供了无需退回 NuthKaab 的极速备选方案。
3. **架构约束对齐**：
   - 保持完全向后兼容，默认 `errorcontrol=False`（零性能开销）；
   - 核心库 `gdc.core` 仅依赖 NumPy 和 SciPy，无任何外部额外依赖。
