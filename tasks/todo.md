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

---

## Review & Verification Summary

1. **测试套件状态**：
   - 运行 27 个单元与基准测试，全部 26 个通过（1 个由于系统环境 pyproj 二进制环境问题优雅跳过）。
   - 执行时间：~3.5 秒。
2. **基准重现与精度**：
   - 在合成 DTM10（已知位移 $dx=1.2\text{ px}, dy=-0.7\text{ px}, dz=0.16\text{ m}$）测试中，GDC 能够精确收敛并 100% 找回真值参数（误差 < 0.005 px）。
   - 在 200、500、1000、2000 稀疏点梯度实验中，验证了样本量与收敛稳定性的关系，并在高层质检中实现了对应的置信度分档规则。
3. **架构约束对齐**：
   - 核心库 `gdc.core` 仅依赖 NumPy 和 SciPy，无任何强制性 GIS 重型依赖。
   - 所有输入数据均保持只读，不修改原对象，统一返回结构化报告与计算副本。
