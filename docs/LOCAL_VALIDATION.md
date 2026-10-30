# 本地验证记录（交付之前）

- 测试环境：容器 Python 3.13，PyTorch CPU 2.10；**这不是用户 Ubuntu + sers_ddpm 环境**，因此服务器上仍需独立验证。
- 测试：`python -m pytest VAE/tests -v`，结果：**5 passed**。
- 数据 Excel 实际检查：400 行、63 列（A 为企业编号，B 为年份，C～BJ 为当年4季度×15变量，BK 为风险标签），16 风险、64 正常。
- 分层划分：train=48（风险 10），validation=16（风险 3），test=16（风险 3）。
- 短训练：`python -m VAE.scripts.train --epochs 2 --device cpu`，联合条件模型正常存储。`--class all --risk-number 5 --normal-number 7` 顺利生成两类 Excel，自动评价完成；另外 `--class risk --number 3` 成功。
- **重要：短训练只用于工程烟雾测试；它的指标明显不满足高质量金融模拟数据要求，也不属于论文或简历成果。**
- 正式交付 ZIP **不含**上述短训练 checkpoint、不含短生成的表格或质量图表，避免误用。用户应先在服务器重跑测试、训练正式模型，再评价。
- 不应将此次 `COS` 或 `Coverage` 的测试数值和简历现有结果比较，指标定义和训练集不同。
