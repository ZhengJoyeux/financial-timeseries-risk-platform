# LSTM — 下一阶段（仅建目录，当前不训练）

后续实现 Multi-head LSTM：**15 条独立财务指标序列→15 个小 LSTM head→Concat→三层全连接→Softmax 二分类**。

训练对比组：`real-only`、`real+generated`。归一化依旧仅在真实训练企业上拟合；企业级固定的 48/16/16 划分不能改变，生成样本只能进入训练。后续补充缺失掩码、随机数值遮蔽、整变量遮蔽、早停与 PR-AUC/ROC-AUC/Recall/micro-F1/macro-F1/KS，并在独立测试集上评价。

**提醒：** 你的参考 LSTM 论文使用每变量一个 LSTM head；这里不是 Transformer 多头注意力。目前未实现模型/训练代码，以符合“先写 VAE”的阶段安排。
