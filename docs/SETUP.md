# 服务器部署（目标：/home/wqzheng/Financial platform）

项目压缩包 `Financial-platform-phase1-yearly-layout.zip` 上传到 `/home/wqzheng/` 后执行：

```bash
conda activate sers_ddpm
cd /home/wqzheng
unzip -n 'Financial-platform-phase1-yearly-layout.zip' -d /home/wqzheng
cd '/home/wqzheng/Financial platform'
python -m pip install -r requirements.txt
python -m pytest VAE/tests -v
```

通过后，正式训练 **一个联合条件 TimeVAE**：

```bash
python -m VAE.scripts.train --config config/vae.yaml
```

训练完成后生成两类各 50 家并自动评价：

```bash
python -m VAE.scripts.generate --config config/vae.yaml --class all --number 50 --output VAE/outputs/generated/run_001
```

checkpoint：`VAE/outputs/checkpoints/timevae_conditional_best.pt`。日志：`VAE/outputs/logs/conditional_history.csv`。Excel：`VAE/outputs/generated/run_001/`。质量报告：`VAE/outputs/evaluation/run_001/`。

为了安全，`unzip -n` 避免覆盖。请先确认 `pwd` 显示 `/home/wqzheng/Financial platform`。整个阶段不需要进入现有的 SERS 项目文件夹，也不要触碰那两个仓库、checkpoint 或数据。
