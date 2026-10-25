# 教授要求的 FPFCS 固定预算对照

队列：`run_professor_fpfcs_100epoch.py`。结果目录：`formal_paper_professor_fpfcs_100ep_v1/`。

共六次从头训练，按种子配对顺序运行：Full 42、No FPFCS 42、Full 43、No FPFCS 43、Full 44、No FPFCS 44。

- 主干：PicoSemaNet（代码名称 `picolite_fullpath_distill`），推理参数 4,844；两组保持同一结构和初始化规则。
- Full：Segmentation + 九阶段 FPFCS + KD；FPFCS 边界/纹理/SDF 权重为 0.18/0.05/0.12。
- No FPFCS：Segmentation + KD；只将上述三个 FPFCS 损失权重设为零。
- 两组关闭 EC（`lambda_edge=0`），关闭 decoder deep supervision；固定教师 `formal_paper_seed42_v1/teacher_performance_context_seed42/best.pt`，KD 权重 0.25、温度 2。
- 数据固定为 `paper_split_70_10_20_seed20260912.json`：1815/259/520 张训练/验证/测试图像；不使用样本截断。
- 320×320，batch 8，AdamW，初始学习率 0.001，weight decay 0.0001；ReduceLROnPlateau 使用验证 Dice，patience 5、factor 0.5、最低学习率 1e-6；EMA 0.995。
- 完整训练 100 轮，早停 patience 设为 101，保证不会在本预算内提前停止；按最高验证 Dice 保留 best.pt，结束后载入该 checkpoint 测试。
- 保留原辅助项 warmup 8 轮、floor 0.15、aux/seg cap 0.25。两组所有共同超参数一致。

## BF1

预测最终分割 logits 经 sigmoid 后以 0.5 二值化，参考掩膜以 0.5 二值化。两者均在 320×320 评估空间提取 3×3 形态学梯度边界。边界以离散圆盘分别膨胀，包含 dx²+dy²≤4 的像素，即两像素欧氏距离容差。

Precision = 预测边界中距任意参考边界像素不超过 2 像素的比例；Recall = 参考边界中距任意预测边界像素不超过 2 像素的比例。采用邻域存在性匹配，允许多对多，不进行一对一配对。BF1=2PR/(P+R)，P+R 为零时结果为零；两边界皆空且掩膜相同时为 1。按图像计算后取均值。

本队列显式使用 `--boundary-distance euclidean`。历史运行仍保留其原来的 Chebyshev 方形邻域定义；历史 BF1 不能直接与本队列 BF1 混用。

## 运行与结果

在项目目录使用 `.venv/bin/python run_professor_fpfcs_100epoch.py` 启动；`--prepare-only` 只登记；`--status` 查看状态。

queue.json 保存六次命令、输入/源码 SHA256、运行状态和 PID；每次训练独立保存日志、配置、best.pt、last.pt 与测试指标。队列使用文件锁防止重复启动，不覆盖未完成的训练目录。每次完成后更新 results_per_seed.csv 和 summary.json，报告 Dice/BF1/IoU/Accuracy 的均值、样本标准差（ddof=1），以及同种子 Full−No FPFCS 的差值和汇总。失败时保留现场并停止后续队列。
