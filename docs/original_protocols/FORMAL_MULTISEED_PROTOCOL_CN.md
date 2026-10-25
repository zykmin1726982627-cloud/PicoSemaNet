# 正式多随机种子补充实验

## 实验范围

在已有 seed=42 结果的基础上，对以下四种设置新增 seed=43、44：

1. 3M FullPath；
2. 0.01M FullPath；
3. 0.005M FullPath；
4. 0.005M 无深监督（当前消融中最强）。

完成后，每种设置均拥有 seed=42、43、44 共三次试验。统计文件同时保留每次原始数据，并计算样本标准差（分母 \(n-1\)）。seed=45 因时间原因取消。

## 受控变量

- 固定数据划分：1815/259/520；
- 固定分辨率：320×320；
- 每次训练上限：200轮；
- 早停 patience：40；
- 其余优化器、调度器、EMA和损失权重与 seed=42 完全相同；
- 四种学生均使用同一个 seed=42 Performance Context 教师。

因此，这组统计衡量的是“在固定教师条件下的学生训练方差”。如果论文需要声称完整系统的端到端方差，则还需要为每个随机种子重新训练教师；这不在本轮四类模型的范围内。

## 输出

- 各次训练：`formal_paper_multiseed_v1/<variant>_seed<seed>/`；
- 逐次记录：`formal_paper_multiseed_v1/multiseed_runs.csv`；
- 均值和标准差：`formal_paper_multiseed_v1/multiseed_summary.json`。
