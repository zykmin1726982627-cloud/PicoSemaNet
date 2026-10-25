# 最后30小时补充实验协议

## 队列目标

在不改变锁定数据划分、不接触测试集调参且不开发新模型结构的前提下，补齐论文最关键的统计证据。

## 优先级与顺序

1. 原始U-Net与4,844参数Tiny U-Net的Seeds 43、44；
2. FullPath-NDS去除KD的Seeds 43、44；
3. M4无边界、M5无纹理、M6无SDF的Seeds 43、44；
4. 若时间允许，去除Edge loss的Seeds 43、44。

基线顺序采用广度优先：U-Net 43、Tiny 43、U-Net 44、Tiny 44。M4–M6同样先完成全部Seed 43，再进行Seed 44。

## 固定协议

- 数据：`paper_split_70_10_20_seed20260912.json`；
- 图像尺寸：320×320；batch size：8；AMP开启；
- 常规实验：200轮上限、40轮早停耐心；
- M4–M6：100轮上限、20轮早停耐心，与既有Seed 42预算一致；
- 学习率：1e-3，ReduceLROnPlateau耐心5轮，最低学习率1e-6；
- 每个模型只依据验证Dice保存`best.pt`，训练结束后测试一次。
