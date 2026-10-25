# Projection-head与基线补充实验协议

## 已有结果映射

`MicroSemaNet No-FPFCS`对应M2：保留4,844参数MicroSemaNet/PicoLite部署核心和KD、Edge训练目标，但关闭边界、纹理、SDF全路径特征概念监督。Seeds 42、43、44已经完成，因此不重复训练。

`Projection-head`对应`picolite_learned_head_fullpath_distill`：在U形路径九个阶段分别使用普通可学习`1x1`投影头预测边界、纹理和SDF。投影头只在训练期存在，推理核心仍为4,844参数。

## 新增正式基线

- 原始U-Net：标准双3x3卷积、四次下采样/上采样，base channels为32，共7,763,041个训练及推理参数；
- 约5K Tiny U-Net：只保留与主模型完全相同的4,844参数部署核心，仅用分割损失训练，不含FPFCS、Projection head、解码器深监督、KD或Edge loss。

## 训练顺序

在当前`no-edge`运行完成后，按广度优先原则执行：

1. Projection-head Seed 42；
2. 原始U-Net Seed 42；
3. Tiny U-Net 5K Seed 42；
4. Projection-head Seed 43；
5. Projection-head Seed 44；
6. 恢复此前尚未运行的encoder-only语义路径消融。

所有新运行使用锁定的70/10/20无泄漏划分、320分辨率、batch size 8、200轮上限和40轮早停耐心。
