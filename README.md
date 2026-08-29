# Flow Matching 学习项目

这是一个面向初学者的二维 Flow Matching 学习项目，随后扩展到条件生成、最优传输和简单图像生成。项目强调：先理解速度场，再理解高维图像 latent，最后再连接科学图像生成与 SIQA/SIU2A。

## 项目结构

```text
src/utils          公共基础工具：采样、积分、绘图、指标
src/modules        模型、时间/条件嵌入、损失、OT耦合
src/pipelines      完整训练与评估流程
src/cli            命令行和pipeline分发
configs            实验配置
outputs            每个实验的独立结果
main.py            统一入口
```

三个二维实验已完成L4拆分：公共采样、ODE和绘图位于`src/utils/`，模型与训练目标位于`src/modules/`，完整实验位于`src/pipelines/`。

## 当前实验

### 1. 无条件二维 Flow Matching

```powershell
python main.py --pipeline unconditional_2d --steps 4000 --particles 1000
```

从二维高斯噪声生成双月牙，观察 `overview`、速度场、ODE轨迹和loss。

### 2. 条件二维 Flow Matching

```powershell
python main.py --pipeline conditional_2d --steps 4000 --particles 1000
```

同一个网络通过条件标签学习两套不同的速度场：条件0为双月牙，条件1为圆环。

### 3. 最优传输耦合

```powershell
python main.py --pipeline ot_flow --steps 2500 --particles 700
```

使用Sinkhorn近似寻找更合理的噪声—目标配对，并观察运输距离与路径变化。

### 4. MNIST像素空间Flow Matching

MNIST训练集与测试集存放在根目录`data/MNIST/`，数据不会提交到Git。运行：

```powershell
python main.py --config configs/mnist_flow.yaml
```

CPU快速学习测试：

```powershell
python main.py --config configs/mnist_flow.yaml --steps 200 --batch-size 32 --particles 16 --ode-steps 30 --subset-size 2000 --no-animation
```

该pipeline直接把一张`1×28×28`图像视为784维样本，在像素空间学习速度场。输出包括真实图像/噪声/生成图像总览、ODE生成轨迹、不同时间预测的clean image、loss曲线和采样动画。
### 5. Label条件MNIST与空间特征轨迹

```powershell
python main.py --config configs/mnist_conditional_flow.yaml
```

该pipeline把数字标签`0-9`作为条件输入速度网络，使同一组初始噪声在不同label下沿不同速度场生成指定数字。除了像素生成轨迹，还输出：

- `label_grid.png`：按0-9排列的条件生成结果；
- `feature_pca_trajectory.png`：使用1×1、2×2、4×4空间金字塔保留粗粒度空间信息，再对特征轨迹统一PCA；
- `spatial_feature_pca.png`：对每个空间位置的通道特征做共享PCA并映射为RGB，观察空间结构形成；
- `feature_activation.png`：通道特征范数热力图；
- `predicted_clean.png`：每个时间点预测的最终干净图像。

支持加载已有模型继续训练或只做分析：

```powershell
python main.py --config configs/mnist_conditional_flow.yaml --checkpoint-path outputs/.../checkpoint.pt
```
## 数学核心

```text
x_t = (1-t) x_noise + t x_data
u_t = x_data - x_noise
v_theta(x_t, t, c) ≈ u_t
dx/dt = v_theta(x, t, c)
```

二维实验中的一个粒子对应高维生图模型中的一整张图像 latent，而不是一个像素。条件 `c` 对应生图模型的文本语义条件。

## 交互式教程

打开 [docs/flow_matching_tutorial.html](docs/flow_matching_tutorial.html)，可以拖动时间 `t`，观察二维粒子如何沿速度场移动，并切换无条件、双月牙和圆环条件。该文件可以直接部署到 GitHub Pages。

## 输出规范

每个pipeline应在 `outputs/<pipeline>/` 下保存独立实验结果，至少包含总览图、loss曲线、轨迹/速度场图、指标和实际配置。实验完成后可让Codex根据这些输出生成逐图说明文档。

## 学习顺序

1. 无条件速度场；
2. 条件如何改变速度场；
3. ODE采样轨迹；
4. OT如何改变噪声—数据耦合；
5. MNIST等简单图像；
6. 图像 latent、VAE、MMDiT与科学图像瓶颈诊断。

详细研究讨论见 `SCIENTIFIC_IMAGE_FLOW_MATCHING_NOTES.md`，项目规范见 `AGENTS.md`。
