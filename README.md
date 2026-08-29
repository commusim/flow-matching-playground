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

该pipeline把数字标签`0-9`作为条件输入速度网络，使同一组初始噪声在不同label下沿不同速度场生成指定数字。 当前条件网络使用逐层AdaGN/FiLM式残差块：每个Block都由时间和Label生成scale/shift，在GroupNorm之后调制特征，避免入口加法条件被归一化削弱。除了像素生成轨迹，还输出：

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

## MNIST模型版本与Checkpoint兼容

项目明确保留三种模型架构：

| 模型名称 | `model_variant` | 配置文件 | Checkpoint用途 |
|---|---|---|---|
| 无条件 | `unconditional` | `configs/mnist_flow.yaml` | 原始无条件MNIST模型 |
| 有条件（入口加法） | `conditional_additive` | `configs/mnist_conditional_additive.yaml` | 兼容第一版Label条件模型 |
| AdaGN条件 | `conditional_adagn` | `configs/mnist_conditional_adagn.yaml` | 每个残差块进行条件调制的新模型 |

加载时建议使用`--model-variant auto`，系统会根据checkpoint参数名自动识别架构和hidden维度：

```powershell
python main.py --config configs/mnist_conditional_additive.yaml --checkpoint-path path/to/checkpoint.pt --model-variant auto --steps 0
```

`steps=0`表示只加载模型、执行生成和可视化；设置大于0则在已有权重上继续训练。无条件checkpoint必须使用`mnist_flow`，两种条件checkpoint使用对应条件pipeline。
## 三条图像生成学习路线

### 识别模型与语义评价

```powershell
python main.py --config configs/mnist_classifier.yaml
```

输出分类器checkpoint、混淆矩阵和监督语义特征t-SNE。该分类器冻结后用于评价条件生成准确率和目标类别置信度。

### VAE压缩与latent Flow

先训练VAE：

```powershell
python main.py --config configs/mnist_vae.yaml
```

再把VAE和分类器checkpoint写入`configs/mnist_latent_flow.yaml`或通过CLI传入：

```powershell
python main.py --config configs/mnist_latent_flow.yaml --vae-checkpoint path/to/vae.pt --classifier-checkpoint path/to/classifier.pt
```

VAE把`1×28×28`像素压缩为`8×7×7` latent，Flow Matching在latent空间中训练，最终由Decoder还原图像。

### 多尺度条件U-Net Flow

```powershell
python main.py --config configs/mnist_unet_flow.yaml --classifier-checkpoint path/to/classifier.pt
```

U-Net通过`28×28 → 14×14 → 7×7 → 14×14 → 28×28`获得全局感受野，在每个残差块中使用AdaGN条件，并通过Label Dropout与Classifier-Free Guidance强化类别控制。

### 2026-08-29阶段结果

| 实验 | 结果 |
|---|---:|
| MNIST识别模型 | 测试准确率98.87% |
| MNIST VAE | 重建MSE 0.0030 |
| VAE latent Flow | 条件准确率62.5%，目标置信度58.7% |
| 条件U-Net Flow（1200步，base=16） | 条件准确率97.5%，目标置信度97.9% |

这组对照表明：VAE能够显著压缩生成空间并保持良好重建，但当前latent Flow仍需要更强的latent速度网络或更长训练；多尺度U-Net对MNIST全局数字结构和Label控制最有效。
## 通用图像配置

U-Net实现现已合并到`src/modules/image_velocity.py`，图像速度模型统一放在同一模块。旧的模型类名与checkpoint参数键保持兼容。

图像pipeline不再固定为`1×28×28`。以下参数可以在YAML或CLI中设置：

```yaml
dataset: mnist          # mnist / fashion_mnist / cifar10 / image_folder
image_size: 32          # VAE和两级U-Net要求能被4整除
input_channels: 1       # 灰度为1，RGB为3
num_classes: 10
data_root: null         # null表示项目根目录data/
download: true
```

CLI覆盖示例：

```powershell
python main.py --config configs/mnist_unet_flow.yaml --image-size 32 --input-channels 1 --num-classes 10
```

项目还提供通用pipeline别名：

```text
image_classifier
image_vae
image_latent_flow
image_unet_flow
```

Fashion-MNIST示例：

```powershell
python main.py --config configs/fashion_mnist_unet_flow.yaml
```

CIFAR-10 RGB示例：

```powershell
python main.py --config configs/cifar10_unet_flow.yaml
```

自定义数据采用`ImageFolder`目录结构，并在config中设置`dataset: image_folder`、`data_root`、`input_channels`和`num_classes`。
## 统一分类器语义轨迹比较

不同生成器自身的隐藏特征不处于同一个空间，不能直接比较。新的`semantic_trajectory_comparison` pipeline将所有实验的中间图像输入同一个冻结监督分类器，提取同一层128维语义特征；随后把全部实验、全部时间和全部Label的特征拼接，只拟合一次t-SNE，再按实验拆分绘图。

```powershell
python main.py --config configs/semantic_trajectory_comparison.yaml
```

默认比较：入口加法条件CNN、AdaGN条件CNN、VAE latent Flow和条件U-Net。输出包括：

- `shared_tsne_semantic_trajectories.png`：所有样本的共享t-SNE轨迹；
- `shared_tsne_label_centroid_trajectories.png`：更清晰的Label中心轨迹；
- `shared_tsne_final_embeddings.png`：最终语义特征位置；
- `semantic_accuracy_over_time.png`：随Flow时间变化的目标类别准确率；
- `semantic_confidence_over_time.png`：随Flow时间变化的目标类别置信度；
- `shared_tsne_embeddings.npz`：同一降维器产生的坐标，可复用分析。

本次每类2张、共20张的统一对比结果：加法条件20%、AdaGN条件5%、latent Flow 80%、条件U-Net 100%。样本量较小，数值用于轨迹机制观察，不作为最终统计结论。
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
