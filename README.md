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

打开[GitHub Pages交互教程](https://commusim.github.io/flow-matching-playground/)或本地`docs/index.html`。教程现分为五章：基础Flow Matching理论、二维无条件/有条件速度场、MNIST多模型比较、监督语义流形与速度场、结构差异结论。支持时间`t`、模型、Label和动力学指标交互。

本地预览需要HTTP服务器：

```powershell
cd docs
python -m http.server 8765
```

然后访问`http://127.0.0.1:8765/`。教程数据资产可通过以下命令从现有Checkpoint重新生成：

```powershell
python scripts/build_tutorial_assets.py
```

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

- `semantic_trajectories_with_real_clusters.png`：以真实类别簇为背景的目标Label中心轨迹；
- `final_predictions_with_real_clusters.png`：最终分类器预测；圆点正确、方块错误、叉号unknown；
- `unknown_aware_confusions.png`：包含unknown列的条件生成混淆矩阵；
- `strict_accuracy_over_time.png`：低置信度计为unknown后的严格准确率；
- `known_rate_over_time.png`与`target_confidence_over_time.png`：可识别比例与目标置信度；
- `shared_tsne_embeddings.npz`：同一降维器产生的坐标，可复用分析。

当前正式对比使用每个目标Label生成10张（共100张/实验），并从真实测试集每类采样100张作为共同语义簇参照。分类器最大概率低于0.8时记为`unknown`，不强制分配类别。

Unknown感知结果：

| 实验 | 严格准确率 | Known rate | Known中的准确率 | Unknown rate |
|---|---:|---:|---:|---:|
| 入口加法条件 | 8% | 52% | 15.4% | 48% |
| AdaGN条件CNN | 16% | 50% | 32% | 50% |
| VAE latent Flow | 58% | 81% | 71.6% | 19% |
| 条件U-Net | 100% | 100% | 100% | 0% |

轨迹起点按目标Label着色：相同基础噪声会复制给0-9，初始位置重合，随后因条件不同而分叉。终点颜色与类别完全由分类器判定，而不是使用目标Label伪装为预测结果。
### 分类器特征流形上的语义速度场

分类器语义特征为`f_t=C_encoder(x_t)∈R^128`。语义速度使用相邻时间有限差分：

```text
semantic_velocity(t) ≈ [f(t+Δt)-f(t)] / Δt
```

这等价于对`J_C(x_t)·v_theta(x_t,t,y)`的数值近似。由于t-SNE是非线性且没有可靠向量变换，真实语义速度使用同一个线性PCA投影位置和速度，t-SNE仍只用于邻域可视化。

新增输出：

- `classifier_semantic_velocity_field_pca.png`：真实类别簇背景上的语义速度箭头；
- `semantic_speed_over_time.png`：128维语义速度范数；
- `target_alignment_over_time.png`：速度与“当前位置→真实目标类别中心”方向的余弦对齐；
- `target_cluster_distance_over_time.png`：到真实目标类别中心的128维距离；
- `classifier_semantic_features.npz`：原始128维特征，可复用分析。

需要注意：后期目标中心对齐度可能转负、中心距离可能回升，但分类准确率仍保持很高。这通常表示模型已经进入正确类别流形，随后从类别平均中心移动到不同的具体书写风格，而不一定是语义退化。
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

## GitHub Pages教程更新（2026-08-30）

教程已经重构为理论先行、证据分层的Nature式学习页面：公式使用MathJax LaTeX渲染；二维无条件与条件Flow分为独立章节；MNIST按入口加法CNN、AdaGN CNN、VAE latent Flow和条件U-Net分别讨论；同一目标Label提供横向时间演进与单模型细粒度交互；分类器语义流形提供真实簇、语义速度和动力学指标。

教程资产现在使用41个时间点：

```text
t = 0.000, 0.025, 0.050, ..., 1.000
```

这比原来的10个时间点更适合观察早期结构形成、中期语义分叉和后期细节修正。运行`python scripts/build_tutorial_assets.py`可根据现有Checkpoint重新生成资产。
