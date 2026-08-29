# 项目Checklist

## 基础结构

- [x] `src/utils/` 已建立，承载公共基础能力与工具
- [x] `src/modules/` 已建立
- [x] `src/pipelines/` 已建立
- [x] `src/cli/` 已建立
- [x] 根目录独立存在 `configs/`、`data/`、`outputs/`、`main.py`
- [x] 统一入口支持 `--pipeline`
- [x] 所有旧实验已完成内部模块化拆分

## Flow Matching原理

- [x] 无条件二维速度场拟合
- [x] 条件二维速度场
- [x] 多时间速度场可视化
- [x] ODE采样轨迹可视化
- [x] 同噪声不同条件轨迹对比
- [x] 条件速度差分场
- [x] Sinkhorn最优传输耦合
- [x] OT运输距离和路径可视化

## 简单图像扩展

- [x] MNIST数据加载pipeline
- [ ] Fashion-MNIST或CIFAR-10子集
- [x] 简单图像的像素空间Flow Matching
- [ ] 小型VAE编码/解码实验
- [ ] 图像latent轨迹可视化
- [x] Label条件MNIST生成
- [x] 空间金字塔特征PCA轨迹
- [x] 空间特征PCA图与激活热力图
- [x] MNIST checkpoint加载

## 科学图像研究准备

- [ ] VAE重建前后SIQA对比
- [ ] MMDiT与Decoder模块瓶颈实验
- [ ] 正确/错误Prompt最小差异实验
- [ ] 正确与错误语义的latent可分性实验
- [ ] SIU2A评估接入
- [ ] 科学图像生成pipeline

## 文档与展示

- [x] README已更新
- [x] 经验理论文档已建立
- [x] 交互式HTML教程已建立
- [ ] GitHub Pages自动部署
- [ ] 每个pipeline生成独立图像说明文档
