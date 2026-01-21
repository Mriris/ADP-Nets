# ADP-Nets

基于 APDNet 的图像去噪网络。

## 环境配置

```bash
# 使用 uv 安装依赖
uv sync
```

## 训练

```bash
python3 ./train_ag.py --arch APDNet --batch_size 16 --gpu '0' --nepoch 1000 \
    --train_ps 256 \
    --train_gt_dir <训练集干净图像目录> \
    --train_input_dir <训练集噪声图像目录> \
    --val_gt_dir <验证集干净图像目录> \
    --val_input_dir <验证集噪声图像目录> \
    --embed_dim 64 --warmup --checkpoint 500 \
    --env <实验名称> --noiseL 15 --lr_initial 0.0001
```

**参数说明**:
- `--noiseL`: 训练时添加的高斯噪声标准差（sigma），默认为 15
- `--embed_dim`: 模型嵌入维度，默认为 64
- `--nepoch`: 训练轮数
- `--checkpoint`: 每隔多少轮保存一次模型

## 推理

### 方式一：使用合成噪声测试

对干净图像添加指定级别的高斯噪声进行测试：

```bash
uv run python3 ./test_pad_ad.py --arch APDNet --gpus '0' \
    --input_dir <干净图像目录> \
    --gt_dir <干净图像目录> \
    --result_dir <输出目录> \
    --weights <模型权重路径> \
    --embed_dim 64 --noiseL 15
```

### 方式二：使用真实噪声图像

直接对已有噪声的图像进行去噪（无 ground truth）：

```bash
uv run python3 ./test_pad_ad.py --arch APDNet --gpus '0' \
    --input_dir <噪声图像目录> \
    --result_dir <输出目录> \
    --weights <模型权重路径> \
    --embed_dim 64 --no_gt
```

### 方式三：使用真实噪声图像并计算 PSNR

需要同时提供噪声图像和对应的干净图像：

```bash
uv run python3 ./test_pad_ad.py --arch APDNet --gpus '0' \
    --input_dir <噪声图像目录> \
    --gt_dir <干净图像目录> \
    --result_dir <输出目录> \
    --weights <模型权重路径> \
    --embed_dim 64
```

**注意**: 使用此方式前需修改 `test_pad_ad.py` 中的代码，将合成噪声改为直接使用真实噪声图像：

```python
# 将
shape = (data_test[0].cuda()).shape
inputs = torch.cuda.FloatTensor(shape).normal_(mean=0, std=(args.noiseL)/255.)
inputs = inputs + data_test[0].cuda()

# 改为
inputs = data_test[1].cuda()
```

## 注意事项

**噪声级别匹配**: 模型性能与噪声级别密切相关。如果输入图像的噪声级别与训练时使用的 `--noiseL` 不匹配，去噪效果会大幅下降。

例如：
- 模型训练时 `--noiseL 15`（sigma=15）
- 输入图像噪声 sigma=42
- 结果：PSNR 仅 19 dB

解决方案：根据实际噪声级别重新训练模型，或使用真实噪声图像对进行训练。

## 项目结构

```
ADP-Nets/
├── APDNet.py           # 模型定义
├── train_ag.py         # 训练脚本
├── test_pad_ad.py      # 推理脚本
├── train.sh            # 训练示例
├── test.sh             # 推理示例
├── dataset.py          # 数据集定义
├── losses.py           # 损失函数
├── modules.py          # 模型模块
├── Regularize.py       # 正则化模块
├── options.py          # 参数配置
├── utils/              # 工具函数
│   ├── loader.py       # 数据加载
│   ├── model_utils.py  # 模型工具
│   └── image_utils.py  # 图像工具
└── log/                # 训练日志和模型权重
```
