# 心理健康数据分析项目

## 项目简介

本项目是一个基于深度学习的心理健康数据分析工具，主要用于对心理健康相关的时序数据进行分类分析。项目支持PyTorch和Jittor两种深度学习框架，提供了Transformer等模型用于数据处理和分析。

## 数据准备

### 数据结构

在指定的`root_path`下需要包含以下内容：

1. **特征文件夹**：保存处理过的时序npy文件
   - 每个npy文件的尺寸应为：(时间维度, 特征维度)
   - 文件类型应为：np.float32
   - 支持的特征类型：`crop_clip`(去除背景、裁剪面部的clip特征), `crop_dino`(去除背景、裁剪面部的dino特征), `csv`(midiapipe输出的面部特征点), `src_clip`(原始clip特征), `src_dino`(原始dino特征), `wobg_clip`(仅去除背景的clip特征), `wobg_dino`(仅去除背景的dino特征)

2. **data.csv**：包含样本标签信息
   - 必须包含`Case_id`列，用于关联特征文件
   - 包含各种心理健康评估结果列，如`result-1`至`result-15`
   - 对于`suiside`标签，会根据SDS-1到SDS-6的分数按照权重(1,2,6,10,10,4)加权求和，总分≥6分为有风险(1)，<6分为无风险(0)

### 数据示例

```
root_path/
├── crop_clip/
│   ├── case_001.npy  # 尺寸: (时间维度, 特征维度), 类型: np.float32
│   ├── case_002.npy
│   └── ...
├── wobg_clip/
│   ├── case_001.npy
│   ├── case_002.npy
│   └── ...
└── data.csv  # 包含Case_id和各评估结果
```

## 环境准备

### 依赖项

- Python 3.8+
- PyTorch 1.10+ (用于run.sh)
- Jittor 1.3+ (用于runjt.sh)
- pandas
- numpy
- scikit-learn

### 安装依赖

```bash
# 安装PyTorch环境
pip install torch torchvision torchaudio

# 安装Jittor环境
pip install jittor

# 安装其他依赖
pip install pandas numpy scikit-learn
```

## 运行项目

### 使用PyTorch运行 (run.sh)

```bash
chmod +x run.sh
./run.sh
```

### 使用Jittor运行 (runjt.sh)

```bash
chmod +x runjt.sh
./runjt.sh
```

## 配置说明

### 主要参数

- `--task_name`: 任务类型，默认为`classification`
- `--model`: 模型名称，支持`Transformer`, `Informer`等
- `--data`: 数据类型，支持`Mental`(PyTorch)和`Mental_jt`(Jittor)
- `--features`: 特征类型，如`wobg_clip`, `crop_clip`等
- `--target`: 目标标签，如`depression`, `anxiety`, `suiside`等
- `--seq_len`: 输入序列长度
- `--enc_in`: 编码器输入维度
- `--learning_rate`: 学习率

### 自定义配置

可以修改`run.sh`或`runjt.sh`文件中的参数组合来进行不同的实验：

```bash
# 修改参数组合
seq_lens=(256)
enc_ins=(64)
featuress=(wobg_clip)
lrs=(0.0002)
```

### 目录结构

```
jittor_mental/
├── run.py              # 主运行脚本，解析命令行参数并启动实验
├── run.sh              # PyTorch环境运行脚本，包含参数组合
├── runjt.sh            # Jittor环境运行脚本，包含参数组合
├── data_factory.py     # 数据工厂，根据框架类型选择相应的数据加载器
├── exp_basic.py        # 实验基础类，定义了实验的基本流程
├── mental.py           # PyTorch版的心理健康数据加载器
├── mental_jt.py        # Jittor版的心理健康数据加载器
├── tools.py            # 工具函数
├── layers/             # PyTorch版的模型层
│   ├── Embed.py                # 嵌入层实现
│   ├── SelfAttention_Family.py # 注意力机制家族实现
│   ├── Transformer_EncDec.py   # Transformer编码器-解码器实现
│   └── masking.py              # 掩码相关函数
├── layers_jt/          # Jittor版的模型层
│   ├── Embed.py                # 嵌入层实现
│   ├── SelfAttention_Family.py # 注意力机制家族实现
│   ├── Transformer_EncDec.py   # Transformer编码器-解码器实现
│   └── masking.py              # 掩码相关函数
├── models/             # 模型定义
│   ├── Transformer.py          # PyTorch版的Transformer模型
│   └── Transformer_jt.py       # Jittor版的Transformer模型
└── tests/              # 测试脚本
    ├── compare.py              # 框架间对比测试
    ├── compare_emb.py          # 嵌入层对比测试
    ├── compare_encdec.py       # 编码器-解码器对比测试
    └── compare_trans.py        # Transformer模型对比测试
```


## 支持的标签类型

- `depression`: 抑郁症评估结果
- `anxiety`: 焦虑症评估结果
- `suiside`: 自杀风险评估结果
- 以及其他多种心理健康评估标签

## 结果输出

训练完成后，结果文件将保存在`results/`目录下，模型检查点将保存在`checkpoints/`目录下。

## 注意事项

1. 确保数据文件的路径和格式正确
2. 根据实际硬件资源调整batch_size和学习率
3. 对于大型数据集，建议使用GPU加速训练
4. 可以通过修改代码中的参数来调整模型结构和训练策略

## 致谢

本项目的开发受到了以下开源项目的启发和帮助：

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library.git) - 提供了先进的时间序列分析模型和基准测试
- [Jittor](https://github.com/Jittor/jittor.git) - 高性能的深度学习框架，支持即时编译和元操作

感谢这些项目的开发者们为开源社区做出的贡献！
