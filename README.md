# 分子性质预测系统

基于Transformer的分子性质预测系统，支持SMILES预训练和下游任务微调。

## 项目概述

本项目实现了一个完整的分子性质预测流水线，包括：

1. **SMILES数据处理**：规范化、符号化、数据增强
2. **预训练模型**：基于掩码语言模型的双编码器架构
3. **下游任务**：分子性质分类和回归预测
4. **训练框架**：完整的训练、验证和推理流程

## 系统架构

### 数据处理流程

```
原始SMILES → 规范化 → 符号化 → 数据增强 → 训练数据
     ↓
   验证有效性 → 长度过滤 → 符号字典构建 → ID序列转换
```

**输入**：SMILES字符串文件
**输出**：

- 符号字典（symbol_dict.pkl）
- 预训练数据对（pretrain_data.pkl）
- 原始SMILES列表（smiles_list.pkl）

### 模型架构

#### 1. 预训练模型（MLM）

```
规范化SMILES → 编码器1 → 分子嵌入
                           ↓
掩码SMILES → 编码器2 → 符号预测
```

**核心组件**：

- **符号嵌入层**：将SMILES符号转换为向量表示
- **位置编码器**：添加序列位置信息
- **双编码器**：两个独立的Transformer编码器
- **掩码机制**：随机掩码SMILES符号进行预测

**训练目标**：预测被掩码的SMILES符号

#### 2. 下游任务模型

```
SMILES → 预训练编码器 → CLS表示 → 分类/回归头 → 预测结果
```

**核心组件**：

- **预训练编码器**：加载预训练权重，可选择冻结
- **任务头**：针对具体任务的分类或回归层
- **输出激活**：Sigmoid/Softmax（分类）或Identity（回归）

## 目录结构

```
new/
├── config/                    # 配置文件
│   ├── __init__.py
│   ├── model_config.py       # 模型配置
│   └── data_config.py        # 数据配置
├── src/                      # 源代码
│   ├── __init__.py
│   ├── data/                 # 数据处理模块
│   │   ├── __init__.py
│   │   ├── symbol_dictionary.py    # 符号字典
│   │   ├── smiles_processor.py     # SMILES处理器
│   │   └── dataset_builder.py      # 数据集构建器
│   ├── models/               # 模型模块
│   │   ├── __init__.py
│   │   ├── transformer_model.py    # Transformer基础组件
│   │   ├── pretrain_model.py       # 预训练模型
│   │   └── property_model.py       # 性质预测模型
│   └── training/             # 训练模块
│       ├── __init__.py
│       ├── pretrain_trainer.py     # 预训练训练器
│       └── property_trainer.py     # 性质预测训练器
├── main.py                   # 主运行脚本
└── README.md                 # 项目文档
```

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install rdkit-pypi
pip install scikit-learn
pip install numpy pandas
pip install tdc  # 可选，用于TDC数据集
```

## 使用方法

### 1. 构建预训练数据

#### 生成smiles.txt

```bash
python prepare.py
```

#### 生成symbol_dict等

```bash
python main.py build_data \
    --smiles_file data/smiles.txt \
    --output_dir data/pretrain \
    --max_samples 100000
```

**参数说明**：

- `--smiles_file`：包含SMILES字符串的文本文件，每行一个SMILES
- `--output_dir`：输出目录，将保存处理后的数据
- `--max_samples`：最大样本数量（可选）

**输出文件**：

- `symbol_dict.pkl`：符号字典
- `pretrain_data.pkl`：预训练数据对
- `smiles_list.pkl`：原始SMILES列表

### 2. 预训练模型

```bash
python main.py pretrain \
    --data_dir data/pretrain \
    --save_dir models/pretrain \
    --device cuda
```

**参数说明**：

- `--data_dir`：预训练数据目录
- `--save_dir`：模型保存目录
- `--device`：训练设备（cuda/cpu）
- `--resume`：从检查点恢复训练（可选）

**输出文件**：

- `best_model.pt`：最佳模型检查点
- `latest_checkpoint.pt`：最新检查点
- `final_model.pt`：最终模型
- 编码器权重文件（定期保存）

### 3. 下游任务微调

#### 训练多标签模型

```bash
python main.py finetune \
    --task multi_label_odor \
    --csv_file data/Multi-Labelled_Smiles_Odors_dataset.csv \
    --pretrain_model models/pretrain/best_model.pt \
    --save_dir models/multi_label_odor \
    --smiles_column nonStereoSMILES \
    --min_label_frequency 5 \
    --device cuda
```

可选参数：

- `--pretrain_model`: 预训练模型路径
- `--smiles_column`: SMILES列名（默认：nonStereoSMILES）
- `--min_label_frequency`: 最小标签频次（默认：5）
- `--max_smiles_length`: 最大SMILES长度（默认：200）

#### （忽略）气味描述符预测

```bash
python main.py finetune \
    --task odor \
    --pretrain_model models/pretrain/best_model.pt \
    --save_dir models/odor \
    --odorant_file data/odorant.txt \
    --odorless_file data/odorless.txt
```

#### （忽略）TDC数据集任务

```bash
python main.py finetune \
    --task BBBP \
    --pretrain_model models/pretrain/best_model.pt \
    --save_dir models/bbbp \
    --tdc_data_path data/tdc
```

**参数说明**：

- `--task`：任务名称（odor, BBBP, ClinTox等）
- `--pretrain_model`：预训练模型路径
- `--save_dir`：模型保存目录
- `--odorant_file`：有气味分子文件（气味任务）
- `--odorless_file`：无气味分子文件（气味任务）
- `--tdc_data_path`：TDC数据路径（TDC任务）

### 4. 模型推理

```bash
python main.py inference \
    --model_path models/odor/best_model.pt \
    --input_file data/test_smiles.txt \
    --output_file results/predictions.txt
```

## 数据格式

### SMILES文件格式

```
CCO
CC(C)O
c1ccccc1
CCCCCCCC
...
```

### 气味数据格式

**有气味分子文件**：

```
1 CCO fruity sweet
2 CC(C)O alcoholic
3 c1ccccc1 aromatic benzene
...
```

**无气味分子文件**：

```
1 CCCCCCCC odorless
2 CC(C)(C)C odorless
...
```

格式：`ID SMILES 描述符1 描述符2 ...`

## 配置说明

### 模型配置（config/model_config.py）

```python
class PretrainConfig:
    # 模型结构
    dim_embed = 256          # 嵌入维度
    num_head = 8             # 注意力头数
    num_layers = 6           # 编码器层数
    dim_tf_hidden = 1024     # 前馈网络隐藏层维度
    
    # 训练参数
    batch_size = 32          # 批次大小
    learning_rate = 1e-4     # 学习率
    num_epoch = 100          # 训练轮数
    mask_rate = 0.15         # 掩码率
    
    # 其他参数
    dropout = 0.1            # Dropout率
    activation = 'gelu'      # 激活函数
    norm_first = True        # 是否先进行层归一化
```

### 数据配置（config/data_config.py）

```python
class DataConfig:
    # SMILES处理
    limit_smiles_length = 200    # 最大SMILES长度
    limit_freq_od = 10           # 气味描述符最小频次
    
    # 文件名
    pretrain_smiles_file = 'smiles_list.pkl'
    symbol_dict_file = 'symbol_dict.pkl'
    pretrain_data_file = 'pretrain_data.pkl'
```

## 训练监控

训练过程中会输出以下信息：

### 预训练监控

- **Loss**：掩码语言模型损失
- **Accuracy**：掩码位置预测准确率
- **Learning Rate**：当前学习率

### 微调监控

- **分类任务**：Loss, Accuracy, AUC
- **回归任务**：Loss, MSE, MAE, R²

## 模型评估

### 分类指标

- **准确率（Accuracy）**：正确预测的比例
- **AUC**：ROC曲线下面积（二分类）

### 回归指标

- **MSE**：均方误差
- **MAE**：平均绝对误差
- **R²**：决定系数

## 技术特点

### 1. 数据处理

- **SMILES规范化**：使用RDKit确保分子表示一致性
- **数据增强**：生成非规范化SMILES变体增加训练数据
- **符号字典**：高效的符号到ID映射
- **长度过滤**：移除过长或无效的SMILES

### 2. 模型设计

- **双编码器架构**：分离分子理解和序列预测
- **位置编码**：正弦余弦位置编码
- **掩码策略**：随机掩码15%的符号
- **迁移学习**：预训练权重可用于下游任务

### 3. 训练优化

- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：余弦退火和自适应调整
- **早停机制**：防止过拟合
- **检查点保存**：支持训练中断恢复

### 4. 工程实践

- **模块化设计**：清晰的代码结构
- **配置管理**：集中的参数配置
- **错误处理**：完善的异常处理机制
- **日志记录**：详细的训练日志

## 扩展功能

### 添加新任务

1. 在`DatasetBuilder`中添加数据加载方法
2. 在`PropertyTrainer`中添加任务配置
3. 更新主脚本的命令行参数

### 自定义模型

1. 继承`BaseTransformerModel`
2. 实现特定的前向传播逻辑
3. 添加对应的训练器

### 新的数据格式

1. 在`SMILESProcessor`中添加处理方法
2. 更新`DatasetBuilder`的数据加载逻辑

## 常见问题

### Q: 训练过程中内存不足怎么办？

A: 减小`batch_size`或`limit_smiles_length`参数。

### Q: 如何调整模型大小？

A: 修改`dim_embed`、`num_layers`、`num_head`等参数。

### Q: 预训练需要多长时间？

A: 取决于数据量和硬件配置，通常需要几小时到几天。

### Q: 如何评估模型性能？

A: 查看验证集指标，使用测试集进行最终评估。

## 引用

如果使用本项目，请引用相关论文和数据集。

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。
