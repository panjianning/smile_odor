# 推理功能实现文档

本文档详细说明了多标签气味预测系统的推理功能实现。

## 功能概述

推理功能已完全实现，支持：

1. **多标签气味预测** - 从CSV数据集训练和预测多种气味特征
2. **单个SMILES推理** - 对单个分子进行气味预测
3. **批量推理** - 处理大量分子的批量预测
4. **结果解释** - 提供概率值和置信度评估

## 核心组件

### 1. 推理引擎 (`src/inference/inference_engine.py`)

主要功能：

- 模型加载和初始化
- SMILES预处理和验证
- 批量和单个预测
- 结果后处理和输出

```python
from src.inference.inference_engine import InferenceEngine

# 初始化推理引擎
engine = InferenceEngine(model_path="models/best_model.pt", device="cuda")

# 单个预测
result = engine.predict_single("CCO")  # 乙醇

# 批量预测
engine.predict_from_file("input.csv", "output.csv")
```

### 2. CSV数据处理器 (`src/data/csv_processor.py`)

专门处理多标签气味数据集：

- 解析CSV格式的多标签数据
- 标签频次统计和过滤
- 数据预处理和验证
- 支持多种CSV格式

### 3. 数据集构建器更新 (`src/data/dataset_builder.py`)

新增多标签数据集构建方法：

```python
def build_multi_label_odor_dataset(self, csv_path, smiles_column, ...):
    # 构建多标签气味数据集
```

### 4. 训练器更新 (`src/training/property_trainer.py`)

支持多标签CSV数据集训练：

- 自动检测多标签任务
- 支持CSV文件配置
- 多标签损失函数和评估指标

## 使用方法

### 1. 完整流程示例

运行完整的端到端流程：

```bash
python examples/multi_label_odor_example.py
```

这将执行：

1. 数据集分析
2. 模型训练
3. 推理演示
4. 结果分析

### 2. 使用主脚本

#### 运行推理

```bash
python main.py inference \
    --model_path models/multi_label_odor/best_model.pt \
    --input_file input_smiles.csv \
    --output_file predictions.csv \
    --device cuda
```

### 3. 编程接口

```python
# 1. 数据集分析
from src.data.csv_processor import CSVProcessor
processor = CSVProcessor()
stats = processor.analyze_dataset("data/Multi-Labelled_Smiles_Odors_dataset.csv")

# 2. 训练模型
from src.training.property_trainer import PropertyTrainer
from config.model_config import PropertyPredConfig

config = PropertyPredConfig()
data_config = {
    'csv_file': 'data/Multi-Labelled_Smiles_Odors_dataset.csv',
    'min_label_frequency': 5
}

trainer = PropertyTrainer(config, data_config, "models/odor")
trainer.train('multi_label_odor')

# 3. 推理
from src.inference.inference_engine import InferenceEngine
engine = InferenceEngine("models/odor/best_model.pt")

# 单个预测
result = engine.predict_single("CCO")
print(result['predicted_labels'])

# 批量预测
engine.predict_from_file("input.csv", "output.csv")
```

## 数据格式

### 输入CSV格式

数据集应包含以下列：

- `nonStereoSMILES`: SMILES字符串
- 多个气味标签列（如 `floral`, `fruity`, `sweet` 等）
- 可选的 `descriptors` 列

示例：

```csv
nonStereoSMILES,floral,fruity,sweet,minty
CCO,0,1,1,0
CC(C)CO,0,0,1,0
```

### 输出格式

推理结果包含：

- `SMILES`: 输入的SMILES字符串
- `Valid`: SMILES是否有效
- `{label}_predicted`: 每个标签的预测结果（0/1）
- `{label}_probability`: 每个标签的预测概率
- `{label}_confident`: 是否高置信度预测

## 配置选项

### 数据配置

```python
data_config = {
    'csv_file': 'path/to/dataset.csv',           # CSV文件路径
    'smiles_column': 'nonStereoSMILES',          # SMILES列名
    'descriptor_column': 'descriptors',           # 描述符列名（可选）
    'min_label_frequency': 5,                     # 最小标签频次
    'max_smiles_length': 200                      # 最大SMILES长度
}
```

### 模型配置

```python
from config.model_config import PropertyPredConfig

config = PropertyPredConfig()
config.num_epoch = 50                    # 训练轮数
config.batch_size = 32                   # 批大小
config.learning_rate = 1e-4              # 学习率
config.early_stopping_patience = 10      # 早停耐心值
```

## 性能优化

### 1. 批量推理优化

- 支持自定义批大小
- 自动填充和对齐序列
- GPU内存优化

### 2. 数据预处理优化

- SMILES验证和规范化
- 低频标签过滤
- 无效数据跳过

### 3. 模型优化

- 支持多标签二分类
- 概率阈值调整
- 置信度评估

## 错误处理

系统包含完善的错误处理：

1. **数据错误**
   - 无效SMILES检测
   - 缺失数据处理
   - 格式错误提示

2. **模型错误**
   - 模型文件检查
   - 设备兼容性
   - 内存不足处理

3. **推理错误**
   - 批量处理异常
   - 结果保存错误
   - 进度监控

## 扩展性

系统设计具有良好的扩展性：

1. **新数据格式支持**
   - 可轻松添加新的CSV格式解析器
   - 支持其他数据源（JSON、数据库等）

2. **新任务类型**
   - 支持回归任务
   - 支持多分类任务
   - 支持序列预测

3. **新模型架构**
   - 模块化设计便于替换模型
   - 支持不同的预训练模型
   - 支持集成学习

## 示例结果

### 单个SMILES预测示例

```
SMILES: CCO
预测标签:
  sweet: 是 (概率: 0.823)
  alcoholic: 是 (概率: 0.756)
  fruity: 否 (概率: 0.234)
```

### 批量预测统计

```
处理样本数: 1000
有效样本数: 987
预测标签统计:
  sweet: 234 个样本
  floral: 156 个样本
  fruity: 189 个样本
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批大小
   - 使用CPU推理

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型版本兼容性

3. **数据格式错误**
   - 检查CSV列名
   - 确认SMILES格式

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 总结

推理功能已完全实现，支持：

- ✅ 多标签气味预测
- ✅ 单个和批量推理
- ✅ CSV数据集处理
- ✅ 结果解释和分析
- ✅ 完整的端到端流程
- ✅ 错误处理和优化

系统现在可以处理您提供的多标签气味数据集，进行训练和预测。
