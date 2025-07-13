"""模型配置文件"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PretrainConfig:
    """预训练模型配置"""
    # 模型架构参数
    dim_embed: int = 256          # 嵌入维度
    dim_tf_hidden: int = 256      # Transformer隐藏层维度
    num_head: int = 16            # 多头注意力头数
    num_layers: int = 10          # Transformer层数
    dropout: float = 0.1          # Dropout率
    norm_first: bool = True       # 是否先进行归一化
    activation: str = 'gelu'      # 激活函数

    # 训练参数
    learning_rate: float = 0.0005  # 学习率
    num_epoch: int = 800          # 训练轮数
    batch_size: int = 128         # 批次大小
    mask_rate: float = 0.5        # 掩码率
    init_range: float = 0.1       # 参数初始化范围

    # 数据参数
    limit_smiles_length: int = 100  # SMILES最大长度
    total_size: int = 100000        # 训练数据总量


@dataclass
class PropertyPredConfig:
    """分子性质预测模型配置"""
    # 模型架构参数
    dim_embed: int = 256          # 嵌入维度
    dim_tf_hidden: int = 512      # Transformer隐藏层维度
    num_head: int = 8             # 多头注意力头数
    num_layers: int = 10          # Transformer层数
    dropout: float = 0.1          # Dropout率
    norm_first: bool = True       # 是否先进行归一化
    activation: str = 'gelu'      # 激活函数

    # 训练参数
    learning_rate: float = 0.0001  # 学习率
    num_epoch: int = 50           # 训练轮数
    batch_size: int = 32          # 批次大小
    init_range: float = 0.1       # 参数初始化范围

    # 迁移学习参数
    pretrain_path: Optional[str] = None  # 预训练模型路径
    freeze_encoder: bool = True          # 是否冻结编码器
    fine_tune_last_layer: bool = True    # 是否微调最后一层
    unfreeze_last_layers: int = 2        # 解冻最后几层
    transfer_learning_rate_ratio: float = 1.0  # 迁移学习学习率比例

    # 训练控制参数
    early_stopping_patience: int = 10   # 早停耐心值

    # 数据参数
    limit_smiles_length: int = 100  # SMILES最大长度
    fold_cv: int = 5                # 交叉验证折数


@dataclass
class SpecialTokens:
    """特殊标记配置"""
    PAD_ID: int = 0      # 填充标记
    CLS_ID: int = 1      # 分类标记
    BOS_ID: int = 2      # 开始标记
    EOS_ID: int = 3      # 结束标记
    MSK_ID: int = 4      # 掩码标记

    PAD_TOKEN: str = ' '
    CLS_TOKEN: str = '[CLS]'
    BOS_TOKEN: str = '[BOS]'
    EOS_TOKEN: str = '[EOS]'
    MSK_TOKEN: str = '▪'


# 默认配置实例
DEFAULT_PRETRAIN_CONFIG = PretrainConfig()
DEFAULT_PROPERTY_CONFIG = PropertyPredConfig()
SPECIAL_TOKENS = SpecialTokens()
