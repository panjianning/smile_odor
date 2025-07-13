"""预训练模型

该模块实现用于SMILES预训练的双编码器模型。
主要功能：
1. 掩码语言模型（MLM）预训练
2. 双编码器架构
3. 分子嵌入生成
"""

import torch
import torch.nn as nn
import random
from typing import Optional, Tuple, Dict
from .transformer_model import BaseTransformerModel
from config.model_config import SPECIAL_TOKENS, PretrainConfig


class MLMPretrainModel(nn.Module):
    """掩码语言模型预训练模型

    实现双编码器架构，用于SMILES序列的自监督预训练。
    第一个编码器处理规范化SMILES，第二个编码器处理掩码SMILES。
    """

    def __init__(self, config: PretrainConfig, num_tokens: int):
        """初始化预训练模型

        Args:
            config: 预训练配置
            num_tokens: 词汇表大小
        """
        super().__init__()

        self.config = config
        self.num_tokens = num_tokens
        self.mask_rate = config.mask_rate

        # 共享的符号嵌入层和位置编码
        from .transformer_model import SymbolEncoder, PositionalEncoder
        self.symbol_encoder = SymbolEncoder(
            num_tokens, config.dim_embed, config.init_range
        )
        self.positional_encoder = PositionalEncoder(config.dim_embed)

        # 两个独立的Transformer编码器
        from .transformer_model import TransformerEncoder
        self.smiles_encoder1 = TransformerEncoder(
            config.dim_embed, config.num_head, config.dim_tf_hidden,
            config.num_layers, config.dropout, config.norm_first,
            config.activation
        )

        self.smiles_encoder2 = TransformerEncoder(
            config.dim_embed, config.num_head, config.dim_tf_hidden,
            config.num_layers, config.dropout, config.norm_first,
            config.activation
        )

        # Dropout层
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        # 输出层
        self.output_layer = nn.Linear(config.dim_embed, num_tokens)

    def forward(self, canonical_smiles: torch.Tensor,
                masked_smiles: torch.Tensor,
                use_molecular_embedding: bool = True) -> torch.Tensor:
        """前向传播

        Args:
            canonical_smiles: 规范化SMILES序列 [batch_size, seq_len]
            masked_smiles: 掩码SMILES序列 [batch_size, seq_len]
            use_molecular_embedding: 是否使用分子嵌入

        Returns:
            torch.Tensor: 预测的符号概率 [batch_size, seq_len, num_tokens]
        """
        batch_size = canonical_smiles.size(0)
        device = canonical_smiles.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID

        molecular_embedding = None

        if use_molecular_embedding:
            # 第一个编码器：处理规范化SMILES，生成分子嵌入
            x1 = torch.cat([cls_tokens, canonical_smiles], dim=1)
            padding_mask1 = (x1 == SPECIAL_TOKENS.PAD_ID)

            # 符号嵌入 + 位置编码
            x1_embed = self.symbol_encoder(x1) + self.positional_encoder(x1)
            x1_embed = self.dropout1(x1_embed)

            # 第一个编码器
            x1_encoded = self.smiles_encoder1(x1_embed, padding_mask1)

            # 提取分子嵌入（CLS位置的输出）
            # [batch_size, dim_embed]
            molecular_embedding = x1_encoded[:, 0, :]

        # 第二个编码器：处理掩码SMILES
        x2 = torch.cat([cls_tokens, masked_smiles], dim=1)
        padding_mask2 = (x2 == SPECIAL_TOKENS.PAD_ID)

        # 符号嵌入
        x2_embed = self.symbol_encoder(x2)

        # 如果使用分子嵌入，替换CLS位置的嵌入
        if use_molecular_embedding and molecular_embedding is not None:
            x2_embed[:, 0, :] = molecular_embedding

        # 添加位置编码
        x2_embed = x2_embed + self.positional_encoder(x2_embed)
        x2_embed = self.dropout2(x2_embed)

        # 第二个编码器
        x2_encoded = self.smiles_encoder2(x2_embed, padding_mask2)

        # 输出层（跳过CLS位置）
        output = self.output_layer(x2_encoded[:, 1:, :])

        return output

    def apply_masking(self, smiles_ids: torch.Tensor,
                      mask_rate: Optional[float] = None) -> torch.Tensor:
        """对SMILES序列应用掩码

        Args:
            smiles_ids: SMILES ID序列 [batch_size, seq_len]
            mask_rate: 掩码率，如果为None则使用配置中的值

        Returns:
            torch.Tensor: 掩码后的序列
        """
        if mask_rate is None:
            mask_rate = self.mask_rate

        masked_smiles = smiles_ids.clone()

        # 生成掩码
        mask = torch.rand_like(smiles_ids.float()) < mask_rate

        # 不对特殊标记应用掩码
        special_tokens = [
            SPECIAL_TOKENS.PAD_ID, SPECIAL_TOKENS.CLS_ID,
            SPECIAL_TOKENS.BOS_ID, SPECIAL_TOKENS.EOS_ID
        ]
        for token_id in special_tokens:
            mask = mask & (smiles_ids != token_id)

        # 应用掩码
        masked_smiles[mask] = SPECIAL_TOKENS.MSK_ID

        return masked_smiles

    def get_molecular_embedding(self, canonical_smiles: torch.Tensor) -> torch.Tensor:
        """获取分子嵌入

        Args:
            canonical_smiles: 规范化SMILES序列 [batch_size, seq_len]

        Returns:
            torch.Tensor: 分子嵌入 [batch_size, dim_embed]
        """
        batch_size = canonical_smiles.size(0)
        device = canonical_smiles.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, canonical_smiles], dim=1)
        padding_mask = (x == SPECIAL_TOKENS.PAD_ID)

        # 编码
        x_embed = self.symbol_encoder(x) + self.positional_encoder(x)
        x_embed = self.dropout1(x_embed)
        x_encoded = self.smiles_encoder1(x_embed, padding_mask)

        # 返回CLS位置的嵌入
        return x_encoded[:, 0, :]

    def save_encoders(self, save_dir: str, epoch: int):
        """保存编码器

        Args:
            save_dir: 保存目录
            epoch: 训练轮数
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 构建文件名
        symbol_encoder_name = f'OdorCode-40 symbol_encoder D{self.config.dim_embed}.Hidden{self.config.dim_tf_hidden}.Head{self.config.num_head}.L{self.config.num_layers}.R{self.config.mask_rate}.S{self.config.total_size}-epoch.{epoch}'
        smiles_encoder_name = f'OdorCode-40 smiles_encoder D{self.config.dim_embed}.Hidden{self.config.dim_tf_hidden}.Head{self.config.num_head}.L{self.config.num_layers}.R{self.config.mask_rate}.S{self.config.total_size}-epoch.{epoch}'

        # 保存模型
        torch.save(
            self.symbol_encoder.state_dict(),
            os.path.join(save_dir, symbol_encoder_name)
        )
        torch.save(
            self.smiles_encoder1.state_dict(),
            os.path.join(save_dir, smiles_encoder_name)
        )

    def load_encoders(self, load_dir: str, epoch: int):
        """加载编码器

        Args:
            load_dir: 加载目录
            epoch: 训练轮数
        """
        import os

        # 构建文件名
        symbol_encoder_name = f'OdorCode-40 symbol_encoder D{self.config.dim_embed}.Hidden{self.config.dim_tf_hidden}.Head{self.config.num_head}.L{self.config.num_layers}.R{self.config.mask_rate}.S{self.config.total_size}-epoch.{epoch}'
        smiles_encoder_name = f'OdorCode-40 smiles_encoder D{self.config.dim_embed}.Hidden{self.config.dim_tf_hidden}.Head{self.config.num_head}.L{self.config.num_layers}.R{self.config.mask_rate}.S{self.config.total_size}-epoch.{epoch}'

        # 加载模型
        self.symbol_encoder.load_state_dict(
            torch.load(os.path.join(load_dir, symbol_encoder_name))
        )
        self.smiles_encoder1.load_state_dict(
            torch.load(os.path.join(load_dir, smiles_encoder_name))
        )

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'MLM Pretrain Model',
            'num_tokens': self.num_tokens,
            'dim_embed': self.config.dim_embed,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_head,
            'mask_rate': self.mask_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class PretrainLoss(nn.Module):
    """预训练损失函数"""

    def __init__(self, ignore_index: int = SPECIAL_TOKENS.PAD_ID):
        """初始化损失函数

        Args:
            ignore_index: 忽略的标记ID（通常是PAD）
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='mean')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失

        Args:
            predictions: 预测结果 [batch_size, seq_len, num_tokens]
            targets: 目标序列 [batch_size, seq_len]

        Returns:
            torch.Tensor: 损失值
        """
        # 重塑张量以适应CrossEntropyLoss
        batch_size, seq_len, num_tokens = predictions.shape
        predictions_flat = predictions.view(batch_size * seq_len, num_tokens)
        targets_flat = targets.view(batch_size * seq_len)

        return self.criterion(predictions_flat, targets_flat)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                       masked_positions: torch.Tensor = None) -> Dict[str, float]:
    """计算预测准确率

    Args:
        predictions: 预测结果 [batch_size, seq_len, num_tokens]
        targets: 目标序列 [batch_size, seq_len]
        masked_positions: 掩码位置 [batch_size, seq_len]

    Returns:
        Dict[str, float]: 准确率统计
    """
    # 获取预测的符号ID
    predicted_ids = torch.argmax(predictions, dim=-1)

    # 计算正确预测
    correct = (predicted_ids == targets)

    # 非填充位置
    non_pad_mask = (targets != SPECIAL_TOKENS.PAD_ID)

    # 总体准确率
    total_correct = correct & non_pad_mask
    total_accuracy = total_correct.sum().float() / non_pad_mask.sum().float()

    result = {'total_accuracy': total_accuracy.item()}

    # 掩码位置准确率
    if masked_positions is not None:
        mask_correct = correct & masked_positions
        mask_accuracy = mask_correct.sum().float() / masked_positions.sum().float()
        result['mask_accuracy'] = mask_accuracy.item()

    return result
