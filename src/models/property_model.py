"""分子性质预测模型

该模块实现基于预训练Transformer的分子性质预测模型。
主要功能：
1. 分类任务预测
2. 回归任务预测
3. 迁移学习支持
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .transformer_model import BaseTransformerModel
from config.model_config import SPECIAL_TOKENS, PropertyPredConfig


class PropertyPredictionModel(nn.Module):
    """分子性质预测模型

    基于预训练Transformer编码器的分子性质预测模型，
    支持分类和回归任务。
    """

    def __init__(self, config: PropertyPredConfig, num_tokens: int,
                 num_labels: int, task_type: str = 'classification'):
        """初始化分子性质预测模型

        Args:
            config: 模型配置
            num_tokens: 词汇表大小
            num_labels: 标签数量
            task_type: 任务类型 ('classification' 或 'regression')
        """
        super().__init__()

        self.config = config
        self.num_tokens = num_tokens
        self.num_labels = num_labels
        self.task_type = task_type

        # 基础Transformer模型
        self.transformer = BaseTransformerModel(
            num_tokens=num_tokens,
            d_model=config.dim_embed,
            num_heads=config.num_head,
            d_hidden=config.dim_tf_hidden,
            num_layers=config.num_layers,
            dropout=config.dropout,
            norm_first=config.norm_first,
            activation=config.activation,
            init_range=config.init_range
        )

        # 分类/回归头
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_embed, config.dim_embed // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_embed // 2, num_labels)
        )

        # 输出激活函数
        if task_type == 'classification':
            if num_labels == 1:
                self.output_activation = nn.Sigmoid()
            else:
                self.output_activation = nn.Softmax(dim=-1)
        else:  # regression
            self.output_activation = nn.Identity()

    def forward(self, smiles_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播

        Args:
            smiles_ids: SMILES ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 预测结果 [batch_size, num_labels]
        """
        batch_size = smiles_ids.size(0)
        device = smiles_ids.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, smiles_ids], dim=1)

        # 创建填充掩码
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, attention_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)

        # Transformer编码
        encoded = self.transformer(x, padding_mask)

        # 使用CLS位置的表示进行分类/回归
        cls_representation = encoded[:, 0, :]  # [batch_size, d_model]

        # 分类/回归头
        logits = self.classifier(cls_representation)

        # 应用输出激活函数
        output = self.output_activation(logits)

        return output

    def load_pretrained_encoder(self, pretrain_model_path: str,
                                freeze_encoder: bool = True,
                                unfreeze_last_layers: int = 0):
        """加载预训练编码器

        Args:
            pretrain_model_path: 预训练模型路径
            freeze_encoder: 是否冻结编码器
            unfreeze_last_layers: 解冻最后几层
        """
        # 加载预训练权重
        checkpoint = torch.load(pretrain_model_path,
                                map_location='cpu', weights_only=False)

        if 'symbol_encoder' in checkpoint:
            self.transformer.symbol_encoder.load_state_dict(
                checkpoint['symbol_encoder'], strict=False
            )

        if 'transformer_encoder' in checkpoint:
            self.transformer.transformer_encoder.load_state_dict(
                checkpoint['transformer_encoder'], strict=False
            )

        # 冻结参数
        if freeze_encoder:
            self.transformer.freeze_parameters(
                freeze_embedding=True,
                freeze_encoder=True,
                unfreeze_last_layers=unfreeze_last_layers
            )

    def get_molecular_embedding(self, smiles_ids: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取分子嵌入表示

        Args:
            smiles_ids: SMILES ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 分子嵌入 [batch_size, d_model]
        """
        batch_size = smiles_ids.size(0)
        device = smiles_ids.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, smiles_ids], dim=1)

        # 创建填充掩码
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, attention_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)

        # Transformer编码
        with torch.no_grad():
            encoded = self.transformer(x, padding_mask)

        # 返回CLS位置的表示
        return encoded[:, 0, :]

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'model_type': f'Property Prediction ({self.task_type})',
            'num_tokens': self.num_tokens,
            'num_labels': self.num_labels,
            'd_model': self.config.dim_embed,
            'num_layers': self.config.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


class PropertyLoss(nn.Module):
    """分子性质预测损失函数"""

    def __init__(self, task_type: str = 'classification',
                 num_labels: int = 1, class_weights: Optional[torch.Tensor] = None):
        """初始化损失函数

        Args:
            task_type: 任务类型
            num_labels: 标签数量
            class_weights: 类别权重（用于不平衡数据）
        """
        super().__init__()
        self.task_type = task_type
        self.num_labels = num_labels

        if task_type == 'classification':
            if num_labels == 1:
                # 二分类
                self.criterion = nn.BCELoss()
            else:
                # 多分类或多标签
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:  # regression
            self.criterion = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失

        Args:
            predictions: 预测结果 [batch_size, num_labels]
            targets: 目标标签 [batch_size, num_labels]

        Returns:
            torch.Tensor: 损失值
        """
        if self.task_type == 'classification':
            if self.num_labels == 1:
                # 二分类：确保目标为float类型
                return self.criterion(predictions.squeeze(), targets.float().squeeze())
            elif self.num_labels == 0:
                # 处理标签数量为0的情况，使用二分类损失
                criterion = nn.BCELoss()
                return criterion(torch.sigmoid(predictions).squeeze(), targets.float().squeeze())
            else:
                # 多分类：目标应为long类型的类别索引
                if targets.dim() > 1 and targets.size(1) > 1:
                    # 多标签分类
                    return nn.functional.binary_cross_entropy(predictions, targets.float())
                else:
                    # 多分类
                    return self.criterion(predictions, targets.long().squeeze())
        else:  # regression
            return self.criterion(predictions, targets.float())


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                      task_type: str = 'classification') -> Dict[str, float]:
    """计算评估指标

    Args:
        predictions: 预测结果 [batch_size, num_labels]
        targets: 目标标签 [batch_size, num_labels]
        task_type: 任务类型

    Returns:
        Dict[str, float]: 评估指标
    """
    metrics = {}

    if task_type == 'classification':
        # 分类指标
        if predictions.size(1) == 1:
            # 二分类
            pred_labels = (predictions > 0.5).float()
            accuracy = (pred_labels.squeeze() ==
                        targets.float().squeeze()).float().mean()
            metrics['accuracy'] = accuracy.item()
        else:
            # 多分类
            pred_labels = torch.argmax(predictions, dim=1)
            if targets.dim() > 1 and targets.size(1) > 1:
                # 多标签分类
                pred_binary = (predictions > 0.5).float()
                accuracy = (pred_binary == targets.float()).float().mean()
                metrics['accuracy'] = accuracy.item()
            else:
                # 多分类
                accuracy = (pred_labels == targets.long().squeeze()
                            ).float().mean()
                metrics['accuracy'] = accuracy.item()

    else:  # regression
        # 回归指标
        mse = nn.functional.mse_loss(predictions, targets.float())
        mae = nn.functional.l1_loss(predictions, targets.float())

        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['rmse'] = torch.sqrt(mse).item()

    return metrics


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min'):
        """初始化早停机制

        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """检查是否应该早停

        Args:
            score: 当前分数

        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否更好"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:  # mode == 'max'
            return current > best + self.min_delta
