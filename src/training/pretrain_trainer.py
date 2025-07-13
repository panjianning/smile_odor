"""预训练训练器

该模块实现SMILES预训练模型的训练流程。
主要功能：
1. 预训练数据加载和处理
2. 模型训练循环
3. 损失和指标监控
4. 模型保存和加载
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.models.pretrain_model import MLMPretrainModel, PretrainLoss, calculate_accuracy
from src.data.dataset_builder import DatasetBuilder, PretrainDataset, collate_pretrain_batch
from config.model_config import PretrainConfig, SPECIAL_TOKENS


class PretrainTrainer:
    """预训练训练器

    负责SMILES预训练模型的完整训练流程。
    """

    def __init__(self, config: PretrainConfig, data_dir: str,
                 save_dir: str, device: str = 'cuda'):
        """初始化预训练训练器

        Args:
            config: 预训练配置
            data_dir: 数据目录
            save_dir: 模型保存目录
            device: 训练设备
        """
        self.config = config
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化组件
        self.dataset_builder = DatasetBuilder()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = PretrainLoss()

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.train_accuracies = []

    def setup_model(self, num_tokens: int):
        """设置模型

        Args:
            num_tokens: 词汇表大小
        """
        print(f"初始化预训练模型...")
        self.model = MLMPretrainModel(self.config, num_tokens)
        self.model.to(self.device)

        # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epoch,
            eta_min=self.config.learning_rate * 0.1
        )

    def load_data(self) -> Tuple[DataLoader, int]:
        """加载预训练数据

        Returns:
            Tuple[DataLoader, int]: (数据加载器, 词汇表大小)
        """
        print("加载预训练数据...")

        # 加载数据
        canonical_ids, target_ids, symbol_dict = self.dataset_builder.load_pretrain_data(
            self.data_dir
        )

        print(f"数据统计:")
        print(f"  - 训练样本数: {len(canonical_ids)}")
        print(f"  - 词汇表大小: {symbol_dict.vocab_size}")

        # 创建数据集
        dataset = PretrainDataset(
            canonical_ids, target_ids, symbol_dict, self.config.mask_rate
        )

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_pretrain_batch
        )

        return dataloader, symbol_dict.vocab_size

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch

        Args:
            dataloader: 数据加载器

        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # 数据移到设备
            canonical_smiles = batch['canonical_smiles'].to(self.device)
            target_smiles = batch['target_smiles'].to(self.device)
            masked_smiles = batch['masked_smiles'].to(self.device)

            # 前向传播
            predictions = self.model(canonical_smiles, masked_smiles)

            # 计算损失
            loss = self.criterion(predictions, target_smiles)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 计算准确率
            with torch.no_grad():
                # 创建掩码位置标记
                mask_positions = (masked_smiles == SPECIAL_TOKENS.MSK_ID)
                accuracy_metrics = calculate_accuracy(
                    predictions, target_smiles, mask_positions
                )

            # 累计指标
            total_loss += loss.item()
            total_accuracy += accuracy_metrics.get('mask_accuracy', 0.0)
            num_batches += 1

            # 打印进度
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {accuracy_metrics.get('mask_accuracy', 0.0):.4f}")

        # 更新学习率
        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点

        Args:
            epoch: 当前epoch
            metrics: 训练指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存
        if metrics['loss'] < self.best_loss:
            self.best_loss = metrics['loss']
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  保存最佳模型 (loss: {metrics['loss']:.4f})")

        # 定期保存编码器
        if epoch % 50 == 0 or epoch == self.config.num_epoch:
            self.model.save_encoders(self.save_dir, epoch)
            print(f"  保存编码器 (epoch {epoch})")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(checkpoint_path):
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])

            print(f"成功加载检查点 (epoch {self.current_epoch})")
            return True

        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False

    def train(self, resume_from_checkpoint: bool = False):
        """开始训练

        Args:
            resume_from_checkpoint: 是否从检查点恢复训练
        """
        print("开始预训练...")

        # 加载数据
        dataloader, num_tokens = self.load_data()

        # 设置模型
        self.setup_model(num_tokens)
        self.setup_optimizer()

        # 尝试恢复训练
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(
                self.save_dir, 'latest_checkpoint.pt')
            self.load_checkpoint(checkpoint_path)

        # 训练循环
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epoch):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epoch}")
            print("-" * 50)

            # 训练一个epoch
            epoch_start = time.time()
            metrics = self.train_epoch(dataloader)
            epoch_time = time.time() - epoch_start

            # 记录指标
            self.train_losses.append(metrics['loss'])
            self.train_accuracies.append(metrics['accuracy'])

            # 打印结果
            print(f"训练结果:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Learning Rate: {metrics['learning_rate']:.6f}")
            print(f"  Time: {epoch_time:.2f}s")

            # 保存检查点
            self.save_checkpoint(epoch + 1, metrics)

            # 更新当前epoch
            self.current_epoch = epoch + 1

        total_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {total_time:.2f}s")

        # 保存最终模型
        final_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_tokens': num_tokens
        }, final_path)
        print(f"最终模型已保存到: {final_path}")

    def evaluate_sample(self, canonical_smiles: str, symbol_dict) -> Dict[str, any]:
        """评估单个样本

        Args:
            canonical_smiles: 规范化SMILES字符串
            symbol_dict: 符号字典

        Returns:
            Dict: 评估结果
        """
        self.model.eval()

        with torch.no_grad():
            # 转换为ID序列
            smiles_ids = symbol_dict.smiles_to_ids(canonical_smiles)
            smiles_tensor = torch.tensor(
                [smiles_ids], dtype=torch.long, device=self.device)

            # 应用掩码
            masked_smiles = self.model.apply_masking(smiles_tensor)

            # 预测
            predictions = self.model(smiles_tensor, masked_smiles)

            # 获取预测的符号
            predicted_ids = torch.argmax(
                predictions, dim=-1).squeeze().cpu().tolist()
            predicted_smiles = symbol_dict.ids_to_smiles(predicted_ids)

            # 获取分子嵌入
            molecular_embedding = self.model.get_molecular_embedding(
                smiles_tensor)

            return {
                'original_smiles': canonical_smiles,
                'masked_smiles': symbol_dict.ids_to_smiles(masked_smiles.squeeze().cpu().tolist()),
                'predicted_smiles': predicted_smiles,
                'molecular_embedding': molecular_embedding.cpu().numpy(),
                'embedding_dim': molecular_embedding.size(-1)
            }

    def get_training_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies
        }
