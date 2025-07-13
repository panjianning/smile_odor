"""CSV数据处理器

该模块专门处理CSV格式的多标签气味数据集。
主要功能：
1. 解析CSV格式的多标签数据
2. 标签频次统计和过滤
3. 数据预处理和验证
4. 支持多种CSV格式
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
import os

from .symbol_dictionary import SymbolDictionary
from .smiles_processor import SMILESProcessor


class CSVProcessor:
    """CSV数据处理器

    专门处理多标签气味数据集的CSV文件。
    """

    def __init__(self, smiles_processor: Optional[SMILESProcessor] = None):
        """初始化CSV处理器

        Args:
            smiles_processor: SMILES处理器实例
        """
        self.smiles_processor = smiles_processor or SMILESProcessor()

    def load_multi_label_csv(self, csv_path: str,
                             smiles_column: str = 'nonStereoSMILES',
                             descriptor_column: str = 'descriptors',
                             min_label_frequency: int = 5,
                             max_smiles_length: int = 200) -> Tuple[List[str], List[List[int]], Dict[str, int], List[str]]:
        """加载多标签CSV数据

        Args:
            csv_path: CSV文件路径
            smiles_column: SMILES列名
            descriptor_column: 描述符列名（可选）
            min_label_frequency: 最小标签频次
            max_smiles_length: 最大SMILES长度

        Returns:
            Tuple: (SMILES列表, 标签矩阵, 标签到ID映射, 标签名列表)
        """
        print(f"加载多标签CSV数据: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"原始数据形状: {df.shape}")

        # 检查必要的列
        if smiles_column not in df.columns:
            raise ValueError(f"找不到SMILES列: {smiles_column}")

        # 获取所有可能的标签列（除了SMILES和描述符列）
        exclude_columns = {smiles_column}
        if descriptor_column in df.columns:
            exclude_columns.add(descriptor_column)

        label_columns = [
            col for col in df.columns if col not in exclude_columns]
        print(f"发现 {len(label_columns)} 个标签列")

        # 统计标签频次
        label_frequencies = {}
        for col in label_columns:
            if col in df.columns:
                freq = df[col].sum() if df[col].dtype in [
                    'int64', 'float64'] else 0
                label_frequencies[col] = freq

        # 过滤低频标签
        filtered_labels = [label for label, freq in label_frequencies.items()
                           if freq >= min_label_frequency]
        filtered_labels.sort()  # 保持一致的顺序

        print(f"过滤后保留 {len(filtered_labels)} 个标签 (频次 >= {min_label_frequency})")

        # 创建标签到ID的映射
        label_to_id = {label: i for i, label in enumerate(filtered_labels)}

        # 处理数据
        valid_smiles = []
        valid_labels = []
        skipped_count = 0

        for idx, row in df.iterrows():
            smiles = str(row[smiles_column]).strip()

            # 验证SMILES
            if not self._is_valid_smiles(smiles, max_smiles_length):
                skipped_count += 1
                continue

            # 规范化SMILES
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol)
            except:
                skipped_count += 1
                continue

            # 构建标签向量
            label_vector = [0] * len(filtered_labels)
            for i, label in enumerate(filtered_labels):
                if label in df.columns:
                    value = row[label]
                    # 处理不同的标签格式
                    if pd.isna(value):
                        label_vector[i] = 0
                    elif isinstance(value, (int, float)):
                        label_vector[i] = int(value > 0.5)  # 阈值化
                    else:
                        label_vector[i] = 1 if str(value).lower() in [
                            '1', 'true', 'yes'] else 0

            valid_smiles.append(canonical_smiles)
            valid_labels.append(label_vector)

        print(f"处理完成:")
        print(f"  有效样本: {len(valid_smiles)}")
        print(f"  跳过样本: {skipped_count}")
        print(f"  标签维度: {len(filtered_labels)}")

        return valid_smiles, valid_labels, label_to_id, filtered_labels

    def _is_valid_smiles(self, smiles: str, max_length: int = 200) -> bool:
        """验证SMILES有效性

        Args:
            smiles: SMILES字符串
            max_length: 最大长度

        Returns:
            bool: 是否有效
        """
        if not smiles or len(smiles) > max_length:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def analyze_dataset(self, csv_path: str,
                        smiles_column: str = 'nonStereoSMILES') -> Dict[str, any]:
        """分析数据集统计信息

        Args:
            csv_path: CSV文件路径
            smiles_column: SMILES列名

        Returns:
            Dict: 统计信息
        """
        print(f"分析数据集: {csv_path}")

        df = pd.read_csv(csv_path)

        # 基本统计
        stats = {
            'total_samples': len(df),
            'total_columns': len(df.columns),
            'smiles_column': smiles_column
        }

        # SMILES统计
        if smiles_column in df.columns:
            smiles_lengths = df[smiles_column].astype(str).str.len()
            stats['smiles_stats'] = {
                'min_length': int(smiles_lengths.min()),
                'max_length': int(smiles_lengths.max()),
                'avg_length': float(smiles_lengths.mean()),
                'valid_smiles': sum(1 for s in df[smiles_column] if self._is_valid_smiles(str(s)))
            }

        # 标签统计
        exclude_columns = {smiles_column, 'descriptors'}
        label_columns = [
            col for col in df.columns if col not in exclude_columns]

        label_stats = {}
        for col in label_columns:
            if df[col].dtype in ['int64', 'float64']:
                positive_count = int(df[col].sum())
                label_stats[col] = {
                    'positive_samples': positive_count,
                    'negative_samples': len(df) - positive_count,
                    'positive_ratio': positive_count / len(df)
                }

        stats['label_stats'] = label_stats
        stats['num_labels'] = len(label_columns)

        # 标签共现统计
        if len(label_columns) > 1:
            label_matrix = df[label_columns].values
            cooccurrence = np.dot(label_matrix.T, label_matrix)
            stats['label_cooccurrence'] = {
                'max_cooccurrence': int(cooccurrence.max()),
                'avg_labels_per_sample': float(label_matrix.sum(axis=1).mean())
            }

        return stats

    def create_balanced_split(self, smiles_list: List[str],
                              labels_list: List[List[int]],
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1,
                              random_seed: int = 42) -> Tuple[Dict[str, List], Dict[str, List]]:
        """创建平衡的数据集划分

        Args:
            smiles_list: SMILES列表
            labels_list: 标签列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_seed: 随机种子

        Returns:
            Tuple: (数据划分, 统计信息)
        """
        np.random.seed(random_seed)

        total_samples = len(smiles_list)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # 计算划分点
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        # 划分索引
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # 创建数据集
        splits = {
            'train': {
                'smiles': [smiles_list[i] for i in train_indices],
                'labels': [labels_list[i] for i in train_indices]
            },
            'val': {
                'smiles': [smiles_list[i] for i in val_indices],
                'labels': [labels_list[i] for i in val_indices]
            },
            'test': {
                'smiles': [smiles_list[i] for i in test_indices],
                'labels': [labels_list[i] for i in test_indices]
            }
        }

        # 统计信息
        stats = {
            'total_samples': total_samples,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'train_ratio': len(train_indices) / total_samples,
            'val_ratio': len(val_indices) / total_samples,
            'test_ratio': len(test_indices) / total_samples
        }

        # 每个集合的标签分布
        for split_name, split_data in splits.items():
            if split_data['labels']:
                label_matrix = np.array(split_data['labels'])
                stats[f'{split_name}_label_distribution'] = {
                    'avg_labels_per_sample': float(label_matrix.sum(axis=1).mean()),
                    'total_positive_labels': int(label_matrix.sum())
                }

        return splits, stats

    def save_processed_data(self, smiles_list: List[str],
                            labels_list: List[List[int]],
                            label_to_id: Dict[str, int],
                            label_names: List[str],
                            output_dir: str,
                            symbol_dict: Optional[SymbolDictionary] = None):
        """保存处理后的数据

        Args:
            smiles_list: SMILES列表
            labels_list: 标签列表
            label_to_id: 标签到ID映射
            label_names: 标签名列表
            output_dir: 输出目录
            symbol_dict: 符号字典
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存SMILES和标签
        data = {
            'smiles': smiles_list,
            'labels': labels_list,
            'label_to_id': label_to_id,
            'label_names': label_names
        }

        import pickle
        with open(os.path.join(output_dir, 'processed_data.pkl'), 'wb') as f:
            pickle.dump(data, f)

        # 保存为CSV格式（便于查看）
        df_data = {'SMILES': smiles_list}
        for i, label_name in enumerate(label_names):
            df_data[label_name] = [labels[i] for labels in labels_list]

        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)

        # 保存符号字典
        if symbol_dict:
            symbol_dict.save(os.path.join(output_dir, 'symbol_dict.pkl'))

        # 保存元数据
        metadata = {
            'num_samples': len(smiles_list),
            'num_labels': len(label_names),
            'label_names': label_names,
            'label_to_id': label_to_id
        }

        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"数据已保存到: {output_dir}")
        print(f"  样本数量: {len(smiles_list)}")
        print(f"  标签数量: {len(label_names)}")
