"""数据集构建器

该模块负责构建用于预训练和下游任务的数据集。
主要功能：
1. 预训练数据集构建（MLM任务）
2. 分子性质预测数据集构建
3. 气味描述符数据集构建
4. TDC数据集处理
"""

import os
import pickle
import random
from typing import List, Tuple, Dict, Optional, Any
import torch
from torch.utils.data import Dataset
from rdkit import Chem

from .symbol_dictionary import SymbolDictionary
from .smiles_processor import SMILESProcessor
from .csv_processor import CSVProcessor
from config.model_config import SPECIAL_TOKENS
from config.data_config import DEFAULT_DATA_CONFIG, DEFAULT_TDC_CONFIG


def collate_pretrain_batch(batch):
    """预训练数据的自定义collate函数

    处理不同长度的序列，进行填充对齐
    """
    # 获取最大长度
    max_len = max(max(len(item['canonical_smiles']), len(item['target_smiles']),
                      len(item['masked_smiles'])) for item in batch)

    # 填充序列
    canonical_batch = []
    target_batch = []
    masked_batch = []

    for item in batch:
        # 填充到最大长度
        canonical = item['canonical_smiles'] + [SPECIAL_TOKENS.PAD_ID] * \
            (max_len - len(item['canonical_smiles']))
        target = item['target_smiles'] + [SPECIAL_TOKENS.PAD_ID] * \
            (max_len - len(item['target_smiles']))
        masked = item['masked_smiles'] + [SPECIAL_TOKENS.PAD_ID] * \
            (max_len - len(item['masked_smiles']))

        canonical_batch.append(canonical)
        target_batch.append(target)
        masked_batch.append(masked)

    return {
        'canonical_smiles': torch.tensor(canonical_batch, dtype=torch.long),
        'target_smiles': torch.tensor(target_batch, dtype=torch.long),
        'masked_smiles': torch.tensor(masked_batch, dtype=torch.long)
    }


def collate_property_batch(batch):
    """分子性质预测数据的自定义collate函数"""
    # 获取最大长度
    max_len = max(len(item['smiles']) for item in batch)

    # 填充序列
    smiles_batch = []
    labels_batch = []

    for item in batch:
        # 填充SMILES序列
        smiles = item['smiles'] + [SPECIAL_TOKENS.PAD_ID] * \
            (max_len - len(item['smiles']))
        smiles_batch.append(smiles)
        labels_batch.append(item['labels'])

    return {
        'smiles': torch.tensor(smiles_batch, dtype=torch.long),
        'labels': torch.stack(labels_batch)
    }


class PretrainDataset(Dataset):
    """预训练数据集

    用于掩码语言模型预训练的数据集。
    包含规范化SMILES和对应的非规范化SMILES对。
    """

    def __init__(self, canonical_smiles_ids: List[List[int]],
                 target_smiles_ids: List[List[int]],
                 symbol_dict: SymbolDictionary,
                 mask_rate: float = 0.1):
        """初始化预训练数据集

        Args:
            canonical_smiles_ids: 规范化SMILES的ID序列列表
            target_smiles_ids: 目标SMILES的ID序列列表
            symbol_dict: 符号字典
            mask_rate: 掩码率
        """
        self.canonical_smiles_ids = canonical_smiles_ids
        self.target_smiles_ids = target_smiles_ids
        self.symbol_dict = symbol_dict
        self.mask_rate = mask_rate

    def __len__(self) -> int:
        return len(self.canonical_smiles_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        canonical_ids = self.canonical_smiles_ids[idx]
        target_ids = self.target_smiles_ids[idx]

        # 应用掩码
        masked_ids = self._apply_mask(target_ids.copy())

        return {
            'canonical_smiles': canonical_ids,  # 返回列表，稍后在collate中处理
            'target_smiles': target_ids,
            'masked_smiles': masked_ids
        }

    def _apply_mask(self, ids: List[int]) -> List[int]:
        """对ID序列应用掩码

        Args:
            ids: 原始ID序列

        Returns:
            List[int]: 掩码后的ID序列
        """
        masked_ids = []
        for id_val in ids:
            if random.random() < self.mask_rate:
                masked_ids.append(SPECIAL_TOKENS.MSK_ID)
            else:
                masked_ids.append(id_val)
        return masked_ids


class PropertyDataset(Dataset):
    """分子性质预测数据集"""

    def __init__(self, smiles_ids: List[List[int]],
                 labels: List[List[float]],
                 task_type: str = 'classification'):
        """初始化分子性质预测数据集

        Args:
            smiles_ids: SMILES的ID序列列表
            labels: 标签列表
            task_type: 任务类型（'classification' 或 'regression'）
        """
        self.smiles_ids = smiles_ids
        self.labels = labels
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.smiles_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles_ids = self.smiles_ids[idx]
        label = self.labels[idx]

        return {
            'smiles': smiles_ids,  # 返回列表，稍后在collate中处理
            'labels': torch.tensor(label, dtype=torch.float)
        }


class DatasetBuilder:
    """数据集构建器

    负责构建各种类型的数据集，包括预训练数据集和下游任务数据集。
    """

    def __init__(self, config=None):
        """初始化数据集构建器

        Args:
            config: 数据配置对象
        """
        self.config = config or DEFAULT_DATA_CONFIG
        self.smiles_processor = SMILESProcessor(
            self.config.limit_smiles_length)
        self.symbol_dict = SymbolDictionary()
        self.csv_processor = CSVProcessor(self.smiles_processor)

    def build_pretrain_dataset(self, smiles_files: List[str],
                               max_samples: int = None) -> Tuple[List[str], SymbolDictionary]:
        """构建预训练数据集

        Args:
            smiles_files: SMILES文件路径列表
            max_samples: 最大样本数量

        Returns:
            Tuple[List[str], SymbolDictionary]: (SMILES列表, 符号字典)
        """
        print("开始构建预训练数据集...")

        # 1. 处理SMILES文件，构建符号字典
        all_smiles = []
        total_skipped = 0

        for file_path in smiles_files:
            print(f"处理文件: {file_path}")
            smiles_list, skipped = self.smiles_processor.process_smiles_file(
                file_path, max_samples
            )

            # 添加符号到字典
            for smiles in smiles_list:
                self.symbol_dict.add_symbols_from_smiles(smiles)

            all_smiles.extend(smiles_list)
            total_skipped += skipped

            if max_samples and len(all_smiles) >= max_samples:
                all_smiles = all_smiles[:max_samples]
                break

        # 2. 完成符号字典构建
        self.symbol_dict.finalize_dictionary()

        print(f"处理完成:")
        print(f"  - 有效SMILES数量: {len(all_smiles)}")
        print(f"  - 跳过的SMILES数量: {total_skipped}")
        print(f"  - 符号字典大小: {self.symbol_dict.vocab_size}")

        return all_smiles, self.symbol_dict

    def create_pretrain_pairs(self, smiles_list: List[str]) -> Tuple[List[str], List[str]]:
        """创建预训练数据对

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            Tuple[List[str], List[str]]: (规范化SMILES, 非规范化SMILES)
        """
        print("创建预训练数据对...")

        canonical_smiles = []
        target_smiles = []

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"已处理: {i}", end=' ')
            if i % 10000 == 0:
                print()

            # 生成非规范化变体
            variants = self.smiles_processor.generate_noncanonical_smiles(
                smiles)

            # 创建训练对
            for variant in variants:
                canonical_smiles.append(smiles)
                target_smiles.append(variant)

        print(f"\n创建了 {len(canonical_smiles)} 个训练对")
        return canonical_smiles, target_smiles

    def save_pretrain_data(self, smiles_list: List[str],
                           canonical_smiles: List[str],
                           target_smiles: List[str],
                           output_dir: str):
        """保存预训练数据

        Args:
            smiles_list: 原始SMILES列表
            canonical_smiles: 规范化SMILES列表
            target_smiles: 目标SMILES列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始SMILES列表
        smiles_file = os.path.join(
            output_dir, self.config.pretrain_smiles_file)
        with open(smiles_file, 'wb') as f:
            pickle.dump(smiles_list, f)
        print(f"已保存SMILES列表到: {smiles_file}")

        # 保存符号字典
        dict_file = os.path.join(output_dir, self.config.symbol_dict_file)
        self.symbol_dict.save(dict_file)
        print(f"已保存符号字典到: {dict_file}")

        # 转换为ID序列
        print("转换SMILES为ID序列...")
        canonical_ids = [self.symbol_dict.smiles_to_ids(
            smiles) for smiles in canonical_smiles]
        target_ids = [self.symbol_dict.smiles_to_ids(
            smiles) for smiles in target_smiles]

        # 保存预训练数据
        pretrain_data = [canonical_ids, target_ids]
        data_file = os.path.join(output_dir, self.config.pretrain_data_file)
        with open(data_file, 'wb') as f:
            pickle.dump(pretrain_data, f)
        print(f"已保存预训练数据到: {data_file}")

    def load_pretrain_data(self, data_dir: str) -> Tuple[List[List[int]], List[List[int]], SymbolDictionary]:
        """加载预训练数据

        Args:
            data_dir: 数据目录

        Returns:
            Tuple: (规范化SMILES_IDs, 目标SMILES_IDs, 符号字典)
        """
        # 加载符号字典
        dict_file = os.path.join(data_dir, self.config.symbol_dict_file)
        self.symbol_dict.load(dict_file)

        # 加载预训练数据
        data_file = os.path.join(data_dir, self.config.pretrain_data_file)
        with open(data_file, 'rb') as f:
            canonical_ids, target_ids = pickle.load(f)

        return canonical_ids, target_ids, self.symbol_dict

    def build_odor_dataset(self, odorant_file: str, odorless_file: str) -> Tuple[List[List[int]], List[List[float]], Dict]:
        """构建气味描述符数据集

        Args:
            odorant_file: 有气味分子文件路径
            odorless_file: 无气味分子文件路径

        Returns:
            Tuple: (SMILES_IDs, 标签, 标签信息)
        """
        print("构建气味描述符数据集...")

        # 1. 统计气味描述符频次
        label_freq = {}

        # 处理有气味分子文件
        with open(odorant_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                smiles_str = parts[1]
                if len(smiles_str) > self.config.limit_smiles_length:
                    continue

                mol = Chem.MolFromSmiles(smiles_str)
                if mol is None:
                    continue

                # 统计气味描述符
                odor_descriptors = parts[2:]
                for od in odor_descriptors:
                    if od == 'odorless':
                        continue
                    label_freq[od] = label_freq.get(od, 0) + 1

        # 2. 创建标签映射
        label_to_id = {}
        id_to_label = []
        label_id = 0

        # 按频次排序，只保留频次大于阈值的标签
        sorted_labels = sorted(
            label_freq.items(), key=lambda x: x[1], reverse=True)
        for label, freq in sorted_labels:
            # 对于演示数据，使用更低的阈值
            freq_threshold = min(self.config.limit_freq_od, 1)
            if freq >= freq_threshold:
                label_to_id[label] = label_id
                id_to_label.append(label)
                label_id += 1

        print(f"保留了 {len(id_to_label)} 个气味描述符")

        # 3. 构建数据集
        smiles_ids = []
        labels = []

        def process_file(filepath: str):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    smiles_str = parts[1]
                    if len(smiles_str) > self.config.limit_smiles_length:
                        continue

                    mol = Chem.MolFromSmiles(smiles_str)
                    if mol is None:
                        continue

                    # 规范化SMILES
                    canonical_smiles = Chem.MolToSmiles(mol)
                    smiles_id = self.symbol_dict.smiles_to_ids(
                        canonical_smiles)

                    # 创建标签向量
                    label_vec = [0.0] * len(id_to_label)
                    if len(parts) > 2:  # 有气味描述符
                        odor_descriptors = parts[2:]
                        for od in odor_descriptors:
                            if od in label_to_id:
                                label_vec[label_to_id[od]] = 1.0

                    smiles_ids.append(smiles_id)
                    labels.append(label_vec)

        # 处理两个文件
        process_file(odorant_file)
        process_file(odorless_file)

        label_info = {
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(id_to_label),
            'task_type': 'classification'
        }

        print(f"构建完成，共 {len(smiles_ids)} 个样本")
        return smiles_ids, labels, label_info

    def build_tdc_dataset(self, task_name: str, data_path: str = None) -> Tuple[List[List[int]], List[List[float]], Dict]:
        """构建TDC数据集

        Args:
            task_name: 任务名称
            data_path: TDC数据路径

        Returns:
            Tuple: (SMILES_IDs, 标签, 任务信息)
        """
        try:
            import tdc
        except ImportError:
            raise ImportError("需要安装TDC库: pip install PyTDC")

        print(f"构建TDC数据集: {task_name}")

        if data_path is None:
            data_path = DEFAULT_TDC_CONFIG.tdc_data_path

        # 加载TDC数据
        tdc_group = tdc.benchmark_group.admet_group(path=data_path)
        benchmark = tdc_group.get(task_name)
        train_val, test_set = benchmark['train_val'], benchmark['test']

        # 确定任务类型
        is_regression = DEFAULT_TDC_CONFIG.is_regression_task(task_name)
        task_type = 'regression' if is_regression else 'classification'

        def process_tdc_data(smiles_list: List[str], label_list: List[float]) -> Tuple[List[List[int]], List[List[float]]]:
            """处理TDC数据"""
            processed_smiles = []
            processed_labels = []

            for smiles_str, label in zip(smiles_list, label_list):
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is None:
                    continue

                # 规范化SMILES并截断
                canonical_smiles = Chem.MolToSmiles(mol)
                if len(canonical_smiles) > self.config.limit_smiles_length:
                    canonical_smiles = canonical_smiles[:
                                                        self.config.limit_smiles_length]

                smiles_id = self.symbol_dict.smiles_to_ids(canonical_smiles)
                processed_smiles.append(smiles_id)
                processed_labels.append([float(label)])

            return processed_smiles, processed_labels

        # 处理训练验证数据
        train_smiles_ids, train_labels = process_tdc_data(
            train_val['Drug'].tolist(), train_val['Y'].tolist()
        )

        # 处理测试数据
        test_smiles_ids, test_labels = process_tdc_data(
            test_set['Drug'].tolist(), test_set['Y'].tolist()
        )

        task_info = {
            'task_name': task_name,
            'task_type': task_type,
            'num_labels': 1,
            'train_size': len(train_smiles_ids),
            'test_size': len(test_smiles_ids)
        }

        print(f"构建完成:")
        print(f"  - 训练样本: {len(train_smiles_ids)}")
        print(f"  - 测试样本: {len(test_smiles_ids)}")
        print(f"  - 任务类型: {task_type}")

        return (train_smiles_ids + test_smiles_ids,
                train_labels + test_labels,
                task_info)

    def build_multi_label_odor_dataset(self, csv_path: str,
                                       smiles_column: str = 'nonStereoSMILES',
                                       descriptor_column: str = 'descriptors',
                                       min_label_frequency: int = 5,
                                       max_smiles_length: int = 200) -> Tuple[List[List[int]], List[List[float]], Dict]:
        """构建多标签气味数据集（从CSV文件）

        Args:
            csv_path: CSV文件路径
            smiles_column: SMILES列名
            descriptor_column: 描述符列名（可选）
            min_label_frequency: 最小标签频次
            max_smiles_length: 最大SMILES长度

        Returns:
            Tuple: (SMILES_IDs, 标签, 任务信息)
        """
        print(f"构建多标签气味数据集: {csv_path}")

        # 使用CSV处理器加载数据
        smiles_list, labels_list, label_to_id, label_names = self.csv_processor.load_multi_label_csv(
            csv_path, smiles_column, descriptor_column, min_label_frequency, max_smiles_length
        )

        # 转换SMILES为ID序列
        print("转换SMILES为ID序列...")
        smiles_ids = []
        valid_labels = []

        for i, smiles in enumerate(smiles_list):
            try:
                smiles_id = self.symbol_dict.smiles_to_ids(smiles)
                smiles_ids.append(smiles_id)
                valid_labels.append([float(x) for x in labels_list[i]])
            except Exception as e:
                print(f"处理SMILES失败: {smiles} - {e}")
                continue

        # 创建任务信息
        task_info = {
            'task_type': 'classification',
            'num_labels': len(label_names),
            'label_to_id': label_to_id,
            'id_to_label': label_names,
            'label_names': label_names,
            'csv_path': csv_path,
            'smiles_column': smiles_column,
            'min_label_frequency': min_label_frequency
        }

        print(f"多标签气味数据集构建完成:")
        print(f"  样本数量: {len(smiles_ids)}")
        print(f"  标签数量: {len(label_names)}")
        print(f"  符号字典大小: {self.symbol_dict.vocab_size}")

        return smiles_ids, valid_labels, task_info

    def get_dataset_statistics(self, smiles_ids: List[List[int]],
                               labels: List[List[float]] = None) -> Dict[str, Any]:
        """获取数据集统计信息

        Args:
            smiles_ids: SMILES ID序列列表
            labels: 标签列表（可选）

        Returns:
            Dict: 统计信息
        """
        if not smiles_ids:
            return {}

        lengths = [len(ids) for ids in smiles_ids]
        stats = {
            'total_samples': len(smiles_ids),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'vocab_size': self.symbol_dict.vocab_size
        }

        if labels:
            label_lengths = [len(label) for label in labels]
            stats.update({
                'num_labels': max(label_lengths) if label_lengths else 0,
                'label_distribution': self._get_label_distribution(labels)
            })

        return stats

    def _get_label_distribution(self, labels: List[List[float]]) -> Dict[str, int]:
        """获取标签分布统计"""
        if not labels or not labels[0]:
            return {}

        num_labels = len(labels[0])
        positive_counts = [0] * num_labels

        for label_vec in labels:
            for i, val in enumerate(label_vec):
                if val > 0.5:  # 假设0.5为阈值
                    positive_counts[i] += 1

        return {f'label_{i}': count for i, count in enumerate(positive_counts)}
