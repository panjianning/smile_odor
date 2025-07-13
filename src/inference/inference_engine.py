"""推理引擎

该模块实现分子性质预测的推理功能。
主要功能：
1. 模型加载和初始化
2. 单个和批量SMILES预测
3. 结果后处理和输出
4. 多标签气味预测
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from rdkit import Chem
import pickle

from src.models.property_model import PropertyPredictionModel
from src.data.symbol_dictionary import SymbolDictionary
from src.data.smiles_processor import SMILESProcessor
from config.model_config import PropertyPredConfig, SPECIAL_TOKENS
from src.data.dataset_builder import DatasetBuilder


class InferenceEngine:
    """推理引擎

    负责加载训练好的模型并进行预测。
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        """初始化推理引擎

        Args:
            model_path: 训练好的模型路径
            device: 推理设备
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.symbol_dict = None
        self.smiles_processor = None
        self.task_info = None
        self.config = None

        # 加载模型
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载训练好的模型

        Args:
            model_path: 模型路径
        """
        print(f"加载模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载检查点
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False)

        # 获取配置信息
        self.config = checkpoint.get('config', PropertyPredConfig())

        # 获取任务信息
        self.task_info = checkpoint.get('task_info', {
            'task_type': 'classification',
            'num_labels': 138,
            'id_to_label': ['unknown']
        })

        self.dataset_builder = DatasetBuilder()
        self.dataset_builder.load_pretrain_data('./data/pretrain')

        self.symbol_dict = self.dataset_builder.symbol_dict

        # 初始化SMILES处理器
        self.smiles_processor = SMILESProcessor()

        # 创建模型
        self.model = PropertyPredictionModel(
            config=self.config,
            num_tokens=self.symbol_dict.vocab_size,
            num_labels=self.task_info['num_labels'],
            task_type=self.task_info['task_type']
        )

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成!")
        print(f"  任务类型: {self.task_info['task_type']}")
        print(f"  标签数量: {self.task_info['num_labels']}")
        print(f"  词汇表大小: {self.symbol_dict.vocab_size}")

    def preprocess_smiles(self, smiles_list: List[str]) -> Tuple[List[List[int]], List[bool]]:
        """预处理SMILES字符串

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            Tuple[List[List[int]], List[bool]]: (ID序列列表, 有效性标记列表)
        """
        processed_ids = []
        valid_flags = []

        for smiles in smiles_list:
            try:
                # 验证SMILES有效性
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    processed_ids.append([])
                    valid_flags.append(False)
                    continue

                # 规范化SMILES
                canonical_smiles = Chem.MolToSmiles(mol)

                # 转换为ID序列
                smiles_ids = self.symbol_dict.smiles_to_ids(canonical_smiles)

                processed_ids.append(smiles_ids)
                valid_flags.append(True)

            except Exception as e:
                print(f"处理SMILES失败: {smiles} - {e}")
                processed_ids.append([])
                valid_flags.append(False)

        return processed_ids, valid_flags

    def predict_batch(self, smiles_list: List[str],
                      batch_size: int = 32,
                      return_probabilities: bool = True) -> Dict[str, Union[List, np.ndarray]]:
        """批量预测

        Args:
            smiles_list: SMILES字符串列表
            batch_size: 批处理大小
            return_probabilities: 是否返回概率值

        Returns:
            Dict: 预测结果
        """
        print(f"开始批量预测，共 {len(smiles_list)} 个SMILES...")

        # 预处理SMILES
        smiles_ids_list, valid_flags = self.preprocess_smiles(smiles_list)

        all_predictions = []
        all_probabilities = []

        # 批量处理
        for i in range(0, len(smiles_ids_list), batch_size):
            batch_ids = smiles_ids_list[i:i + batch_size]
            batch_valid = valid_flags[i:i + batch_size]

            # 过滤有效的SMILES
            valid_batch_ids = [ids for ids, valid in zip(
                batch_ids, batch_valid) if valid]

            if not valid_batch_ids:
                # 如果批次中没有有效SMILES，添加默认预测
                batch_predictions = [
                    np.zeros(self.task_info['num_labels'])] * len(batch_ids)
                batch_probabilities = [
                    np.zeros(self.task_info['num_labels'])] * len(batch_ids)
            else:
                # 进行预测
                batch_predictions, batch_probabilities = self._predict_batch_ids(
                    valid_batch_ids)

                # 将结果映射回原始顺序
                pred_idx = 0
                final_batch_predictions = []
                final_batch_probabilities = []

                for valid in batch_valid:
                    if valid:
                        final_batch_predictions.append(
                            batch_predictions[pred_idx])
                        final_batch_probabilities.append(
                            batch_probabilities[pred_idx])
                        pred_idx += 1
                    else:
                        final_batch_predictions.append(
                            np.zeros(self.task_info['num_labels']))
                        final_batch_probabilities.append(
                            np.zeros(self.task_info['num_labels']))

                batch_predictions = final_batch_predictions
                batch_probabilities = final_batch_probabilities

            all_predictions.extend(batch_predictions)
            all_probabilities.extend(batch_probabilities)

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"已处理 {min(i + batch_size, len(smiles_ids_list))} / {len(smiles_ids_list)} 个样本")

        results = {
            'smiles': smiles_list,
            'valid': valid_flags,
            'predictions': all_predictions,
        }

        if return_probabilities:
            results['probabilities'] = all_probabilities

        return results

    def _predict_batch_ids(self, batch_ids: List[List[int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """对ID序列批次进行预测

        Args:
            batch_ids: ID序列批次

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (预测标签, 概率值)
        """
        if not batch_ids:
            return [], []

        # 填充序列到相同长度
        max_len = max(len(ids) for ids in batch_ids)
        padded_batch = []

        for ids in batch_ids:
            padded = ids + [SPECIAL_TOKENS.PAD_ID] * (max_len - len(ids))
            padded_batch.append(padded)

        # 转换为张量
        input_tensor = torch.tensor(
            padded_batch, dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # 转换为numpy数组
        outputs_np = outputs.cpu().numpy()

        predictions = []
        probabilities = []

        for output in outputs_np:
            if self.task_info['task_type'] == 'classification':
                if self.task_info['num_labels'] == 1:
                    # 二分类
                    prob = float(output[0])
                    pred = 1 if prob > 0.5 else 0
                    predictions.append(np.array([pred]))
                    probabilities.append(np.array([prob]))
                else:
                    # 多分类或多标签
                    if len(output.shape) == 1 or output.shape[0] == 1:
                        # 多标签二分类（每个标签独立的二分类）
                        probs = output
                        preds = (probs > 0.5).astype(int)
                    else:
                        # 多分类
                        probs = output
                        preds = np.zeros_like(probs)
                        preds[np.argmax(probs)] = 1

                    predictions.append(preds)
                    probabilities.append(probs)
            else:
                # 回归
                predictions.append(output)
                probabilities.append(output)  # 回归任务中概率就是预测值

        return predictions, probabilities

    def predict_single(self, smiles: str, return_probabilities: bool = True) -> Dict[str, Union[str, np.ndarray, Dict]]:
        """单个SMILES预测

        Args:
            smiles: SMILES字符串
            return_probabilities: 是否返回概率值

        Returns:
            Dict: 预测结果
        """
        results = self.predict_batch(
            [smiles], batch_size=1, return_probabilities=return_probabilities)

        result = {
            'smiles': smiles,
            'valid': results['valid'][0],
            'predictions': results['predictions'][0],
        }

        if return_probabilities:
            result['probabilities'] = results['probabilities'][0]

        # 如果有标签名称，添加标签解释
        if 'id_to_label' in self.task_info and results['valid'][0]:
            result['predicted_labels'] = self._interpret_predictions(
                results['predictions'][0],
                results['probabilities'][0] if return_probabilities else None
            )

        return result

    def _interpret_predictions(self, predictions: np.ndarray,
                               probabilities: Optional[np.ndarray] = None) -> Dict[str, Union[bool, float]]:
        """解释预测结果

        Args:
            predictions: 预测标签
            probabilities: 概率值

        Returns:
            Dict: 标签解释
        """
        interpreted = {}
        id_to_label = self.task_info.get('id_to_label', [])

        for i, (pred, label) in enumerate(zip(predictions, id_to_label)):
            if probabilities is not None:
                interpreted[label] = {
                    'predicted': bool(pred),
                    'probability': float(probabilities[i])
                }
            else:
                interpreted[label] = bool(pred)

        return interpreted

    def predict_from_file(self, input_file: str, output_file: str,
                          batch_size: int = 32,
                          probability_threshold: float = 0.5) -> None:
        """从文件读取SMILES并预测，结果保存到文件

        Args:
            input_file: 输入文件路径（支持txt, csv格式）
            output_file: 输出文件路径
            batch_size: 批处理大小
            probability_threshold: 概率阈值
        """
        print(f"从文件预测: {input_file} -> {output_file}")

        # 读取输入文件
        smiles_list = self._read_smiles_file(input_file)
        print(f"读取到 {len(smiles_list)} 个SMILES")

        # 批量预测
        results = self.predict_batch(smiles_list, batch_size=batch_size)

        # 保存结果
        self._save_results(results, output_file, probability_threshold)
        print(f"结果已保存到: {output_file}")

    def _read_smiles_file(self, file_path: str) -> List[str]:
        """读取SMILES文件

        Args:
            file_path: 文件路径

        Returns:
            List[str]: SMILES列表
        """
        smiles_list = []

        if file_path.endswith('.csv'):
            # CSV文件
            df = pd.read_csv(file_path)
            if 'SMILES' in df.columns:
                smiles_list = df['SMILES'].tolist()
            elif 'smiles' in df.columns:
                smiles_list = df['smiles'].tolist()
            elif 'nonStereoSMILES' in df.columns:
                smiles_list = df['nonStereoSMILES'].tolist()
            else:
                # 假设第一列是SMILES
                smiles_list = df.iloc[:, 0].tolist()
        else:
            # 文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 如果行包含多个字段，取第一个作为SMILES
                        smiles = line.split()[0] if ' ' in line else line
                        smiles_list.append(smiles)

        return smiles_list

    def _save_results(self, results: Dict, output_file: str,
                      probability_threshold: float = 0.5) -> None:
        """保存预测结果

        Args:
            results: 预测结果
            output_file: 输出文件路径
            probability_threshold: 概率阈值
        """
        # 准备输出数据
        output_data = []
        id_to_label = self.task_info.get(
            'id_to_label', [f'label_{i}' for i in range(self.task_info['num_labels'])])

        for i, smiles in enumerate(results['smiles']):
            row = {
                'SMILES': smiles,
                'Valid': results['valid'][i]
            }

            if results['valid'][i]:
                predictions = results['predictions'][i]
                probabilities = results.get(
                    'probabilities', [None] * len(results['predictions']))[i]

                # 添加每个标签的预测结果
                for j, label in enumerate(id_to_label):
                    if j < len(predictions):
                        row[f'{label}_predicted'] = bool(predictions[j])
                        if probabilities is not None and j < len(probabilities):
                            row[f'{label}_probability'] = float(
                                probabilities[j])
                            row[f'{label}_confident'] = float(
                                probabilities[j]) > probability_threshold
            else:
                # 无效SMILES的默认值
                for label in id_to_label:
                    row[f'{label}_predicted'] = False
                    row[f'{label}_probability'] = 0.0
                    row[f'{label}_confident'] = False

            output_data.append(row)

        # 保存为CSV
        df = pd.DataFrame(output_data)
        df.to_csv(output_file, index=False)

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息

        Returns:
            Dict: 模型信息
        """
        info = {
            'task_type': self.task_info.get('task_type', 'unknown'),
            'num_labels': self.task_info.get('num_labels', 0),
            'vocab_size': self.symbol_dict.vocab_size if self.symbol_dict else 0,
            'device': str(self.device),
            'labels': self.task_info.get('id_to_label', [])
        }

        if self.model:
            model_info = self.model.get_model_info()
            info.update(model_info)

        return info
