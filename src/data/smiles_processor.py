"""SMILES处理器

该模块负责SMILES字符串的处理和验证。
主要功能：
1. SMILES字符串验证和规范化
2. 生成非规范化SMILES变体
3. 分子结构验证
4. 数据清洗和过滤
"""

import random
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from config.data_config import DEFAULT_DATA_CONFIG


class SMILESProcessor:
    """SMILES处理器

    负责SMILES字符串的各种处理操作，包括验证、规范化、
    生成变体等功能。
    """

    def __init__(self, limit_length: int = None):
        """初始化SMILES处理器

        Args:
            limit_length: SMILES最大长度限制
        """
        self.limit_length = limit_length or DEFAULT_DATA_CONFIG.limit_smiles_length

    def is_valid_smiles(self, smiles_str: str) -> bool:
        """验证SMILES字符串是否有效

        Args:
            smiles_str: SMILES字符串

        Returns:
            bool: 是否有效
        """
        try:
            if len(smiles_str) > self.limit_length:
                return False

            # 检查是否包含同位素标记（数字在[]内）
            if self._contains_isotope(smiles_str):
                return False

            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return False

            # 验证SMILES的一致性
            canonical_smiles = Chem.MolToSmiles(mol, rootedAtAtom=-1)
            mol_tmp = Chem.MolFromSmiles(canonical_smiles)
            canonical_smiles2 = Chem.MolToSmiles(mol_tmp, rootedAtAtom=-1)

            return canonical_smiles == canonical_smiles2

        except Exception:
            return False

    def _contains_isotope(self, smiles_str: str) -> bool:
        """检查SMILES是否包含同位素标记

        Args:
            smiles_str: SMILES字符串

        Returns:
            bool: 是否包含同位素标记
        """
        i = 0
        while i < len(smiles_str):
            if smiles_str[i] == '[' and i + 1 < len(smiles_str):
                if smiles_str[i + 1].isdigit():
                    return True
            i += 1
        return False

    def canonicalize_smiles(self, smiles_str: str) -> Optional[str]:
        """将SMILES字符串规范化

        Args:
            smiles_str: 输入SMILES字符串

        Returns:
            Optional[str]: 规范化的SMILES字符串，失败返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return None

            canonical_smiles = Chem.MolToSmiles(mol, rootedAtAtom=-1)
            return canonical_smiles

        except Exception:
            return None

    def generate_noncanonical_smiles(self, smiles_str: str,
                                     num_variants: int = None) -> List[str]:
        """生成非规范化SMILES变体

        Args:
            smiles_str: 规范化SMILES字符串
            num_variants: 生成变体数量

        Returns:
            List[str]: 非规范化SMILES变体列表
        """
        if num_variants is None:
            num_variants = DEFAULT_DATA_CONFIG.num_tries_noncanonical

        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return []

            # 跳过包含'.'的分子（多组分）
            if '.' in smiles_str:
                return []

            variants = []
            num_atoms = mol.GetNumAtoms()
            atom_indices = list(range(num_atoms))

            for _ in range(num_variants):
                try:
                    # 随机选择根原子
                    root_atom = random.choice(atom_indices)
                    variant_smiles = Chem.MolToSmiles(
                        mol, rootedAtAtom=root_atom)

                    # 确保变体长度不超过限制
                    if len(variant_smiles) <= self.limit_length:
                        variants.append(variant_smiles)
                    else:
                        # 如果变体太长，使用原始SMILES
                        variants.append(smiles_str)

                except Exception:
                    # 如果生成失败，使用原始SMILES
                    variants.append(smiles_str)

            return variants

        except Exception:
            return []

    def process_smiles_file(self, filepath: str, max_count: int = None) -> Tuple[List[str], int]:
        """处理SMILES文件

        Args:
            filepath: SMILES文件路径
            max_count: 最大处理数量

        Returns:
            Tuple[List[str], int]: (有效SMILES列表, 跳过的数量)
        """
        valid_smiles = []
        skipped_count = 0
        processed_count = 0

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_count and processed_count >= max_count:
                        break

                    processed_count += 1
                    if processed_count % 10000 == 0:
                        print(f"已处理: {processed_count}", end=' ')
                    if processed_count % 100000 == 0:
                        print()

                    line = line.strip()
                    if not line:
                        continue

                    # 解析文件格式（假设格式为：ID SMILES 或只有SMILES）
                    parts = line.split()
                    if len(parts) >= 2:
                        smiles_str = parts[1]  # 第二列是SMILES
                    else:
                        smiles_str = parts[0]  # 只有SMILES

                    if self.is_valid_smiles(smiles_str):
                        canonical_smiles = self.canonicalize_smiles(smiles_str)
                        if canonical_smiles:
                            valid_smiles.append(canonical_smiles)
                    else:
                        skipped_count += 1

        except FileNotFoundError:
            print(f"文件未找到: {filepath}")
        except Exception as e:
            print(f"处理文件时出错: {e}")

        return valid_smiles, skipped_count

    def create_training_pairs(self, canonical_smiles_list: List[str]) -> Tuple[List[str], List[str]]:
        """创建训练数据对（规范化 -> 非规范化）

        Args:
            canonical_smiles_list: 规范化SMILES列表

        Returns:
            Tuple[List[str], List[str]]: (输入SMILES列表, 目标SMILES列表)
        """
        input_smiles = []
        target_smiles = []

        for canonical_smiles in canonical_smiles_list:
            if len(canonical_smiles) < 2:  # 跳过太短的SMILES
                continue

            # 生成非规范化变体
            variants = self.generate_noncanonical_smiles(canonical_smiles)

            # 为每个变体创建训练对
            for variant in variants:
                input_smiles.append(canonical_smiles)
                target_smiles.append(variant)

        return input_smiles, target_smiles

    def apply_random_root(self, smiles_str: str) -> str:
        """对SMILES应用随机根原子（数据增强）

        Args:
            smiles_str: 输入SMILES字符串

        Returns:
            str: 随机根化的SMILES字符串
        """
        try:
            # 跳过包含'.'的分子
            if '.' in smiles_str:
                return smiles_str

            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return smiles_str

            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return smiles_str

            # 随机选择根原子
            root_atom = random.choice(range(num_atoms))
            new_smiles = Chem.MolToSmiles(mol, rootedAtAtom=root_atom)

            # 检查长度限制
            if len(new_smiles) <= self.limit_length:
                return new_smiles
            else:
                return smiles_str

        except Exception:
            return smiles_str

    def get_statistics(self, smiles_list: List[str]) -> dict:
        """获取SMILES列表的统计信息

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            dict: 统计信息
        """
        if not smiles_list:
            return {}

        lengths = [len(smiles) for smiles in smiles_list]

        return {
            'total_count': len(smiles_list),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'unique_count': len(set(smiles_list))
        }

    def filter_by_length(self, smiles_list: List[str],
                         min_length: int = 2,
                         max_length: int = None) -> List[str]:
        """按长度过滤SMILES列表

        Args:
            smiles_list: SMILES字符串列表
            min_length: 最小长度
            max_length: 最大长度

        Returns:
            List[str]: 过滤后的SMILES列表
        """
        if max_length is None:
            max_length = self.limit_length

        return [smiles for smiles in smiles_list
                if min_length <= len(smiles) <= max_length]

    def remove_duplicates(self, smiles_list: List[str]) -> List[str]:
        """去除重复的SMILES

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            List[str]: 去重后的SMILES列表
        """
        return list(set(smiles_list))
