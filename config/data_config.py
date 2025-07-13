"""数据配置文件"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """数据处理配置"""
    # SMILES处理参数
    limit_smiles_length: int = 100      # SMILES最大长度
    num_tries_noncanonical: int = 5     # 生成非规范SMILES的尝试次数

    # 数据文件路径
    chembl_smiles_file: str = 'all_data/webScrapping/chembl_smi.txt'
    tgsc_odorant_file: str = 'webScrapping/tgsc_odorant_1020.txt'
    tgsc_odorless_file: str = 'webScrapping/tgsc_odorless_1020.txt'

    # 输出文件路径
    output_dir: str = 'CHEMBL'
    pretrain_smiles_file: str = 'OdorCode-40 Pretrain_SMILES_STR_LIST'
    symbol_dict_file: str = 'OdorCode-40 Symbol Dictionary'
    pretrain_data_file: str = 'OdorCode-40 Pretrain MLM_data'

    # 数据量控制
    num_nbdc_smiles: int = 100000       # 使用的SMILES数量
    limit_freq_od: int = 49             # 气味描述符最小频次


@dataclass
class TDCConfig:
    """TDC数据集配置"""
    # TDC数据路径
    tdc_data_path: str = '/tf/haha/all_data/TDC/admet_group/'

    # 回归任务列表
    regression_tasks: List[str] = None

    # 分类任务列表
    classification_tasks: List[str] = None

    def __post_init__(self):
        if self.regression_tasks is None:
            self.regression_tasks = [
                'caco2_wang',
                'lipophilicity_astrazeneca',
                'solubility_aqsoldb',
                'ppbr_az',
                'vdss_lombardo',
                'half_life_obach',
                'clearance_hepatocyte_az',
                'clearance_microsome_az',
                'ld50_zhu',
            ]

        if self.classification_tasks is None:
            self.classification_tasks = [
                'hia_hou',
                'pgp_broccatelli',
                'bioavailability_ma',
                'bbb_martins',
                'cyp2d6_veith',
                'cyp3a4_veith',
                'cyp2c9_veith',
                'cyp2c9_substrate_carbonmangels',
                'cyp2d6_substrate_carbonmangels',
                'cyp3a4_substrate_carbonmangels',
                'herg',
                'ames',
                'dili',
            ]

    @property
    def all_tasks(self) -> List[str]:
        """获取所有任务列表"""
        return self.regression_tasks + self.classification_tasks

    def is_regression_task(self, task_name: str) -> bool:
        """判断是否为回归任务"""
        return task_name in self.regression_tasks


# 默认配置实例
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_TDC_CONFIG = TDCConfig()
