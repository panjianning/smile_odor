"""使用示例

该脚本演示如何使用分子性质预测系统进行完整的训练和推理流程。
"""

from src.training.property_trainer import PropertyTrainer
from src.training.pretrain_trainer import PretrainTrainer
from src.data.dataset_builder import DatasetBuilder
from config.data_config import DEFAULT_DATA_CONFIG
from config.model_config import PretrainConfig, PropertyPredConfig
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def create_sample_data():
    """创建示例数据文件"""
    print("创建示例数据...")

    # 创建示例SMILES数据
    sample_smiles = [
        "CCO",  # 乙醇
        "CC(C)O",  # 异丙醇
        "c1ccccc1",  # 苯
        "CCc1ccccc1",  # 乙苯
        "CC(C)(C)O",  # 叔丁醇
        "CCCCCCCC",  # 辛烷
        "c1ccc(cc1)O",  # 苯酚
        "CC(=O)O",  # 乙酸
        "CCN(CC)CC",  # 三乙胺
        "c1ccc2c(c1)ccccc2"  # 萘
    ]

    # 保存SMILES文件
    os.makedirs("data", exist_ok=True)
    with open("data/sample_smiles.txt", "w") as f:
        for smiles in sample_smiles:
            f.write(f"{smiles}\n")

    # 创建气味数据示例
    odorant_data = [
        "1 CCO alcoholic sweet",
        "2 CC(C)O alcoholic",
        "3 c1ccccc1 aromatic benzene",
        "4 CCc1ccccc1 aromatic sweet",
        "5 c1ccc(cc1)O phenolic medicinal"
    ]

    odorless_data = [
        "1 CCCCCCCC odorless",
        "2 CC(C)(C)O odorless",
        "3 CC(=O)O odorless",
        "4 CCN(CC)CC odorless",
        "5 c1ccc2c(c1)ccccc2 odorless"
    ]

    with open("data/odorant.txt", "w") as f:
        for line in odorant_data:
            f.write(f"{line}\n")

    with open("data/odorless.txt", "w") as f:
        for line in odorless_data:
            f.write(f"{line}\n")

    print("示例数据创建完成!")


def example_pretrain():
    """预训练示例"""
    print("\n" + "="*60)
    print("预训练示例")
    print("="*60)

    # 1. 构建预训练数据
    print("1. 构建预训练数据...")
    builder = DatasetBuilder(DEFAULT_DATA_CONFIG)

    # 构建数据集
    smiles_list, symbol_dict = builder.build_pretrain_dataset(
        ["data/sample_smiles.txt"], max_samples=None
    )

    # print(f"smiles_list = {smiles_list}")
    # print(f"symbol_dict = {symbol_dict}")

    # 创建预训练数据对
    canonical_smiles, target_smiles = builder.create_pretrain_pairs(
        smiles_list)

    # print(f"canonical_smiles = {canonical_smiles}")
    # print(f"target_smiles = {target_smiles}")

    # 保存数据
    os.makedirs("data/pretrain", exist_ok=True)
    builder.save_pretrain_data(
        smiles_list, canonical_smiles, target_smiles, "data/pretrain"
    )

    print(f"预训练数据构建完成!")
    print(f"- 样本数量: {len(smiles_list)}")
    print(f"- 词汇表大小: {symbol_dict.vocab_size}")

    # 2. 预训练模型
    print("\n2. 开始预训练...")
    config = PretrainConfig()
    # 为了演示，使用较小的配置
    config.num_epoch = 5
    config.batch_size = 4

    trainer = PretrainTrainer(
        config=config,
        data_dir="data/pretrain",
        save_dir="models/pretrain_demo",
        device="cpu"  # 使用CPU进行演示
    )

    trainer.train()
    print("预训练完成!")


def example_finetune():
    """微调示例"""
    print("\n" + "="*60)
    print("微调示例")
    print("="*60)

    # 配置
    config = PropertyPredConfig()
    config.num_epoch = 3
    config.batch_size = 2
    config.early_stopping_patience = 5

    # 数据配置
    data_config = {
        'odorant_file': 'data/odorant.txt',
        'odorless_file': 'data/odorless.txt'
    }

    # 创建训练器
    trainer = PropertyTrainer(
        config=config,
        data_config=data_config,
        save_dir="models/odor_demo",
        device="cpu"
    )

    # 开始训练
    test_metrics = trainer.train(
        task_name="odor",
        pretrain_model_path="models/pretrain_demo/best_model.pt"
    )

    print(f"微调完成!")
    print(f"测试集结果: {test_metrics}")


def example_inference():
    """推理示例"""
    print("\n" + "="*60)
    print("推理示例")
    print("="*60)

    # 这里可以添加推理代码
    # 加载训练好的模型，对新的SMILES进行预测
    print("推理功能演示...")

    # 示例：加载模型并预测
    test_smiles = ["CCO", "c1ccccc1", "CCCCCCCC"]

    print("测试SMILES:")
    for i, smiles in enumerate(test_smiles):
        print(f"  {i+1}. {smiles}")

    print("预测结果:")
    print("  (实际推理代码需要加载训练好的模型)")
    print("  1. CCO -> 有气味 (概率: 0.85)")
    print("  2. c1ccccc1 -> 有气味 (概率: 0.92)")
    print("  3. CCCCCCCC -> 无气味 (概率: 0.78)")


def main():
    """主函数"""
    print("分子性质预测系统使用示例")
    print("="*60)

    # 创建示例数据
    create_sample_data()

    # 预训练示例
    example_pretrain()

    # 微调示例
    example_finetune()

    # 推理示例
    example_inference()

    print("\n" + "="*60)
    print("示例运行完成!")
    print("="*60)

    print("\n生成的文件:")
    print("- data/sample_smiles.txt: 示例SMILES数据")
    print("- data/odorant.txt: 有气味分子数据")
    print("- data/odorless.txt: 无气味分子数据")
    print("- data/pretrain/: 预训练数据")
    print("- models/pretrain_demo/: 预训练模型")
    print("- models/odor_demo/: 微调模型")

    print("\n下一步:")
    print("1. 准备更大规模的SMILES数据集")
    print("2. 调整模型配置参数")
    print("3. 使用GPU进行训练")
    print("4. 评估模型性能")


if __name__ == "__main__":
    main()
