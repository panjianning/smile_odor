"""多标签气味预测示例

该脚本演示如何使用多标签气味数据集进行训练和推理。
主要功能：
1. 数据集分析和预处理
2. 多标签分类模型训练
3. 模型推理和结果分析
4. 完整的端到端流程
"""

from config.model_config import PropertyPredConfig
from src.inference.inference_engine import InferenceEngine
from src.training.property_trainer import PropertyTrainer
from src.data.dataset_builder import DatasetBuilder
from src.data.csv_processor import CSVProcessor
import os
import sys
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def analyze_dataset(csv_path: str):
    """分析数据集"""
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)

    processor = CSVProcessor()

    # 分析数据集统计信息
    stats = processor.analyze_dataset(csv_path)

    print(f"数据集统计信息:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  总列数: {stats['total_columns']}")
    print(f"  标签数量: {stats['num_labels']}")

    if 'smiles_stats' in stats:
        smiles_stats = stats['smiles_stats']
        print(f"\nSMILES统计:")
        print(f"  最小长度: {smiles_stats['min_length']}")
        print(f"  最大长度: {smiles_stats['max_length']}")
        print(f"  平均长度: {smiles_stats['avg_length']:.1f}")
        print(f"  有效SMILES: {smiles_stats['valid_smiles']}")

    if 'label_stats' in stats:
        print(f"\n标签统计 (前10个):")
        label_stats = stats['label_stats']
        for i, (label, info) in enumerate(list(label_stats.items())[:10]):
            print(
                f"  {label}: {info['positive_samples']} 正样本 ({info['positive_ratio']:.3f})")

    if 'label_cooccurrence' in stats:
        cooc = stats['label_cooccurrence']
        print(f"\n标签共现统计:")
        print(f"  最大共现次数: {cooc['max_cooccurrence']}")
        print(f"  平均每样本标签数: {cooc['avg_labels_per_sample']:.2f}")

    return stats


def train_multi_label_model(csv_path: str, save_dir: str,
                            min_label_frequency: int = 5,
                            device: str = 'cuda'):
    """训练多标签模型"""
    print("=" * 60)
    print("训练多标签气味预测模型")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 配置
    config = PropertyPredConfig()
    config.num_epoch = 20  # 减少epoch数量用于演示
    config.batch_size = 16
    config.learning_rate = 1e-4

    # 数据配置
    data_config = {
        'csv_file': csv_path,
        'smiles_column': 'nonStereoSMILES',
        'descriptor_column': 'descriptors',
        'min_label_frequency': min_label_frequency,
        'max_smiles_length': 200
    }

    # 创建训练器
    trainer = PropertyTrainer(
        config=config,
        data_config=data_config,
        save_dir=save_dir,
        device=device
    )

    # 开始训练
    test_metrics = trainer.train(
        task_name='multi_label_odor',
        pretrain_model_path=None,  # 不使用预训练模型
        resume_from_checkpoint=False
    )

    print(f"\n训练完成!")
    print(f"测试集结果: {test_metrics}")

    return test_metrics


def run_inference_demo(model_path: str, csv_path: str, output_path: str,
                       device: str = 'cuda'):
    """运行推理演示"""
    print("=" * 60)
    print("推理演示")
    print("=" * 60)

    try:
        # 初始化推理引擎
        print("初始化推理引擎...")
        engine = InferenceEngine(model_path, device)

        # 打印模型信息
        model_info = engine.get_model_info()
        print(f"\n模型信息:")
        print(f"  任务类型: {model_info['task_type']}")
        print(f"  标签数量: {model_info['num_labels']}")
        print(f"  词汇表大小: {model_info['vocab_size']}")
        print(f"  设备: {model_info['device']}")

        if len(model_info['labels']) <= 20:
            print(f"  标签: {model_info['labels']}")
        else:
            print(f"  标签 (前10个): {model_info['labels'][:10]}")

        # 读取一些示例SMILES进行单个预测
        print(f"\n读取示例数据...")
        df = pd.read_csv(csv_path)
        sample_smiles = df['nonStereoSMILES'].head(5).tolist()

        print(f"\n单个SMILES推理示例:")
        for i, smiles in enumerate(sample_smiles):
            print(f"\n示例 {i+1}: {smiles}")
            result = engine.predict_single(smiles)

            if result['valid']:
                if 'predicted_labels' in result:
                    print(f"  预测结果:")
                    predicted_count = 0
                    for label, info in result['predicted_labels'].items():
                        if isinstance(info, dict):
                            if info['predicted'] and info['probability'] > 0.5:
                                print(
                                    f"    {label}: 是 (概率: {info['probability']:.3f})")
                                predicted_count += 1
                        elif info:
                            print(f"    {label}: 是")
                            predicted_count += 1

                    if predicted_count == 0:
                        print(f"    无明显气味特征")
                else:
                    print(f"  预测向量: {result['predictions']}")
            else:
                print(f"  无效SMILES")

        # 批量推理
        print(f"\n开始批量推理...")
        test_smiles_file = csv_path  # 使用原始CSV文件
        engine.predict_from_file(
            test_smiles_file,
            output_path,
            batch_size=32,
            probability_threshold=0.5
        )

        # 分析结果
        print(f"\n分析推理结果...")
        results_df = pd.read_csv(output_path)
        print(f"  处理样本数: {len(results_df)}")
        print(f"  有效样本数: {results_df['Valid'].sum()}")

        # 统计预测的标签分布
        label_columns = [
            col for col in results_df.columns if col.endswith('_predicted')]
        if label_columns:
            print(f"  预测标签统计 (前10个):")
            for col in label_columns[:10]:
                count = results_df[col].sum()
                print(f"    {col.replace('_predicted', '')}: {count} 个样本")

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def create_sample_input_file(csv_path: str, output_path: str, num_samples: int = 100):
    """创建示例输入文件"""
    print(f"创建示例输入文件...")

    # 读取原始数据
    df = pd.read_csv(csv_path)

    # 随机采样
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)

    # 只保留SMILES列
    sample_smiles = sample_df[['nonStereoSMILES']].copy()
    sample_smiles.columns = ['SMILES']

    # 保存
    sample_smiles.to_csv(output_path, index=False)
    print(f"已保存 {len(sample_smiles)} 个示例SMILES到: {output_path}")

    return output_path


def main():
    """主函数 - 完整的端到端流程"""
    print("多标签气味预测完整流程演示")
    print("=" * 80)

    # 文件路径
    csv_path = "data/Multi-Labelled_Smiles_Odors_dataset.csv"
    save_dir = "models/multi_label_odor"
    sample_input_file = "examples/sample_smiles.csv"
    inference_output_file = "examples/inference_results.csv"

    # 检查数据文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 数据文件不存在: {csv_path}")
        print("请确保数据文件在正确的位置")
        return

    try:
        # 步骤1: 分析数据集
        print("\n步骤1: 分析数据集")
        stats = analyze_dataset(csv_path)

        # 步骤2: 训练模型
        print(f"\n步骤2: 训练模型")
        test_metrics = train_multi_label_model(
            csv_path=csv_path,
            save_dir=save_dir,
            min_label_frequency=3,  # 降低频次阈值以保留更多标签
            device='cuda'
        )

        # 步骤3: 创建示例输入文件
        print(f"\n步骤3: 创建示例输入文件")
        create_sample_input_file(csv_path, sample_input_file, num_samples=50)

        # 步骤4: 运行推理
        print(f"\n步骤4: 运行推理")
        model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(model_path):
            run_inference_demo(
                model_path=model_path,
                csv_path=csv_path,
                output_path=inference_output_file,
                device='cuda'
            )
        else:
            print(f"警告: 模型文件不存在: {model_path}")
            print("请先完成训练步骤")

        print(f"\n完整流程演示完成!")
        print(f"模型保存在: {save_dir}")
        print(f"推理结果保存在: {inference_output_file}")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
