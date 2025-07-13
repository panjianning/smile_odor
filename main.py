"""主运行脚本

该脚本提供了完整的训练和推理流程。
支持的功能：
1. 预训练数据构建
2. 模型预训练
3. 下游任务微调
4. 模型推理和评估
"""

from src.training.property_trainer import PropertyTrainer
from src.training.pretrain_trainer import PretrainTrainer
from src.data.dataset_builder import DatasetBuilder
from config.data_config import DEFAULT_DATA_CONFIG
from config.model_config import PretrainConfig, PropertyPredConfig
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def build_pretrain_data(args):
    """构建预训练数据"""
    print("=" * 60)
    print("构建预训练数据")
    print("=" * 60)

    # 初始化数据集构建器
    builder = DatasetBuilder(DEFAULT_DATA_CONFIG)

    # 构建预训练数据集
    smiles_files = [args.smiles_file] if args.smiles_file else []
    if not smiles_files:
        print("错误: 请提供SMILES文件路径")
        return

    # 构建数据集
    smiles_list, symbol_dict = builder.build_pretrain_dataset(
        smiles_files, args.max_samples
    )

    # 创建预训练数据对
    canonical_smiles, target_smiles = builder.create_pretrain_pairs(
        smiles_list)

    # 保存数据
    builder.save_pretrain_data(
        smiles_list, canonical_smiles, target_smiles, args.output_dir
    )

    print(f"\n预训练数据构建完成!")
    print(f"输出目录: {args.output_dir}")


def run_pretrain(args):
    """运行预训练"""
    print("=" * 60)
    print("开始预训练")
    print("=" * 60)

    # 创建配置
    config = PretrainConfig()
    if args.config_file:
        # 这里可以添加从文件加载配置的逻辑
        pass

    # 创建训练器
    trainer = PretrainTrainer(
        config=config,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=args.device
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"\n预训练完成!")
    print(f"模型保存目录: {args.save_dir}")


def run_finetune(args):
    """运行微调"""
    print("=" * 60)
    print(f"开始微调任务: {args.task}")
    print("=" * 60)

    # 创建配置
    config = PropertyPredConfig()
    if args.config_file:
        # 这里可以添加从文件加载配置的逻辑
        pass

    # 数据配置
    data_config = {}

    # 根据任务类型配置数据
    if args.task == 'multi_label_odor':
        # 多标签CSV任务
        if not args.csv_file:
            print("错误: multi_label_odor任务需要提供--csv_file参数")
            return

        data_config.update({
            'csv_file': args.csv_file,
            'smiles_column': args.smiles_column,
            'descriptor_column': 'descriptors',
            'min_label_frequency': args.min_label_frequency,
            'max_smiles_length': args.max_smiles_length
        })

        print(f"多标签CSV配置:")
        print(f"  CSV文件: {args.csv_file}")
        print(f"  SMILES列: {args.smiles_column}")
        print(f"  最小标签频次: {args.min_label_frequency}")
        print(f"  最大SMILES长度: {args.max_smiles_length}")

    elif args.task == 'odor':
        # 传统odor任务
        if not args.odorant_file or not args.odorless_file:
            print("错误: odor任务需要提供--odorant_file和--odorless_file参数")
            return

        data_config.update({
            'odorant_file': args.odorant_file,
            'odorless_file': args.odorless_file
        })

        print(f"传统odor配置:")
        print(f"  有气味文件: {args.odorant_file}")
        print(f"  无气味文件: {args.odorless_file}")

    else:
        # TDC任务或其他
        data_config.update({
            'tdc_data_path': args.tdc_data_path
        })

        print(f"TDC任务配置:")
        print(f"  TDC数据路径: {args.tdc_data_path}")

    # 创建训练器
    trainer = PropertyTrainer(
        config=config,
        data_config=data_config,
        save_dir=args.save_dir,
        device=args.device
    )

    # 开始训练
    test_metrics = trainer.train(
        task_name=args.task,
        pretrain_model_path=args.pretrain_model,
        resume_from_checkpoint=args.resume
    )

    print(f"\n微调完成!")
    print(f"模型保存目录: {args.save_dir}")
    print(f"测试集结果: {test_metrics}")


def run_inference(args):
    """运行推理"""
    print("=" * 60)
    print("运行推理")
    print("=" * 60)

    from src.inference.inference_engine import InferenceEngine

    try:
        # 初始化推理引擎
        print(f"初始化推理引擎...")
        engine = InferenceEngine(args.model_path, args.device)

        # 打印模型信息
        model_info = engine.get_model_info()
        print(f"\n模型信息:")
        for key, value in model_info.items():
            if key == 'labels' and len(value) > 10:
                print(f"  {key}: {len(value)} 个标签 (前10个: {value[:10]})")
            else:
                print(f"  {key}: {value}")

        # 执行推理
        if args.input_file and args.output_file:
            # 从文件批量推理
            print(f"\n开始批量推理...")
            engine.predict_from_file(
                args.input_file,
                args.output_file,
                batch_size=getattr(args, 'batch_size', 32),
                probability_threshold=getattr(args, 'threshold', 0.5)
            )
        elif hasattr(args, 'smiles') and args.smiles:
            # 单个SMILES推理
            print(f"\n对单个SMILES进行推理: {args.smiles}")
            result = engine.predict_single(args.smiles)

            print(f"\n推理结果:")
            print(f"  SMILES: {result['smiles']}")
            print(f"  有效: {result['valid']}")

            if result['valid'] and 'predicted_labels' in result:
                print(f"  预测标签:")
                for label, info in result['predicted_labels'].items():
                    if isinstance(info, dict):
                        pred = info['predicted']
                        prob = info['probability']
                        print(
                            f"    {label}: {'是' if pred else '否'} (概率: {prob:.3f})")
                    else:
                        print(f"    {label}: {'是' if info else '否'}")
        else:
            print("错误: 请提供输入文件和输出文件，或者单个SMILES字符串")

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分子性质预测系统")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 构建预训练数据命令
    build_parser = subparsers.add_parser('build_data', help='构建预训练数据')
    build_parser.add_argument('--smiles_file', type=str, required=True,
                              help='SMILES文件路径')
    build_parser.add_argument('--output_dir', type=str, required=True,
                              help='输出目录')
    build_parser.add_argument('--max_samples', type=int, default=None,
                              help='最大样本数量')

    # 预训练命令
    pretrain_parser = subparsers.add_parser('pretrain', help='运行预训练')
    pretrain_parser.add_argument('--data_dir', type=str, required=True,
                                 help='预训练数据目录')
    pretrain_parser.add_argument('--save_dir', type=str, required=True,
                                 help='模型保存目录')
    pretrain_parser.add_argument('--config_file', type=str, default=None,
                                 help='配置文件路径')
    pretrain_parser.add_argument('--device', type=str, default='cuda',
                                 help='训练设备')
    pretrain_parser.add_argument('--resume', action='store_true',
                                 help='从检查点恢复训练')

    # 微调命令
    finetune_parser = subparsers.add_parser('finetune', help='运行微调')
    finetune_parser.add_argument('--task', type=str, required=True,
                                 help='任务名称 (odor, multi_label_odor, BBBP, ClinTox, 等)')
    finetune_parser.add_argument('--pretrain_model', type=str, default=None,
                                 help='预训练模型路径')
    finetune_parser.add_argument('--save_dir', type=str, required=True,
                                 help='模型保存目录')

    # 传统odor任务参数
    finetune_parser.add_argument('--odorant_file', type=str, default=None,
                                 help='有气味分子文件路径 (用于odor任务)')
    finetune_parser.add_argument('--odorless_file', type=str, default=None,
                                 help='无气味分子文件路径 (用于odor任务)')

    # 多标签CSV任务参数
    finetune_parser.add_argument('--csv_file', type=str, default=None,
                                 help='多标签CSV文件路径 (用于multi_label_odor任务)')
    finetune_parser.add_argument('--smiles_column', type=str, default='nonStereoSMILES',
                                 help='SMILES列名')
    finetune_parser.add_argument('--min_label_frequency', type=int, default=5,
                                 help='最小标签频次')
    finetune_parser.add_argument('--max_smiles_length', type=int, default=200,
                                 help='最大SMILES长度')

    # 其他参数
    finetune_parser.add_argument('--tdc_data_path', type=str, default=None,
                                 help='TDC数据路径')
    finetune_parser.add_argument('--config_file', type=str, default=None,
                                 help='配置文件路径')
    finetune_parser.add_argument('--device', type=str, default='cuda',
                                 help='训练设备')
    finetune_parser.add_argument('--resume', action='store_true',
                                 help='从检查点恢复训练')

    # 推理命令
    inference_parser = subparsers.add_parser('inference', help='运行推理')
    inference_parser.add_argument('--model_path', type=str, required=True,
                                  help='模型路径')
    inference_parser.add_argument('--input_file', type=str, required=True,
                                  help='输入SMILES文件')
    inference_parser.add_argument('--output_file', type=str, required=True,
                                  help='输出结果文件')
    inference_parser.add_argument('--device', type=str, default='cuda',
                                  help='推理设备')

    args = parser.parse_args()

    if args.command == 'build_data':
        build_pretrain_data(args)
    elif args.command == 'pretrain':
        run_pretrain(args)
    elif args.command == 'finetune':
        run_finetune(args)
    elif args.command == 'inference':
        run_inference(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
