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
    data_config = {
        'odorant_file': args.odorant_file,
        'odorless_file': args.odorless_file,
        'tdc_data_path': args.tdc_data_path
    }

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

    # 这里可以添加推理逻辑
    # 加载模型，处理输入SMILES，输出预测结果
    print("推理功能待实现...")


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
                                 help='任务名称 (odor, BBBP, ClinTox, 等)')
    finetune_parser.add_argument('--pretrain_model', type=str, default=None,
                                 help='预训练模型路径')
    finetune_parser.add_argument('--save_dir', type=str, required=True,
                                 help='模型保存目录')
    finetune_parser.add_argument('--odorant_file', type=str, default=None,
                                 help='有气味分子文件路径')
    finetune_parser.add_argument('--odorless_file', type=str, default=None,
                                 help='无气味分子文件路径')
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
