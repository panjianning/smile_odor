"""多标签气味预测微调示例

演示如何使用main.py的finetune命令进行多标签气味预测模型训练
"""

import subprocess
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_finetune_example():
    """运行微调示例"""
    print("=" * 80)
    print("多标签气味预测微调示例")
    print("=" * 80)

    # 检查数据文件是否存在
    csv_file = "data/Multi-Labelled_Smiles_Odors_dataset_example.csv"
    if not os.path.exists(csv_file):
        print(f"错误: 数据文件不存在: {csv_file}")
        print("请确保数据文件在正确的位置")
        return

    # 构建命令
    cmd = [
        "python", "main.py", "finetune",
        "--task", "multi_label_odor",
        "--csv_file", csv_file,
        "--save_dir", "models/multi_label_odor_finetune",
        "--smiles_column", "nonStereoSMILES",
        "--min_label_frequency", "3",  # 降低频次阈值以保留更多标签
        "--max_smiles_length", "200",
        "--device", "cuda"
    ]

    # 可选：添加预训练模型
    pretrain_model = "models/pretrain/best_model.pt"
    if os.path.exists(pretrain_model):
        cmd.extend(["--pretrain_model", pretrain_model])
        print(f"使用预训练模型: {pretrain_model}")
    else:
        print("未找到预训练模型，将从头开始训练")

    print(f"\n执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)

    try:
        # 执行命令
        result = subprocess.run(cmd, check=True, capture_output=False)

        print("\n" + "=" * 80)
        print("微调完成!")
        print(f"模型保存在: models/multi_label_odor_finetune")

        # 检查生成的文件
        save_dir = "models/multi_label_odor_finetune"
        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            print(f"\n生成的文件:")
            for file in files:
                print(f"  {file}")

    except subprocess.CalledProcessError as e:
        print(f"微调过程中出现错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n用户中断训练")
        return False

    return True


def run_inference_example():
    """运行推理示例"""
    print("\n" + "=" * 80)
    print("推理示例")
    print("=" * 80)

    model_path = "models/multi_label_odor_finetune/best_model.pt"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先完成微调训练")
        return False

    # 创建示例输入文件
    input_file = "examples/sample_input.csv"
    output_file = "examples/sample_output.csv"

    # 创建示例SMILES
    sample_smiles = [
        "CCO",  # 乙醇
        "CC(C)CO",  # 异丙醇
        "C1=CC=C(C=C1)CO",  # 苯甲醇
        "CC(=O)OCC",  # 乙酸乙酯
        "C1=CC=CC=C1"  # 苯
    ]

    import pandas as pd
    df = pd.DataFrame({"SMILES": sample_smiles})
    df.to_csv(input_file, index=False)
    print(f"创建示例输入文件: {input_file}")

    # 构建推理命令
    cmd = [
        "python", "main.py", "inference",
        "--model_path", model_path,
        "--input_file", input_file,
        "--output_file", output_file,
        "--device", "cuda"
    ]

    print(f"\n执行推理命令:")
    print(" ".join(cmd))

    try:
        # 执行推理
        result = subprocess.run(cmd, check=True, capture_output=False)

        print(f"\n推理完成!")
        print(f"结果保存在: {output_file}")

        # 显示结果
        if os.path.exists(output_file):
            results_df = pd.read_csv(output_file)
            print(f"\n推理结果预览:")
            print(results_df.head())

    except subprocess.CalledProcessError as e:
        print(f"推理过程中出现错误: {e}")
        return False

    return True


def main():
    """主函数"""
    print("多标签气味预测完整流程示例")

    # 步骤1: 微调训练
    print("\n步骤1: 微调训练")
    success = run_finetune_example()

    if success:
        # 步骤2: 推理演示
        print("\n步骤2: 推理演示")
        run_inference_example()

    print("\n示例完成!")


if __name__ == "__main__":
    main()
