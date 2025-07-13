from src.inference.inference_engine import InferenceEngine
import numpy as np
from src.data.dataset_builder import DatasetBuilder


dataset_builder = DatasetBuilder()
_, _, task_info = dataset_builder.build_multi_label_odor_dataset(
    './data/Multi-Labelled_Smiles_Odors_dataset.csv')

label_to_id = task_info['label_to_id']

# 初始化推理引擎
engine = InferenceEngine(
    model_path="models/multi_label_odor/best_model.pt", device="cuda")

# 单个预测
result = engine.predict_single("OCc1ccc(O)cc1")

print(np.max(result['probabilities']))
print(np.argmax(result['probabilities']))

# 真实标签是fishy
print(result['probabilities'][label_to_id['fruity']])
