import torch
import transformers
import datasets
import numpy as np
import pandas as pd
import matplotlib
import tqdm

print("=" * 50)
print("环境验证结果:")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"Datasets 版本: {datasets.__version__}")
print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Matplotlib 版本: {matplotlib.__version__}")
print(f"TQDM 版本: {tqdm.__version__}")
print("=" * 50)