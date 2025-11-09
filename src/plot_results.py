# src/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():

    large_history_file = "./results/transformer_large_training_history.csv"
    small_history_file = "./results/transformer_small_training_history.csv"
    
    # 检查文件是否存在
    if not os.path.exists(large_history_file):
        print(f"Error: {large_history_file} not found.")
        return
    if not os.path.exists(small_history_file):
        print(f"Error: {small_history_file} not found.")
        return

    # 读取数据
    large_df = pd.read_csv(large_history_file)
    small_df = pd.read_csv(small_history_file)

    if large_df['epoch'].iloc[0] == 0:
        large_df['epoch'] += 1
    if small_df['epoch'].iloc[0] == 0:
        small_df['epoch'] += 1

    # 创建单个图
    plt.figure(figsize=(10, 7))

    plt.plot(large_df['epoch'], large_df['train_loss'], label='Transformer (Large) - Train', color='blue', linestyle='-')
    plt.plot(large_df['epoch'], large_df['val_loss'], label='Transformer (Large) - Val', color='blue', linestyle='--')

    plt.plot(small_df['epoch'], small_df['train_loss'], label='Transformer (Small) - Train', color='red', linestyle='-')
    plt.plot(small_df['epoch'], small_df['val_loss'], label='Transformer (Small) - Val', color='red', linestyle='--')

    plt.title('Training and Validation Loss Comparison (Size Ablation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('./results/size_ablation_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close() # 关闭图形以释放内存

    print("Size ablation comparison plot saved to ./results/size_ablation_loss_comparison.png")

if __name__ == "__main__":
    plot_comparison()