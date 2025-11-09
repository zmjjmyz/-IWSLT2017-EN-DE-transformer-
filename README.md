# Transformer 架构实现

本仓库包含一个用于在 IWSLT2017 数据集（英语到德语）上进行机器翻译的 Transformer 模型的实现。

## 功能

- Transformer 架构（编码器 + 解码器）
- 支持 IWSLT2017 数据集（EN↔DE）
- 训练稳定性技术：AdamW 优化器、学习率调度、梯度裁剪
- 模型保存/加载功能
- 训练曲线可视化
- 模型大小对比的消融研究


## 安装

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/w18650/LLM-Mid.git
    cd LLM-Mid-main
    ```

2.  **创建 Python 环境（推荐）：**
    ```bash
    conda create -n transformer_env python=3.10
    conda activate transformer_env
    ```

3.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```

## 训练

```bash
bash scripts/run.sh
```

此脚本将执行以下操作：训练大 Transformer 模型（默认参数为 d_model=512、nhead=8、num_encoder_layers=6、num_decoder_layers=6、dim_feedforward=2048），训练小 Transformer 模型（参数为 d_model=256、nhead=4、num_encoder_layers=2、num_decoder_layers=2、dim_feedforward=512），并在./results/ 目录下生成相应的对比图。

## 项目结构

```
├── src/
│   ├── transformer.py     
│   ├── dataloader.py     
│   ├── trainer.py              
│   ├── main.py                
│   └── plot_results.py         
├── scripts/
│   └── run.sh                 
├── results/                   
├── requirements.txt           
└── README.md                   
```

## 结果

运行 `scripts/run.sh` 后，以下内容将保存在 `results` 目录中：
- `transformer_large_checkpoint.pth`: 训练好的大模型检查点
- `transformer_large_training_curves.png`: 大模型的训练/验证曲线
- `transformer_large_training_history.csv`: 大模型的训练历史
- `transformer_small_checkpoint.pth`: 训练好的小模型检查点
- `transformer_small_training_curves.png`: 小模型的训练/验证曲线
- `transformer_small_training_history.csv`: 小模型的训练历史
- `size_ablation_loss_comparison.png`: 大小模型损失的对比图。
