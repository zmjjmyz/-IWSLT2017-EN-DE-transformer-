import torch

# 加载checkpoint（确保路径正确）
checkpoint_path = "/root/autodl-tmp/LLM-Mid-main/results/transformer_large_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)

# 1. 打印训练/验证损失（简单直观）
print("===== 训练损失（train_losses） =====")
print("类型：", type(checkpoint['train_losses']))  # 通常是列表
print("长度（训练步数/轮数）：", len(checkpoint['train_losses']))
print("前5个损失值：", checkpoint['train_losses'][:5])  # 打印前5个
print("\n===== 验证损失（val_losses） =====")
print("类型：", type(checkpoint['val_losses']))
print("长度（验证次数）：", len(checkpoint['val_losses']))
print("前5个损失值：", checkpoint['val_losses'][:5])


# 2. 打印模型参数（model_state_dict）的部分内容（全部打印会非常长）
print("\n===== 模型参数（model_state_dict） =====")
print("类型：", type(checkpoint['model_state_dict']))  # 通常是OrderedDict或dict
print("包含的层参数键名（前10个）：", list(checkpoint['model_state_dict'].keys())[:10])  # 查看有哪些层
# 打印某一层的具体参数（比如第一个层的权重）
first_key = list(checkpoint['model_state_dict'].keys())[0]
print(f"第一个参数（{first_key}）的值（部分）：\n", checkpoint['model_state_dict'][first_key][:2, :2])  # 只打印前2x2的张量


# 3. 打印优化器状态（optimizer_state_dict）的部分内容
print("\n===== 优化器状态（optimizer_state_dict） =====")
print("包含的键：", checkpoint['optimizer_state_dict'].keys())  # 通常有param_groups、state等
# 打印学习率参数组（param_groups）
print("优化器参数组（学习率等）：", checkpoint['optimizer_state_dict']['param_groups'])
# 打印第一个参数的优化器状态（如动量）
if 'state' in checkpoint['optimizer_state_dict'] and checkpoint['optimizer_state_dict']['state']:
    first_param_id = next(iter(checkpoint['optimizer_state_dict']['state'].keys()))
    print(f"第一个参数的优化器状态：", checkpoint['optimizer_state_dict']['state'][first_param_id])


# 4. 打印学习率调度器状态（scheduler_state_dict）
print("\n===== 调度器状态（scheduler_state_dict） =====")
print("调度器状态内容：", checkpoint['scheduler_state_dict'])  # 通常包含last_epoch等信息