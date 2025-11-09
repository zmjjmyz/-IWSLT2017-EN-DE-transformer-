import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from transformer import generate_square_subsequent_mask

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab, 
                 device, d_model=512, lr=0.0003, warmup_steps=2000):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        
        self.scheduler = LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: self._lr_scale(step)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['<pad>'], reduction='sum')
        self.train_losses = []
        self.val_losses = []
    
    def _lr_scale(self, step):
        """Learning rate scaling function with warmup."""
        step = max(step, 0)
        step += 1 

        return min(step ** -0.5, step * (self.warmup_steps ** -1.5))
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc="Training")):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]  
            tgt_output = tgt[:, 1:]  
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            src_mask = (src != self.src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(self.device)
            
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            
            non_pad_mask = (tgt_output != self.tgt_vocab['<pad>'])
            num_tokens = non_pad_mask.sum().item()
            

            if torch.isnan(loss):
                print(f"NaN found in loss at batch {batch_idx}!")
                print("Output sample:", output.view(-1, output.size(-1))[:10, :10]) 
                print("Target sample:", tgt_output.contiguous().view(-1)[:10]) 
                continue 
            
            self.optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"NaN/Inf gradient norm found at batch {batch_idx}!")
                continue 
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_tokens += num_tokens
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = float('inf') 
        
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Validating"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
                src_mask = (src != self.src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(self.device)
                
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                
                if torch.isnan(loss):
                    print("NaN found in validation loss!")
                    continue
                
                non_pad_mask = (tgt_output != self.tgt_vocab['<pad>'])
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item()
                total_tokens += num_tokens
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = float('inf') 
        
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def plot_training_curves(self, save_path):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        
        ax.plot(self.train_losses, label='Train Loss', color='blue')
        ax.plot(self.val_losses, label='Validation Loss', color='red')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)