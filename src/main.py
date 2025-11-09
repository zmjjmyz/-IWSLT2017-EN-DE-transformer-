import torch
import argparse
from transformer import Transformer
from dataloader import get_dataloaders
from trainer import Trainer, count_parameters
import os

def main():
    parser = argparse.ArgumentParser(description='Transformer for IWSLT2017')
    parser.add_argument('--data_path', type=str, default='./data/2017-01-trnted/texts/en/de',
                        help='Path to the IWSLT2017 data (directory containing en-de.zip)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='Model dimension for the standard model')
    parser.add_argument('--nhead', type=int, default=8, 
                        help='Number of attention heads for the standard model')
    parser.add_argument('--num_encoder_layers', type=int, default=6, 
                        help='Number of encoder layers for the standard model')
    parser.add_argument('--num_decoder_layers', type=int, default=6, 
                        help='Number of decoder layers for the standard model')
    parser.add_argument('--dim_feedforward', type=int, default=2048, 
                        help='Dimension of feedforward network for the standard model')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples to use for training')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_small_model', action='store_true',
                        help='Use the small Transformer model for ablation study')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("Loading data...")
    train_loader, val_loader, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv = get_dataloaders(
        args.data_path, 
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length
    )
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    if args.use_small_model:
        print("Using Small Transformer Model.")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        model_name = "transformer_small"
    else:
        print("Using Standard (Large) Transformer Model.")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        model_name = "transformer_large"
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        d_model=model.d_model,
        lr=args.lr
    )
    
    print("Starting training...")
    trainer.train(epochs=args.epochs)
    
    model_save_path = os.path.join(args.save_path, f'{model_name}_checkpoint.pth')
    curves_save_path = os.path.join(args.save_path, f'{model_name}_training_curves.png')
    history_save_path = os.path.join(args.save_path, f'{model_name}_training_history.csv')
    
    trainer.save_checkpoint(model_save_path)
    trainer.plot_training_curves(curves_save_path)
    
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(trainer.train_losses) + 1),
        'train_loss': trainer.train_losses,
        'val_loss': trainer.val_losses,
    })
    history_df.to_csv(history_save_path, index=False)
    
    print("Training completed!")
    print(f"Results saved to {args.save_path}")

if __name__ == "__main__":
    main()