import os

os.environ['HF_HOME'] = '../Huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "30"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import torch
import random
import numpy as np
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# 导入 dataset 和模型模块
from data.translation_datasets import Seq2SeqDataModule
from models.model import TransformerMT

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer Machine Translation Model')

    # 数据参数
    parser.add_argument('--dataset_name', type=str, default='iwslt2017', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='iwslt2017-en-de', help='Dataset configuration')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
    parser.add_argument('--model_name', type=str, default='Helsinki-NLP/opus-mt-en-de', help='Tokenizer model name')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--limit_train_samples', type=int, default=0, help='Limit training samples (0 means no limit)')

    # 模型参数
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ffn', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout_emb', type=float, default=0.1, help='Embedding dropout')
    parser.add_argument('--dropout_atten', type=float, default=0.1, help='Attention dropout')
    parser.add_argument('--dropout_ffn', type=float, default=0.1, help='Feedforward dropout')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    # 路径参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--result_dir', type=str, default='../results', help='Result save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch["src_input"].to(device)
        tgt = batch["tgt_input"].to(device)

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        optimizer.zero_grad()
        outputs = model(src, decoder_input)
        logits = outputs.reshape(-1, outputs.size(-1))
        labels = labels.reshape(-1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        src = batch["src_input"].to(device)
        tgt = batch["tgt_input"].to(device)

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        outputs = model(src, decoder_input)
        logits = outputs.reshape(-1, outputs.size(-1))
        labels = labels.reshape(-1)

        loss = criterion(logits, labels)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def plot_loss_curve(train_losses, val_losses, save_dir):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s', linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curve_0.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")

    plt.close()


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, best_val_loss, path):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model_config': {
            'num_layers': model.num_layers,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_ffn': model.d_ffn,
            'max_seq_len': model.max_seq_len,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch + 1}")
    return epoch, train_losses, val_losses, best_val_loss


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model_save_path = os.path.join(args.save_dir, "best_checkpoint.pt")
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")

    # 加载数据
    data_module = Seq2SeqDataModule(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        limit_train_samples=args.limit_train_samples,
    )
    train_loader, val_loader, _, tokenizer = data_module.prepare_dataloaders()

    # 确保有 bos/eos
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = '<s>'
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = '</s>'

    # 初始化模型
    model = TransformerMT(
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ffn=args.d_ffn,
        max_seq_len=args.max_len,
        dropout_emb=args.dropout_emb,
        dropout_atten=args.dropout_atten,
        dropout_ffn=args.dropout_ffn
    ).to(device)

    # 打印模型参数数量
    model.print_parameters()

    # 生成所需 token id
    model.tgt_bos_id = tokenizer.bos_token_id
    model.tgt_eos_id = tokenizer.eos_token_id

    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 训练循环变量初始化
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 断点续训
    if args.resume and os.path.exists(args.resume):
        model, train_state = TransformerMT.load_model(args.resume, tokenizer, tokenizer, optimizer, device)
        start_epoch = train_state.get('epoch', 0) + 1  # 从下一轮开始
        train_losses = train_state.get('train_losses', [])
        val_losses = train_state.get('val_losses', [])
        best_val_loss = train_state.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最佳模型（只保存模型权重）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(model_save_path)  # 不保存训练状态
            print(f"Best model saved to {model_save_path}")

        # 保存检查点（保存完整训练状态）
        model.save_model(checkpoint_path, optimizer, epoch, train_losses, val_losses, best_val_loss, save_train_state=True)
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, args.result_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()