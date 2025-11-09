import os

os.environ['HF_HOME'] = '../Huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "30"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import torch
from tqdm import tqdm
import evaluate
from models.model import TransformerMT
from data.translation_datasets import Seq2SeqDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Test Transformer Machine Translation Model')

    # 数据参数
    parser.add_argument('--dataset_name', type=str, default='iwslt2017', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='iwslt2017-en-de', help='Dataset configuration')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
    parser.add_argument('--model_name', type=str, default='Helsinki-NLP/opus-mt-en-de', help='Tokenizer model name')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # 模型参数
    parser.add_argument('--model_ckpt', type=str, default='./checkpoints/Transformer-MT.pt',
                        help='Model checkpoint path')

    return parser.parse_args()


def evaluate_bleu(model, dataloader, tokenizer, device):
    """在测试集上计算 BLEU 分数"""
    bleu = evaluate.load("sacrebleu")

    all_preds, all_refs = [], []

    for batch in tqdm(dataloader, desc="Evaluating BLEU"):
        src = batch["src_input"].to(device)
        tgt = batch["tgt_input"].to(device)

        # 使用模型自带的 generate()
        pred_ids = model.generate(src, max_new_tokens=tokenizer.model_max_length)
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(tgt, skip_special_tokens=True)

        for p, r in zip(preds, refs):
            all_preds.append(p)
            all_refs.append([r])  # sacrebleu 需要二维列表

    bleu_result = bleu.compute(predictions=all_preds, references=all_refs)
    print(f"BLEU score: {bleu_result['score']:.2f}")
    return bleu_result['score']


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data_module = Seq2SeqDataModule(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        limit_train_samples=0,
    )
    _, _, test_loader, tokenizer = data_module.prepare_dataloaders()

    # 确保有 bos/eos
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = '<s>'
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = '</s>'

    # 加载模型
    model, _ = TransformerMT.load_model(args.model_ckpt, src_tokenizer=tokenizer, tgt_tokenizer=tokenizer)
    model.to(device)
    model.eval()

    # 打印模型参数数量
    model.print_parameters()

    # 生成所需 token id
    model.tgt_bos_id = tokenizer.bos_token_id
    model.tgt_eos_id = tokenizer.eos_token_id

    # BLEU 评估
    bleu = evaluate_bleu(model, test_loader, tokenizer, device)

    # 展示样例
    batch = next(iter(test_loader))
    src = batch["src_input"].to(device)
    pred_ids = model.generate(src, max_new_tokens=tokenizer.model_max_length)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(batch["tgt_input"], skip_special_tokens=True)

    print("\nExample translations:")
    for i in range(3):
        print(f"[SRC]  {tokenizer.batch_decode([batch['src_input'][i]], skip_special_tokens=True)[0]}")
        print(f"[PRED] {preds[i]}")
        print(f"[REF]  {refs[i]}\n")


if __name__ == "__main__":
    main()