# dataset.py
import torch
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer


class Seq2SeqDataModule:
    """
    通用的 Seq2Seq 数据加载模块
    支持 IWSLT2017、TED Talks、Gigaword、CNN/DailyMail 等数据集。
    通过统一接口 prepare_dataloaders() 返回 DataLoader + tokenizer。
    """

    def __init__(
        self,
        dataset_name: str = "iwslt2017",
        dataset_config: str = "iwslt2017-en-de",
        src_lang: str = "en",
        tgt_lang: str = "de",
        model_name: str = "Helsinki-NLP/opus-mt-en-de",
        max_len: int = 128,
        batch_size: int = 64,
        limit_train_samples: int = 20000,
        num_workers: int = 2,
        use_fast_tokenizer: bool = True,
    ):
        """
        参数:
            dataset_name: Hugging Face 数据集名称
            dataset_config: 数据集配置名（如 iwslt2017-en-de）
            src_lang: 源语言 key
            tgt_lang: 目标语言 key
            model_name: 对应 tokenizer 模型名
            max_len: 最大序列长度
            batch_size: batch 大小
            limit_train_samples: 限制训练样本数（调试时使用）
            num_workers: DataLoader 并行线程
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.limit_train_samples = limit_train_samples
        self.num_workers = num_workers
        self.use_fast_tokenizer = use_fast_tokenizer

        self.tokenizer = None
        self.pad_id = None
        self.eos_id = None

    # ---------------------- #
    # Step 1. 初始化 tokenizer #
    # ---------------------- #
    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        print(f"Tokenizer loaded from {self.model_name}, vocab size={len(self.tokenizer)}")

    # ---------------------- #
    # Step 2. 加载原始数据集 #
    # ---------------------- #
    # def load_raw_dataset(self):
    #     dataset = load_dataset(
    #         self.dataset_name,
    #         self.dataset_config,
    #         split={'train': 'train', 'validation': 'validation'},
    #         trust_remote_code=True
    #     )
    #     train_raw, val_raw = dataset['train'], dataset['validation']
    #
    #     if self.limit_train_samples > 0:
    #         train_raw = train_raw.select(range(min(self.limit_train_samples, len(train_raw))))
    #     print(f"Loaded {len(train_raw)} training samples, {len(val_raw)} validation samples")
    #     return train_raw, val_raw
    def load_raw_dataset(self):
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split={
                'train': 'train',
                'validation': 'validation',
                'test': 'test'
            },
            trust_remote_code=True
        )

        train_raw, val_raw, test_raw = dataset['train'], dataset['validation'], dataset['test']

        if self.limit_train_samples > 0:
            train_raw = train_raw.select(range(min(self.limit_train_samples, len(train_raw))))

        print(f"Loaded {len(train_raw)} train, {len(val_raw)} val, {len(test_raw)} test samples")
        return train_raw, val_raw, test_raw

    # ---------------------- #
    # Step 3. Tokenize 批处理 #
    # ---------------------- #
    def tokenize_batch(self, examples):
        """
        对一批样本进行编码，输出 input_ids 和 labels。
        """
        src_texts = [t[self.src_lang] for t in examples['translation']]
        tgt_texts = [t[self.tgt_lang] for t in examples['translation']]

        enc = self.tokenizer(src_texts, truncation=True, padding=False, max_length=self.max_len)
        dec = self.tokenizer(tgt_texts, truncation=True, padding=False, max_length=self.max_len - 1)

        # Decoder 的输出要加上 <eos>
        return {
            "input_ids": enc["input_ids"],
            "labels": [ids + [self.eos_id] for ids in dec["input_ids"]],
        }

    # ---------------------- #
    # Step 4. 构造 collate_fn #
    # ---------------------- #
    def collate_fn(self, batch):
        input_ids = [b['input_ids'] for b in batch]
        labels = [b['labels'] for b in batch]
        max_src = max(len(x) for x in input_ids)
        max_tgt = max(len(x) for x in labels)

        src_padded = [x + [self.pad_id] * (max_src - len(x)) for x in input_ids]
        tgt_padded = [x + [self.pad_id] * (max_tgt - len(x)) for x in labels]

        return {
            "src_input": torch.tensor(src_padded, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_padded, dtype=torch.long),
        }

    # ---------------------- #
    # Step 5. 构造 Dataloader #
    # ---------------------- #
    # def prepare_dataloaders(self):
    #     if self.tokenizer is None:
    #         self.setup_tokenizer()
    #
    #     train_raw, val_raw = self.load_raw_dataset()
    #
    #     train_tok = train_raw.map(
    #         partial(self.tokenize_batch),
    #         batched=True,
    #         remove_columns=train_raw.column_names
    #     )
    #     val_tok = val_raw.map(
    #         partial(self.tokenize_batch),
    #         batched=True,
    #         remove_columns=val_raw.column_names
    #     )
    #
    #     train_loader = DataLoader(
    #         train_tok,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         collate_fn=self.collate_fn,
    #         num_workers=self.num_workers
    #     )
    #
    #     val_loader = DataLoader(
    #         val_tok,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         collate_fn=self.collate_fn,
    #         num_workers=self.num_workers
    #     )
    #
    #     print(f"Dataloaders ready | batch_size={self.batch_size}")
    #     return train_loader, val_loader, self.tokenizer
    def prepare_dataloaders(self):
        if self.tokenizer is None:
            self.setup_tokenizer()

        train_raw, val_raw, test_raw = self.load_raw_dataset()

        # Tokenize
        train_tok = train_raw.map(partial(self.tokenize_batch), batched=True, remove_columns=train_raw.column_names)
        val_tok = val_raw.map(partial(self.tokenize_batch), batched=True, remove_columns=val_raw.column_names)
        test_tok = test_raw.map(partial(self.tokenize_batch), batched=True, remove_columns=test_raw.column_names)

        # DataLoaders
        train_loader = DataLoader(train_tok, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=self.collate_fn, num_workers=self.num_workers)
        val_loader = DataLoader(val_tok, batch_size=self.batch_size, shuffle=False,
                                collate_fn=self.collate_fn, num_workers=self.num_workers)
        test_loader = DataLoader(test_tok, batch_size=self.batch_size, shuffle=False,
                                 collate_fn=self.collate_fn, num_workers=self.num_workers)

        print(f"Dataloaders ready | batch_size={self.batch_size}")
        return train_loader, val_loader, test_loader, self.tokenizer