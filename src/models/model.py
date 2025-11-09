import torch
import numpy as np

from torch import nn


# 多头注意力定义
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_k, dim_v, dim_model, num_heads, dropout=0.):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # 线性层
        self.W_Q = nn.Linear(dim_model, dim_k * num_heads)
        self.W_K = nn.Linear(dim_model, dim_k * num_heads)
        self.W_V = nn.Linear(dim_model, dim_v * num_heads)
        self.W_O = nn.Linear(dim_v * num_heads, dim_model)

        # 线性层初始化，专门为ReLU设计，而Xavier Uniform为线性/sigmoid/tanh设计
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_v)))
        nn.init.normal_(self.W_O.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_v)))

    def forward(self, Q, K, V, atten_mask):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        dim_k, dim_v = self.dim_k, self.dim_v
        num_heads = self.num_heads

        # 多头划分
        Q = self.W_Q(Q).reshape(N, -1, num_heads, dim_k).transpose(1, 2)  # (batch_size, seq_len, dim_model)->(batch_size, num_heads, seq_len, dim_k)
        K = self.W_K(K).reshape(N, -1, num_heads, dim_k).transpose(1, 2)  # (batch_size, seq_len, dim_model)->(batch_size, num_heads, seq_len, dim_k)
        V = self.W_V(V).reshape(N, -1, num_heads, dim_v).transpose(1, 2)  # (batch_size, seq_len, dim_model)->(batch_size, num_heads, seq_len, dim_v)

        # 预处理掩码
        if atten_mask is not None:
            assert atten_mask.shape == (N, q_len, k_len)
            atten_mask = atten_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
            atten_mask = atten_mask.bool()

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(dim_k))
        if atten_mask is not None:
            scores.masked_fill_(atten_mask, -1e4)
        atten_scores = torch.softmax(scores, dim=-1)
        atten_scores = self.dropout(atten_scores)  # 被dropout的位置权重会置零，然后重新归一化（依旧保持期望保持不变，且加和为1）,让模型学会在部分注意力连接缺失时仍能工作

        outputs = atten_scores @ V

        # 多头合并
        outputs = outputs.transpose(1, 2).reshape(N, -1, num_heads * dim_v)  # (batch_size, num_heads, seq_len, dim_v)->(batch_size, seq_len, num_heads*dim_v)
        outputs = self.W_O(outputs)

        return outputs


# 位置编码（使用 正弦余弦位置编码）
def pos_sinusoid_embedding(seq_len, dim_model):
    embeddings = torch.zeros((seq_len, dim_model))
    for i in range(dim_model):
        func = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = func(torch.arange(0, seq_len) / torch.pow(torch.tensor(10000.0), (
                    2 * (i // 2) / dim_model)))  # 2 * (i // 2): 将索引映射到成对的频率索引
    return embeddings.float()


# 逐位置前馈网络定义
class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ffn, dropout=0.):
        super().__init__()
        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.dropout = nn.Dropout(dropout)

        # 使用一维卷积实现MLP
        self.conv1 = nn.Conv1d(in_channels=self.dim_model, out_channels=self.dim_ffn, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.dim_ffn, out_channels=self.dim_model, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        outputs = self.relu(self.conv1(X.transpose(1,
                                                   2)))  # (batch_size, seq_len, dim_model)-> (batch_size, dim_model, seq_len)->(batch_size, dim_ffn, seq_len)-> (batch_size, dim_ffn, seq_len)
        outputs = self.conv2(outputs).transpose(1,
                                                2)  # (batch_size, dim_ffn, seq_len)->(batch_size, dim_model, seq_len)->(batch_size, seq_len, dim_model)
        outputs = self.dropout(outputs)  # 形状保持不变
        return outputs


# 编码器Layer定义
class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ffn, num_heads, dropout_atten=0., dropout_ffn=0.):
        super().__init__()

        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.dropout_atten_p = dropout_atten
        self.dropout_ffn_p = dropout_ffn

        assert dim_model % num_heads == 0  # 确保可以正确划分
        self.dim_each_head = dim_model // num_heads  # 获取每个头处理的维度

        # 归一化层
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

        # 多头注意力层
        self.multi_headim_atten = MultiHeadAttention(dim_k=self.dim_each_head, dim_v=self.dim_each_head,
                                                     dim_model=self.dim_model, num_heads=self.num_heads,
                                                     dropout=self.dropout_atten_p)

        # 前馈网络层
        self.poswise_ffn = FeedForward(dim_model=self.dim_model, dim_ffn=self.dim_ffn, dropout=self.dropout_ffn_p)

    def forward(self, enc_in, atten_mask):
        residual = enc_in

        # 多头注意力层forward
        context = self.multi_headim_atten(Q=enc_in, K=enc_in, V=enc_in, atten_mask=atten_mask)

        # 残差连接+层归一化
        outputs = self.norm1(residual + context)

        residual = outputs

        # 前馈网络forward
        outputs = self.poswise_ffn(outputs)

        # 残差连接+层归一化
        outputs = self.norm2(residual + outputs)

        return outputs


# 编码器定义
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, num_layers, dim_enc, num_heads, dim_ffn, seq_max_len, dropout_emb=0.,
                 dropout_atten=0., dropout_ffn=0.):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.num_layers = num_layers
        self.dim_enc = dim_enc
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.seq_max_len = seq_max_len
        self.dropout_emb_p = dropout_emb
        self.dropout_atten_p = dropout_atten
        self.dropout_ffn_p = dropout_ffn

        # 输入嵌入层
        self.in_emb = nn.Embedding(
            num_embeddings=self.src_vocab_size,  # 词汇表大小
            embedding_dim=self.dim_enc,  # 嵌入维度
            padding_idx=0  # 可选：填充符索引
        )

        # 位置编码层，加载固定的
        self.pos_emb = nn.Embedding.from_pretrained(embeddings=pos_sinusoid_embedding(self.seq_max_len, self.dim_enc),
                                                    freeze=True)
        self.dropout_emb = nn.Dropout(self.dropout_emb_p)

        # 编码器Layers
        self.layers = nn.ModuleList([
            EncoderLayer(dim_model=self.dim_enc, dim_ffn=self.dim_ffn, num_heads=self.num_heads,
                         dropout_atten=self.dropout_atten_p, dropout_ffn=self.dropout_ffn_p) for _ in
            range(self.num_layers)
        ])

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.shape

        # 1. 输入嵌入 ← 新增步骤
        X = self.in_emb(input_ids)  # (batch_size, seq_len) -> (batch_size, seq_len, dim_enc)

        # 2. 增加位置编码
        outputs = X + self.pos_emb(torch.arange(seq_len, device=X.device))
        outputs = self.dropout_emb(outputs)

        # 3. Encoder Layers forward
        for layer in self.layers:
            outputs = layer(outputs, mask)

        return outputs  # 形状: (batch_size, seq_len, dim_enc)


# 解码器Layer定义
class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ffn, num_heads, dropout_atten=0., dropout_ffn=0.):
        super().__init__()
        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.dropout_atten_p = dropout_atten
        self.dropout_ffn_p = dropout_ffn

        assert dim_model % num_heads == 0
        self.num_each_head = dim_model // num_heads

        # 归一化层
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        #
        self.poswise_ffn = FeedForward(
            dim_model=self.dim_model,
            dim_ffn=self.dim_ffn,
            dropout=self.dropout_ffn_p
        )

        # 多头注意力层（self-attention和encoder-decoder cross attention）
        self.enc_atten = MultiHeadAttention(dim_k=self.num_each_head, dim_v=self.num_each_head,
                                            dim_model=self.dim_model, num_heads=self.num_heads,
                                            dropout=self.dropout_atten_p)
        self.enc_dec_cross_atten = MultiHeadAttention(dim_k=self.num_each_head, dim_v=self.num_each_head,
                                                      dim_model=self.dim_model, num_heads=self.num_heads,
                                                      dropout=self.dropout_atten_p)

    def forward(self, dec_in, enc_out, dec_mask, enc_dec_atten_mask, cache=None, freqs_cis=None):
        residual = dec_in

        # self-attention forward
        context = self.enc_atten(Q=dec_in, K=dec_in, V=dec_in, atten_mask=dec_mask)
        outputs = self.norm1(residual + context)

        residual = outputs
        # encoder-decoder cross attention forward
        context = self.enc_dec_cross_atten(Q=outputs, K=enc_out, V=enc_out, atten_mask=enc_dec_atten_mask)
        outputs = self.norm2(residual + context)

        residual = outputs
        # 前馈网络forward
        outputs = self.poswise_ffn(outputs)
        outputs = self.norm3(residual + outputs)

        return outputs


# 解码器定义
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, num_layers, dim_dec, num_heads, dim_ffn, seq_max_len, dropout_emb=0.,
                 dropout_atten=0., dropout_ffn=0.):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.dim_dec = dim_dec
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.seq_max_len = seq_max_len
        self.dropout_emb_p = dropout_emb
        self.dropout_atten_p = dropout_atten
        self.dropout_ffn_p = dropout_ffn

        # outputs embedding层
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim_dec)
        self.dropout_emb = nn.Dropout(self.dropout_emb_p)

        # 位置编码层
        self.pos_emb = nn.Embedding.from_pretrained(embeddings=pos_sinusoid_embedding(seq_max_len, dim_dec),
                                                    freeze=True)

        # 解码器 layers
        self.layers = nn.ModuleList([
            DecoderLayer(dim_model=self.dim_dec, dim_ffn=self.dim_ffn, num_heads=self.num_heads,
                         dropout_atten=self.dropout_atten_p, dropout_ffn=self.dropout_ffn_p) for _ in
            range(self.num_layers)
        ])

    def forward(self, labels, emc_out, dec_mask, dec_enc_atten_mask, cache=None):
        tgt_emb = self.tgt_emb(labels)

        ## 调试位置编码的最长长度极限
        # print("labels.shape[1]:", labels.shape[1])
        # print("self.pos_emb.num_embeddings:", self.pos_emb.num_embeddings)

        pos_emb = self.pos_emb(torch.arange(labels.shape[1], device=labels.device))

        outputs = self.dropout_emb(tgt_emb + pos_emb)

        # decoder layers forward
        for layer in self.layers:
            outputs = layer(outputs, emc_out, dec_mask, dec_enc_atten_mask)

        return outputs


# 完整的TransformerMT模型
class TransformerMT(nn.Module):
    def __init__(self,
                 src_tokenizer, tgt_tokenizer,
                 num_layers, d_model, num_heads, d_ffn, max_seq_len,
                 dropout_emb=0., dropout_atten=0., dropout_ffn=0.
                 ):
        super().__init__()
        # 从 tokenizer 提取关键信息
        self.src_tokenizer = src_tokenizer
        self.src_vocab_size = len(self.src_tokenizer)
        self.src_pad_id = src_tokenizer.pad_token_id
        self.src_bos_id = getattr(src_tokenizer, "bos_token_id", None)
        self.src_eos_id = getattr(src_tokenizer, "eos_token_id", None)

        self.tgt_tokenizer = tgt_tokenizer
        self.tgt_vocab_size = len(self.tgt_tokenizer)
        self.tgt_pad_id = tgt_tokenizer.pad_token_id
        self.tgt_bos_id = getattr(tgt_tokenizer, "bos_token_id", None)
        self.tgt_eos_id = getattr(tgt_tokenizer, "eos_token_id", None)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.max_seq_len = max_seq_len
        self.dropout_emb_p = dropout_emb
        self.dropout_atten_p = dropout_atten
        self.dropout_ffn_p = dropout_ffn

        self.encoder = Encoder(
            src_vocab_size=self.src_vocab_size,
            num_layers=self.num_layers,
            dim_enc=self.d_model,
            num_heads=self.num_heads,
            dim_ffn=self.d_ffn,
            seq_max_len=self.max_seq_len,
            dropout_emb=self.dropout_emb_p,
            dropout_atten=self.dropout_atten_p,
            dropout_ffn=self.dropout_ffn_p
        )

        self.decoder = Decoder(
            tgt_vocab_size=self.tgt_vocab_size,
            num_layers=self.num_layers,
            dim_dec=self.d_model,
            num_heads=self.num_heads,
            dim_ffn=self.d_ffn,
            seq_max_len=self.max_seq_len,
            dropout_emb=self.dropout_emb_p,
            dropout_atten=self.dropout_atten_p,
            dropout_ffn=self.dropout_ffn_p
        )

        # 输出层
        self.output_layer = nn.Linear(d_model, self.tgt_vocab_size)

    def count_parameters(self):
        """统计模型参数数量（以M为单位）"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params

    def print_parameters(self):
        """打印模型参数数量"""
        total_params = self.count_parameters()
        total_params_m = total_params / 1_000_000  # 转换为百万单位
        print(f"Total trainable parameters: {total_params}({total_params_m:.4f}M)")

    # encoder self-attention mask：处理输入序列长度不一样的问题，用于padding
    def get_src_mask(self, src_ids):
        """
        基于 token id（含 pad token）生成 Encoder 的自注意力 mask。
        循环方式实现，形状为 [B, max_len, max_len]。
        True 表示该位置被 mask（即不参与注意力计算）。

        参数：
            src_ids: [B, max_len]，输入序列的 token id
            pad_idx: 表示 <pad> 的 token id（默认 0）

        返回：
            attn_mask: [B, max_len, max_len]，bool类型，True 为屏蔽
        """
        device = src_ids.device
        batch_size, max_seq_len = src_ids.size()
        mask = torch.ones((batch_size, max_seq_len, max_seq_len), device=device, dtype=torch.bool)

        # 对每个样本(token向量)逐个构造mask矩阵
        for i in range(batch_size):  # todo 优化，不使用循环
            # 当前样本有效token数（非padding部分）
            valid_len = (src_ids[i] != self.src_pad_id).sum().item()

            # 前 valid_len 个token可见（即False），后面padding部分被mask（True）
            mask[i, :, :valid_len] = False

        return mask

    # decoder的self attention的causal mask
    def get_tgt_mask(self, tgt_ids):
        device = tgt_ids.device
        batch_size, max_seq_len = tgt_ids.size()

        # 防止看到未来的mask
        seq_mask = torch.triu(torch.ones((batch_size, max_seq_len, max_seq_len), device=device, dtype=torch.bool),
                              diagonal=1)

        # padding mask，类似 get_src_mask
        pad_mask = torch.ones((batch_size, max_seq_len, max_seq_len), device=device, dtype=torch.bool)
        for i in range(batch_size):  # todo 优化，不使用循环
            valid_len = (tgt_ids[i] != self.tgt_pad_id).sum().item()
            pad_mask[i, :, :valid_len] = False

        mask = torch.logical_or(seq_mask, pad_mask)  # seq_mask | pad_mask
        return mask

    def get_enc_dec_mask(self, src_ids, tgt_ids):
        """
        生成 Encoder-Decoder Cross Attention 的掩码。
        仅屏蔽 encoder 中的 <PAD> 部分。

        参数：
            src_ids: [B, src_len]，encoder 输入 token id
            tgt_ids: [B, tgt_len]，decoder 输入 token id

        返回：
            enc_dec_mask: [B, tgt_len, src_len]，bool 类型，True 表示被屏蔽
        """
        device = src_ids.device
        batch_size, src_len = src_ids.size()
        tgt_len = tgt_ids.size(1)

        # 初始化为全 True（全部屏蔽）
        enc_dec_mask = torch.ones((batch_size, tgt_len, src_len), device=device, dtype=torch.bool)

        # 对每个样本，标记非 <PAD> 的 encoder token 可见（即 False）
        for i in range(batch_size):  # todo 优化，不使用循环
            valid_len = (src_ids[i] != self.src_pad_id).sum().item()
            enc_dec_mask[i, :, :valid_len] = False  # 前 valid_len 不屏蔽

        return enc_dec_mask

    # encoder和decoder的cross - attention的mask
    def forward(self, src_ids, tgt_ids):

        # 获取三种类型的掩码
        src_mask = self.get_src_mask(src_ids)
        tgt_mask = self.get_tgt_mask(tgt_ids)
        enc_dec_mask = self.get_enc_dec_mask(src_ids, tgt_ids)

        # 编码器前向传播
        enc_output = self.encoder(src_ids, src_mask)

        # 解码器前向传播
        dec_output = self.decoder(tgt_ids, enc_output, tgt_mask, enc_dec_mask)

        # 输出层
        output = self.output_layer(dec_output)

        return output

    @torch.no_grad()
    def generate(self, src_ids, max_new_tokens=128, beam_size=1):
        """
        基于自回归的序列生成函数，用于机器翻译等 seq2seq 任务。
        支持贪心搜索（beam_size=1），可扩展为 beam search。

        参数：
            src_ids: [B, src_len]，源语言输入（已包含 <PAD>）
            max_new_tokens: 最多生成多少个新 token
            beam_size: beam search 大小（暂时仅支持1，即贪心）

        返回：
            translations: List[List[int]]，每个样本生成的目标 token 序列（含 EOS）
        """
        device = src_ids.device
        batch_size = src_ids.size(0)

        # 1 编码器部分：只执行一次
        src_mask = self.get_src_mask(src_ids)
        enc_output = self.encoder(src_ids, src_mask)

        # 2 初始化 decoder 输入：以 <BOS> 开始
        if self.tgt_bos_id is None:
            raise ValueError("Target tokenizer must define bos_token_id for generation.")
        if self.tgt_eos_id is None:
            raise ValueError("Target tokenizer must define eos_token_id for generation.")

        tgt_ids = torch.full(
            (batch_size, 1), self.tgt_bos_id, device=device, dtype=torch.long
        )  # [B, 1]

        # 标记是否已生成EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3 自回归循环生成
        for _ in range(max_new_tokens):
            tgt_mask = self.get_tgt_mask(tgt_ids)
            enc_dec_mask = self.get_enc_dec_mask(src_ids, tgt_ids)

            dec_output = self.decoder(tgt_ids, enc_output, tgt_mask, enc_dec_mask)
            logits = self.output_layer(dec_output)  # [B, seq_len, vocab]
            next_token_logits = logits[:, -1, :]  # 取最后一个时间步 [B, vocab]

            # 贪心选取
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]

            # 拼接新token
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            # 更新finished状态（如果生成了EOS）
            finished |= (next_token.squeeze(-1) == self.tgt_eos_id)
            if finished.all():
                break

        # 4 转为List输出
        translations = tgt_ids.tolist()
        return translations

    # 模型保存
    def save_model(self, path: str, optimizer=None, epoch=None, train_losses=None, val_losses=None, best_val_loss=None,
                   save_train_state=False):
        """保存模型
        Args:
            save_train_state: True表示保存训练状态（用于检查点），False表示只保存模型（用于最佳模型）
        """
        state = {
            'model_state_dict': self.state_dict(),
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ffn': self.d_ffn,
            'max_seq_len': self.max_seq_len
        }

        # 只有在保存训练状态时才添加这些信息
        if save_train_state:
            if optimizer is not None:
                state['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                state['epoch'] = epoch
            if train_losses is not None:
                state['train_losses'] = train_losses
            if val_losses is not None:
                state['val_losses'] = val_losses
            if best_val_loss is not None:
                state['best_val_loss'] = best_val_loss

        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    @classmethod
    def load_model(cls, path: str, src_tokenizer, tgt_tokenizer, optimizer=None, device='cpu'):
        """从文件中加载模型，可选返回训练状态"""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            num_layers=checkpoint['num_layers'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            d_ffn=checkpoint['d_ffn'],
            max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Model loaded from {path}")

        # 返回训练状态信息（如果存在）
        train_state = {}
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_state['optimizer'] = optimizer
        if 'epoch' in checkpoint:
            train_state['epoch'] = checkpoint['epoch']
        if 'train_losses' in checkpoint:
            train_state['train_losses'] = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            train_state['val_losses'] = checkpoint['val_losses']
        if 'best_val_loss' in checkpoint:
            train_state['best_val_loss'] = checkpoint['best_val_loss']

        return model, train_state