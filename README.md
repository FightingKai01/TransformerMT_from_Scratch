# Transformer_from_Scratch
本项目实现了一个**从零**构建的 Transformer 模型（Encoder-Decoder 架构），在 IWSLT2017 英德翻译数据集（en→de） 上进行训练、验证并进行测试，最终得到英德翻译模型TransformerMT。
# 安装

```
cd .../TransformerMT/
pip install -r requirements.txt
```

# 环境与硬件要求

|     组件      |     推荐     |
| :-----------: | :----------: |
|    Python     |     3.10     |
|    torch      | 2.3.0+cu121  |
| transformers  |    4.50.0    |
|   datasets    |    3.0.0     |
| sentencepiece |    0.2.1     |
|     numpy     |    1.23.5    |
|  matplotlib   |    3.10.7    |
|   sacrebleu   |    2.5.1     |
|   evaluate    |    0.4.6     |
|      GPU      | RTX 3090 24G |
|   操作系统    | Linux 20.04  |

# 项目结构

```python
Transformer_from_Scratch
|
|-- results
|   |-- img.png
|   |-- loss_curve_1.png
|   |-- loss_curve_2.png
|-- src
|   |-- checkpoints
|   |   |-- best_checkpoint.pt
|   |   |-- checkpoint.pt
|   |-- data
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   |-- translation_datasets.cpython-310.pyc
|   |   |-- translation_datasets.py
|   |-- demo.py
|   |-- models
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   |-- model.cpython-310.pyc
|   |   |-- model.py
|   |-- test.py
|   |-- train.py
|   
|-- README.md
|-- requirements.txt

```

# 运行

## 方式1：运行脚本

```python
#训练
bash scripts/train.sh
#测试
bash scripts/test.sh
```



## 方式2：CLI

### 训练

```python
python train.py \
  --dataset_name "iwslt2017" \
  --dataset_config "iwslt2017-en-de" \
  --src_lang "en" \
  --tgt_lang "de" \
  --model_name "Helsinki-NLP/opus-mt-en-de" \
  --max_len 1024 \
  --batch_size 64 \
  --limit_train_samples 0 \
  --num_layers 2 \
  --d_model 256 \
  --num_heads 8 \
  --d_ffn 1024 \
  --dropout_emb 0.1 \
  --dropout_atten 0.1 \
  --dropout_ffn 0.1 \
  --num_epochs 50 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --save_dir "./checkpoints" \
  --result_dir "../results"  \
  --resume "./checkpoints/checkpoint.pt"

```

### 测试

```python
python test.py \
  --dataset_name "iwslt2017" \
  --dataset_config "iwslt2017-en-de" \
  --src_lang "en" \
  --tgt_lang "de" \
  --model_name "Helsinki-NLP/opus-mt-en-de" \
  --max_len 1024 \
  --batch_size 64 \
  --model_ckpt "./checkpoints/best_checkpoint.pt"
```

## 实验可复现性

为确保实验结果完全可重复，代码中固定了所有随机种子。

```
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
```

- 数据加载与划分：使用 Hugging Face datasets 提供的官方 IWSLT2017 (en→de) 版本（206112 train, 888 val, 8079 test samples）
- Tokenizer：Helsinki-NLP/opus-mt-en-de
- 样本数量：limit_train_samples = 0 无限制
- 优化器：AdamW(lr=5e-4, weight_decay 0.01)
- Loss：CrossEntropy(ignore_index=tokenizer.pad_token_id)

## 输出结果

训练结束后，可视化损失曲线图会自动保存在 `results/` 目录下。

模型检查点存储在`./src/checkpoints/`下，`checkpoint.pt`用于断点续训，保存上次训练终止的模型最终状态，`best_checkpoint.pt`用于测试，保存训练过程中在验证集上最好的模型。

checkpoints下载链接： https://pan.baidu.com/s/1lT7E5caMYbU_MEQPSWdxQA?pwd=0110 提取码: 0110

目前最好的结果:BLEU=22.15。

## models/model.py结构概览

|          模块          |             功能              |
| :--------------------: | :---------------------------: |
|   MultiHeadAttention   |         多头自注意力          |
|      FeedForward       |         前馈神经网络          |
| pos_sinusoid_embedding |         正弦位置编码          |
|      EncoderLayer      |    自注意力 + FFN + 残差层    |
|        Encoder         |            编码器             |
|      DecoderLayer      |     交叉注意力+FFN+残差层     |
|        decoder         |            解码器             |
|     TransformerMT      | 整体 Encoder-Decoder 框架包含 |

