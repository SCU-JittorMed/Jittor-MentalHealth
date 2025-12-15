import jittor as jt
from jittor import nn
from layers_jt.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers_jt.SelfAttention_Family import FullAttention, AttentionLayer
from layers_jt.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Jittor 中通常直接使用 nn.gelu
        self.act = nn.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)


    def classification(self, x_enc, x_mark_enc):
        import time
        t0 = time.time()
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        t1 = time.time()
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        t2 = time.time()
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        t3 = time.time()
        # zero-out padding embeddings
        # Jittor 支持类似的广播机制
        output = output * x_mark_enc.unsqueeze(-1)  
        t4 = time.time()
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        t5 = time.time()
        # 打印耗时
        # print(f"---------------------------------------")
        # print(f"1. Embedding Cost : {t1 - t0:.6f} s")
        # print(f"2. Encoder Cost   : {t2 - t1:.6f} s")
        # print(f"3. Act+Drop Cost  : {t3 - t2:.6f} s")
        # print(f"4. Masking Cost   : {t4 - t3:.6f} s")
        # print(f"5. Project Cost   : {t5 - t4:.6f} s")
        # print(f"   Total Cost     : {t5 - t0:.6f} s")
        # print(f"---------------------------------------")
        return output

    def execute(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Jittor 模型通常使用 execute 代替 forward，但为了兼容性 forward 也是可以的
        # 这里保留代码逻辑，如果是在 Time-Series-Library 框架下，通常调用 forward
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]
    
    # 保持 forward 接口以兼容大部分训练代码
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.execute(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)