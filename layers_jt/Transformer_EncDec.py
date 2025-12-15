import jittor as jt
from jittor import nn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # PyTorch: padding=2, padding_mode='circular'
        # 手动实现 circular padding，所以这里 padding=0
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=0) 
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        
        # FIX: 使用 (Height=D, Width=L) 的策略避免 Jittor 维度检查错误
        # Input视作 [B, 1, D, L]
        # kernel_size=(1, 3): D维度不变，L维度做池化
        # stride=(1, 2): D维度步长1，L维度步长2
        # padding=(0, 1): D维度不填充，L维度填充1
        self.maxPool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def execute(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)
        
        # 1. Manual Circular Padding (padding=2)
        # [B, D, L] -> [B, D, L+4]
        x = jt.concat([x[:, :, -2:], x, x[:, :, :2]], dim=2)
        
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # 2. MaxPool1d logic using MaxPool2d trick
        # 当前 x: [B, D, L_out]
        # 变为: [B, 1, D, L_out] (视 D 为 Height, L 为 Width)
        x = x.unsqueeze(1) 
        
        # 池化后: [B, 1, D, L_pooled]
        x = self.maxPool(x)
        
        # 还原: [B, D, L_pooled]
        x = x.squeeze(1)
        
        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.relu if activation == "relu" else nn.gelu

    def execute(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        
        # FFN: [B, L, D] -> transpose -> [B, D, L]
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def execute(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.relu if activation == "relu" else nn.gelu

    def execute(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Self Attention
        res = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0]
        x = x + self.dropout(res)
        x = self.norm1(x)

        # Cross Attention
        res = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0]
        x = x + self.dropout(res)

        y = x = self.norm2(x)
        
        # FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def execute(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x