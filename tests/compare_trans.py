import sys
import os
import numpy as np
import time

# 定义配置类，模拟 argparse 的参数
class Configs:
    def __init__(self):
        self.task_name = 'classification' # 根据代码逻辑，forward调用了classification
        self.pred_len = 0
        self.seq_len = 96      # 输入序列长度
        self.enc_in = 7        # Encoder 输入特征数
        self.dec_in = 7        # Decoder 输入特征数
        self.c_out = 7         # 输出特征数
        self.d_model = 512     # Embedding 维度
        self.n_heads = 8       # 多头注意力头数
        self.e_layers = 3      # Encoder 层数
        self.d_layers = 1      # Decoder 层数 (虽然 classification 没用到，但 init 需要)
        self.d_ff = 2048       # FFN 维度
        self.factor = 1        # Attention factor
        self.embed = 'timeF'   # Embedding 类型
        self.freq = 'h'        # 频率
        self.dropout = 0.1     # Dropout
        self.activation = 'gelu' 
        self.num_class = 10    # 分类类别数

def test_pytorch(configs):
    print("=" * 20 + " Testing PyTorch Model " + "=" * 20)
    try:
        import torch
        from models.Transformer import Model as PyTorchModel
        
        # 1. 初始化模型
        model = PyTorchModel(configs)
        model.eval()
        print("PyTorch Model built successfully.")

        # 2. 创建假数据 (Batch, Seq_Len, Features)
        batch_size = 2
        x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
        # x_mark_enc 在 classification 中作为 padding mask 使用 (B, Seq_Len)
        x_mark_enc = torch.ones(batch_size, configs.seq_len) 
        
        # Decoder 输入 (虽然 forward 里没用到，但为了接口完整性传入)
        x_dec = torch.randn(batch_size, configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)

        # 3. 前向传播
        start_time = time.time()
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        end_time = time.time()

        print(f"Input Shape: {x_enc.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Expected Shape: ({batch_size}, {configs.num_class})")
        print(f"Inference Time: {end_time - start_time:.4f}s")
        
        if output.shape == (batch_size, configs.num_class):
            print("\033[92mPyTorch Test Passed!\033[0m")
        else:
            print("\033[91mPyTorch Output Shape Mismatch!\033[0m")

    except Exception as e:
        print(f"\033[91mPyTorch Test Failed with error:\033[0m {e}")
        import traceback
        traceback.print_exc()


def test_jittor(configs):
    print("\n" + "=" * 20 + " Testing Jittor Model " + "=" * 20)
    try:
        import jittor as jt
        # 启用 CUDA 如果可用，否则使用 CPU
        if jt.has_cuda:
            jt.flags.use_cuda = 1
            print("Jittor using CUDA.")
        else:
            print("Jittor using CPU.")
            
        from models.Transformer_jt import Model as JittorModel

        # 1. 初始化模型
        model = JittorModel(configs)
        model.eval()
        print("Jittor Model built successfully.")

        # 2. 创建假数据 (Batch, Seq_Len, Features)
        batch_size = 2
        x_enc = jt.randn(batch_size, configs.seq_len, configs.enc_in)
        # x_mark_enc 在 classification 中作为 padding mask
        x_mark_enc = jt.ones(batch_size, configs.seq_len)
        
        x_dec = jt.randn(batch_size, configs.pred_len, configs.dec_in)
        x_mark_dec = jt.randn(batch_size, configs.pred_len, 4)

        # 3. 前向传播
        # Jittor 第一次运行会进行编译，可能较慢
        start_time = time.time()
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # 强制同步以获取准确时间 (Jittor 是异步的)
        output.sync() 
        end_time = time.time()

        print(f"Input Shape: {x_enc.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Expected Shape: ({batch_size}, {configs.num_class})")
        print(f"Inference Time (including compile overhead if first run): {end_time - start_time:.4f}s")

        if output.shape == [batch_size, configs.num_class]:
            print("\033[92mJittor Test Passed!\033[0m")
        else:
            print("\033[91mJittor Output Shape Mismatch!\033[0m")

    except ImportError as e:
         print(f"\033[91mJittor Test Failed on Import:\033[0m {e}")
         print("Please ensure 'layers_jt' folder exists and contains necessary modules.")
    except Exception as e:
        print(f"\033[91mJittor Test Failed with error:\033[0m {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    configs = Configs()
    
    # 运行测试
    test_pytorch(configs)
    test_jittor(configs)