# import os
# from data_factory import data_provider
# from tools import EarlyStopping, cal_accuracy
# import torch
# import torch.nn as nn
# from torch import optim
# from jittor import optim as jt_optim
# import os
# import time
# import numpy as np

# from models import Transformer, Transformer_jt

# class Exp_Basic(object):
#     def __init__(self, args):
#         self.args = args
#         self.model_dict = {
#             'Transformer': Transformer,
#             'Transformer_jt': Transformer_jt,
#         }
#         self.device = self._acquire_device()
#         if self.args.model == "Transformer_jt":
#             self.model = self._build_model()
#         else:
#             self.model = self._build_model().to(self.device)
#     def _build_model(self):
#         # model input depends on data
#         train_data, train_loader = self._get_data(flag='TRAIN')
#         test_data, test_loader = self._get_data(flag='TEST')
#         self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
#         self.args.pred_len = 0
#         if self.args.enc_in==7:#default setting
#             self.args.enc_in = train_data.feature_df.shape[1]
#         self.args.num_class = len(train_data.class_names)
#         # model init
#         if self.args.model == "Transformer_jt":
#             model = self.model_dict[self.args.model].Model(self.args)
#         else:
#             model = self.model_dict[self.args.model].Model(self.args).float()
#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _acquire_device(self):
#         if self.args.use_gpu and self.args.gpu_type == 'cuda':
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(
#                 self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
#             device = torch.device('cuda:{}'.format(self.args.gpu))
#             print('Use GPU: cuda:{}'.format(self.args.gpu))
#         elif self.args.use_gpu and self.args.gpu_type == 'mps':
#             device = torch.device('mps')
#             print('Use GPU: mps')
#         else:
#             device = torch.device('cpu')
#             print('Use CPU')
#         return device

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         if self.args.model == "Transformer_jt":
#             model_optim = jt_optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         else:
#             model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.CrossEntropyLoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         preds = []
#         trues = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 padding_mask = padding_mask.float().to(self.device)
#                 label = label.to(self.device)

#                 outputs = self.model(batch_x, padding_mask, None, None)

#                 pred = outputs.detach()
#                 loss = criterion(pred, label.long().view(-1))
#                 total_loss.append(loss.item())

#                 preds.append(outputs.detach())
#                 trues.append(label)

#         total_loss = np.average(total_loss)

#         preds = torch.cat(preds, 0)
#         trues = torch.cat(trues, 0)
#         probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
#         predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
#         trues = trues.flatten().cpu().numpy()
#         accuracy = cal_accuracy(predictions, trues)

#         self.model.train()
#         return total_loss, accuracy

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='TRAIN')
#         vali_data, vali_loader = self._get_data(flag='TEST')
#         test_data, test_loader = self._get_data(flag='TEST')

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()

#             for i, (batch_x, label, padding_mask) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()

#                 batch_x = batch_x.float().to(self.device)
#                 padding_mask = padding_mask.float().to(self.device)
#                 label = label.to(self.device)

#                 outputs = self.model(batch_x, padding_mask, None, None)
#                 loss = criterion(outputs, label.long().view(-1))
#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
#                 model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
#             test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

#             print(
#                 "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
#                 .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
#             early_stopping(-val_accuracy, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

import os
from data_factory import data_provider
from tools import EarlyStopping, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np

# 引入 Jittor
import jittor as jt
import jittor.nn as jnn

from models import Transformer, Transformer_jt

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Transformer_jt': Transformer_jt,
        }
        
        # 判断是否使用 Jittor
        self.is_jittor = (self.args.model == 'Transformer_jt')
        
        self.device = self._acquire_device()
        
        if self.is_jittor:
            self.model = self._build_model()
        else:
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        
        # 统一 seq_len
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        
        # 自动设置 enc_in
        if self.args.enc_in == 7: # default setting
            # 尝试获取特征维度，兼容 DataFrame (Pandas) 和直接的 shape 读取
            if hasattr(train_data, 'feature_df'):
                self.args.enc_in = train_data.feature_df.shape[1]
            else:
                # 假如没有 feature_df，尝试通过第一条数据获取维度
                sample_x, _ = train_data[0]
                self.args.enc_in = sample_x.shape[1] if len(sample_x.shape) > 1 else 1

        self.args.num_class = len(train_data.class_names)
        
        # model init
        model_cls = self.model_dict[self.args.model].Model
        
        if self.is_jittor:
            model = model_cls(self.args)
        else:
            model = model_cls(self.args).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.is_jittor:
            if self.args.use_gpu:
                jt.flags.use_cuda = 1
            else:
                jt.flags.use_cuda = 0
            return None
        else:
            if self.args.use_gpu and self.args.gpu_type == 'cuda':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            elif self.args.use_gpu and self.args.gpu_type == 'mps':
                device = torch.device('mps')
                print('Use GPU: mps')
            else:
                device = torch.device('cpu')
                print('Use CPU')
            return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.is_jittor:
            # Jittor 常用 Adam，也可以用 jt.optim.SGD 等
            model_optim = jt.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.is_jittor:
            criterion = jnn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        
        # Jittor 上下文管理
        if self.is_jittor:
            context = jt.no_grad()
        else:
            context = torch.no_grad()

        with context:
            for i, batch_data in enumerate(vali_loader):
                # 数据解包适配
                if self.is_jittor:
                    batch_x, label, padding_mask = batch_data
                else:
                    batch_x, label, padding_mask = batch_data
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                # Forward
                if self.is_jittor:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    # Jittor loss 计算
                    loss = criterion(outputs, label)
                    total_loss.append(loss.item())
                    
                    preds.append(outputs.numpy())
                    trues.append(label.numpy())
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    pred = outputs.detach()
                    loss = criterion(pred, label.long().view(-1))
                    total_loss.append(loss.item())

                    preds.append(outputs.detach().cpu())
                    trues.append(label.cpu())

        total_loss = np.average(total_loss)

        if self.is_jittor:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            # Softmax
            # Jittor numpy 手动 softmax 或使用 jt.nn.softmax
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / e_x.sum(axis=1, keepdims=True)
            probs = softmax(preds)
            predictions = np.argmax(probs, axis=1)
            trues = trues.flatten()
        else:
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            probs = torch.nn.functional.softmax(preds, dim=1) 
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            trues = trues.flatten().cpu().numpy()

        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL') # 通常 vali 用 VAL 集
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        
        # 注意：EarlyStopping 内部通常包含 torch.save，Jittor 需要适配
        # 这里假设 EarlyStopping 还是原来的，如果是 Jittor，我们可能需要手动处理保存逻辑
        # 或者在 EarlyStopping 传入 model 时做特殊处理。
        # 为兼容性，这里我们手动处理最佳模型的保存逻辑，或者依赖 tools.py 已经修改过
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                
                if self.is_jittor:
                    batch_x, label, padding_mask = batch_data
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)
                    
                    # Jittor: Forward -> Loss -> Step(loss)
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label)
                    
                    model_optim.step(loss)
                    
                    train_loss.append(loss.item())
                    
                else:
                    batch_x, label, padding_mask = batch_data
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().view(-1))
                    train_loss.append(loss.item())
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # Validation
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            # Early Stopping Check
            if self.is_jittor:
                # Jittor 保存逻辑手动处理，绕过可能不兼容的 EarlyStopping 保存部分
                # 假设 EarlyStopping 只是用来判断是否停止
                # 我们需要临时修改 EarlyStopping 或者在这里手动保存
                
                # 简易版 Early Stopping 逻辑适配 Jittor
                score = -val_accuracy # 假设 EarlyStopping 使用 validation loss 越小越好，如果是 Accuracy 则是越大越好。原代码传入 -val_accuracy 说明逻辑是“越小越好”
                # 注意：原代码 `early_stopping(-val_accuracy, ...)` 意味着它期望 monitor 指标下降。
                # 如果 early_stopping 内部有 save_checkpoint，传递 Jittor model 可能会报错（如果内部用了 torch.save）
                
                # 为了安全，这里我们只用 early_stopping 的计数功能，保存自己来做
                early_stopping(score, self.model, path) 
                # 如果 early_stopping 内部保存失败（catch exception），或者我们需要覆盖
                if early_stopping.val_loss_min == score: # 意味着这是最佳模型
                     print(f"Jittor: Saving best model to {path}/checkpoint.pkl")
                     self.model.save(os.path.join(path, 'checkpoint.pkl'))

            else:
                early_stopping(-val_accuracy, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load Best Model
        if self.is_jittor:
            best_model_path = os.path.join(path, 'checkpoint.pkl')
            print(f"Loading best model from {best_model_path}")
            self.model.load(best_model_path)
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model