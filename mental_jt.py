import os
import pandas as pd
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
import jittor.nn as nn
import argparse 
import matplotlib.pyplot as plt

class MentalLoader(Dataset):
    """
    Jittor Dataset class for the Mental Health dataset.
    Loads features from specified subdirectories and labels from a central CSV file.
    Handles variable-length time series by resampling to a fixed length.
    """
    
    VALID_FEATURE_TYPES = [
        'crop_clip', 'crop_dino', 'csv', 
        'src_clip', 'src_dino', 'wobg_clip', 'wobg_dino'
    ]
    
    target_dict = {
                    "depression": "result-1", "angery":"result-2","provoke":"result-3", "mania":"result-4","anxiety":"result-5",
                   "symptom":"result-6","attention":"result-7","suicide":"result-8","psychosis":"result-9","sleep":"result-10",
                   "memory":"result-11","repeat":"result-12","disociation":"result-13","personality":"result-14","material":"result-15",
                   "suiside":"suiside"
                   }
    
    def __init__(self, args, root_path, flag='train', feature_sampler=None):
        super().__init__()
        self.args = args
        self.root_path = root_path
        self.feature_type = args.features
        try:
            self.label_column = self.target_dict[args.target]
        except KeyError:
            raise ValueError(f"Invalid target '{args.target}'. "
                             f"Must be one of {list(self.target_dict.keys())}")
            
        self.flag = flag
        self.seq_len = args.seq_len
        self.max_seq_len = args.seq_len
        self.feature_sampler = args.enc_in
        self.class_names = ["No", "Yes"]
        
        # Validate feature_type
        if self.feature_type not in self.VALID_FEATURE_TYPES:
            raise ValueError(f"Invalid feature_type '{self.feature_type}'. "
                             f"Must be one of {self.VALID_FEATURE_TYPES}")
        
        # 1. Load and process labels from data.csv
        labels_path = os.path.join(self.root_path, 'data.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label file not found at: {labels_path}")
        all_labels_df = pd.read_csv(labels_path)
        
        if self.label_column == "suiside":
            # SDS-1 到SDS-6 按照 1，2，6，10，10，4的权重加权求和；
            # 总分值大于等于6分，为1：有风险，小于6，为0：无风险
            for indx in range(6):
                all_labels_df[self.label_column] += all_labels_df[f"SDS-{indx+1}"] * [1,2,6,10,10,4][indx]
            all_labels_df[self.label_column] = (all_labels_df[self.label_column] >= 6).astype(int)
        else:
            # Preprocess labels: replace -1 with 0
            all_labels_df[self.label_column] = all_labels_df[self.label_column].replace(-1, 0)
        
        # Use Case_id as the index for easy lookup
        all_labels_df.set_index('Case_id', inplace=True)
        
        # 2. Split data into train/val/test sets
        self._split_data(all_labels_df)
        
        # Jittor specific: Set total length explicitly so the Dataset knows how many items it has
        self.total_len = len(self.case_ids)
        
        # 如果需要设置 shuffle 或 batch_size，可以在这里或者外部调用 set_attrs
        # 默认 behavior: Jittor dataset 自己也是 DataLoader
        shuffle = True if flag == 'train' else False
        self.set_attrs(shuffle=shuffle)
        
        print(f"Loaded {self.flag} set with {len(self.case_ids)} samples.")

    def _split_data(self, all_labels_df):
        all_case_ids = all_labels_df.index.unique().tolist()
        # filter some cases 如果不是suiside，则过滤掉Case_id非 纯数字字符串 的样本
        if self.label_column != "suiside":
            all_case_ids = [case_id for case_id in all_case_ids if str(case_id).isdigit()]
        
        # Use a fixed seed for reproducibility of splits
        np.random.seed(42)
        np.random.shuffle(all_case_ids)
        n_cases = len(all_case_ids)
        n_train = int(0.7 * n_cases)
        n_val = int(0.15 * n_cases)
        train_ids = all_case_ids[:n_train]
        val_ids = all_case_ids[n_train : n_train + n_val]
        test_ids = all_case_ids[n_train + n_val:]
        
        if self.flag == 'TRAIN':
            self.case_ids = train_ids
        elif self.flag == 'VAL':
            self.case_ids = val_ids
        elif self.flag == 'TEST':
            self.case_ids = test_ids
        else:
            raise ValueError(f"Invalid flag '{self.flag}'. Must be 'TRAIN', 'VAL', or 'TEST'.")
            
        # Filter the main labels dataframe to only include IDs for the current split
        self.labels_df = all_labels_df.loc[self.case_ids]

    def _resample_tensor(self, tensor: jt.Var, target_len: int) -> jt.Var:
        """
        Resamples a 2D tensor (T, F) to a new time dimension (target_len, F).
        Fix: Jittor interpolate requires (N, C, H, W). We add dummy dims to satisfy this.
        """
        if tensor.shape[0] == target_len:
            return tensor
        
        # 1. Transform (T, F) -> (F, T)
        # 2. Unsqueeze to (1, F, 1, T) -> effectively (Batch, Channels, Height=1, Width=Time)
        tensor_reshaped = tensor.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        
        # 3. Resample
        # use mode='bilinear' which is standard for 4D tensors (Height, Width)
        resampled = nn.interpolate(tensor_reshaped, size=(1, target_len), mode='bilinear', align_corners=False)
        
        # 4. Reshape back
        # (1, F, 1, target_len) -> squeeze -> (F, target_len) -> transpose -> (target_len, F)
        return resampled.squeeze(2).squeeze(0).transpose(0, 1)
        
    def __len__(self):
        return len(self.case_ids)
    
    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            # PyTorch: mean(0, keepdim=True) -> Jittor: mean(0, keepdims=True)
            mean = case.mean(0, keepdims=True)
            case = case - mean
            
            # PyTorch: var(dim=1, unbiased=False)
            # unbiased=False means dividing by N (Population Variance). 
            # Jittor doesn't have unbiased arg directly in all versions, 
            # so we calculate explicitly: mean(x^2) - mean(x)^2 or mean((x-mu)^2).
            # Since 'case' is already centered (case - mean) along dim 0, 
            # but we need variance along dim 1:
            # Note: The original code calculated mean on dim 0, subtracted it, 
            # then calculated var on dim 1. This is preserved here.
            
            # var(dim=1) implementation:
            var_dim1 = case.sqr().mean(1, keepdims=True)
            stdev = jt.sqrt(var_dim1 + 1e-5)
            
            case /= stdev
            return case
        else:
            return case
        
    def __getitem__(self, idx):
        # 1. Get Case ID and corresponding label
        case_id = self.case_ids[idx]
        label = self.labels_df.loc[case_id, self.label_column]
        
        # 2. Find and load feature file(s)
        feature_dir = os.path.join(self.root_path, self.feature_type)
        file_ext = '.csv' if self.feature_type == 'csv' else '.npy'
        
        # Check for Video1 and Video2 and load them
        data_parts = []
        for video_prefix in ['Video1_', 'Video2_']:
            file_path = os.path.join(feature_dir, f"{video_prefix}{case_id}{file_ext}")
            if os.path.exists(file_path):
                if file_ext == '.npy':
                    data = np.load(file_path)
                else: # .csv
                    data = pd.read_csv(file_path).values
                data_parts.append(data)
        
        if not data_parts:
            # Jittor datasets handle exceptions similarly, but be careful in parallel workers
            raise FileNotFoundError(f"No feature files found for Case_id {case_id} in {feature_dir}")
            
        # 3. Concatenate if multiple video parts exist
        full_data = np.concatenate(data_parts, axis=0)
        
        # 4. Convert to tensor and resample
        # torch.from_numpy().float() -> jt.array().float32()
        feature_tensor = jt.array(full_data).float32()
        
        # Ensure tensor is not empty
        if feature_tensor.shape[0] == 0:
             print(f"Warning: Empty feature tensor for Case_id {case_id}. Returning zeros.")
             feature_tensor = jt.zeros((1, feature_tensor.shape[1]), dtype=jt.float32)

        resampled_tensor = self._resample_tensor(feature_tensor, self.seq_len)
        
        # 5. Prepare label tensor
        label_tensor = np.array(label, dtype=np.int64)
        # label_tensor = jt.array(label).int64() # Using int64 (Long) to match Torch
        
        # 6. Instance normalization
        resampled_tensor = self.instance_norm(resampled_tensor)
        
        # 7. Sample feature dimensions if feature_sampler is provided
        if self.feature_sampler is not None:
            if isinstance(self.feature_sampler, int):
                # Sample to fixed dimension length
                if resampled_tensor.shape[1] > self.feature_sampler:
                    # Use random sampling if there are more features than requested
                    np.random.seed(42)  # For reproducibility
                    selected_indices = np.random.choice(
                        resampled_tensor.shape[1], 
                        size=self.feature_sampler, 
                        replace=False
                    )
                    resampled_tensor = resampled_tensor[:, selected_indices]
            elif isinstance(self.feature_sampler, list) or isinstance(self.feature_sampler, np.ndarray):
                # Select specific feature indices
                resampled_tensor = resampled_tensor[:, self.feature_sampler]
            else:
                raise ValueError(f"feature_sampler must be int, list, or np.ndarray, got {type(self.feature_sampler).__name__}")
        padding_mask = np.ones(self.seq_len, dtype=np.float32)
        return resampled_tensor, label_tensor, padding_mask