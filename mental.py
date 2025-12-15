import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os




class MentalLoader(Dataset):
    """
    Dataset class for the Mental Health dataset.
    Loads features from specified subdirectories and labels from a central CSV file.
    Handles variable-length time series by resampling to a fixed length.

    Args:
        args (Namespace): A namespace object containing script arguments, must have `seq_len`.
        root_path (str): The root directory of the dataset.
        feature_type (str): The type of feature to load. Must be one of the subdirectory names.
                           e.g., 'crop_clip', 'csv', 'src_dino'.
        label_column (str): The name of the column in data.csv to be used as the label.
                            e.g., 'result-1', 'result-2'.
        flag (str): One of 'train', 'val', or 'test' to select the dataset split.
        feature_sampler (int or list, optional): Method for sampling feature dimensions.
                                                If int, sample to this fixed dimension length.
                                                If list, select specific feature indices.
                                                Defaults to None (use all features).
        
    Attributes:
        seq_len (int): The target sequence length for resampling.
        feature_type (str): The selected feature type.
        label_column (str): The selected label column.
        case_ids (list): A list of Case_ids for the current data split.
        labels_df (pd.DataFrame): A dataframe containing the labels for the case_ids in the split.
        feature_sampler (int or list or None): Method for sampling feature dimensions.
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
            # 1到5分，无风险。
            # 6到9分，低风险。
            # 10分以上，高风险。
        else:
            # Preprocess labels: replace -1 with 0
            all_labels_df[self.label_column] = all_labels_df[self.label_column].replace(-1, 0)
        
        # Use Case_id as the index for easy lookup
        all_labels_df.set_index('Case_id', inplace=True)
        
        
        # 2. Split data into train/val/test sets
        self._split_data(all_labels_df)
        print(f"Loaded {self.flag} set with {len(self.case_ids)} samples.")

    def _split_data(self, all_labels_df):
        all_case_ids = all_labels_df.index.unique().tolist()
        # filter some cases 如果不是suiside，则过滤掉Case_id非 纯数字字符串 的样本
        if self.label_column!="suiside":
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

    def _resample_tensor(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resamples a 2D tensor (T, F) to a new time dimension (target_len, F).
        """
        if tensor.shape[0] == target_len:
            return tensor
        # F.interpolate works on (N, C, L_in), so we reshape our (T, F) tensor
        # (T, F) -> (F, T) -> (1, F, T)
        tensor_reshaped = tensor.T.unsqueeze(0)
        # Resample
        resampled = F.interpolate(tensor_reshaped, size=target_len, mode='linear', align_corners=False)
        # Reshape back to (target_len, F)
        # (1, F, target_len) -> (F, target_len) -> (target_len, F)
        return resampled.squeeze(0).T
        
    def __len__(self):
        return len(self.case_ids)
    
    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
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
            raise FileNotFoundError(f"No feature files found for Case_id {case_id} in {feature_dir}")
            
        # 3. Concatenate if multiple video parts exist
        full_data = np.concatenate(data_parts, axis=0)
        # 4. Convert to tensor and resample
        feature_tensor = torch.from_numpy(full_data).float()
        
        # Ensure tensor is not empty
        if feature_tensor.shape[0] == 0:
             # Handle empty features, e.g., by returning zeros or raising error
             print(f"Warning: Empty feature tensor for Case_id {case_id}. Returning zeros.")
             feature_tensor = torch.zeros((1, feature_tensor.shape[1]), dtype=torch.float)

        resampled_tensor = self._resample_tensor(feature_tensor, self.seq_len)
        
        # 5. Prepare label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
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
        return resampled_tensor, label_tensor