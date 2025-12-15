import torch
from torch.utils.data import DataLoader
import jittor as jt
import argparse
import os
import numpy as np

# 假设 mental.py 和 mental_jt.py 在同一目录下
from mental import MentalLoader
from mental_jt import MentalLoader as MentalLoaderJT

data_dict = {
    'Mental': MentalLoader,
    'Mental_jt': MentalLoaderJT,
}

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step).
    (PyTorch helper function)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def collate_fn(data, max_len=None):
    """
    PyTorch Collate function to handle variable length (though MentalLoader resamples to fixed length)
    and generate padding masks.
    """
    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features
    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    # Handle labels
    # If labels are 0-d tensors (scalars), stack them.
    targets = torch.stack(labels, dim=0)

    # Generate masks
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    return X, targets, padding_masks

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if args.data == 'Mental_jt':
        # --- Jittor Logic ---
        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
        
        # Instantiate Jittor Dataset
        dataset = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )
        
        # Jittor dataset behaves like a DataLoader when set_attrs is used
        dataset.set_attrs(
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False # usually False for val/test, adjustable if needed
        )
        
        # For Jittor, the dataset IS the loader
        return dataset, dataset

    else:
        # --- PyTorch Logic ---
        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
        drop_last = False # usually False to evaluate all data
        
        dataset = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        
        return dataset, data_loader


# ==========================================
# Main Function for Testing
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mental Health Data Provider Test')
    
    # Argument definition matches what MentalLoader expects
    parser.add_argument('--root_path', type=str, default='/data2/lx/mental/preprocess/output', help='root path of the data')
    parser.add_argument('--data', type=str, default='Mental', help='Mental or Mental_jt')
    parser.add_argument('--features', type=str, default='crop_clip', help='feature type')
    parser.add_argument('--target', type=str, default='depression', help='target label')
    parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--enc_in', type=int, default=None, help='feature dimension sampler (int or None)')

    args = parser.parse_args()

    print(f"Testing with Data: {args.data}")
    
    # Create dummy data for testing if not exists (Optional structure check)
    if not os.path.exists(os.path.join(args.root_path, 'data.csv')):
        print(f"Error: {os.path.join(args.root_path, 'data.csv')} not found.")
        print("Please ensure your dataset path is correct before running the test.")
        exit(1)

    try:
        # 1. Get Provider
        dataset, loader = data_provider(args, flag='TRAIN')
        print(f"Dataset Size: {len(dataset)}")

        # 2. Iterate through one batch
        print("Iterating through one batch...")
        
        if args.data == 'Mental_jt':
            # Jittor Iteration
            for i, (batch_x, batch_y) in enumerate(loader):
                print(f"\n[Jittor Batch {i}]")
                print(f"Input Shape (batch, seq, feat): {batch_x.shape}")
                print(f"Label Shape: {batch_y.shape}")
                print(f"Input Type: {type(batch_x)}")
                
                # Check for NaNs
                if jt.isnan(batch_x).any():
                    print("Warning: NaNs found in input!")
                
                break # Only test one batch
        else:
            # PyTorch Iteration
            for i, (batch_x, batch_y, batch_mask) in enumerate(loader):
                print(f"\n[PyTorch Batch {i}]")
                print(f"Input Shape (batch, seq, feat): {batch_x.shape}")
                print(f"Label Shape: {batch_y.shape}")
                print(f"Mask Shape: {batch_mask.shape}")
                print(f"Input Type: {type(batch_x)}")
                
                break # Only test one batch

        print("\nTest passed successfully!")

    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        import traceback
        traceback.print_exc()