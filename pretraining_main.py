import argparse
from models import load_tokenizer
import torch
from dataset import dataset
from torch.utils.data import DataLoader, random_split


def get_args():
    parser = argparse.ArgumentParser(description='PhyLLM scripts of model pre-training')

    # -------------------------------- Pre-training Group-----------------------------
    preprocess_args = parser.add_argument_group('Model pre-training parameters')
    preprocess_args.add_argument('--tokenizer_weight_path', type=str, help='Path to pretrained model')
    parser.add_argument('--signal_total_length', default=1500, type=int, help='Total length L of input signal per channel')
    parser.add_argument('--tokenizer_patch_size', default=300, type=int, help='Patch size T for ECGTokenizer rearrange')
    parser.add_argument('--num_channels', default=12, type=int, help='Number of ECG channels N')
    parser.add_argument('--overlap_ratio', default=0.5, type=float, help='Overlap ratio of input signal')


    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("--- Model Pre-training Stage ---")
    # ------------ load pre-trained tokenizer -----------
    frozen_tokenizer = load_tokenizer.load_pretrained_tokenizer(args)

    device = torch.device(args.device)

    # ------------ load dataset and build dataloader ------------
    aggregated_dataset_2 = dataset.AggregatedECGDataset(
        file_paths=args.processed_data_paths,
        window_size=args.signal_total_length,
        stride_size=int(args.signal_total_length * args.overlap_ratio),
        dataset_key='ecg_data',
    )

    dataset_size = len(aggregated_dataset_2)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(aggregated_dataset_2, [train_size, val_size])

    train_sampler = None
    val_sampler = None
    shuffle_train = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )

    print(f"\nCreated DataLoader instances:")
    print(f"Train DataLoader: {len(train_dataloader)} batches of size {args.batch_size} (approx)")
    print(f"Validation DataLoader: {len(val_dataloader)} batches of size {args.batch_size} (approx)")





