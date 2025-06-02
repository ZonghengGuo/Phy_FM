import argparse
import json
import os
from datetime import datetime
from train import tokenizer_engine
from preprocessing import processors
from models import tokenizer
from train import utils
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from dataset import dataset
from torch.utils.data import DataLoader, Subset, random_split
from train.optim_factory import create_optimizer
from train.utils import NativeScalerWithGradNormCount as NativeScaler
import time


def get_args():
    parser = argparse.ArgumentParser(description='PhyLLM scripts：preprocessing、tokenizer training、pre-training')

    parser.add_argument('--stage', type=str, required=True,
                        choices=['preprocessing', 'train_tokenizer', 'pre-training', 'finetune'],
                        help='Specify the process stages to be executed: preprocess, train_tokenizer, pre-training, finetune。')

    # -------------------------------- Preprocessing Group--------------------------------
    preprocess_args = parser.add_argument_group('Data preprocessing parameters')
    preprocess_args.add_argument('--dataset_names', type=str, nargs='+', help='list of the dataset names')
    preprocess_args.add_argument('--raw_data_paths', type=str, nargs='+', help='list of dataset input paths')
    preprocess_args.add_argument('--processed_data_paths', type=str, nargs='+', help='list of save path of HDF5 files')
    preprocess_args.add_argument('--l_freq', type=float, default=1.0, help='Low-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--h_freq', type=float, default=30.0, help='high-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--rsfreq', type=int, default=300, help='resampling rate (Hz)')

    # -------------------------------- Tokenizer Training Group--------------------------------
    preprocess_args = parser.add_argument_group('Tokenizer preprocessing parameters')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    # ECGTokenizer Model parameters for encoder and decoder
    parser.add_argument('--signal_total_length', default=1500, type=int, help='Total length L of input signal per channel')
    parser.add_argument('--overlap_ratio', default=0.5, type=float, help='Overlap ratio of input signal')
    parser.add_argument('--tokenizer_patch_size', default=300, type=int, help='Patch size T for ECGTokenizer rearrange')
    parser.add_argument('--num_channels', default=12, type=int, help='Number of ECG channels N')
    parser.add_argument('--embed_dim_transformer', default=100, type=int, help='Embedding dimension for NeuralTransformer components')
    parser.add_argument('--depth_transformer_enc', default=12, type=int, help='Depth of NeuralTransformer encoder in ECGTokenizer')
    parser.add_argument('--depth_transformer_dec', default=3, type=int, help='Depth of NeuralTransformer decoder in ECGTokenizer')
    parser.add_argument('--num_heads_transformer', default=10, type=int, help='Number of attention heads for NeuralTransformer')

    # Quantizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='Number of embeddings in the codebook (K)')
    parser.add_argument('--codebook_embed_dim', default=100, type=int, help='Dimension of each embedding in the codebook (D)')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='EMA decay for quantizer')
    parser.add_argument('--quantize_kmeans_init', action='store_true', help='Enable kmeans_init for quantizer')
    parser.add_argument('--no_quantize_kmeans_init', action='store_false', dest='quantize_kmeans_init')
    parser.set_defaults(quantize_kmeans_init=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', default=[0.9, 0.99], type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='Learning rate') #Peak LR
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='Epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='Lower lr bound for cosine scheduler')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.stage == 'preprocessing':
        print("--- Preprocessing Stage ---")

        if "cpsc" in args.dataset_names:
            print("Start processing CPSC dataset")
            cpsc = processors.BaseProcessor(args, args.dataset_names[0], args.raw_data_paths[0], args.processed_data_paths[0])
            cpsc.process_record()

        if "georgia" in args.dataset_names:
            print("Start processing Georgia dataset")
            georgia = processors.BaseProcessor(args, args.dataset_names[1], args.raw_data_paths[1], args.processed_data_paths[1])
            georgia.process_record()

    elif args.stage == "train_tokenizer":
        print("--- Training Tokenizer Stage ---")

        # -------------------------------- Model Instantiation ------------------------------------
        # Todo: change args.num_channels to not fixed
        encoder_config = tokenizer.get_model_default_params(args.signal_total_length, args.tokenizer_patch_size, args.num_channels)
        decoder_config = tokenizer.get_model_default_params(args.signal_total_length, args.tokenizer_patch_size, args.num_channels)

        # modify decoder parameters
        decoder_config['sig_size'] = encoder_config['sig_size'] // encoder_config['patch_size']
        decoder_config['patch_size'] = 1
        decoder_config['depth'] = 3
        decoder_config['embed_dim'] = 33 * (12 // args.num_channels) + 1
        decoder_config['patch_embed'] = tokenizer.PatchEmbed(
            EEG_size=decoder_config['sig_size'],
            patch_size=decoder_config['patch_size'],
            in_chans=decoder_config['in_chans'],
            embed_dim=decoder_config['embed_dim']
        )

        print(args)
        device = torch.device(args.device)

        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        model = tokenizer.ECGTokenizer(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            n_embed=args.codebook_size,
            embed_dim=args.codebook_embed_dim,
            patch_size=args.tokenizer_patch_size,
            decoder_out_dim=args.tokenizer_patch_size  # Predict spectrum for each patch
        )

        model.to(device)
        model_without_ddp = model

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters: {n_parameters / 1e6:.2f} M")

        # -------------------------------- Dataset and Dataloader ------------------------------------
        aggregated_dataset_2 = dataset.AggregatedECGDataset(
            file_paths=args.processed_data_paths,
            window_size=args.signal_total_length,
            stride_size=int(args.signal_total_length*args.overlap_ratio),
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

        # -------------------------------- Training Model ------------------------------------
        if utils.get_rank() == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        optimizer = create_optimizer(args, model_without_ddp)
        loss_scaler = NativeScaler()

        num_training_steps_per_epoch = len(train_dataset) // (args.batch_size * utils.get_world_size())
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs
        )

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler
        )

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_val_metric = -float('inf')

        for epoch in range(args.epochs):
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch)

            train_stats = tokenizer_engine.train_one_epoch(
                model, train_dataloader, optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, args=args
            )

            if args.output_dir:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    save_ckpt_freq=args.save_ckpt_freq
                )

            val_stats = {}
            if val_dataloader is not None:
                val_stats = tokenizer_engine.evaluate(val_dataloader, model, device, log_writer, epoch, args)
                print(f"Validation loss of the network: {val_stats['loss']:.4f}")
                if log_writer is not None:
                    log_writer.update(**{f"val_{k}": v for k, v in val_stats.items()}, head="val_epoch",
                                      step=epoch)  # log_writer.update(**val_stats, head="val/loss")

                if val_stats['loss'] < (max_val_metric if max_val_metric != -float('inf') else float('inf')):
                    max_val_metric = val_stats['loss']
                    if args.output_dir:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                            save_ckpt_freq=1
                        )
                    print(f"New best val_loss: {max_val_metric:.4f}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))