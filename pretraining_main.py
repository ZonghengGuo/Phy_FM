import argparse
import json
import os
from datetime import time, datetime
from train.pretrain_engine import train_one_epoch_pretrain, evaluate_pretrain
from models import load_tokenizer
import torch
from dataset import dataset
from torch.utils.data import DataLoader, random_split
from train import utils
from models import transformer
from train.optim_factory import create_optimizer


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
    # Todo: 构建不同通道数的数据集
    args = get_args()
    print("--- Model Pre-training Stage ---")
    # ------------ load pre-trained tokenizer -----------
    frozen_tokenizer = load_tokenizer.load_pretrained_tokenizer(args)

    device = torch.device(args.device)

    # ------------ load dataset and build dataloader ------------
    datasets_train = [
        ["D:/database/ECG_12l/cpsc_2018/cpsc.hdf5", "D:/database/ECG_12l/georgia/georgia.hdf5"],
    ]


    signal_length_train = [
        1500, # 12 channels are matched with 1500 length
    ]

    dataset_list = []
    for file_paths, window_size in zip(datasets_train, signal_length_train):
        stride_size = int(window_size * args.overlap_ratio)
        dataset_obj = dataset.AggregatedECGDataset(
            file_paths=file_paths,
            window_size=window_size,
            stride_size=stride_size,
            dataset_key='data',  # 或者 args.dataset_key
        )
        if len(dataset_obj) > 0:
            dataset_list.append(dataset_obj)

    train_dataloader_list = []
    val_dataloader_list = []

    print("\n--- Preparing DataLoaders for each dataset group ---")
    for i, dataset_obj in enumerate(dataset_list):
        print(f"\nProcessing dataset group {i + 1} with {len(dataset_obj)} samples.")
        dataset_size = len(dataset_obj)
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size

        if train_size <= 0 or val_size <= 0:
            print(f"Warning: Dataset group {i + 1} is too small to split. Skipping this group.")
            continue

        train_subset, val_subset = random_split(
            dataset_obj,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        # 为这个子集创建 DataLoader
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        train_dataloader_list.append(train_loader)
        val_dataloader_list.append(val_loader)

    if not train_dataloader_list:
        print("ERROR: No valid DataLoaders were created. Check dataset paths and sizes.")
        exit()

    print(f"\nCreated {len(train_dataloader_list)} group(s) of DataLoaders.")

    num_A_segments = args.signal_total_length // args.tokenizer_segment_length
    student_core_model = transformer.NeuralTransformerForMaskedPhyModeling(
        num_channels=args.num_channels,
        max_time_segments=num_A_segments,
        segment_length=args.tokenizer_segment_length,
        temporal_conv_out_chans=args.labram_temporal_conv_out_chans,
        vocab_size=args.codebook_size,
        embed_dim=args.labram_embed_dim,
        depth=args.labram_depth,
        num_heads=args.labram_num_heads,
        mlp_ratio=args.labram_mlp_ratio,
        drop_path_rate=args.labram_drop_path_rate,
        init_values=args.labram_layer_scale_init_value
    )

    # 2. 用包装类创建包含学生和教师的完整预训练模型
    # LaBraM的默认momentum是0.996
    pretraining_model = transformer.NeuralTransformerForPretraining(student_core_model, momentum=0.996)

    pretraining_model.to(device)
    model_without_ddp = pretraining_model

    n_parameters = sum(p.numel() for p in pretraining_model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {n_parameters / 1e6:.2f} M")

    optimizer = create_optimizer(args, pretraining_model)
    # 创建学习率调度器 (使用正确的总步数)
    num_training_steps_per_epoch = sum(len(dl) for dl in train_dataloader_list)
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr_labram, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs_labram
    )

    # 创建loss_scaler
    loss_scaler = utils.NativeScalerWithGradNormCount()

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    print(f"Start pre-training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0  # 用于追踪最佳验证准确率

    for epoch in range(args.start_epoch, args.epochs):
        # (如果需要分布式，这里会有 sampler.set_epoch(epoch))

        # 调用训练引擎进行一轮训练
        train_stats = train_one_epoch_pretrain(
            main_model=pretraining_model,
            frozen_tokenizer=frozen_tokenizer,
            data_loader_list=train_dataloader_list,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            args=args
        )

        # --- 评估和模型保存逻辑 ---

        # 定期保存检查点
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            utils.save_model(
                args=args, epoch=epoch, model=pretraining_model,
                model_without_ddp=pretraining_model,  # 单机模式下 model 和 model_without_ddp 是同一个
                optimizer=optimizer, loss_scaler=loss_scaler)

        # 在验证集上评估模型
        val_stats = {}
        if val_dataloader_list:
            val_stats = evaluate_pretrain(
                main_model=pretraining_model,
                frozen_tokenizer=frozen_tokenizer,
                data_loader_list=val_dataloader_list,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args
            )
            print(f"Validation Accuracy of the model on the val set: {val_stats['accuracy']:.4f}")

            # 检查是否是当前最好的模型
            if val_stats['accuracy'] > max_accuracy:
                max_accuracy = val_stats['accuracy']
                if args.output_dir:
                    print(f"*** New best validation accuracy: {max_accuracy:.4f} ***")
                    utils.save_model(
                        args=args, epoch="best", model=pretraining_model,
                        model_without_ddp=pretraining_model,
                        optimizer=optimizer, loss_scaler=loss_scaler)

        # 记录日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))














