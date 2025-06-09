import torch
import torch.nn as nn
import numpy as np
import sys
from einops import rearrange
from train import utils
from models.tokenizer import ECGTokenizer
from models.transformer import NeuralTransformerForPretraining


def train_one_epoch_pretrain(
        main_model: nn.Module,  # 您的 ECGLabramForPretraining 实例
        frozen_tokenizer: nn.Module,  # 预训练好的、冻结的 ECGTokenizer
        data_loader_list: list,  # 包含一个或多个 DataLoader 的列表
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,  # NativeScaler
        log_writer=None,
        start_steps=None,
        lr_schedule_values=None,
        args=None,
):
    """
    ECG-LaBraM主模型的单轮训练函数。
    """
    main_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}] (Pre-training)'
    print_freq = args.print_freq if hasattr(args, 'print_freq') else 20

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    step_offset = 0  # 用于在多个dataloader之间累计步数

    # 1. 遍历每个DataLoader
    for data_loader in data_loader_list:

        # 2. 遍历当前DataLoader中的所有数据批次
        for step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

            it = start_steps + step + step_offset  # 计算全局训练步数

            # 更新学习率
            if lr_schedule_values is not None and it < len(lr_schedule_values):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

            # --- 核心预训练逻辑开始 ---

            # 将数据移动到指定设备
            if isinstance(batch_data, (list, tuple)):
                ecg_signals = batch_data[0]
            else:
                ecg_signals = batch_data
            ecg_signals = ecg_signals.float().to(device, non_blocking=True)

            # a. 获取目标Token IDs (在torch.no_grad()环境下)
            with torch.no_grad():
                # ECGTokenizer的forward方法将(B, N, L)转换为(B, N, A, T)
                # Tokenizer内部的rearrange和patch_embed处理形状
                input_ids = frozen_tokenizer.get_codebook_indices(ecg_signals)
                # input_ids 的形状: (B, NumPatches), 例如 (64, 360)

            # b. 随机掩码
            # 生成一个布尔掩码 bool_masked_pos，True表示被掩码的位置
            # LaBraM使用的方法是生成随机噪声然后取top-k
            noise = torch.rand(input_ids.shape, device=device)
            num_patches_to_mask = int(args.mask_ratio * input_ids.shape[1])
            masked_indices = torch.topk(noise, k=num_patches_to_mask, dim=1).indices
            bool_masked_pos = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
            bool_masked_pos.scatter_(1, masked_indices, True)

            # c. 准备标签 (Labels)
            labels = input_ids[bool_masked_pos]

            # d. 主模型前向传播
            # 模型输入是原始的、连续的ECG信号和布尔掩码
            # 模型内部会将掩码位置的patch替换为mask_token
            # ecg_signals 在ECGTokenizer中可能被reshape了，我们需要确保
            # 主模型接收的也是同样格式的输入 (B, N, A, T)
            # 假设 tokenizer_segment_length (T_segment) 是主模型TemporalEncoder的输入长度
            reshaped_ecg_signals = rearrange(ecg_signals, 'b n (a t) -> b n a t', t=args.tokenizer_segment_length)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # ECGLabramForPretraining.forward 返回 (student_preds, teacher_preds)
                # student_preds 是学生模型对掩码位置的预测
                student_preds, teacher_preds = main_model(reshaped_ecg_signals, bool_masked_pos)

                # e. 计算损失
                # LaBraM使用对称掩码策略 (Symmetric Masking) 来提升效率
                # 学生模型预测被掩码的部分 (mask_ratio)
                loss_student = criterion(student_preds, labels)

                # 教师模型预测未被掩码的部分 (1 - mask_ratio)
                # 这需要我们准备反向掩码的标签
                labels_sym = input_ids[~bool_masked_pos]  # 使用反向掩码
                loss_teacher = criterion(teacher_preds, labels_sym)

                # 总损失是两者之和
                loss = loss_student + loss_teacher

            loss_value = loss.item()

            if not np.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at epoch {epoch}, step {it}")
                sys.exit(1)

            # f. 反向传播与优化
            optimizer.zero_grad()
            # loss_scaler 会自动处理混合精度下的梯度缩放
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad if hasattr(args, 'clip_grad') else None,
                        parameters=main_model.student.parameters(), create_graph=False)

            # --- 核心预训练逻辑结束 ---

            # 日志记录
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # 计算准确率 (预测正确的token占所有被掩码token的比例)
            with torch.no_grad():
                acc_student = (student_preds.max(-1)[1] == labels).float().mean().item()
                acc_teacher = (teacher_preds.max(-1)[1] == labels_sym).float().mean().item()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_student=loss_student.item())
            metric_logger.update(loss_teacher=loss_teacher.item())
            metric_logger.update(acc_student=acc_student)
            metric_logger.update(acc_teacher=acc_teacher)

            current_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=current_lr)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="train/loss")
                log_writer.update(loss_student=loss_student.item(), head="train/loss")
                log_writer.update(loss_teacher=loss_teacher.item(), head="train/loss")
                log_writer.update(acc_student=acc_student, head="train/acc")
                log_writer.update(acc_teacher=acc_teacher, head="train/acc")
                log_writer.update(lr=current_lr, head="opt")
                log_writer.set_step()

        step_offset += len(data_loader)

    # 打印整个epoch的平均统计信息
    metric_logger.synchronize_between_processes()  # 在单机模式下此为空操作
    print("Averaged stats for epoch:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pretrain(
        main_model: nn.Module,
        frozen_tokenizer: nn.Module,
        data_loader_list: list,
        device: torch.device,
        epoch: int,
        log_writer=None,
        args=None,
):
    """
    在验证集上评估ECG-LaBraM主模型的性能。
    """
    main_model.eval()  # 将模型设置为评估模式
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Val Epoch: [{epoch}] (Pre-training)'
    criterion = torch.nn.CrossEntropyLoss()

    # 遍历每个验证集的 DataLoader
    for data_loader in data_loader_list:
        for batch_data in metric_logger.log_every(data_loader, 10, header):
            # 加载数据到设备
            if isinstance(batch_data, (list, tuple)):
                ecg_signals = batch_data[0]
            else:
                ecg_signals = batch_data
            ecg_signals = ecg_signals.float().to(device, non_blocking=True)

            # 获取目标Token IDs
            # 注意：这里的tokenizer和模型输入的reshape逻辑需要与训练时完全一致
            input_ids = frozen_tokenizer.get_codebook_indices(ecg_signals)

            # 随机掩码 (在验证时也需要掩码来评估预测能力)
            noise = torch.rand(input_ids.shape, device=device)
            num_patches_to_mask = int(args.mask_ratio * input_ids.shape[1])
            masked_indices = torch.topk(noise, k=num_patches_to_mask, dim=1).indices
            bool_masked_pos = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
            bool_masked_pos.scatter_(1, masked_indices, True)

            labels = input_ids[bool_masked_pos]

            # 调整输入形状以匹配主模型
            reshaped_ecg_signals = rearrange(ecg_signals, 'b n (a t) -> b n a t', t=args.tokenizer_segment_length)

            # 模型前向传播
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # 在评估时，我们通常只关心学生模型的性能
                # ECGLabramForPretraining的forward应能处理active_teacher=False的情况
                # 或者我们直接调用学生模型
                if hasattr(main_model, 'student'):
                    student_preds = main_model.student(reshaped_ecg_signals, bool_masked_pos)
                else:  # 如果模型没有学生/教师结构，直接调用
                    student_preds = main_model(reshaped_ecg_signals, bool_masked_pos)

                loss = criterion(student_preds, labels)

            # 计算准确率
            acc = (student_preds.max(-1)[1] == labels).float().mean().item()

            # 更新日志
            metric_logger.update(loss=loss.item())
            metric_logger.update(accuracy=acc)

    # 聚合和打印日志
    metric_logger.synchronize_between_processes()  # 在单机模式下为空操作
    print(f"Averaged validation stats for epoch {epoch}:", metric_logger)

    # 准备返回的统计数据
    final_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.update(**{f"val_epoch/{k}": v for k, v in final_stats.items()}, step=epoch)

    return final_stats


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """
    保存模型检查点。
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_paths = [output_dir / f'checkpoint-{epoch_name}.pth']

    # 如果是最佳模型，额外保存一份
    if epoch_name == "best":
        checkpoint_paths.append(output_dir / 'checkpoint-best.pth')

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
        torch.save(to_save, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")