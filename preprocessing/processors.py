import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import argparse
from scipy.signal import butter, filtfilt, iirnotch, resample
import h5py

class H5Dataset:
    def __init__(self, path, name: str): # path 可以是 str 或 Path
        self.__name = name
        _path_obj = Path(path) # 在这里转换为 Path 对象
        self.__file_path = _path_obj / f'{name}.hdf5'
        _path_obj.mkdir(parents=True, exist_ok=True)
        self.__f = None

    def _open_if_not(self):
        if self.__f is None:
            import h5py
            self.__f = h5py.File(self.__file_path, 'a')

    def add_group(self, grp_name: str):
        self._open_if_not()
        return self.__f.create_group(grp_name)

    def add_dataset(self, grp, ds_name: str, arr: np.ndarray, chunks: tuple = None):
        self._open_if_not()
        if chunks is None:
            default_chunk_len = 256
            if arr.ndim == 1:
                current_chunk_len = arr.shape[0] if arr.shape[0] > 0 else default_chunk_len
                chunks = (min(default_chunk_len, current_chunk_len),)
            elif arr.ndim == 2:
                current_samples = arr.shape[1] if arr.shape[1] > 0 else default_chunk_len
                num_channels = arr.shape[0] if arr.shape[0] > 0 else 1
                chunks = (num_channels, min(default_chunk_len, current_samples))
            else:
                chunks = True
        try:
            return grp.create_dataset(ds_name, data=arr, chunks=chunks)
        except TypeError as e:
            print(f"创建数据集 '{ds_name}' 时发生分块错误 (chunks={chunks}, data_shape={arr.shape}): {e}")
            print("将尝试不使用分块创建。")
            return grp.create_dataset(ds_name, data=arr)

    def add_attributes(self, src, attr_name: str, attr_value):
        self._open_if_not()
        if isinstance(attr_value, list) and all(isinstance(item, str) for item in attr_value):
            import h5py
            attr_value = np.array(attr_value, dtype=h5py.string_dtype(encoding='utf-8'))
        src.attrs[f'{attr_name}'] = attr_value

    def save(self): # 在LaBraM的实现中，这个方法是关闭文件
        if self.__f is not None:
            self.__f.close()
            self.__f = None
        print(f"HDF5 文件已保存到: {self.__file_path}")

    @property
    def name(self):
        return self.__name

    @property
    def file_path(self):
        return self.__file_path


class BaseProcessor:
    def __init__(self, args: argparse.Namespace):
        self.raw_data_path = args.raw_data_path
        self.processed_data_path = args.processed_data_path

        self.target_sfreq = args.rsfreq
        self.lowcut = args.l_freq
        self.highcut = args.h_freq
        self.powerline_freq = getattr(args, 'powerline_freq', 50.0)
        self.dataset_h5_name = args.dataset_name

        self.h5_writer = H5Dataset(self.processed_data_path, self.dataset_h5_name)

        self.overall_stats = {
            "total_records_found": 0,
            "successfully_processed_records": 0,
            "total_original_duration_seconds": 0.0,
            "failed_records": []
        }
        print(f"简化处理器初始化: 原始数据路径 '{self.raw_data_path}'")
        print(f"处理参数: 目标采样率={self.target_sfreq}Hz, 滤波范围=[{self.lowcut}Hz, {self.highcut}Hz], "
              f"工频干扰={self.powerline_freq}Hz")


    def _butter_bandpass_filter(self, data: np.ndarray, fs: int, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        can_lowpass = self.highcut > 0 and self.highcut < nyq
        can_highpass = self.lowcut > 0 and self.lowcut < nyq

        if can_bandpass := (can_lowpass and can_highpass and self.lowcut < self.highcut):
            low = self.lowcut / nyq
            high = self.highcut / nyq
            b, a = butter(order, [low, high], btype='band', analog=False)
        elif can_highpass:
            low = self.lowcut / nyq
            b, a = butter(order, low, btype='high', analog=False)
        elif can_lowpass:
            high = self.highcut / nyq
            b, a = butter(order, high, btype='low', analog=False)
        else:
            return data # 不滤波

        if len(data) <= order * 3:
            print(f"警告: 数据长度 {len(data)} 过短，无法进行阶数为 {order} 的滤波。跳过滤波。")
            return data
        y = filtfilt(b, a, data)
        return y

    def _notch_filter(self, data: np.ndarray, fs: int, quality_factor: float = 30.0) -> np.ndarray:
        if self.powerline_freq <= 0:
            return data
        nyq = 0.5 * fs
        freq = self.powerline_freq / nyq
        if not (0 < freq < 1):
            return data

        if len(data) <= 8:
            print(f"警告: 数据长度 {len(data)} 过短，无法进行陷波滤波。跳过。")
            return data
        b, a = iirnotch(freq, quality_factor)
        y = filtfilt(b, a, data)
        return y

    def resample_waveform(self, original_sfreq, signal):
        num_original_samples = len(signal)
        num_target_samples = int(num_original_samples * (self.target_sfreq / original_sfreq))
        if num_target_samples == 0 and num_original_samples > 0:
            num_target_samples = 1

        if num_original_samples > 0 and num_target_samples > 0:
             resampled_data = resample(signal, num_target_samples)
        elif num_original_samples > 0 and num_target_samples == 0:
             resampled_data = np.array([])
             print(f"警告: 重采样目标长度为0，原始长度{num_original_samples}。信号变为空。")
        else:
             resampled_data = np.array([])
        return resampled_data

    def normalize_to_minus_one_to_one(self, data):
        """将单通道信号标准化到 [-1, 1] 区间。"""
        if data.size == 0 or np.all(data == data[0]):
            return data

        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            return np.zeros_like(data)

        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized_data

    def interpolate_nan_multichannel(self, sig):
        # sig: shape (channels, time)
        interpolated = []
        for channel in sig:
            interpolated_channel = pd.Series(channel).interpolate(method='linear', limit_direction='both').to_numpy()
            interpolated.append(interpolated_channel)
        return np.array(interpolated)

    def process_single_record(self, sig, fs, time):
        num_samples_original, num_original_channels = sig.shape
        processed_channels_data = []

        for i in range(num_original_channels):
            channel_data = sig[:, i]
            # filter
            filtered_channel_data = self._butter_bandpass_filter(channel_data, fs)
            filtered_channel_data = self._notch_filter(filtered_channel_data, fs)

            # resample
            resampled_channel_data = self.resample_waveform(fs, filtered_channel_data)

            # Normalize
            normalized_channel_data = self.normalize_to_minus_one_to_one(resampled_channel_data)

            processed_channels_data.append(normalized_channel_data)

        min_len = min(len(ch) for ch in processed_channels_data)
        processed_signal_array = np.array([ch[:min_len] for ch in processed_channels_data])

        processed_signal_array = self.interpolate_nan_multichannel(processed_signal_array)

        if np.any(np.isnan(processed_signal_array)) or np.any(np.isinf(processed_signal_array)):
            print(
                f"Info: NaNs or Infs still present after interpolation or were Infs. Applying np.nan_to_num to zero them out.")
            processed_signal_array = np.nan_to_num(processed_signal_array, nan=0.0, posinf=0.0, neginf=0.0)

        metadata = {"original_sfreq": fs,
                    "duration_seconds_original": time,
                    "processed_signal": processed_signal_array,
                    "target_sfreq": self.target_sfreq}
        return metadata


    def process_record(self):
        # read record
        dataset_path = self.raw_data_path
        for subject_name in os.listdir(dataset_path):
            subject_path = os.path.join(dataset_path, subject_name)
            record_names = set(os.path.splitext(f)[0] for f in os.listdir(subject_path))
            for wave_name in record_names:
                wave_path = os.path.join(subject_path, wave_name)
                try:
                    record = wfdb.rdrecord(wave_path)
                except Exception as e:
                    print(f"读取记录 {wave_path} 时发生错误: {e}")
                    continue
                fs = record.fs
                time = record.sig_len / fs
                waves = record.p_signal

                # preprocessing
                meta_data = self.process_single_record(waves, fs, time)

                # save meta_data
                processed_signal_to_save = meta_data.pop("processed_signal")
                h5_group_name = f"{subject_name}_{wave_name}"
                try:
                    grp = self.h5_writer.add_group(h5_group_name)
                    # 假设 processed_signal_to_save 是 (channels, samples) 格式
                    dset = self.h5_writer.add_dataset(grp, 'ecg_data', processed_signal_to_save)

                    # 将 meta_data 中剩余的项作为属性存储
                    for key, value in meta_data.items():
                        if value is not None:  # 确保值不是 None
                            try:
                                self.h5_writer.add_attributes(dset, key, value)
                            except Exception as e_attr:
                                print(f"  WARNING: Can't for {h5_group_name} save key: {key} (value: {value}): {e_attr}")

                    print(f"Successfully process waveforms and going to save: {h5_group_name}")
                    if hasattr(self, 'overall_stats'):
                        self.overall_stats["successfully_processed_records"] += 1
                        if "duration_seconds_original" in meta_data:  # 确保键存在
                            self.overall_stats["total_original_duration_seconds"] += meta_data[
                                "duration_seconds_original"]

                except Exception as e_h5:
                    print(f"  ERROR: in {h5_group_name} failed to save: {e_h5}")
                    if hasattr(self, 'overall_stats') and "failed_records" in self.overall_stats:
                        self.overall_stats["failed_records"].append(f"{h5_group_name} (HDF5 write error: {e_h5})")