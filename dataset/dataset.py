import h5py
import bisect
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset


class AdaptedECGDataset(Dataset):
    """
    Reads a single HDF5 file containing ECG data,
    adapted from SingleShockDataset.
    """

    def __init__(self,
                 file_path: Union[str, Path],
                 window_size: int = 300,  # Default to 1 second at 300Hz
                 stride_size: int = 1,
                 start_percentage: float = 0.0,
                 end_percentage: float = 1.0,
                 dataset_key: str = 'ecg_data',  # Key for the ECG data in HDF5
                 ):
        '''
        Extracts datasets from file_path.

        :param file_path: Path to the target HDF5 data file.
        :param window_size: Length of a single sample (in time points).
        :param stride_size: Interval between two adjacent samples.
        :param start_percentage: Percentage index of the first sample in the data file (inclusive).
        :param end_percentage: Percentage index of the end of the dataset sample in the data file (exclusive).
        :param dataset_key: The key/name of the dataset within HDF5 groups (e.g., 'ecg_data').
        '''
        self.__file_path = Path(file_path)  # Ensure it's a Path object
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__dataset_key = dataset_key

        self.__file = None
        self.__length = 0  # Initialize to 0
        self.__feature_size = None

        self.__group_names = []  # Renamed from __subjects for clarity
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        try:
            self.__file = h5py.File(str(self.__file_path), 'r')
        except Exception as e:
            print(f"Error opening HDF5 file {self.__file_path}: {e}")
            # Raise the exception or handle it as per your application's needs
            raise

        self.__group_names = [key for key in self.__file.keys()]
        if not self.__group_names:
            print(f"Warning: No groups (subjects/records) found in HDF5 file: {self.__file_path}")
            self.__length = 0
            return

        global_idx = 0
        for group_name in self.__group_names:
            if self.__dataset_key not in self.__file[group_name]:
                print(
                    f"Warning: Dataset key '{self.__dataset_key}' not found in group '{group_name}' in file {self.__file_path}. Skipping this group.")
                continue

            self.__global_idxes.append(global_idx)  # Start index of samples from this group

            # Assuming data shape is (num_channels, num_time_points)
            # The second dimension (index 1) is the signal length
            try:
                signal_length = self.__file[group_name][self.__dataset_key].shape[1]
            except IndexError:  # Handle cases where dataset might be 1D or have unexpected shape
                print(
                    f"Warning: Dataset '{self.__dataset_key}' in group '{group_name}' has an unexpected shape: {self.__file[group_name][self.__dataset_key].shape}. Expected 2D (channels, samples). Skipping this group.")
                # Remove the last added global_idx if this group is skipped
                if self.__global_idxes and self.__global_idxes[-1] == global_idx:
                    self.__global_idxes.pop()
                continue

            # Total number of possible windows
            if signal_length < self.__window_size:
                print(
                    f"Warning: Signal length ({signal_length}) in group '{group_name}' is less than window_size ({self.__window_size}). No windows can be extracted. Skipping this group.")
                if self.__global_idxes and self.__global_idxes[-1] == global_idx:
                    self.__global_idxes.pop()
                continue

            total_sample_num = (signal_length - self.__window_size) // self.__stride_size + 1
            if total_sample_num <= 0:
                print(
                    f"Warning: No valid windows for group '{group_name}' with signal_length={signal_length}, window_size={self.__window_size}, stride_size={self.__stride_size}. Skipping.")
                if self.__global_idxes and self.__global_idxes[-1] == global_idx:
                    self.__global_idxes.pop()
                continue

            # Calculate start and end indices for windowing based on percentages
            actual_start_sample_idx = int(total_sample_num * self.__start_percentage)
            actual_end_sample_idx = int(total_sample_num * self.__end_percentage)  # exclusive end

            # Convert sample indices back to time point indices in the original signal
            start_time_point_idx = actual_start_sample_idx * self.__stride_size
            # The last possible start for a window is (actual_end_sample_idx - 1) * stride_size
            # So, the number of windows to consider is (actual_end_sample_idx - actual_start_sample_idx)

            num_windows_in_segment = actual_end_sample_idx - actual_start_sample_idx

            if num_windows_in_segment <= 0:
                # print(f"Debug: No windows for group {group_name} after percentage slicing.")
                if self.__global_idxes and self.__global_idxes[-1] == global_idx:  # clean up if no windows
                    self.__global_idxes.pop()
                continue

            self.__local_idxes.append(start_time_point_idx)  # Store the starting time point for this group's segment
            global_idx += num_windows_in_segment

        self.__length = global_idx

        # Determine feature size from the first valid group and dataset
        for group_name in self.__group_names:
            if self.__dataset_key in self.__file[group_name]:
                try:
                    data_shape = self.__file[group_name][self.__dataset_key].shape
                    if len(data_shape) >= 2:  # Expecting at least (num_channels, num_samples)
                        self.__feature_size = [data_shape[0], self.__window_size]
                        break
                except Exception as e:
                    print(f"Could not determine feature_size from group {group_name}: {e}")

        if self.__feature_size is None and self.__length > 0:
            print(
                f"Warning: Could not determine feature_size for {self.__file_path}, but dataset length is {self.__length}.")
        elif self.__length == 0:
            print(f"Info: Dataset at {self.__file_path} is empty after initialization.")

    @property
    def feature_size(self):
        """Returns the shape of a single window: [num_channels, window_size]"""
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__length:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__length}")

        # Find which group this global index falls into
        # bisect_left returns the insertion point, so group_idx-1 is the correct group
        group_idx = bisect.bisect_right(self.__global_idxes, idx) - 1

        if group_idx < 0 or group_idx >= len(self.__group_names):
            # This case should ideally not be reached if __length is calculated correctly
            # and __global_idxes is populated correctly.
            raise RuntimeError(f"Could not map global index {idx} to a valid group_idx. "
                               f"Global_idxes: {self.__global_idxes}, num_groups: {len(self.__group_names)}")

        group_name = self.__group_names[group_idx]

        # Calculate the local window index within this group's segment
        local_window_idx = idx - self.__global_idxes[group_idx]

        # Calculate the starting time point for this specific window in the original signal
        item_start_time_point = self.__local_idxes[group_idx] + (local_window_idx * self.__stride_size)

        # Retrieve the data window
        # Data is assumed to be (num_channels, time_points)
        data_window = self.__file[group_name][self.__dataset_key][:,
                      item_start_time_point: item_start_time_point + self.__window_size]
        return data_window

    def free(self) -> None:
        """Closes the HDF5 file."""
        if self.__file:
            self.__file.close()
            self.__file = None
            print(f"HDF5 file {self.__file_path} closed.")

    def __del__(self):
        """Ensures the file is closed when the object is deleted."""
        self.free()


class AggregatedECGDataset(Dataset):
    """
    Integrates multiple HDF5 files by creating an AdaptedECGDataset for each.
    Similar to ShockDataset from LaBraM.
    """

    def __init__(self,
                 file_paths: List[Union[str, Path]],  # list contains some HDF5 paths
                 window_size: int = 300,
                 stride_size: int = 1,
                 start_percentage: float = 0.0,
                 end_percentage: float = 1.0,
                 dataset_key: str = 'ecg_data',
                 ):
        '''
        :param file_paths: List of paths to HDF5 data files.
        :param window_size: Length of a single sample.
        :param stride_size: Interval between two adjacent samples.
        :param start_percentage: Start percentage for data extraction from each file.
        :param end_percentage: End percentage for data extraction from each file.
        :param dataset_key: The key for ECG data within HDF5 groups.
        '''
        self.__file_paths = [Path(fp) for fp in file_paths]
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__dataset_key = dataset_key

        self.__datasets: List[AdaptedECGDataset] = []
        self.__length = 0
        self.__feature_size = None
        self.__dataset_idxes = []  # Stores the starting global index for each sub-dataset

        self.__init_dataset()

    def __init_dataset(self) -> None:
        current_total_length = 0
        for file_path in self.__file_paths:
            if not file_path.exists():
                print(f"WARNING: cannot find {file_path}, skip it.")
                continue

            # Create an instance of AdaptedECGDataset for each HDF5 file
            single_dataset = AdaptedECGDataset(
                file_path=file_path,
                window_size=self.__window_size,
                stride_size=self.__stride_size,
                start_percentage=self.__start_percentage,
                end_percentage=self.__end_percentage,
                dataset_key=self.__dataset_key,
            )

            if len(single_dataset) > 0:
                self.__datasets.append(single_dataset)
                self.__dataset_idxes.append(current_total_length)
                current_total_length += len(single_dataset)
            else:
                print(f"Message: Dataset {file_path} is empty or could not be loaded, skipped.")
                single_dataset.free()

        self.__length = current_total_length

        # Get feature_size from the first successfully loaded data set
        if self.__datasets:
            self.__feature_size = self.__datasets[0].feature_size
        else:
            print("Warning: No dataset was loaded successfully, AggregatedECGDataset is empty.")

    @property
    def feature_size(self):
        """Returns the shape of a single window: [num_channels, window_size]"""
        if self.__feature_size is None and self.__datasets:
            self.__feature_size = self.__datasets[0].feature_size
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__length:
            raise IndexError(f"Global index {idx} out of range, aggregate dataset total length {self.__length}")

        # Use bisect_right to find which subdataset this global index belongs to
        dataset_internal_idx = bisect.bisect_right(self.__dataset_idxes, idx) - 1

        if dataset_internal_idx < 0:
            raise RuntimeError(f"Unable to find subdataset for global index {idx}. dataset_idxes: {self.__dataset_idxes}")

        local_idx = idx - self.__dataset_idxes[dataset_internal_idx]

        return self.__datasets[dataset_internal_idx][local_idx]

    def free(self) -> None:
        """Closes all underlying HDF5 files."""
        for dataset in self.__datasets:
            dataset.free()
        print("All aggregated HDF5 files have been closed.")



# --- 验证代码 ---
if __name__ == '__main__':
    if __name__ == '__main__':
        hdf5_file_path_1 = Path(r"D:\database\ECG_12l\georgia\georgia.hdf5")
        hdf5_file_path_2 = Path(r"D:\database\ECG_12l\cpsc_2018\cpsc.hdf5")

        if not hdf5_file_path_1.exists():
            print(f"错误: 文件 {hdf5_file_path_1} 不存在！请检查路径。")
            exit()
        if not hdf5_file_path_2.exists():
            print(f"错误: 文件 {hdf5_file_path_2} 不存在！请检查路径。")
            exit()

        print(f"将使用以下HDF5文件进行测试: \n1. {hdf5_file_path_1}\n2. {hdf5_file_path_2}")

        # --- 第一个 AggregatedECGDataset 实例 (例如，只包含 Georgia 数据集) ---
        print(f"\n--- 测试 AggregatedECGDataset 实例 1 (仅 Georgia) ---")
        try:
            # 假设采样率为300Hz, 窗口大小为600 (2秒), 步长为300 (1秒)
            # dataset_key 和 channel_attr_key 需要根据您HDF5文件的实际情况调整
            aggregated_dataset_1 = AggregatedECGDataset(
                file_paths=[hdf5_file_path_1],  # 只传入一个文件
                window_size=600,
                stride_size=300,
                dataset_key='ecg_data',  # 假设HDF5中数据集名为 'ecg_data'
            )

            print(f"实例1 - 数据集总样本数: {len(aggregated_dataset_1)}")
            if len(aggregated_dataset_1) > 0:
                print(f"实例1 - 单个样本特征大小: {aggregated_dataset_1.feature_size}")

                print("\n实例1 - 获取前2个样本:")
                for i in range(min(2, len(aggregated_dataset_1))):
                    sample = aggregated_dataset_1[i]
                    print(f"  实例1 - 样本 {i} 形状: {sample.shape}")
            else:
                print("实例1 - 数据集为空。")
            aggregated_dataset_1.free()

        except Exception as e:
            print(f"测试实例1时发生错误: {e}")
            import traceback

            traceback.print_exc()

        # --- 第二个 AggregatedECGDataset 实例 (包含 Georgia 和 CPSC 数据集) ---
        print(f"\n--- 测试 AggregatedECGDataset 实例 2 (Georgia + CPSC) ---")
        try:
            aggregated_dataset_2 = AggregatedECGDataset(
                file_paths=[hdf5_file_path_1, hdf5_file_path_2],  # 传入两个文件
                window_size=600,
                stride_size=300,
                dataset_key='ecg_data',
            )

            print(f"实例2 - 数据集总样本数: {len(aggregated_dataset_2)}")

            if len(aggregated_dataset_2) > 0:
                print(f"实例2 - 单个样本特征大小: {aggregated_dataset_2.feature_size}")

                print("\n实例2 - 获取前几个样本:")
                num_samples_to_show = 5
                # 尝试获取一些样本，包括可能跨越文件边界的样本
                # 首先计算第一个文件在聚合数据集中的大致样本数，以测试边界
                temp_ds1_for_len_check = AdaptedECGDataset(
                    file_path=hdf5_file_path_1, window_size=600, stride_size=300,
                    dataset_key='ecg_data'
                )
                len_ds1 = len(temp_ds1_for_len_check)
                temp_ds1_for_len_check.free()
                print(f"(信息: 文件1 '{hdf5_file_path_1.name}' 包含约 {len_ds1} 个窗口)")

                indices_to_check = [0, 1]
                if len_ds1 > 1:
                    indices_to_check.append(len_ds1 - 1)  # 第一个文件的最后一个窗口
                if len(aggregated_dataset_2) > len_ds1:
                    indices_to_check.append(len_ds1)  # 第二个文件的第一个窗口
                if len(aggregated_dataset_2) > len_ds1 + 1:
                    indices_to_check.append(len_ds1 + 1)  # 第二个文件的第二个窗口
                if len(aggregated_dataset_2) - 1 not in indices_to_check and len(
                        aggregated_dataset_2) > 0:  # 确保最后一个也被检查
                    indices_to_check.append(len(aggregated_dataset_2) - 1)  # 整个聚合数据集的最后一个窗口

                indices_to_check = sorted(list(set(idx for idx in indices_to_check if idx < len(aggregated_dataset_2))))

                for i in indices_to_check:
                    try:
                        sample = aggregated_dataset_2[i]
                        print(f"  实例2 - 样本 {i} 形状: {sample.shape}")
                    except IndexError:
                        print(f"  实例2 - 样本 {i} 索引超出范围 (数据集长度: {len(aggregated_dataset_2)})")
                    except Exception as e_item:
                        print(f"  实例2 - 获取样本 {i} 时出错: {e_item}")


            else:
                print("实例2 - 数据集为空。")
            aggregated_dataset_2.free()

        except Exception as e:
            print(f"测试实例2时发生错误: {e}")
            import traceback

            traceback.print_exc()

        print("\n--- HDF5读取测试结束 ---")