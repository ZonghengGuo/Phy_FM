import argparse
from preprocessing import processors


def get_args():
    parser = argparse.ArgumentParser(description='PhyLLM scripts：preprocessing、tokenizer training、pre-training')

    # -------------------------------- Preprocessing Group--------------------------------
    preprocess_args = parser.add_argument_group('Data preprocessing parameters')
    preprocess_args.add_argument('--dataset_name', type=str, help='list of the dataset names')
    preprocess_args.add_argument('--raw_data_path', type=str, help='list of dataset input paths')
    preprocess_args.add_argument('--processed_data_path', type=str, help='list of save path of HDF5 files')
    preprocess_args.add_argument('--l_freq', type=float, default=1.0, help='Low-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--h_freq', type=float, default=30.0, help='high-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--rsfreq', type=int, default=300, help='resampling rate (Hz)')

    return parser.parse_args()

if __name__ == '__main__':
    # Todo: 如果通道数目不一样，在这一步应该把通道数目划分整齐，同一个数据集内的通道数目应该一致
    args = get_args()

    print("Start processing dataset")
    database = processors.BaseProcessor(args)
    database.process_record()