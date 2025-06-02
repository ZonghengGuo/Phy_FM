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
    args = get_args()

    if args.dataset_name == "cpsc":
        print("Start processing CPSC dataset")
        cpsc = processors.BaseProcessor(args)
        cpsc.process_record()

    elif args.dataset_name == "georgia":
        print("Start processing Georgia dataset")
        georgia = processors.BaseProcessor(args)
        georgia.process_record()