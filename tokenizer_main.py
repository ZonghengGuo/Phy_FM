import argparse
from preprocessing import processors

def get_args():
    parser = argparse.ArgumentParser(description='PhyLLM scripts：preprocessing、tokenizer training、pre-training')

    # --- 阶段控制参数 ---
    parser.add_argument('--stage', type=str, required=True,
                        choices=['preprocessing', 'train_tokenizer', 'pre-training', 'finetune'],
                        help='指定要执行的流程阶段: preprocess, train_tokenizer, pre-training, finetune。')

    # -------------------------------- preprocessing group--------------------------------
    preprocess_args = parser.add_argument_group('数据预处理参数')
    preprocess_args.add_argument('--dataset_name', type=str,
                                 help='(预处理阶段) the name of dataset。')
    preprocess_args.add_argument('--raw_data_path', type=str,
                                 help='(预处理阶段) dataset input path。')
    preprocess_args.add_argument('--processed_data_path', type=str,
                                 help='(预处理阶段) save path of HDF5 files')
    preprocess_args.add_argument('--l_freq', type=float, default=1.0,
                                 help='(预处理阶段) 带通滤波器的低频截止频率 (Hz)。')
    preprocess_args.add_argument('--h_freq', type=float, default=30.0,
                                 help='(预处理阶段) 带通滤波器的高频截止频率 (Hz)。')
    preprocess_args.add_argument('--rsfreq', type=int, default=300,
                                 help='(预处理阶段) resampling rate (Hz)。')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.stage == 'preprocessing':
        print("--- Preprocessing Stage ---")
        if not args.raw_data_path or not args.processed_data_path or not args.dataset_name:
            parser.error("in --stage=preprocess, --raw_data_path, --processed_data_path, --dataset_name 是必需的。")
        cpsc = processors.BaseProcessor(args)
        cpsc.process_record()
