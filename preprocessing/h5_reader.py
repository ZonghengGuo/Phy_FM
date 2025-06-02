import h5py

# 定义 HDF5 文件的路径
file_path = r"D:\database\ECG_12l\cpsc_2018\cpsc.hdf5"

try:
    # 以只读模式打开 HDF5 文件
    with h5py.File(file_path, 'r') as hf:
        print(f"成功打开文件: {file_path}")

        # --- 探索文件内容 ---

        # 1. 列出文件中的顶层组和数据集
        print("\n文件中的顶层内容:")

        # 2. 访问特定的数据集或组
        # 假设你知道文件中有一个名为 'dataset_name' 的数据集
        # 如果你知道具体名称，可以取消下面代码的注释并替换 'dataset_name'
        #
        # if 'dataset_name' in hf:
        #     dataset = hf['dataset_name']
        #     print(f"\n访问数据集 'dataset_name':")
        #     print(f"数据集形状: {dataset.shape}")
        #     print(f"数据类型: {dataset.dtype}")
        #
        #     # 读取数据 (例如，读取整个数据集)
        #     # data = dataset[:]
        #     # print(f"前5个数据点: {data[:5]}") # 打印前5个数据点作为示例
        # else:
        #     print("\n文件中未找到 'dataset_name'。请检查 HDF5 文件结构。")

        # --- 如何进一步探索 ---
        # 你可以使用 hf.visit(print) 或 hf.visititems(lambda name, obj: print(f"{name}: {obj}"))
        # 来递归地打印文件中的所有项目及其类型。

        print("\n递归打印文件结构:")
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        hf.visititems(print_attrs) # 打印名称和属性

        # 如果你想读取某个具体的数据集，你需要知道它的路径。
        # 例如，如果文件中有个数据集路径是 'group1/subgroupA/my_data'
        # 你可以这样访问：
        #
        # if 'group1/subgroupA/my_data' in hf:
        #     my_specific_data = hf['group1/subgroupA/my_data'][:]
        #     print(f"\n读取 'group1/subgroupA/my_data' 的数据，形状: {my_specific_data.shape}")
        # else:
        #     print("\n未找到示例路径 'group1/subgroupA/my_data'")


except FileNotFoundError:
    print(f"错误: 文件未找到 {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")