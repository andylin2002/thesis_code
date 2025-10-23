import numpy as np

# 定义要读取的文件名
file_name = 'csi_sample_swapped.npy'

try:
    # 使用 numpy.load() 加载 .npy 文件
    data = np.load(file_name)
    
    # 获取并打印数组的形状
    shape = data.shape
    print(f"成功加载文件: {file_name}")
    print(f"数组的形状 (Shape) 是: {shape}")
    
    # 也可以打印数组的其他信息，例如数据类型和维度数量
    print(f"数组的数据类型 (Dtype) 是: {data.dtype}")
    print(f"数组的维度数量 (Ndim) 是: {data.ndim}")

except FileNotFoundError:
    print(f"错误: 文件 '{file_name}' 未找到。请确保文件在当前目录下或提供正确的路径。")
except Exception as e:
    print(f"加载文件 '{file_name}' 时发生错误: {e}")