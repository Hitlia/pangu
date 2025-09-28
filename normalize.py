import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd

def generate_times(start_date, end_date, freq='6h'):
    return pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()

def compute_dataset_statistics(timestamps, npy_path, tp6hr_path):
    """
    计算数据集中70个通道数据的统计量
    
    Args:
        timestamps: 按时间顺序排列的 datetime 对象列表，间隔为6小时
        npy_path: 存放 npy 文件的根目录
    """
    # 初始化统计量
    channel_means = np.zeros(70)
    channel_m2 = np.zeros(70)  # 用于计算方差的中间值
    channel_counts = np.zeros(70)
    channel_mins = np.full(70, np.inf)
    channel_maxs = np.full(70, -np.inf)
    
    def _get_filepath(timestamp):
        """根据时间戳构造 npy 文件路径"""
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        return os.path.join(npy_path, str(year), file_name)
    
    def _get_tp6hr_filepath(timestamp):
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        file_path = os.path.join(tp6hr_path, str(year), file_name)
        return file_path
    
    # 使用Welford在线算法计算均值和方差
    # 参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    for timestamp in tqdm(timestamps, desc="Processing files"):
        file_path = _get_filepath(timestamp)
        tp_path = _get_tp6hr_filepath(timestamp)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        try:
            # 加载数据并处理NaN值
            data = np.load(file_path)[:70, :, :241]  # [70, 241, 241]
            tp_data = np.load(tp_path)[:, :, :241]
            data[69,:,:] = tp_data[0,:,:]
            data = np.nan_to_num(data, nan=0.0)
            
            # 更新统计量
            for channel in range(70):
                channel_data = data[channel].flatten()
                
                # 更新最小值最大值
                channel_min = np.min(channel_data)
                channel_max = np.max(channel_data)
                if channel_min < channel_mins[channel]:
                    channel_mins[channel] = channel_min
                if channel_max > channel_maxs[channel]:
                    channel_maxs[channel] = channel_max
                
                # 使用Welford算法更新均值和方差
                for value in channel_data:
                    channel_counts[channel] += 1
                    delta = value - channel_means[channel]
                    channel_means[channel] += delta / channel_counts[channel]
                    delta2 = value - channel_means[channel]
                    channel_m2[channel] += delta * delta2
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # 计算标准差
    channel_stds = np.sqrt(channel_m2 / channel_counts)
    
    return {
        'means': channel_means,
        'stds': channel_stds,
        'mins': channel_mins,
        'maxs': channel_maxs,
        'counts': channel_counts
    }

def main():
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Compute statistics for weather dataset')
    parser.add_argument('--npy_path', default='/data/lhy/data/npy-data/', type=str, help='Path to the npy files directory')
    parser.add_argument('--tp6hr_path', default='/data/lhy/data/tp6hr-data/', type=str)
    parser.add_argument('--output', default='./norms_minmax.npy', type=str, help='Output file name')
    
    args = parser.parse_args()
    
    timestamps = generate_times(datetime(2018, 1, 1, 1), datetime(2023, 12, 31, 19))
    # 计算统计量
    statistics = compute_dataset_statistics(timestamps, args.npy_path, args.tp6hr_path)
    
    # 保存结果
    np.savez(args.output, 
             means=statistics['means'],
             stds=statistics['stds'],
             mins=statistics['mins'],
             maxs=statistics['maxs'],
             counts=statistics['counts'])
    
    print("Statistics computed and saved successfully!")
    print(f"Means: {statistics['means']}")
    print(f"Standard deviations: {statistics['stds']}")
    print(f"Minimum values: {statistics['mins']}")
    print(f"Maximum values: {statistics['maxs']}")

if __name__ == '__main__':
    main()