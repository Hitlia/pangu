import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import pandas as pd

from weather_dataset import WeatherPanguData

def generate_times(start_date, end_date, freq='6h'):
    return pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()

def pre_norm_by_npy(all_times, npy_path, tp6hr_path, save_norm_path):
    var_num = 70
    space_num = 241 * 241

    device = torch.device("cuda")

    # 初始化累积张量
    var_sum_input = torch.zeros(var_num, dtype=torch.float32).to(device)
    var_sum_sq_input = torch.zeros(var_num, dtype=torch.float32).to(device)
    channel_mins = torch.full(tuple([var_num]), torch.inf).to(device)
    channel_maxs = torch.full(tuple([var_num]), -torch.inf).to(device)

    total_count = 0

    dataset = WeatherPanguData(all_times, npy_path, tp6hr_path, input_window_size=1, output_window_size=1)
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                              drop_last=False, shuffle=False, num_workers=0, pin_memory=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Processing files")):
        # [batch_size, input_window_size, var_num, H, W]
        x = batch['input'].to(device).squeeze(1).squeeze(0)                             # [70, 241, 241]
        
        var_max_input, _ = torch.max(x, dim=1)
        var_max_input, _ = torch.max(var_max_input, dim=1)
        var_min_input, _ = torch.min(x, dim=1)
        var_min_input, _ = torch.min(var_min_input, dim=1)
        channel_maxs = torch.max(channel_maxs, var_max_input)
        channel_mins = torch.min(channel_mins, var_min_input)
            
        # 对输入数据 x 计算均值和方差累计（沿空间维度）
        var_sum_input += x.sum(dim=(1, 2))
        var_sum_sq_input += (x ** 2).sum(dim=(1, 2))

        total_count += 1

    var_mean = var_sum_input / (total_count * space_num)
    var_sq_mean = var_sum_sq_input / (total_count * space_num)
    var_std = torch.sqrt(torch.clamp(var_sq_mean - var_mean ** 2, min=0.0))
    norm_params_input = {'var_mean': var_mean, 'var_std': var_std, 'var_max': channel_maxs, 'var_min': channel_mins}
    torch.save(norm_params_input, save_norm_path)
    print("Input norm parameters saved to {}".format(save_norm_path))

    return norm_params_input

def main():
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Compute statistics for weather dataset')
    parser.add_argument('--npy_path', default='/data/lhy/data/npy-data/', type=str, help='Path to the npy files directory')
    parser.add_argument('--tp6hr_path', default='/data/lhy/data/tp6hr-data/', type=str)
    parser.add_argument('--output', default='./norms.npy', type=str, help='Output file name')
    
    args = parser.parse_args()
    
    timestamps = generate_times(datetime(2018, 1, 1, 1), datetime(2024, 12, 31, 19))
    pre_norm_by_npy(timestamps, args.npy_path, args.tp6hr_path, args.output)
    
if __name__ == '__main__':
    main()