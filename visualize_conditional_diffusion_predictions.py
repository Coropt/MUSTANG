"""
可视化conditional diffusion预测结果的时间序列图
包含原始数据值、MUSTANG Fine-tuned预测值、Baseline conditional diffusion预测值
只显示mask=1的评估点
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import sys
import os
import argparse

# 添加项目路径以便导入util模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from util.utils import get_station_ids
except ImportError:
    # 如果无法导入，使用默认函数
    def get_station_ids(num_stations):
        if num_stations == 12:
            # 原始12个站点 (不含04182000)
            return ['04178000', '04183000', '04183500', '04185318', '04186500', 
                    '04188100', '04190000', '04191058', '04191444', '04191500', '04192500', '04193500']
        elif num_stations == 14:
            # Common stations (excluding 04182000 due to high missing rate)
            return ['04178000', '04183000', '04183500', '04184500', '04185318', '04186500', 
                    '04188100', '04188496', '04190000', '04191058', '04191444', '04191500', 
                    '04192500', '04193500']
        elif num_stations == 15:
            return ['04178000', '04183000', '04183500', '04185318', '04186500', 
                    '04188100', '04190000', '04191058', '04191444', '04191500', '04192500', 
                    '04193500', '04194000', '04194500']
        elif num_stations == 20:
            return ['4178000', '4182000', '4183000', '4183500', '4184500', '4185000', '4185318', '4185440', 
                    '4186500', '4188100', '4188496', '4189000', '4190000', '4191058', '4191444', '4191500', 
                    '4192500', '4192574', '4192599', '4193500']
        else:
            return [f"Station_{i+1}" for i in range(num_stations)]


def load_conditional_diffusion_predictions(
    csv_path: Union[str, Path], station_id: str = "Station_1"
):
    """
    加载conditional diffusion预测结果CSV文件
    
    参数:
        csv_path: CSV文件路径
        station_id: 要加载的station ID
    
    返回:
        DataFrame: 包含预测结果的数据框
    """
    df = pd.read_csv(csv_path)
    
    # 检查是否有station_id列
    if 'station_id' in df.columns:
        df = df[df['station_id'] == station_id].copy()
    else:
        # 如果没有station_id列，假设整个文件就是该station的数据
        pass
    
    # 新版列名
    new_required_cols = [
        "timestep",
        "sequence_id",
        "time_in_sequence",
        "imputed_value",
        "ground_truth_value",
        "evaluation_mask",
    ]
    # 旧版列名（兼容）
    old_required_cols = [
        "timestep",
        "sequence_id",
        "time_in_sequence",
        "predicted_value",
        "observed_value",
        "mask",
    ]
    if all(col in df.columns for col in new_required_cols):
        pass
    elif all(col in df.columns for col in old_required_cols):
        df = df.rename(
            columns={
                "predicted_value": "imputed_value",
                "observed_value": "ground_truth_value",
                "mask": "evaluation_mask",
                "is_observed": "is_observed_point",
            }
        )
    else:
        missing_cols = [col for col in new_required_cols if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")
    
    return df


def calculate_actual_dates(df: pd.DataFrame, dataset_start_date: datetime, test_start_idx: int):
    """
    计算实际日期索引
    
    参数:
        df: 数据框
        dataset_start_date: 数据集开始日期
        test_start_idx: test set在原始数据集中的起始索引
    
    返回:
        DataFrame: 添加了actual_date列的数据框
    """
    df = df.copy()
    
    # 计算所有数据点的实际时间索引
    df['actual_time_idx'] = test_start_idx + df['sequence_id'] * 16 + df['time_in_sequence']
    
    # 限制日期范围不超过数据集结束日期
    total_days = (datetime(2024, 9, 30) - dataset_start_date).days + 1
    max_time_idx = total_days - 1
    df['actual_time_idx'] = df['actual_time_idx'].clip(upper=max_time_idx)
    
    # 将实际时间索引转换为日期
    df['actual_date'] = [dataset_start_date + timedelta(days=int(idx)) for idx in df['actual_time_idx'].values]
    
    # 过滤掉超出数据集范围的日期
    dataset_end_date = datetime(2024, 9, 30)
    df = df[df['actual_date'] <= dataset_end_date].copy()
    
    return df


def plot_conditional_diffusion_comparison(
    meta_csv_path: Union[str, Path],
    baseline_csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    station_id: str = "Station_1",
    figsize: tuple = (16, 10),
    dpi: int = 300,
    title: Optional[str] = None,
    show_legend: bool = True,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    use_chinese_labels: bool = False,
    show_all_observed: bool = True,
    max_sequences: Optional[int] = None
):
    """
    绘制conditional diffusion预测结果对比图（MUSTANG Fine-tuned vs Baseline conditional diffusion vs 原始数据）
    
    参数:
        meta_csv_path: Meta fine-tuned预测结果CSV路径
        baseline_csv_path: Baseline conditional diffusion预测结果CSV路径
        output_path: 输出图片路径
        station_id: Station ID
        figsize: 图片大小
        dpi: 图片分辨率
        title: 图片标题
        show_legend: 是否显示图例
        alpha: 透明度
        linewidth: 线条宽度
        use_chinese_labels: 是否使用中文标签
        show_all_observed: 是否显示所有观测值
        max_sequences: 最多显示的序列数
    
    返回:
        fig, ax: matplotlib的figure和axes对象
    """
    # 数据集配置
    dataset_start_date = datetime(2015, 4, 15)
    dataset_end_date = datetime(2024, 9, 30)
    total_days = (dataset_end_date - dataset_start_date).days + 1
    
    # 数据集划分：val_len=0.1, test_len=0.2
    val_len = 0.1
    test_len = 0.2
    test_start_idx = int((1 - test_len) * total_days)
    
    # 加载数据
    print(f"加载Meta fine-tuned预测结果: {meta_csv_path}")
    meta_df = load_conditional_diffusion_predictions(meta_csv_path, station_id)
    meta_df = calculate_actual_dates(meta_df, dataset_start_date, test_start_idx)
    
    print(f"加载Baseline conditional diffusion预测结果: {baseline_csv_path}")
    baseline_df = load_conditional_diffusion_predictions(baseline_csv_path, station_id)
    baseline_df = calculate_actual_dates(baseline_df, dataset_start_date, test_start_idx)
    
    # 只保留 evaluation_mask=1 的点（评估点）
    meta_eval = meta_df[meta_df['evaluation_mask'] == 1].copy()
    baseline_eval = baseline_df[baseline_df['evaluation_mask'] == 1].copy()
    
    if len(meta_eval) == 0:
        raise ValueError("Meta fine-tuned CSV文件中没有evaluation_mask=1的点")
    if len(baseline_eval) == 0:
        raise ValueError("Baseline conditional diffusion CSV文件中没有evaluation_mask=1的点")
    
    # 按日期排序
    meta_eval = meta_eval.sort_values('actual_date')
    baseline_eval = baseline_eval.sort_values('actual_date')
    meta_df = meta_df.sort_values('actual_date')
    baseline_df = baseline_df.sort_values('actual_date')
    
    # 限制序列数
    if max_sequences is not None:
        max_seq_id = max_sequences - 1
        meta_eval = meta_eval[meta_eval['sequence_id'] <= max_seq_id].copy()
        baseline_eval = baseline_eval[baseline_eval['sequence_id'] <= max_seq_id].copy()
        meta_df = meta_df[meta_df['sequence_id'] <= max_seq_id].copy()
        baseline_df = baseline_df[baseline_df['sequence_id'] <= max_seq_id].copy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 根据参数决定observed value使用哪些点
    if show_all_observed:
        # 使用所有真实观测值（从meta_df或baseline_df中取，它们应该相同）
        obs_df = meta_df.copy()
        x_obs = obs_df['actual_date'].values
        y_obs = obs_df['ground_truth_value'].values
    else:
        # 只使用与imputed value相同数量的点（evaluation_mask=1的点）
        obs_df = meta_eval.copy()
        x_obs = obs_df['actual_date'].values
        y_obs = obs_df['ground_truth_value'].values
    
    # imputed value的x轴（只有evaluation_mask=1的点）
    x_meta = meta_eval['actual_date'].values
    x_baseline = baseline_eval['actual_date'].values
    
    # 根据语言设置标签
    if use_chinese_labels:
        obs_label = '原始数据值 (Observed Value)'
        meta_label = 'MUSTANG Fine-tuned 预测值'
        baseline_label = 'Baseline conditional diffusion 预测值'
        xlabel = '日期 (Date)'
        ylabel = '数值 (Value)'
    else:
        obs_label = 'Observed Value'
        meta_label = 'MUSTANG Fine-tuned Prediction'
        baseline_label = 'Baseline conditional diffusion Prediction'
        xlabel = 'Date'
        ylabel = 'Value'
    
    # 绘制原始数据值（ground_truth_value）
    ax.plot(x_obs, y_obs, 
            label=obs_label, 
            color='#006400',  # 深绿色 (DarkGreen)
            linewidth=linewidth,
            alpha=alpha,
            marker='',
            linestyle='-',
            zorder=1)
    
    # 绘制Baseline conditional diffusion预测值
    ax.plot(x_baseline, baseline_eval['imputed_value'].values, 
            label=baseline_label, 
            color='#FF8C00',  # 深橙色 (DarkOrange)
            linewidth=linewidth,
            alpha=alpha,
            marker='',
            linestyle='-',
            zorder=2)
    
    # 绘制MUSTANG Fine-tuned预测值
    ax.plot(x_meta, meta_eval['imputed_value'].values, 
            label=meta_label, 
            color='#DC143C',  # 深红色 (Crimson)
            linewidth=linewidth,
            alpha=alpha,
            marker='',
            linestyle='-',
            zorder=3)
    
    # 设置x轴为日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # 旋转日期标签以避免重叠
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 设置标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        # 尝试从文件名提取station信息
        station_name = station_id
        try:
            # 尝试获取实际的station ID
            station_ids = get_station_ids(15)  # 假设是15个station
            if station_id.startswith('Station_'):
                station_num = int(station_id.split('_')[1])
                if station_num <= len(station_ids):
                    actual_station_id = station_ids[station_num - 1]
                    station_name = f"{station_id} ({actual_station_id})"
        except:
            pass
        
        if use_chinese_labels:
            title = f'conditional diffusion预测结果对比（仅评估点）\n{station_name}'
        else:
            title = f'conditional diffusion Prediction Comparison (Evaluation Points Only)\n{station_name}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 显示图例
    if show_legend:
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"图片已保存至: {output_path}")
    
    return fig, ax


def plot_conditional_diffusion_comparison_by_sequence(
    meta_csv_path: Union[str, Path],
    baseline_csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    station_id: str = "Station_1",
    figsize: tuple = (18, 12),
    dpi: int = 300,
    title: Optional[str] = None,
    max_sequences: int = 9,
    sequences_per_row: int = 3
):
    """
    按序列分组绘制conditional diffusion预测结果对比图（每个序列一个子图）
    
    参数:
        meta_csv_path: Meta fine-tuned预测结果CSV路径
        baseline_csv_path: Baseline conditional diffusion预测结果CSV路径
        output_path: 输出图片路径
        station_id: Station ID
        figsize: 图片大小
        dpi: 图片分辨率
        title: 图片标题
        max_sequences: 最多显示的序列数
        sequences_per_row: 每行显示的序列数
    
    返回:
        fig, axes: matplotlib的figure和axes对象
    """
    # 数据集配置
    dataset_start_date = datetime(2015, 4, 15)
    dataset_end_date = datetime(2024, 9, 30)
    total_days = (dataset_end_date - dataset_start_date).days + 1
    val_len = 0.1
    test_len = 0.2
    test_start_idx = int((1 - test_len) * total_days)
    
    # 加载数据
    meta_df = load_conditional_diffusion_predictions(meta_csv_path, station_id)
    baseline_df = load_conditional_diffusion_predictions(baseline_csv_path, station_id)
    
    # 获取所有唯一的序列ID
    meta_sequence_ids = meta_df['sequence_id'].unique()
    baseline_sequence_ids = baseline_df['sequence_id'].unique()
    sequence_ids = sorted(set(meta_sequence_ids) & set(baseline_sequence_ids))
    
    if max_sequences is not None:
        sequence_ids = sequence_ids[:max_sequences]
    
    n_sequences = len(sequence_ids)
    n_cols = sequences_per_row
    n_rows = (n_sequences + n_cols - 1) // n_cols
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_sequences == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # 为每个序列绘制子图
    for idx, seq_id in enumerate(sequence_ids):
        ax = axes[idx]
        
        # 获取该序列的数据
        meta_seq = meta_df[meta_df['sequence_id'] == seq_id].copy()
        baseline_seq = baseline_df[baseline_df['sequence_id'] == seq_id].copy()
        
        # 只保留 evaluation_mask=1 的点
        meta_eval_seq = meta_seq[meta_seq['evaluation_mask'] == 1].copy()
        baseline_eval_seq = baseline_seq[baseline_seq['evaluation_mask'] == 1].copy()
        
        meta_eval_seq = meta_eval_seq.sort_values('time_in_sequence')
        baseline_eval_seq = baseline_eval_seq.sort_values('time_in_sequence')
        
        if len(meta_eval_seq) == 0 or len(baseline_eval_seq) == 0:
            ax.text(0.5, 0.5, f'No evaluation points\nin sequence {seq_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Sequence {seq_id} (No eval points)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            continue
        
        # 计算实际时间索引并转换为日期
        meta_eval_seq = meta_eval_seq.copy()
        meta_eval_seq['actual_time_idx'] = test_start_idx + meta_eval_seq['sequence_id'] * 16 + meta_eval_seq['time_in_sequence']
        max_time_idx = total_days - 1
        meta_eval_seq['actual_time_idx'] = meta_eval_seq['actual_time_idx'].clip(upper=max_time_idx)
        dates = [dataset_start_date + timedelta(days=int(idx)) for idx in meta_eval_seq['actual_time_idx'].values]
        
        # 过滤掉超出数据集范围的日期
        valid_mask = [d <= dataset_end_date for d in dates]
        meta_eval_seq = meta_eval_seq[valid_mask].reset_index(drop=True)
        dates = [d for d, valid in zip(dates, valid_mask) if valid]
        
        if len(meta_eval_seq) == 0:
            ax.text(0.5, 0.5, f'No valid evaluation points\nin sequence {seq_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Sequence {seq_id} (No valid points)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            continue
        
        # 绘制原始数据、Baseline和Meta预测
        ax.plot(dates, meta_eval_seq['ground_truth_value'].values, 
                label='Observed', color='#006400', linewidth=2.5, alpha=0.9, marker='', linestyle='-')
        ax.plot(dates, baseline_eval_seq['imputed_value'].values, 
                label='Baseline conditional diffusion', color='#FF8C00', linewidth=2.5, alpha=0.9, marker='', linestyle='-')
        ax.plot(dates, meta_eval_seq['imputed_value'].values, 
                label='MUSTANG Fine-tuned', color='#DC143C', linewidth=2.5, alpha=0.9, marker='', linestyle='-')
        
        # 设置x轴为日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'Sequence {seq_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_sequences, len(axes)):
        axes[idx].set_visible(False)
    
    # 设置总标题
    if title is None:
        station_name = station_id
        try:
            station_ids = get_station_ids(15)
            if station_id.startswith('Station_'):
                station_num = int(station_id.split('_')[1])
                if station_num <= len(station_ids):
                    actual_station_id = station_ids[station_num - 1]
                    station_name = f"{station_id} ({actual_station_id})"
        except:
            pass
        title = f'conditional diffusion Prediction Comparison by Sequence (Evaluation Points Only): {station_name}'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"图片已保存至: {output_path}")
    
    return fig, axes


# Backward-compatible aliases
load_csdi_predictions = load_conditional_diffusion_predictions
plot_csdi_comparison = plot_conditional_diffusion_comparison
plot_csdi_comparison_by_sequence = plot_conditional_diffusion_comparison_by_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize conditional diffusion prediction results')
    parser.add_argument('--meta_csv', type=str, required=True,
                        help='Path to MUSTANG fine-tuned predictions CSV file')
    parser.add_argument('--baseline_csv', type=str, required=True,
                        help='Path to baseline conditional diffusion predictions CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for images (default: same as meta_csv directory)')
    parser.add_argument('--station_id', type=str, default='Station_1',
                        help='Station ID to visualize (default: Station_1)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to show (default: all)')
    parser.add_argument('--show_all_observed', action='store_true',
                        help='Show all observed values, not just evaluation points')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = Path(args.meta_csv).parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    station_suffix = args.station_id.replace('_', '')
    output_file = output_dir / f"conditional_diffusion_comparison_{station_suffix}_timeseries.png"
    output_file_by_seq = output_dir / f"conditional_diffusion_comparison_{station_suffix}_by_sequence.png"
    
    print(f"处理 {args.station_id}...")
    
    try:
        # 绘制时间序列对比图
        fig1, ax1 = plot_conditional_diffusion_comparison(
            args.meta_csv,
            args.baseline_csv,
            output_file,
            station_id=args.station_id,
            show_all_observed=args.show_all_observed,
            max_sequences=args.max_sequences
        )
        plt.close(fig1)
        print(f"  ✓ 已保存: {output_file.name}")
        
        # 绘制按序列分组的对比图
        fig2, axes2 = plot_conditional_diffusion_comparison_by_sequence(
            args.meta_csv,
            args.baseline_csv,
            output_file_by_seq,
            station_id=args.station_id,
            max_sequences=args.max_sequences or 9
        )
        plt.close(fig2)
        print(f"  ✓ 已保存: {output_file_by_seq.name}")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
    
    print("\n所有图片生成完成！")
