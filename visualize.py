"""
数据可视化脚本 - 岩性分类项目
包括：数据分布、特征相关性、类别分布、测井曲线等可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


class LithologyVisualizer:
    """岩性数据可视化类"""
    
    def __init__(self, data_path='data/train_all.csv', config_path='config.yaml'):
        """初始化"""
        self.data_path = data_path
        self.config_path = config_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载数据
        self.df = pd.read_csv(data_path)
        print(f"数据加载完成: {len(self.df)} 条记录")
        print(f"列名: {list(self.df.columns)}")
        
        # 特征列
        self.feature_cols = ['SP', 'GR', 'AC', 'DEPTH']
        
        # 创建输出目录
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
    def print_basic_info(self):
        """打印基本统计信息"""
        print("\n" + "="*60)
        print("数据基本信息")
        print("="*60)
        print(f"总样本数: {len(self.df)}")
        print(f"特征数: {len(self.feature_cols)}")
        print(f"井数: {self.df['WELL'].nunique()}")
        print(f"类别数: {self.df['label'].nunique()}")
        
        print("\n各井样本数量:")
        print(self.df['WELL'].value_counts().sort_index())
        
        print("\n类别分布:")
        print(self.df['label'].value_counts().sort_index())
        print("\n类别比例:")
        print(self.df['label'].value_counts(normalize=True).sort_index())
        
        print("\n特征统计信息:")
        print(self.df[self.feature_cols].describe())
        
    def plot_class_distribution(self):
        """Plot class distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        class_counts = self.df['label'].value_counts().sort_index()
        colors = plt.cm.Set3(range(len(class_counts)))
        axes[0].pie(class_counts.values, 
                    labels=[f'Class {i}' for i in class_counts.index],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90)
        axes[0].set_title('Lithology Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        
        # Bar chart
        axes[1].bar(class_counts.index, class_counts.values, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Lithology Class', fontsize=12)
        axes[1].set_ylabel('Sample Count', fontsize=12)
        axes[1].set_title('Lithology Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        axes[1].set_xticks(class_counts.index)
        axes[1].set_xticklabels([f'Class {i}' for i in class_counts.index])
        
        # Add value labels
        for i, v in enumerate(class_counts.values):
            axes[1].text(class_counts.index[i], v, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_class_distribution.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 01_class_distribution.png")
        plt.close()
        
    def plot_feature_distributions(self):
        """Plot feature distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.feature_cols):
            axes[idx].hist(self.df[col], bins=50, color='skyblue', 
                          edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col, fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{col} Distribution', fontsize=13, fontweight='bold')
            axes[idx].axvline(self.df[col].mean(), color='red', 
                             linestyle='--', linewidth=2, label='Mean')
            axes[idx].axvline(self.df[col].median(), color='green', 
                             linestyle='--', linewidth=2, label='Median')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_feature_distributions.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 02_feature_distributions.png")
        plt.close()
        
    def plot_correlation_matrix(self):
        """Plot feature correlation matrix"""
        # Calculate correlation matrix
        corr_matrix = self.df[self.feature_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_correlation_matrix.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 03_correlation_matrix.png")
        plt.close()
        
    def plot_boxplots_by_class(self):
        """Plot feature boxplots by class"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.feature_cols):
            data_to_plot = [self.df[self.df['label'] == i][col].values 
                           for i in sorted(self.df['label'].unique())]
            
            bp = axes[idx].boxplot(data_to_plot, 
                                   tick_labels=[f'Class {i}' for i in sorted(self.df['label'].unique())],
                                   patch_artist=True,
                                   showmeans=True)
            
            # Set colors
            colors = plt.cm.Set3(range(len(data_to_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[idx].set_xlabel('Lithology Class', fontsize=12)
            axes[idx].set_ylabel(col, fontsize=12)
            axes[idx].set_title(f'{col} by Class', fontsize=13, fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_boxplots_by_class.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 04_boxplots_by_class.png")
        plt.close()
        
    def plot_scatter_matrix(self):
        """Plot feature scatter matrix"""
        # Sample data for faster plotting
        sample_size = min(5000, len(self.df))
        df_sample = self.df.sample(n=sample_size, random_state=42)
        
        # Create scatter matrix
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        feature_cols_no_depth = ['SP', 'GR', 'AC']
        colors = ['red', 'blue', 'green']
        
        for i, col1 in enumerate(feature_cols_no_depth):
            for j, col2 in enumerate(feature_cols_no_depth):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    for label_idx, label in enumerate(sorted(df_sample['label'].unique())):
                        data = df_sample[df_sample['label'] == label][col1]
                        ax.hist(data, bins=30, alpha=0.5, color=colors[label_idx], 
                               label=f'Class {label}')
                    ax.set_ylabel('Frequency', fontsize=10)
                    if i == 0:
                        ax.legend(fontsize=8)
                else:
                    # Off-diagonal: scatter plot
                    for label_idx, label in enumerate(sorted(df_sample['label'].unique())):
                        data = df_sample[df_sample['label'] == label]
                        ax.scatter(data[col2], data[col1], 
                                  alpha=0.3, s=10, color=colors[label_idx],
                                  label=f'Class {label}' if i == 0 and j == 1 else '')
                    
                if i == len(feature_cols_no_depth) - 1:
                    ax.set_xlabel(col2, fontsize=10)
                if j == 0:
                    ax.set_ylabel(col1, fontsize=10)
                    
                ax.grid(alpha=0.3)
        
        plt.suptitle('Feature Scatter Matrix (5000 samples)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_scatter_matrix.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 05_scatter_matrix.png")
        plt.close()
        
    def plot_well_logs(self):
        """Plot well logs"""
        # Select a well for display
        well_id = self.df['WELL'].value_counts().index[0]
        well_data = self.df[self.df['WELL'] == well_id].sort_values('DEPTH')
        
        # Limit depth range for clearer display
        depth_range = 200  # Display 200 meters
        start_depth = well_data['DEPTH'].min()
        end_depth = start_depth + depth_range
        well_data = well_data[(well_data['DEPTH'] >= start_depth) & 
                             (well_data['DEPTH'] <= end_depth)]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 10), sharey=True)
        
        log_cols = ['SP', 'GR', 'AC']
        colors_map = {0: 'yellow', 1: 'brown', 2: 'gray'}
        
        # Plot each log curve
        for idx, col in enumerate(log_cols):
            axes[idx].plot(well_data[col], well_data['DEPTH'], 'b-', linewidth=1)
            axes[idx].set_xlabel(col, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{col} Log', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            axes[idx].invert_yaxis()
        
        # Plot lithology column
        ax_litho = axes[3]
        for label in sorted(well_data['label'].unique()):
            label_data = well_data[well_data['label'] == label]
            ax_litho.fill_betweenx(label_data['DEPTH'], 0, 1, 
                                   color=colors_map.get(label, 'white'),
                                   alpha=0.7, label=f'Class {label}')
        
        ax_litho.set_xlabel('Lithology', fontsize=12, fontweight='bold')
        ax_litho.set_title('Lithology Column', fontsize=12, fontweight='bold')
        ax_litho.set_xlim(0, 1)
        ax_litho.set_xticks([])
        ax_litho.legend(loc='upper right')
        ax_litho.invert_yaxis()
        
        axes[0].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Well {well_id} Logs (Depth: {start_depth:.1f}-{end_depth:.1f}m)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_well_logs.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 06_well_logs.png")
        plt.close()
        
    def plot_well_statistics(self):
        """Plot well statistics"""
        well_stats = self.df.groupby('WELL').agg({
            'id': 'count',
            'DEPTH': ['min', 'max'],
            'label': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sample count per well
        well_counts = self.df['WELL'].value_counts().sort_index()
        axes[0, 0].bar(well_counts.index, well_counts.values, 
                      color='steelblue', alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Well ID', fontsize=12)
        axes[0, 0].set_ylabel('Sample Count', fontsize=12)
        axes[0, 0].set_title('Sample Count by Well', fontsize=13, fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # Depth range per well
        for well in sorted(self.df['WELL'].unique()):
            well_data = self.df[self.df['WELL'] == well]
            depth_min = well_data['DEPTH'].min()
            depth_max = well_data['DEPTH'].max()
            axes[0, 1].plot([well, well], [depth_min, depth_max], 
                           'o-', linewidth=3, markersize=8)
        
        axes[0, 1].set_xlabel('Well ID', fontsize=12)
        axes[0, 1].set_ylabel('Depth (m)', fontsize=12)
        axes[0, 1].set_title('Depth Range by Well', fontsize=13, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(alpha=0.3)
        
        # Class distribution by well (stacked)
        well_label_counts = self.df.groupby(['WELL', 'label']).size().unstack(fill_value=0)
        well_label_counts.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                              color=plt.cm.Set3(range(len(well_label_counts.columns))),
                              edgecolor='black', alpha=0.8)
        axes[1, 0].set_xlabel('Well ID', fontsize=12)
        axes[1, 0].set_ylabel('Sample Count', fontsize=12)
        axes[1, 0].set_title('Class Distribution by Well (Stacked)', fontsize=13, fontweight='bold')
        axes[1, 0].legend(title='Class', labels=[f'Class {i}' for i in well_label_counts.columns])
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Class percentage by well
        well_label_pct = well_label_counts.div(well_label_counts.sum(axis=1), axis=0) * 100
        well_label_pct.plot(kind='bar', stacked=True, ax=axes[1, 1],
                           color=plt.cm.Set3(range(len(well_label_pct.columns))),
                           edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Well ID', fontsize=12)
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 1].set_title('Class Percentage by Well', fontsize=13, fontweight='bold')
        axes[1, 1].legend(title='Class', labels=[f'Class {i}' for i in well_label_pct.columns])
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_well_statistics.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 07_well_statistics.png")
        plt.close()
        
    def plot_feature_by_depth(self):
        """Plot features vs depth"""
        # Sample data
        sample_size = min(10000, len(self.df))
        df_sample = self.df.sample(n=sample_size, random_state=42).sort_values('DEPTH')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        log_cols = ['SP', 'GR', 'AC']
        colors = ['red', 'blue', 'green']
        
        for idx, col in enumerate(log_cols):
            for label in sorted(df_sample['label'].unique()):
                label_data = df_sample[df_sample['label'] == label]
                axes[idx].scatter(label_data[col], label_data['DEPTH'], 
                                 alpha=0.3, s=5, c=colors[label],
                                 label=f'Class {label}')
            
            axes[idx].set_xlabel(col, fontsize=12)
            axes[idx].set_ylabel('Depth (m)', fontsize=12)
            axes[idx].set_title(f'{col} vs Depth', fontsize=13, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.suptitle('Features vs Depth (10000 samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '08_feature_by_depth.png', bbox_inches='tight', dpi=150)
        print(f"[OK] Saved: 08_feature_by_depth.png")
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\nGenerating data visualizations...")
        print("="*60)
        
        self.print_basic_info()
        
        print("\nGenerating visualization charts:")
        print("-"*60)
        self.plot_class_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()
        self.plot_boxplots_by_class()
        self.plot_scatter_matrix()
        self.plot_well_logs()
        self.plot_well_statistics()
        self.plot_feature_by_depth()
        
        print("-"*60)
        print(f"\n[COMPLETED] All visualizations saved to: {self.output_dir.absolute()}")
        print("="*60)


def main():
    """主函数"""
    visualizer = LithologyVisualizer(
        data_path='data/train_all.csv',
        config_path='config.yaml'
    )
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()

