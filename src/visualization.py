# src/visualization.py
"""
可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_progress(fitness_history):
    """绘制适应度进展图"""
    plt.figure(figsize=(10, 6))
    
    generations = range(len(fitness_history))
    
    plt.plot(generations, fitness_history, 'b-', linewidth=2, label='平均适应度')
    
    # 添加趋势线
    if len(fitness_history) > 1:
        z = np.polyfit(generations, fitness_history, 3)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "r--", alpha=0.5, label='趋势线')
    
    plt.xlabel('进化代数')
    plt.ylabel('适应度')
    plt.title('适应度进化过程')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('results/fitness_progress.png', dpi=150)
    plt.show()

def plot_population_diversity(population_history):
    """绘制种群多样性图（简化版）"""
    # 这里我们可以计算每代种群中旋律的差异
    pass

def visualize_melody(melody, title=None):
    """可视化旋律（钢琴卷帘）"""
    if not melody.notes:
        print("旋律为空")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制每个音符
    for i, note in enumerate(melody.notes):
        # 获取MIDI编号
        from src.utils import note_name_to_midi
        pitch = note_name_to_midi(note.pitch)
        
        # 绘制矩形
        rect = plt.Rectangle(
            (note.start_time, pitch - 0.4),
            note.duration,
            0.8,
            facecolor='skyblue',
            edgecolor='navy',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # 添加音符标签
        ax.text(
            note.start_time + note.duration / 2,
            pitch,
            note.pitch,
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # 设置图形属性
    ax.set_xlabel('时间（小节）')
    ax.set_ylabel('音高（MIDI编号）')
    ax.set_title(title or '旋律可视化')
    
    # 设置坐标轴范围
    ax.set_xlim(0, melody.total_length)
    
    # 设置Y轴为音符名
    from src.config import NOTES
    y_ticks = []
    y_labels = []
    for i, note_name in enumerate(NOTES[::3]):  # 每3个取一个
        y_ticks.append(note_name_to_midi(note_name))
        y_labels.append(note_name)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图像
    if title:
        filename = title.replace(' ', '_').lower()
        plt.savefig(f'results/{filename}.png', dpi=150)
    
    plt.show()

def create_evolution_report(ga_instance, output_dir):
    """生成进化过程报告"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建报告文件
    report_file = os.path.join(output_dir, 'evolution_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("遗传算法作曲进化报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总进化代数: {ga_instance.generation}\n")
        f.write(f"最终平均适应度: {ga_instance.population.get_average_fitness():.3f}\n")
        f.write(f"最终最佳适应度: {ga_instance.population.get_fittest().fitness:.3f}\n\n")
        
        f.write("适应度历史:\n")
        for gen, fitness in enumerate(ga_instance.fitness_history):
            f.write(f"  第{gen}代: {fitness:.3f}\n")
        
        f.write("\n最佳旋律信息:\n")
        best_melody = ga_instance.population.get_fittest().melody
        f.write(best_melody.to_string())
    
    print(f"报告已保存到: {report_file}")

def compare_melodies(melodies, titles=None):
    """比较多个旋律"""
    if titles is None:
        titles = [f"旋律 {i+1}" for i in range(len(melodies))]
    
    fig, axes = plt.subplots(len(melodies), 1, figsize=(12, 4 * len(melodies)))
    
    if len(melodies) == 1:
        axes = [axes]
    
    for idx, (melody, title) in enumerate(zip(melodies, titles)):
        ax = axes[idx]
        
        # 绘制每个音符
        for i, note in enumerate(melody.notes):
            from src.utils import note_name_to_midi
            pitch = note_name_to_midi(note.pitch)
            
            rect = plt.Rectangle(
                (note.start_time, pitch - 0.4),
                note.duration,
                0.8,
                facecolor=plt.cm.tab20(idx / len(melodies)),
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
        
        ax.set_title(title)
        ax.set_xlabel('时间（小节）')
        ax.set_ylabel('音高')
        ax.set_xlim(0, max(m.total_length for m in melodies))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/melody_comparison.png', dpi=150)
    plt.show()