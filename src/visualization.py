# src/visualization.py
"""
可视化功能
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def get_interval_array(melody):
    """获取旋律的音程数组（忽略休止符）"""
    intervals = []
    notes = melody.notes
    for i in range(len(notes) - 1):
        current_note = notes[i]
        next_note = notes[i + 1]
        #跳过休止符
        if current_note > 0 and next_note > 0:
            interval = next_note - current_note
            intervals.append(interval)
    
    return intervals

def extract_all_melody_features(population):
    """提取所有melody特征向量"""
    feature_list = []
    for i, individual in enumerate(population.inviduals):
        melody = individual.melody
        average_pitch = np.mean(melody)
        std_pitch = np.std(melody,ddof=0)
        intervals = get_interval_array(melody)
        average_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        changes_count = 0
        if len(melody.notes) >= 3:#至少三个轮廓才有变化
            for j in range(2,len(melody.notes)):
                prev2, prev1, curr = melody.notes[j-2], melody.notes[j-1], melody.notes[j]
                dir1 = 1 if prev1 > prev2 else (-1 if prev1 < prev2 else 0)
                dir2 = 1 if curr > prev1 else (-1 if curr < prev1 else 0)
                change = 1 if dir1 != dir2 else 0
                changes_count += change
        features = [
            average_pitch,
            std_pitch,
            average_interval,
            std_interval,
            changes_count
        ]
        feature_list.append(features)
    return feature_list
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
    #提取所有旋律编码的特征向量[平均音高，音高标准差，平均音程，音程标准差，轮廓变化率]，计算距离平均值
    avg_distances=[]
    for gen in range(0,290,300):
        if gen > 290:
            break
        population = utils.load_population("population/checkpoint_gen_{gen}.json")
        feature_list = extract_all_melody_features(population)
        n = X.shape[0]#？
        X = np.array(feature_list)
        differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        squared_diffs = differences ** 2
        distances = np.sqrt(np.sum(squared_diffs, axis=2))
    
        # 取上三角平均（不包括对角线）
        avg_distance = np.mean(distances[np.triu_indices(n, k=1)])
        avg_distances.append(avg_distance)

    plt.figure(figsize=(10, 6))
    generations = range(len(avg_distances))
    # 绘制主曲线
    plt.plot(generations, avg_distances, 'b-', linewidth=2, label='种群多样性')
    
    # 添加趋势线
    if len(avg_distances) > 1:
        z = np.polyfit(generations, avg_distances, 3)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "r--", alpha=0.5, label='趋势线')
    
    plt.xlabel('进化代数')
    plt.ylabel('特征向量平均欧氏距离')
    plt.title('种群多样性演化过程')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('results/diversity_progress.png', dpi=150)
    plt.show()

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