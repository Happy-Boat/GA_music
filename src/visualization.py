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
        melody_no_zero = [note for note in melody.notes if note > 0]#去除休止符
        average_pitch = np.mean(melody_no_zero)
        std_pitch = np.std(melody_no_zero,ddof=0)
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
        ]#需要归一化操作
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
    for gen in range(0,290,10):
        if gen > 290:
            break
        population = utils.load_population("population/checkpoint_gen_{gen}.json")
        feature_list = extract_all_melody_features(population)
        n = X.shape[0]
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
    
    from config import NOTES
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), 
                           gridspec_kw={'height_ratios': [3, 2, 2]})
    
    pitches = np.array(melody)
    time_points = np.arange(len(pitches))
    
    # 1. 主图：钢琴卷帘谱（无间隔版本）
    ax1 = axes[0]
    
    # 分析音符序列
    notes = []  # (开始位置, 音高, 持续时间, 是否包含延长)
    i = 0
    note_counter = 1  # 音符计数器，用于标记
    
    while i < len(pitches):
        if pitches[i] == 0:  # 休止符
            i += 1
        elif pitches[i] == 28:  # 延长音
            # 查找前一个正常音符
            j = i - 1
            while j >= 0 and pitches[j] in [0, 28]:
                j -= 1
            
            if j >= 0:
                # 更新前一个音符的持续时间
                for idx, (start, pitch, dur, has_sustain) in enumerate(notes):
                    if start == j:
                        notes[idx] = (start, pitch, dur + 1, True)
                        break
            i += 1
        else:  # 正常音符
            start = i
            pitch_val = pitches[i]
            duration = 1
            has_sustain = False
            i += 1
            
            # 检查是否有立即的延长音
            while i < len(pitches) and pitches[i] == 28:
                duration += 1
                has_sustain = True
                i += 1
                
            notes.append((start, pitch_val, duration, has_sustain))
    
    # 绘制钢琴卷帘谱
    for note_idx, (start, pitch, duration, has_sustain) in enumerate(notes):
        # 音符主体 - 绘制一个连续矩形，没有间隔
        rect_width = duration  # 宽度等于持续时间
        rect_height = 0.8  # 音符高度
        
        # 绘制整个音符矩形（从开始位置到结束位置）
        rect = plt.Rectangle((start, pitch - rect_height/2), 
                            rect_width, rect_height,
                            facecolor='skyblue', edgecolor='navy',
                            linewidth=2, alpha=1.0, zorder=3)
        ax1.add_patch(rect)
        
        # 如果音符包含延长，在延长部分添加特殊标记
        if has_sustain and duration > 1:
            # 在整个矩形上添加延长标记
            for d in range(1, duration):
                # 在延长部分添加斜线图案
                sustain_rect = plt.Rectangle((start + d - 0.1, pitch - rect_height/2), 
                                           0.2, rect_height,
                                           facecolor='#FF9999', edgecolor='red',
                                           linewidth=1, alpha=0.7, zorder=4)
                ax1.add_patch(sustain_rect)
        
        # 标记音符开始位置（在第一拍上）
        start_marker = plt.Rectangle((start, pitch - rect_height/2), 
                                   0.1, rect_height,
                                   facecolor='darkblue', edgecolor='darkblue',
                                   linewidth=1, alpha=1.0, zorder=5)
        ax1.add_patch(start_marker)
        
        # 标记音符编号
        ax1.text(start + 0.5, pitch + 1.5, f'N{note_counter}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax1.text(start + duration/2, pitch - 1.5, f'{NOTES[pitch - 1]}', 
                    ha='center', va='top', fontsize=8, fontstyle='italic')
        
        # 标记时值
        ax1.text(start + duration/2, pitch - 3, f'{duration}拍',
                ha='center', va='top', fontsize=8, fontstyle='italic')
        
        note_counter += 1
    
    # 绘制休止符
    rest_positions = []
    for i in range(len(pitches)):
        if pitches[i] == 0:
            # 绘制休止符矩形（灰色半透明）
            rest_rect = plt.Rectangle((i, -2 - 0.4), 1, 0.8,
                                     facecolor='gray', edgecolor='darkgray',
                                     linewidth=1, alpha=0.3, zorder=2)
            ax1.add_patch(rest_rect)
            
            # 休止符标记
            ax1.scatter(i + 0.5, -2, color='darkgray', marker='$\u266B$',
                       s=150, linewidths=2, zorder=3)
            rest_positions.append(i)
    
    # 设置坐标轴范围
    all_pitches = [p for p in pitches if p not in [0, 28]]
    if all_pitches:
        min_pitch = min(all_pitches) - 8
        max_pitch = max(all_pitches) + 8
        ax1.set_ylim(min_pitch, max_pitch)
    else:
        ax1.set_ylim(-5, 35)
    
    ax1.set_xlim(-0.5, len(pitches) - 0.5)
    ax1.set_xlabel('时间位置（拍）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('音高值', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - 钢琴卷帘谱', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加时间网格线
    for x in range(0, len(pitches) + 1):
        ax1.axvline(x=x, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)
    
    # 2. 旋律轮廓图（保持不变）
    ax2 = axes[1]
    
    # 创建平滑的轮廓线（考虑延长音）
    contour_pitches = np.full(len(pitches), np.nan)
    current_pitch = np.nan
    
    for i in range(len(pitches)):
        if pitches[i] == 0:
            contour_pitches[i] = np.nan
            current_pitch = np.nan
        elif pitches[i] == 28:
            if not np.isnan(current_pitch):
                contour_pitches[i] = current_pitch
        else:
            contour_pitches[i] = pitches[i]
            current_pitch = pitches[i]
    
    # 绘制阶梯图，更好显示时值
    ax2.step(time_points, contour_pitches, where='post', 
            color='blue', linewidth=3, alpha=0.8, label='旋律轮廓')
    
    # 填充区域
    valid_indices = ~np.isnan(contour_pitches)
    if np.any(valid_indices):
        ax2.fill_between(time_points, 0, contour_pitches, 
                        where=valid_indices, alpha=0.2, color='blue')
    
    # 标记音符开始
    for i in range(len(pitches)):
        if pitches[i] > 0 and pitches[i] != 28:
            ax2.scatter(i, pitches[i], color='darkblue', s=100, 
                       zorder=5, edgecolors='white', linewidths=2)
    
    # 标记休止符位置
    for rest_pos in rest_positions:
        ax2.axvspan(rest_pos, rest_pos + 1, 
                   alpha=0.2, color='gray', zorder=1)
    
    ax2.set_xlabel('时间位置（拍）', fontsize=12, fontweight='bold')
    ax2.set_ylabel('音高值', fontsize=12, fontweight='bold')
    ax2.set_title('旋律轮廓（阶梯图）', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, len(pitches) - 0.5)
    ax2.legend(loc='upper right')
    
    # 3. 音符时值分布图（保持不变）
    ax3 = axes[2]
    
    if notes:
        durations = [note[2] for note in notes]
        note_pitches = [note[1] for note in notes]
        sustain_flags = [note[3] for note in notes]
        
        # 创建颜色数组（有延长音的音符用不同颜色）
        colors = ['#FF9999' if sustain else 'skyblue' for sustain in sustain_flags]
        
        # 绘制时值条形图
        x_positions = np.arange(len(notes))
        bars = ax3.bar(x_positions, durations, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # 添加音高标签
        for i, (bar, pitch) in enumerate(zip(bars, note_pitches)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                    f'{pitch}', ha='center', va='bottom', fontsize=9)
            
            
            ax3.text(bar.get_x() + bar.get_width()/2, -0.5,
                        f'{NOTES[pitch - 1]}', ha='center', va='top', fontsize=8,
                        rotation=45 if len(notes) > 8 else 0)
        
        # 添加延长音标记
        for i, sustain in enumerate(sustain_flags):
            if sustain:
                ax3.text(i, durations[i] + 0.2, '延长', 
                        ha='center', va='bottom', fontsize=8, color='red')
        
        ax3.set_xlabel('音符序号', fontsize=12, fontweight='bold')
        ax3.set_ylabel('时值（拍数）', fontsize=12, fontweight='bold')
        ax3.set_title('音符时值分布', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels([f'#{i+1}' for i in range(len(notes))])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加时值统计信息
        if durations:
            avg_duration = np.mean(durations)
            
            ax3.axhline(y=avg_duration, color='green', linestyle='--', 
                       alpha=0.5, label=f'平均时值: {avg_duration:.1f}拍')
            ax3.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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