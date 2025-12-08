# src/fitness.py
"""
适应度函数
"""

import numpy as np
from src.utils import note_name_to_midi

def pitch_range_fitness(melody):
    """音域范围适应度（避免过大或过小的音域）"""
    if not melody.notes:
        return 0
    
    # 获取所有音符的MIDI编号
    pitches = [note_name_to_midi(note.pitch) for note in melody.notes]
    
    # 计算音域
    pitch_range = max(pitches) - min(pitches)
    
    # 理想音域：1-2个八度（12-24半音）
    # 使用高斯函数评估，峰值在18半音
    ideal_range = 18
    sigma = 6
    
    # 计算适应度（范围在0-1之间）
    fitness = np.exp(-((pitch_range - ideal_range) ** 2) / (2 * sigma ** 2))
    
    return fitness

def rhythm_variety_fitness(melody):
    """节奏多样性适应度"""
    if len(melody.notes) < 2:
        return 0
    
    # 获取所有时长
    durations = [note.duration for note in melody.notes]
    
    # 计算时长的标准差（标准化）
    if len(set(durations)) == 1:
        return 0  # 所有时长相同，多样性最低
    
    std_dev = np.std(durations)
    mean_dur = np.mean(durations)
    
    # 变异系数（标准差/均值）
    if mean_dur == 0:
        return 0
    
    cv = std_dev / mean_dur
    
    # 归一化到0-1之间（假设cv在0-1之间是理想的）
    fitness = min(cv, 1.0)
    
    return fitness

def consonance_fitness(melody):
    """和谐度适应度（基于音程和谐度）"""
    if len(melody.notes) < 2:
        return 0.5  # 单音，中性分数
    
    pitches = [note_name_to_midi(note.pitch) for note in melody.notes]
    
    # 和谐音程（以半音数表示）
    consonant_intervals = [0, 3, 4, 5, 7, 8, 9, 12]
    
    # 计算相邻音符的音程
    consonant_count = 0
    total_intervals = 0
    
    for i in range(len(pitches) - 1):
        interval = abs(pitches[i + 1] - pitches[i]) % 12
        
        # 检查是否和谐
        if interval in consonant_intervals:
            consonant_count += 1
        
        total_intervals += 1
    
    if total_intervals == 0:
        return 0
    
    # 和谐音程的比例
    consonance_ratio = consonant_count / total_intervals
    
    # 给予一些权重，不完全和谐的旋律也可以接受
    fitness = 0.3 + 0.7 * consonance_ratio  # 基础分0.3
    
    return min(fitness, 1.0)

def melodic_contour_fitness(melody):
    """旋律轮廓适应度（避免过多大跳）"""
    if len(melody.notes) < 3:
        return 0.5
    
    pitches = [note_name_to_midi(note.pitch) for note in melody.notes]
    
    # 计算音程大小
    large_leaps = 0
    total_leaps = 0
    
    for i in range(len(pitches) - 1):
        interval = abs(pitches[i + 1] - pitches[i])
        total_leaps += 1
        
        # 大跳定义为超过8度（12半音）
        if interval > 12:
            large_leaps += 1
    
    if total_leaps == 0:
        return 1.0
    
    # 大跳比例
    large_leap_ratio = large_leaps / total_leaps
    
    # 适应度：大跳越少越好
    fitness = 1.0 - large_leap_ratio
    
    # 但也要避免完全没有跳跃（过于平淡）
    # 检查音程多样性
    if total_leaps > 0:
        unique_intervals = len(set(abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)))
        interval_diversity = unique_intervals / min(total_leaps, 10)  # 归一化
        
        # 结合两个因素
        fitness = 0.7 * fitness + 0.3 * interval_diversity
    
    return fitness

def repetition_fitness(melody):
    """重复度适应度（避免过多重复）"""
    if len(melody.notes) < 4:
        return 0.5
    
    pitches = [note_name_to_midi(note.pitch) for note in melody.notes]
    
    # 检查重复模式（简单检查相邻重复）
    repeated_count = 0
    for i in range(len(pitches) - 1):
        if pitches[i] == pitches[i + 1]:
            repeated_count += 1
    
    # 检查更长的重复模式（2-3个音符）
    pattern_repeats = 0
    if len(pitches) >= 6:
        for i in range(len(pitches) - 3):
            pattern = tuple(pitches[i:i+2])
            # 检查后续是否有相同模式
            for j in range(i+2, len(pitches) - 1):
                if tuple(pitches[j:j+2]) == pattern:
                    pattern_repeats += 1
                    break
    
    # 计算适应度
    max_repeats = len(pitches) - 1
    if max_repeats == 0:
        return 1.0
    
    # 惩罚重复
    fitness = 1.0 - (repeated_count / max_repeats) * 0.5 - min(pattern_repeats * 0.1, 0.3)
    
    return max(fitness, 0.1)  # 保证最低适应度

def calculate_fitness(individual, weights):
    """计算总适应度（加权组合）"""
    melody = individual.melody
    
    # 计算各分项适应度
    pitch_fitness = pitch_range_fitness(melody)
    rhythm_fitness = rhythm_variety_fitness(melody)
    consonance_fitness_val = consonance_fitness(melody)
    contour_fitness = melodic_contour_fitness(melody)
    repetition_fitness_val = repetition_fitness(melody)
    
    # 加权求和
    total_fitness = (
        weights.get('pitch_range', 0.2) * pitch_fitness +
        weights.get('rhythm_variety', 0.2) * rhythm_fitness +
        weights.get('consonance', 0.3) * consonance_fitness_val +
        weights.get('contour', 0.2) * contour_fitness +
        weights.get('repetition', 0.1) * repetition_fitness_val
    )
    
    # 归一化到0-1之间
    total_fitness = min(max(total_fitness, 0), 1)
    
    return total_fitness

def dynamic_fitness_adjustment(generation, base_weights):
    """动态调整适应度权重（随进化代数变化）"""
    # 随着进化代数的增加，逐渐增加对和谐度的要求
    adjusted_weights = base_weights.copy()
    
    # 线性调整
    progress = min(generation / 50, 1.0)  # 50代后达到最大调整
    
    # 早期更注重多样性，后期更注重和谐
    adjusted_weights['consonance'] = base_weights['consonance'] * (1 + progress * 0.5)
    adjusted_weights['rhythm_variety'] = base_weights['rhythm_variety'] * (1 - progress * 0.3)
    
    # 归一化权重
    total = sum(adjusted_weights.values())
    for key in adjusted_weights:
        adjusted_weights[key] /= total
    
    return adjusted_weights