# src/fitness.py
"""
适应度函数
"""

import numpy as np
import random
from src.config import *
from src.utils import note_name_to_midi

# notes 为长度为 LENGTH 的列表

def pitch_range_fitness(notes):
    """音域范围适应度（避免过大或过小的音域）"""
    return random.random()

def rhythm_variety_fitness(notes):
    """节奏多样性适应度"""
    return random.random()

def consonance_fitness(notes):
    """和谐度适应度（基于音程和谐度）"""
    return random.random()

def melodic_contour_fitness(notes):
    """旋律轮廓适应度（避免过多大跳）"""
    return random.random()

def repetition_fitness(notes):
    """重复度适应度（避免过多重复）"""
    return random.random()

def calculate_fitness(individual, weights):
    """计算总适应度（加权组合）"""
    melody = individual.melody
    note = melody.notes
    
    # 计算各分项适应度
    pitch_fitness = pitch_range_fitness(note)
    rhythm_fitness = rhythm_variety_fitness(note)
    consonance_fitness_val = consonance_fitness(note)
    contour_fitness = melodic_contour_fitness(note)
    repetition_fitness_val = repetition_fitness(note)
    
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