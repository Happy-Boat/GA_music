# src/fitness.py
"""
适应度函数
"""

import numpy as np
import random
from numpy import sqrt, log, exp
import scipy.stats as ss
from src.config import *
from src.utils import note_name_to_midi
from src.config import *

# notes 为长度为 LENGTH 的列表
def calculate_fitness(individual, weights):
    """计算总适应度（加权组合）"""
    melody = individual.melody
    flatnote = melody.notes
    num_rests=flatnote.count(0)
    intervals=[] # 相邻两音的音程, 忽略休止符
    rhythmic_value=[] # 所有时值, 包括休止符, 八分音符=0.5, 全音符=4
    pitches=[] # 所有音高, 只计起拍
    dissonants=0 # 不和谐音程个数
    seconds=0 # 二度音程个数
    strong_beats=1 # 强拍起拍数
    off_scale_minor=0 # 以第一个音为基准构建小调, 不在该小调中的音数
    off_scale_major=0 # 以第一个音为基准构建大调, 不在该大调中的音数
    larger_than_octave=0 # 超八度音程个数
    prev_note=-1
    initial_note=-1

    for i, cur_note in enumerate(flatnote):
        if i == 0:
            if cur_note == 0 or cur_note == 28:
                return 0
            initial_note = cur_note
            prev_note = cur_note
            rhythmic_value.append(0.5)
            pitches.append(cur_note)

        if i != 0:
            # 如果该排是起拍
            if cur_note != 0 and cur_note != 28:
                rhythmic_value.append(0.5) # 添加时值
                # 以下计算并判断音程
                note_dif=abs(cur_note-prev_note)
                if note_dif == 6 or note_dif == 10 or note_dif == 11 or note_dif >= 13:
                    dissonants += 1
                if note_dif == 1 or note_dif == 2:
                    seconds += 1
                if note_dif >= 13:
                    larger_than_octave += 1
                if (abs(initial_note-cur_note) % 12) not in MINOR:
                    off_scale_minor += 1
                if (abs(initial_note-cur_note) % 12) not in MAJOR:
                    off_scale_major += 1
                intervals.append(note_dif)
                # 以下将该音添加进 pitches
                pitches.append(cur_note)
                prev_note = cur_note
                # 以下判断该拍是否是强拍
                if i % 8 == 0 or i % 8 == 4:
                    strong_beats += 1

            # 如果该拍是休止
            if cur_note == 0:
                if flatnote[i-1] == 0:
                    rhythmic_value[-1] += 0.5
                else:
                    rhythmic_value.append(0.5)

            # 如果该拍是延长
            if cur_note == 28:
                rhythmic_value[-1] += 0.5

    # 计算各分项适应度
    dissonants_ratio=dissonants/len(intervals) # 不和谐音程占比
    seconds_ratio=seconds/len(intervals) # 二度音程占比
    large_oct_ratio=larger_than_octave/len(intervals) # 超八度音程占比
    rests_ratio=num_rests/len(flatnote) # 休止占比
    pitches_mean=(np.mean(pitches)-1)/26 # 平均音高
    pitches_std=(np.std(pitches))/13 # 音高标准差
    log_rhy_val=np.log1p(rhythmic_value) # 各音时值的对数
    rhy_val_mean=np.mean(log_rhy_val)/np.log(5) # 各音时值的对数取平均 / 全音符时值的对数
    rhy_val_std=np.std(log_rhy_val)/np.log(5) # 各音时值的对数的标准差
    strong_beats_ratio=strong_beats/8 # 强拍中是起拍的占比
    
    '''if len(bar_harm_value[0]) == 0 or len(bar_harm_value[1]) == 0 or len(bar_harm_value[2]) == 0 or len(bar_harm_value[3]) == 0:
        return 0
    sum1 = -np.mean(bar_harm_value[0])-np.mean(bar_harm_value[1])-np.mean(bar_harm_value[2])-np.mean(bar_harm_value[3])
    sum2 = -np.var(bar_harm_value[0])-np.var(bar_harm_value[1])-np.var(bar_harm_value[2])-np.var(bar_harm_value[3])'''

    # 加权求和

    total_fitness_1 = (8*exp(-0.5*(dissonants_ratio/0.045)**2)
                     +2*exp(-0.5*((seconds_ratio-0.5)/0.078)**2)
                     +exp(-0.5*((rests_ratio-0.105)/0.092)**2)
                     +exp(-0.5*((large_oct_ratio-0.007)/0.01)**2)
                     +exp(-0.5*((pitches_mean-0.564)/0.065)**2)
                     +exp(-0.5*((pitches_std-0.053)/0.013)**2)
                     +exp(-0.5*((rhy_val_mean-0.282)/0.12)**2)
                     +exp(-0.5*((rhy_val_std-0.156)/0.058)**2)
                     +2*exp(-0.5*((strong_beats_ratio-0.788)/0.219)**2)
                     +4/(min(off_scale_minor,off_scale_major)+1)
    )
    '''total_fitness_2 = (100 + sum1 + sum2 + 1/(min(off_scale_minor,off_scale_major)+1) 
                       +exp(-0.5*((rests_ratio-0.105)/0.092)**2)
                       +exp(-0.5*((rhy_val_mean-0.282)/0.12)**2)
                       +exp(-0.5*((rhy_val_std-0.156)/0.058)**2)
                       +exp(-0.5*((strong_beats_ratio-0.788)/0.219)**2)
    )'''
    return total_fitness_1

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
    
    return base_weights