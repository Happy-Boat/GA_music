# src/music_operations.py
"""
音乐变换操作
"""

import random
from src.utils import note_name_to_midi, midi_to_note_name

def transpose(melody, interval):
    """移调变换（interval为半音数）"""
    transposed_melody = melody.copy()
    
    for note in transposed_melody.notes:
        # 获取当前MIDI编号
        current_midi = note_name_to_midi(note.pitch)
        # 移调
        new_midi = current_midi + interval
        
        # 确保在合理范围内（MIDI 21-108）
        new_midi = max(21, min(108, new_midi))
        
        # 转换回音符名
        note.pitch = midi_to_note_name(new_midi)
    
    return transposed_melody

def inversion(melody, axis_note):
    """倒影变换（以axis_note为轴）"""
    inverted_melody = melody.copy()
    
    # 获取轴音符的MIDI编号
    axis_midi = note_name_to_midi(axis_note)
    
    for note in inverted_melody.notes:
        # 获取当前MIDI编号
        current_midi = note_name_to_midi(note.pitch)
        # 计算倒影
        inverted_midi = 2 * axis_midi - current_midi
        
        # 确保在合理范围内
        inverted_midi = max(21, min(108, inverted_midi))
        
        # 转换回音符名
        note.pitch = midi_to_note_name(inverted_midi)
    
    return inverted_melody

def retrograde(melody):
    """逆行变换（时间顺序反转）"""
    retrograde_melody = melody.copy()
    
    # 获取原始音符列表
    notes = retrograde_melody.notes
    
    if not notes:
        return retrograde_melody
    
    # 获取旋律总时长
    total_duration = melody.get_duration()
    
    # 创建逆行的音符
    new_notes = []
    for i, note in enumerate(reversed(notes)):
        # 计算新的开始时间
        new_start_time = total_duration - (note.start_time + note.duration)
        new_note = note.copy()
        new_note.start_time = max(0, new_start_time)
        new_notes.append(new_note)
    
    # 按开始时间排序
    new_notes.sort(key=lambda x: x.start_time)
    retrograde_melody.notes = new_notes
    
    return retrograde_melody

def rhythmic_variation(melody):
    """节奏变化"""
    varied_melody = melody.copy()
    
    # 可以改变音符的时长或开始时间
    # 这里简单实现：随机改变一些音符的时长
    for note in varied_melody.notes:
        if random.random() < 0.3:  # 30%的概率改变
            # 随机选择新的时长（在合理范围内）
            new_duration = note.duration * random.uniform(0.5, 2.0)
            # 限制时长范围
            note.duration = max(0.125, min(2.0, new_duration))
    
    # 重新排序以确保时间顺序
    varied_melody.notes.sort(key=lambda x: x.start_time)
    
    return varied_melody

def apply_musical_transformation(melody, transformation, **params):
    """应用音乐变换的统一接口"""
    if transformation == 'transpose':
        interval = params.get('interval', 0)
        return transpose(melody, interval)
    elif transformation == 'inversion':
        axis_note = params.get('axis_note', 'C4')
        return inversion(melody, axis_note)
    elif transformation == 'retrograde':
        return retrograde(melody)
    elif transformation == 'rhythmic_variation':
        return rhythmic_variation(melody)
    else:
        raise ValueError(f"未知的变换类型: {transformation}")