# src/utils.py
"""
工具函数
"""

import pretty_midi
from src.config import *

def choose_by_value(val):
    # 按照权重随机选取
    from random import random
    r = random() * sum(val)
    s = 0
    for i in range(len(val)):
        s += val[i]
        if s >= r: return i
    return len(val) - 1

def midi_to_note_name(midi_number):
    """MIDI编号转换为音符名"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_number // 12 - 1
    note_index = midi_number % 12
    return f"{notes[note_index]}{octave}"

def note_name_to_midi(note_name):
    """音符名转换为MIDI编号"""
    pitch_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    # 解析音高字符串
    if len(note_name) == 2:  # 如 "C4"
        note = note_name[0]
        octave = int(note_name[1])
    else:  # 如 "C#4"
        note = note_name[:2]
        octave = int(note_name[2])
    
    return (octave + 1) * 12 + pitch_map[note]

def duration_to_ticks(duration, ticks_per_beat, time_signature=(4, 4)):
    """时值转换为MIDI ticks"""
    # duration以小节为单位
    # 1小节 = 时间签名分子拍
    beats_per_measure = time_signature[0]
    beats = duration * beats_per_measure
    return int(beats * ticks_per_beat)

def parse_time_signature(time_sig_str):
    """解析拍号字符串"""
    if '/' in time_sig_str:
        numerator, denominator = time_sig_str.split('/')
        return (int(numerator), int(denominator))
    return (4, 4)

def validate_melody(melody):
    """验证旋律的合法性"""

    if len(melody.notes) != LENGTH:
        return False, "长度不符"
    
    # 检查形如 0 - 的非法元素
    for i in range(len(melody.notes) - 1):
        if melody.notes[i] == 0 and melody.notes[i + 1] == PITCH_TYPES + 1:
            return False, "休止符后延音"
    
    # 检查过长延音
    lim = int(MAX_DURATION / MIN_DURATION)
    now = 1
    for i in range(len(melody.notes)):
        if melody.notes[i] == PITCH_TYPES + 1:
            now += 1
        elif melody.notes[i] > 0:
            now = 1
        else: now = 0
        if now > lim:
            return False, "延音过长"
    
    return True, "旋律有效"

# 下面两个函数并未被引用

def save_population(population, filename):
    """保存种群到文件（简化版，实际项目可能需要序列化）"""
    import json
    data = {
        'individuals': []
    }
    
    for ind in population.individuals:
        melody_data = {
            'notes': [],
            'time_signature': ind.melody.time_signature,
            'fitness': ind.fitness
        }
        
        for note in ind.melody.notes:
            melody_data['notes'].append({
                'pitch': note.pitch,
                'duration': note.duration,
                'start_time': note.start_time
            })
        
        data['individuals'].append(melody_data)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_population(filename):
    """从文件加载种群（简化版）"""
    import json
    from src.representation import Note, Melody, Individual, Population
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    population = Population()
    
    for ind_data in data['individuals']:
        notes = []
        for note_data in ind_data['notes']:
            note = Note(
                pitch=note_data['pitch'],
                duration=note_data['duration'],
                start_time=note_data['start_time']
            )
            notes.append(note)
        
        melody = Melody(notes, tuple(ind_data['time_signature']))
        individual = Individual(melody, ind_data['fitness'])
        population.add_individual(individual)
    
    return population

