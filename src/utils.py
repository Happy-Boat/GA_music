# src/utils.py
"""
工具函数
"""

import pretty_midi

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
    if not melody.notes:
        return False, "旋律为空"
    
    # 检查音符是否重叠（简单的检查）
    for i in range(len(melody.notes) - 1):
        note1 = melody.notes[i]
        note2 = melody.notes[i + 1]
        if note1.start_time + note1.duration > note2.start_time:
            return False, f"音符{i}和{i+1}重叠"
    
    # 检查是否超出总长度
    total_length = melody.get_duration()
    if total_length > melody.total_length:
        return False, f"旋律长度{total_length:.2f}超过限制{melody.total_length}"
    
    return True, "旋律有效"

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

def generate_random_melody(note_set, num_measures=4, time_signature=(4, 4), min_duration=0.125):
    """随机生成旋律"""
    import random
    from src.representation import Note, Melody
    
    melody = Melody(time_signature=time_signature)
    melody.total_length = num_measures
    
    current_time = 0
    max_time = num_measures
    
    while current_time < max_time:
        # 随机选择音符
        pitch = random.choice(note_set)
        
        # 随机选择时长（避免过长）
        duration = random.choice([0.125, 0.25, 0.375, 0.5, 0.75])
        
        # 确保不超过最大时间
        if current_time + duration > max_time:
            duration = max_time - current_time
        
        # 创建音符
        note = Note(pitch, duration, current_time)
        melody.add_note(note)
        
        # 更新当前时间
        current_time += duration
        
        # 随机添加休止符的概率
        if random.random() < 0.3 and current_time < max_time:
            rest_duration = random.choice([0.125, 0.25, 0.375])
            if current_time + rest_duration <= max_time:
                current_time += rest_duration
    
    return melody