# src/initialization.py
"""
初始种群生成
"""

import os
import random
from src.representation import Individual, Population
from src.utils import generate_random_melody
from config import NOTES

def load_midi_files(directory):
    """加载MIDI文件目录"""
    import pretty_midi
    midi_files = []
    
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.mid', '.midi')):
            filepath = os.path.join(directory, filename)
            try:
                midi_data = pretty_midi.PrettyMIDI(filepath)
                midi_files.append((filename, midi_data))
                print(f"成功加载: {filename}")
            except Exception as e:
                print(f"加载失败 {filename}: {e}")
    
    return midi_files

def extract_segments(midi_data, num_segments=1, segment_length=4):
    """从MIDI中提取片段（简化版）"""
    segments = []
    
    # 获取所有音符
    all_notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)
    
    if not all_notes:
        return segments
    
    # 按时间排序
    all_notes.sort(key=lambda x: x.start)
    
    # 总时长（秒）
    total_time = midi_data.get_end_time()
    
    # 转换为小节（假设120BPM，4/4拍）
    # 1小节 = 4拍 = 4 * 60/120 = 2秒
    total_measures = total_time / 2.0
    
    if total_measures < segment_length:
        return segments
    
    # 随机提取片段
    for _ in range(num_segments):
        # 随机起始位置（以小节为单位）
        start_measure = random.uniform(0, total_measures - segment_length)
        start_time = start_measure * 2.0  # 转换为秒
        end_time = start_time + segment_length * 2.0
        
        # 提取该时间段内的音符
        segment_notes = []
        for note in all_notes:
            if start_time <= note.start < end_time:
                segment_notes.append(note)
        
        if segment_notes:
            segments.append(segment_notes)
    
    return segments

def random_melody_generation(note_set, num_measures, time_signature):
    """随机生成旋律"""
    return generate_random_melody(note_set, num_measures, time_signature)

def initialize_population_from_midi(midi_dir, pop_size):
    """方法(a)：从MIDI文件生成初始种群"""
    from src.representation import Note, Melody, Individual, Population
    from src.utils import midi_to_note_name
    
    population = Population()
    midi_files = load_midi_files(midi_dir)
    
    if not midi_files:
        print("未找到MIDI文件，使用随机生成")
        return initialize_population_random(pop_size, NOTES, 4)
    
    all_segments = []
    
    for filename, midi_data in midi_files:
        segments = extract_segments(midi_data, num_segments=5, segment_length=4)
        
        for segment_notes in segments:
            if segment_notes:
                # 转换为我们的Note对象
                notes = []
                for pm_note in segment_notes:
                    # 转换时间为小节（假设120BPM）
                    start_measure = pm_note.start / 2.0  # 2秒一小节
                    duration_measure = (pm_note.end - pm_note.start) / 2.0
                    
                    # 确保时长合理
                    if duration_measure > 0:
                        note = Note(
                            pitch=midi_to_note_name(pm_note.pitch),
                            duration=min(duration_measure, 1.0),  # 限制最大时长
                            start_time=start_measure
                        )
                        notes.append(note)
                
                if notes:
                    melody = Melody(notes, time_signature=(4, 4))
                    all_segments.append(melody)
    
    # 如果提取的片段不够，用随机生成补充
    while len(all_segments) < pop_size:
        melody = random_melody_generation(NOTES, 4, (4, 4))
        all_segments.append(melody)
    
    # 创建个体
    for i in range(pop_size):
        melody = all_segments[i % len(all_segments)].copy()
        individual = Individual(melody)
        population.add_individual(individual)
    
    return population

def initialize_population_random(pop_size, note_set, num_measures):
    """方法(b)：随机生成初始种群"""
    from src.representation import Individual, Population
    
    population = Population()
    
    for i in range(pop_size):
        melody = random_melody_generation(note_set, num_measures, (4, 4))
        individual = Individual(melody)
        population.add_individual(individual)
    
    return population

def get_initial_population(method='random', **kwargs):
    """获取初始种群（统一接口）"""
    if method == 'midi':
        midi_dir = kwargs.get('midi_dir', 'data/input_midi')
        pop_size = kwargs.get('pop_size', 20)
        return initialize_population_from_midi(midi_dir, pop_size)
    else:  # random
        pop_size = kwargs.get('pop_size', 20)
        note_set = kwargs.get('note_set', NOTES)
        num_measures = kwargs.get('num_measures', 4)
        return initialize_population_random(pop_size, note_set, num_measures)