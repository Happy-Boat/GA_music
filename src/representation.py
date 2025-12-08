# src/representation.py
"""
音乐表示和数据结构
"""

import pretty_midi
import re

class Note:
    """音符类"""
    def __init__(self, pitch, duration, start_time):
        # pitch: 音高字符串，如 'C4'
        # duration: 时值（以小节为单位）
        # start_time: 开始时间（以小节为单位）
        self.pitch = pitch
        self.duration = duration
        self.start_time = start_time
        
    def __repr__(self):
        return f"Note(pitch='{self.pitch}', duration={self.duration:.3f}, start={self.start_time:.3f})"
    
    def copy(self):
        """创建副本"""
        return Note(self.pitch, self.duration, self.start_time)
    
    def get_midi_number(self):
        """获取MIDI编号"""
        pitch_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
            'Cb': 11, 'E#': 5, 'Fb': 4, 'B#': 0,
        }

        # 尝试匹配各种格式
        match = re.match(r'^([A-G])([#bx]?b?)(-?\d+)$', self.pitch)
        if not match:
            # 第二次尝试：简单的#或b
            match = re.match(r'^([A-G])([#b]?)(-?\d+)$', self.pitch)
        
        if match:
            note_name = match.group(1)
            accidental = match.group(2)
            octave = int(match.group(3))
            
            full_name = note_name + accidental if accidental else note_name
            
            if full_name in pitch_map:
                midi_number = 12 * (octave + 1) + pitch_map[full_name]
                
                # 检查是否在有效范围内 (0-127)
                if 0 <= midi_number <= 127:
                    return midi_number
                else:
                    raise ValueError(f"MIDI编号超出范围(0-127): {self.pitch} -> {midi_number}")
        
        # 解析失败，直接报错
        raise ValueError(f"无法解析音高字符串: {self.pitch}")

class Melody:
    """旋律类"""
    def __init__(self, notes=None, time_signature=(4, 4)):
        self.notes = notes if notes is not None else []
        self.time_signature = time_signature
        self.total_length = 4  # 4小节，以小节为单位
        
    def add_note(self, note):
        """添加音符"""
        self.notes.append(note)
    
    def get_duration(self):
        """获取旋律总时长"""
        if not self.notes:
            return 0
        return max(note.start_time + note.duration for note in self.notes)
    
    def to_midi(self, filename, tempo=120):
        """导出为MIDI文件"""
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        for note in self.notes:
            # 转换时间为秒（假设120BPM，4/4拍）
            # 1小节 = 4拍，1拍 = 60/120 = 0.5秒
            start_time = note.start_time * 4 * 60.0 / tempo
            end_time = (note.start_time + note.duration) * 4 * 60.0 / tempo
            
            midi_note = pretty_midi.Note(
                velocity=100,
                pitch=note.get_midi_number(),
                start=start_time,
                end=end_time
            )
            piano.notes.append(midi_note)
        
        midi_data.instruments.append(piano)
        midi_data.write(filename)
    
    def to_string(self):
        """转换为可读字符串"""
        result = f"Melody ({len(self.notes)} notes, {self.time_signature[0]}/{self.time_signature[1]}):\n"
        for i, note in enumerate(self.notes):
            result += f"  {i+1}: {note}\n"
        return result
    
    def copy(self):
        """创建副本"""
        copied_notes = [note.copy() for note in self.notes]
        return Melody(copied_notes, self.time_signature)
    
    def get_pitch_list(self):
        """获取音高列表（MIDI编号）"""
        return [note.get_midi_number() for note in self.notes]
    
    def get_duration_list(self):
        """获取时长列表"""
        return [note.duration for note in self.notes]

class Individual:
    """个体类"""
    def __init__(self, melody, fitness=0):
        self.melody = melody
        self.fitness = fitness
        
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.3f}, notes={len(self.melody.notes)})"
    
    def copy(self):
        """创建副本"""
        return Individual(self.melody.copy(), self.fitness)

class Population:
    """种群类"""
    def __init__(self, individuals=None):
        self.individuals = individuals if individuals is not None else []
    
    def add_individual(self, individual):
        """添加个体"""
        self.individuals.append(individual)
    
    def get_fittest(self):
        """获取最优个体"""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self):
        """计算平均适应度"""
        if not self.individuals:
            return 0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    def size(self):
        """获取种群大小"""
        return len(self.individuals)
    
    def sort_by_fitness(self):
        """按适应度排序"""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    def get_individual(self, index):
        """获取指定位置的个体"""
        if 0 <= index < len(self.individuals):
            return self.individuals[index]
        return None