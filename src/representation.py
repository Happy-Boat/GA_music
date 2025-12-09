# src/representation.py
"""
音乐表示和数据结构
"""

import pretty_midi
import re
from src.config import *
from src.utils import note_name_to_midi

class Melody:
    """旋律类"""
    def __init__(self, notes=None):
        self.notes = notes if notes is not None else []

    def add_note(self, note):
        """添加音符"""
        self.notes.append(note)
    
    def to_midi(self, filename):
        """导出为MIDI文件"""
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        pit, st = 0, 0
        for (i, note) in enumerate(self.notes + [0]):
            # 转换时间为秒（假设120BPM，4/4拍）
            # 1小节 = 4拍，1拍 = 60/120 = 0.5秒
            if i == len(self.notes) or note <= PITCH_TYPES:
                if pit > 0:
                    piano.notes.append(pretty_midi.Note(
                        velocity = 100,
                        pitch = note_name_to_midi(NOTES[pit - 1]),
                        start = st * (240 / TEMPO * MIN_DURATION),
                        end = i * (240 / TEMPO * MIN_DURATION)
                    ))
                pit, st = note, i
        
        midi_data.instruments.append(piano)
        midi_data.write(filename)
    
    def to_string(self):
        """转换为可读字符串"""
        result = f"Melody ({len(self.notes)} notes, {TIME_SIGNATURE[0]}/{TIME_SIGNATURE[1]}):\n"
        for i, note in enumerate(self.notes):
            result += f"  {i+1}: {note}\n"
        return result
    
    def copy(self):
        """创建副本"""
        copied_notes = [note for note in self.notes]
        return Melody(copied_notes)

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