# src/music_operations.py
"""
音乐变换操作
"""

import random
from src.config import *
from src.utils import note_name_to_midi, midi_to_note_name

def modd(x):
    """将音符控制在指定范围内"""
    return (x - 1 + PITCH_TYPES) % PITCH_TYPES + 1

def transpose(melody):
    """移调变换（interval为半音数）"""
    me = melody.copy()
    interval = random.randint(1, PITCH_TYPES - 1)
    for i in range(len(me.notes)):
        if 1 <= me.notes[i] <= PITCH_TYPES:
            me.notes[i] = modd(me.notes[i] + interval)
    
    return me

def inversion(melody):
    """倒影变换（以axis_note为轴）"""
    me = melody.copy()
    axis_note = random.randint(1, PITCH_TYPES * 2)
    for i in range(len(me.notes)):
        if 1 <= me.notes[i] <= PITCH_TYPES:
            me.notes[i] = modd(axis_note - me.notes[i])
    
    return me

def retrograde(melody):
    """逆行变换（时间顺序反转）"""
    son = melody.copy()
    n = len(son.notes)
    for i in range(n - 1):
        if 1 <= son.notes[i] <= PITCH_TYPES and son.notes[i + 1] == PITCH_TYPES + 1:
            z = son.notes[i]; son.notes[i] = son.notes[i + 1]; son.notes[i + 1] = z
    for i in range(n // 2):
        z = son.notes[i]; son.notes[i] = son.notes[n - i - 1]; son.notes[n - i - 1] = z
    return son

def apply_musical_transformation(melody, transformation):
    """应用音乐变换的统一接口"""
    if transformation == 'transpose':
        return transpose(melody)
    elif transformation == 'inversion':
        return inversion(melody)
    elif transformation == 'retrograde':
        return retrograde(melody)
    else:
        raise ValueError(f"未知的变换类型: {transformation}")