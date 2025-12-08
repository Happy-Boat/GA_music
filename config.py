# config.py -- 全局配置参数
"""
全局配置参数设置
"""

# 遗传算法参数
POPULATION_SIZE = 20
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
ELITISM_COUNT = 2

# 音乐参数
TIME_SIGNATURE = (4, 4)  # 4/4拍
NUM_MEASURES = 4         # 4小节
MIN_DURATION = 0.125     # 八分音符（以小节为单位）
MAX_DURATION = 1.0       # 全音符
TICKS_PER_BEAT = 480     # MIDI ticks每拍

# 音符范围
NOTES = [
    'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5'
]

# 适应度函数权重
FITNESS_WEIGHTS = {
    'pitch_range': 0.2,
    'rhythm_variety': 0.2,
    'consonance': 0.3,
    'contour': 0.2,
    'repetition': 0.1
}

# 音符时长选项（以小节为单位）
DURATIONS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]

# 和谐音程（以半音数表示）
CONSONANT_INTERVALS = [0, 3, 4, 5, 7, 8, 9, 12]