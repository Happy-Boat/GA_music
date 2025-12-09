# config.py -- 全局配置参数
"""
全局配置参数设置
"""

# 遗传算法参数
POPULATION_SIZE = 20
GENERATIONS = 300
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TRANSFORM_RATE = 0.1
ELITISM_COUNT = 2 # 每次进化将亲本中最优的 _ 个个体直接复制到下一代

# 音乐参数
TEMPO = 120
TIME_SIGNATURE = (4, 4)  # 4/4拍
NUM_MEASURES = 4         # 4小节
MIN_DURATION = 0.125     # 八分音符（以小节为单位）
MAX_DURATION = 1.0       # 全音符
TICKS_PER_BEAT = 480     # MIDI ticks每拍
LENGTH = NUM_MEASURES * int(1 / MIN_DURATION) # 用于存储乐谱的列表长度，为固定值

# 音符范围
NOTES = [
    'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5'
]
PITCH_TYPES = len(NOTES) # 音高种类数
PROB_TABLE = [1] + [2] * PITCH_TYPES + [20] # 初始随机生成，以及变异时，随机选取音符的概率权重

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

# 大小调
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

