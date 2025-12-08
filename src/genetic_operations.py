# src/genetic_operations.py
"""
遗传操作
"""

import random
import copy

def selection(population, method='tournament', tournament_size=3):
    """选择操作"""
    if method == 'tournament':
        # 锦标赛选择
        tournament = random.sample(population.individuals, min(tournament_size, len(population.individuals)))
        return max(tournament, key=lambda x: x.fitness)
    elif method == 'roulette':
        # 轮盘赌选择
        total_fitness = sum(ind.fitness for ind in population.individuals)
        if total_fitness == 0:
            return random.choice(population.individuals)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population.individuals:
            current += ind.fitness
            if current > pick:
                return ind
        return population.individuals[-1]
    else:
        # 默认使用精英选择
        population.sort_by_fitness()
        return population.individuals[0]

def crossover(parent1, parent2, method='single_point'):
    """交叉操作"""
    melody1 = parent1.melody
    melody2 = parent2.melody
    
    if not melody1.notes or not melody2.notes:
        # 如果任一旋律为空，返回父代副本
        child1 = parent1.copy()
        child2 = parent2.copy()
        return child1, child2
    
    if method == 'single_point':
        # 单点交叉
        # 随机选择一个交叉点（基于音符索引）
        min_notes = min(len(melody1.notes), len(melody2.notes))
        if min_notes <= 1:
            child1 = parent1.copy()
            child2 = parent2.copy()
        else:
            crossover_point = random.randint(1, min_notes - 1)
            
            # 创建子代旋律
            child1_notes = melody1.notes[:crossover_point] + melody2.notes[crossover_point:]
            child2_notes = melody2.notes[:crossover_point] + melody1.notes[crossover_point:]
            
            child1_melody = copy.deepcopy(melody1)
            child1_melody.notes = child1_notes
            
            child2_melody = copy.deepcopy(melody2)
            child2_melody.notes = child2_notes
            
            child1 = parent1.__class__(child1_melody)
            child2 = parent2.__class__(child2_melody)
    
    elif method == 'uniform':
        # 均匀交叉
        max_notes = max(len(melody1.notes), len(melody2.notes))
        child1_notes = []
        child2_notes = []
        
        for i in range(max_notes):
            if i < len(melody1.notes) and i < len(melody2.notes):
                if random.random() < 0.5:
                    child1_notes.append(copy.deepcopy(melody1.notes[i]))
                    child2_notes.append(copy.deepcopy(melody2.notes[i]))
                else:
                    child1_notes.append(copy.deepcopy(melody2.notes[i]))
                    child2_notes.append(copy.deepcopy(melody1.notes[i]))
            elif i < len(melody1.notes):
                child1_notes.append(copy.deepcopy(melody1.notes[i]))
                child2_notes.append(copy.deepcopy(melody1.notes[i]))
            else:
                child1_notes.append(copy.deepcopy(melody2.notes[i]))
                child2_notes.append(copy.deepcopy(melody2.notes[i]))
        
        child1_melody = copy.deepcopy(melody1)
        child1_melody.notes = child1_notes
        
        child2_melody = copy.deepcopy(melody2)
        child2_melody.notes = child2_notes
        
        child1 = parent1.__class__(child1_melody)
        child2 = parent2.__class__(child2_melody)
    
    else:  # 两点交叉
        # 两点交叉
        min_notes = min(len(melody1.notes), len(melody2.notes))
        if min_notes <= 2:
            child1 = parent1.copy()
            child2 = parent2.copy()
        else:
            point1 = random.randint(1, min_notes - 2)
            point2 = random.randint(point1 + 1, min_notes - 1)
            
            child1_notes = (melody1.notes[:point1] + 
                          melody2.notes[point1:point2] + 
                          melody1.notes[point2:])
            child2_notes = (melody2.notes[:point1] + 
                          melody1.notes[point1:point2] + 
                          melody2.notes[point2:])
            
            child1_melody = copy.deepcopy(melody1)
            child1_melody.notes = child1_notes
            
            child2_melody = copy.deepcopy(melody2)
            child2_melody.notes = child2_notes
            
            child1 = parent1.__class__(child1_melody)
            child2 = parent2.__class__(child2_melody)
    
    return child1, child2

def mutation(individual, mutation_rate, note_set):
    """变异操作"""
    melody = individual.melody
    
    for note in melody.notes:
        if random.random() < mutation_rate:
            # 音符替换
            note.pitch = random.choice(note_set)
        
        if random.random() < mutation_rate * 0.5:
            # 时值改变
            note.duration = note.duration * random.uniform(0.5, 1.5)
            # 限制时长范围
            note.duration = max(0.125, min(2.0, note.duration))
    
    # 随机插入/删除音符
    if random.random() < mutation_rate * 0.3 and melody.notes:
        # 删除一个音符
        if len(melody.notes) > 3:
            del_index = random.randint(0, len(melody.notes) - 1)
            del melody.notes[del_index]
    
    if random.random() < mutation_rate * 0.3:
        # 插入一个新音符
        if melody.notes:
            # 在现有音符之间插入
            insert_index = random.randint(0, len(melody.notes))
            if insert_index == 0:
                start_time = 0
            elif insert_index == len(melody.notes):
                start_time = melody.notes[-1].start_time + melody.notes[-1].duration
            else:
                start_time = melody.notes[insert_index-1].start_time + melody.notes[insert_index-1].duration
            
            # 确保不超过总长度
            if start_time < melody.total_length:
                duration = random.choice([0.125, 0.25, 0.375, 0.5])
                if start_time + duration <= melody.total_length:
                    new_note = copy.deepcopy(melody.notes[0])  # 复制一个音符作为模板
                    new_note.pitch = random.choice(note_set)
                    new_note.duration = duration
                    new_note.start_time = start_time
                    melody.notes.insert(insert_index, new_note)
    
    # 重新排序以确保时间顺序
    melody.notes.sort(key=lambda x: x.start_time)
    
    return individual