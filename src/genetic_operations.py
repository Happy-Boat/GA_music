# src/genetic_operations.py
"""
遗传操作
"""

import random
import copy
from src.representation import Individual
from src.utils import choose_by_value, validate_melody
from src.config import *

def selection(population, choose_size = 2):
    """选择操作"""
    if population.size() <= choose_size:
        return population
    ret = []
    prob_table = [x.fitness for x in population.individuals] # 以 fitness 为权重，从亲本中选出 _ 个
    while len(ret) < choose_size:
        i = choose_by_value(prob_table)
        if not(i in ret): ret.append(i)
    return [population.individuals[i] for i in ret]

def crossover(parent1, parent2):
    """交叉操作"""
    melody1 = parent1.melody
    melody2 = parent2.melody
    
    n = min(len(melody1.notes), len(melody2.notes))
    
    try_count = 0
    while try_count < 20:
        m1, m2 = melody1.copy(), melody2.copy()
        l = random.randint(0, n - 1)
        r = random.randint(0, n - 1)# 交换两片段中的 [l, r] 区间
        if l > r: z = l; l = r; r = z
        for i in range(l, r + 1):
            z = m1.notes[i]; m1.notes[i] = m2.notes[i]; m2.notes[i] = z
        if validate_melody(m1) and validate_melody(m2):
            return Individual(m1), Individual(m2)
        try_count += 1

    return parent1, parent2

def mutation(individual):
    """变异操作"""
    melody = individual.melody
    n = len(melody.notes)
    try_count = 0
    while try_count < 20:
        me = melody.copy()
        for i in range(n):
            if random.random() < MUTATION_RATE * 2 / n: # 这里是允许一次变异可能修改多个音符，不知道是否有必要，以及这里的参数可以调整
                me.notes[i] = choose_by_value(PROB_TABLE) # 修改音符

        if validate_melody(me):
            return Individual(me)
        try_count += 1

    return individual