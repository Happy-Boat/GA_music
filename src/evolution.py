# src/evolution.py
"""
进化过程控制
"""

import copy
import random
from src.utils import validate_melody, choose_by_value
from src.config import *
from src.genetic_operations import selection, crossover, mutation
from src.fitness import calculate_fitness, dynamic_fitness_adjustment
from src.representation import Population, Melody, Individual
from src.music_operations import apply_musical_transformation
from src.visualization import current_time

class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self):
        """初始化"""
        self.population = Population()
        self.generation = 0
        self.best_individuals = []  # 每代最佳个体
        self.fitness_history = []   # 每代平均适应度
        self.best_fitness_history = []  # 每代最佳适应度
        self.population_history = [] # 每代种群状态
    
    def record(self, ver = 0):
        # 计算适应度
        self.evaluate_population(self.population)
        self.generation += ver

        # 记录状态
        best = self.population.get_fittest()
        self.best_individuals.append(best)
        self.fitness_history.append(self.population.get_average_fitness())
        self.best_fitness_history.append(best.fitness)

        if self.generation % 10 == 0:
            self.population_history.append(self.population)
            print(f"第 {self.generation} 代: 平均适应度={self.population.get_average_fitness():.3f}, "
                  f"最佳适应度={best.fitness:.3f}")

    def initialize_population(self, method='random'):
        """初始化种群"""
        assert(method == 'random')
        
        while self.population.size() < POPULATION_SIZE:
            # 随机生成旋律
            t = Melody([choose_by_value(PROB_TABLE) for i in range(LENGTH)])
            if validate_melody(t):
                self.population.add_individual(Individual(t))
        self.record()

        print(f"初始种群创建完成，平均适应度: {self.population.get_average_fitness():.3f}")
        
    def evaluate_population(self, population):
        """评估种群"""
        # 动态调整权重
        weights = dynamic_fitness_adjustment(self.generation, FITNESS_WEIGHTS)
        
        for individual in population.individuals:
            individual.fitness = calculate_fitness(individual, weights)
        
        # 按适应度排序
        population.sort_by_fitness()
    
    def evolve(self):
        """执行一代进化"""
        if self.population is None or self.population.size() == 0:
            raise ValueError("种群未初始化或为空")
        
        new_population = Population()
        
        # 精英保留
        self.population.sort_by_fitness()
        for i in range(min(ELITISM_COUNT, self.population.size())):
            new_population.add_individual(self.population.individuals[i].copy())
        
        transformations = ['transpose', 'inversion', 'retrograde']

        # 生成后代
        while new_population.size() < POPULATION_SIZE:
            # 选择父代
            parent1, parent2 = selection(self.population)
            
            # 交叉
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            if random.random() < MUTATION_RATE:
                child1 = mutation(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutation(child2)
            
            # 应用音乐变换（随机）
            if random.random() < TRANSFORM_RATE:  # 10%的概率应用变换
                child1.melody = apply_musical_transformation(child1.melody, random.choice(transformations))
            if random.random() < TRANSFORM_RATE:  # 10%的概率应用变换
                child2.melody = apply_musical_transformation(child2.melody, random.choice(transformations))
            
            assert(validate_melody(child1.melody))
            assert(validate_melody(child2.melody))
            # 添加到新种群
            if new_population.size() < POPULATION_SIZE:
                new_population.add_individual(child1)
            if new_population.size() < POPULATION_SIZE:
                new_population.add_individual(child2)

        self.population = new_population
        self.record(1)
    
    def run(self):
        """运行完整进化过程"""

        print(f"开始进化，共 {GENERATIONS} 代")
        
        for gen in range(GENERATIONS):
            self.evolve()
            
            # 每10代保存检查点
            if gen % 10 == 0:
                self.save_checkpoint(gen)
        
        print(f"进化完成，最终最佳适应度: {self.population.get_fittest().fitness:.3f}")
        
        return self.best_individuals, self.fitness_history,self.population_history
    
    def save_checkpoint(self, generation):
        """保存检查点"""
        import os
        import json
        
        checkpoint_dir = f'data/outputs/{current_time}/population'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_gen_{generation}.json')
        
        # 保存最佳个体
        best_individual = self.population.get_fittest()
        
        checkpoint_data = {
            'generation': generation,
            'best_fitness': best_individual.fitness,
            'average_fitness': self.population.get_average_fitness(),
            'best_melody': {
                'notes': best_individual.melody.notes,
                'time_signature': TIME_SIGNATURE
            }
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # 保存MIDI文件
        midi_file = os.path.join(checkpoint_dir, f'best_gen_{generation}.mid')
        best_individual.melody.to_midi(midi_file)
        
        print(f"检查点保存: {checkpoint_file}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        # 实现检查点加载逻辑
        pass
