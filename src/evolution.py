# src/evolution.py
"""
进化过程控制
"""

import copy
import random
from src.genetic_operations import selection, crossover, mutation
from src.fitness import calculate_fitness, dynamic_fitness_adjustment
from src.representation import Population
import config

class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self):
        """初始化"""
        self.config = config
        self.population = None
        self.generation = 0
        self.best_individuals = []  # 每代最佳个体
        self.fitness_history = []   # 每代平均适应度
        self.best_fitness_history = []  # 每代最佳适应度
        
    def initialize_population(self, method='random'):
        """初始化种群"""
        from src.initialization import get_initial_population
        
        if method == 'midi':
            self.population = get_initial_population(
                method='midi',
                midi_dir='data/input_midi',
                pop_size=self.config.POPULATION_SIZE
            )
        else:  # random
            self.population = get_initial_population(
                method='random',
                pop_size=self.config.POPULATION_SIZE,
                note_set=self.config.NOTES,
                num_measures=self.config.NUM_MEASURES
            )
        
        # 计算初始适应度
        self.evaluate_population(self.population)
        self.generation = 0
        
        # 记录初始状态
        self.best_individuals.append(self.population.get_fittest())
        self.fitness_history.append(self.population.get_average_fitness())
        self.best_fitness_history.append(self.population.get_fittest().fitness)
        
        print(f"初始种群创建完成，平均适应度: {self.population.get_average_fitness():.3f}")
        
    def evaluate_population(self, population):
        """评估种群"""
        # 动态调整权重
        weights = dynamic_fitness_adjustment(self.generation, self.config.FITNESS_WEIGHTS)
        
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
        for i in range(min(self.config.ELITISM_COUNT, self.population.size())):
            new_population.add_individual(self.population.individuals[i].copy())
        
        # 生成后代
        while new_population.size() < self.config.POPULATION_SIZE:
            # 选择父代
            parent1 = selection(self.population, method='tournament')
            parent2 = selection(self.population, method='tournament')
            
            # 交叉
            if random.random() < self.config.CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2, method='uniform')
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            if random.random() < self.config.MUTATION_RATE:
                child1 = mutation(child1, self.config.MUTATION_RATE, self.config.NOTES)
            
            if random.random() < self.config.MUTATION_RATE:
                child2 = mutation(child2, self.config.MUTATION_RATE, self.config.NOTES)
            
            # 应用音乐变换（随机）
            if random.random() < 0.1:  # 10%的概率应用变换
                from src.music_operations import apply_musical_transformation
                transformations = ['transpose', 'inversion', 'retrograde', 'rhythmic_variation']
                transformation = random.choice(transformations)
                
                if transformation == 'transpose':
                    child1.melody = apply_musical_transformation(child1.melody, transformation, interval=random.randint(-5, 5))
                elif transformation == 'inversion':
                    child1.melody = apply_musical_transformation(child1.melody, transformation, axis_note=random.choice(self.config.NOTES))
                else:
                    child1.melody = apply_musical_transformation(child1.melody, transformation)
            
            # 添加到新种群
            if new_population.size() < self.config.POPULATION_SIZE:
                new_population.add_individual(child1)
            if new_population.size() < self.config.POPULATION_SIZE:
                new_population.add_individual(child2)
        
        # 评估新种群
        self.population = new_population
        self.evaluate_population(self.population)
        self.generation += 1
        
        # 记录历史
        best_individual = self.population.get_fittest()
        self.best_individuals.append(best_individual)
        self.fitness_history.append(self.population.get_average_fitness())
        self.best_fitness_history.append(best_individual.fitness)
        
        # 输出进度
        if self.generation % 10 == 0:
            print(f"第 {self.generation} 代: 平均适应度={self.population.get_average_fitness():.3f}, "
                  f"最佳适应度={best_individual.fitness:.3f}")
    
    def run(self, generations=None):
        """运行完整进化过程"""
        if generations is None:
            generations = self.config.GENERATIONS
        
        print(f"开始进化，共 {generations} 代")
        
        for gen in range(generations):
            self.evolve()
            
            # 每10代保存检查点
            if gen % 10 == 0:
                self.save_checkpoint(gen)
        
        print(f"进化完成，最终最佳适应度: {self.population.get_fittest().fitness:.3f}")
        
        return self.best_individuals, self.fitness_history
    
    def save_checkpoint(self, generation):
        """保存检查点"""
        import os
        import json
        
        checkpoint_dir = 'data/population'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_gen_{generation}.json')
        
        # 保存最佳个体
        best_individual = self.population.get_fittest()
        
        checkpoint_data = {
            'generation': generation,
            'best_fitness': best_individual.fitness,
            'average_fitness': self.population.get_average_fitness(),
            'best_melody': {
                'notes': [],
                'time_signature': best_individual.melody.time_signature
            }
        }
        
        for note in best_individual.melody.notes:
            checkpoint_data['best_melody']['notes'].append({
                'pitch': note.pitch,
                'duration': note.duration,
                'start_time': note.start_time
            })
        
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