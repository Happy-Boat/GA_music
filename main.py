# main.py
"""
主程序入口
"""

import argparse
import os
import datetime
from src.evolution import GeneticAlgorithm
from src.visualization import plot_fitness_progress, visualize_melody, create_evolution_report
import src.config as config

from src.visualization import current_time

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='遗传算法机器作曲')
    parser.add_argument('--method', choices=['midi', 'random'], default='random',
                       help='初始种群生成方法')
    parser.add_argument('--generations', type=int, default=100,
                       help='进化代数')
    parser.add_argument('--population', type=int, default=20,
                       help='种群大小')
    parser.add_argument('--output', type=str, default='data/outputs',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                       help='交叉率')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                       help='变异率')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f'data/outputs/{current_time}/population', exist_ok=True)
    os.makedirs('data/input_midi', exist_ok=True)
    
    # 更新配置
    config.GENERATIONS = args.generations
    config.POPULATION_SIZE = args.population
    config.CROSSOVER_RATE = args.crossover_rate
    config.MUTATION_RATE = args.mutation_rate
    
    # 创建遗传算法实例
    ga = GeneticAlgorithm()
    
    # 初始化种群
    print(f"初始化种群（方法: {args.method}）...")
    ga.initialize_population(method=args.method)
    
    # 运行进化
    print(f"开始进化（{config.GENERATIONS}代）...")
    best_individuals, fitness_history = ga.run()
    
    # 保存结果
    print("保存结果...")
    best_melody = ga.population.get_fittest().melody
    # 保存MIDI文件
    midi_dir = os.path.join('data', 'outputs', current_time)  
    midi_path = os.path.join(midi_dir, 'best_melody.mid')
    os.makedirs(midi_dir, exist_ok=True)
    best_melody.to_midi(midi_path)
    print(f"最佳旋律已保存为MIDI: {midi_path}")
    
    # 保存旋律文本
    txt_path = os.path.join(args.output, f"{current_time}/best_melody.txt")
    with open(txt_path, 'w') as f:
        f.write(best_melody.to_string())
    
    # 生成进化报告
    create_evolution_report(ga, midi_dir)
    args.visualize = True
    # 可视化
    if args.visualize:
        plot_fitness_progress(fitness_history)
    
        visualize_melody(best_melody, "最优旋律")
        
        # 比较初始和最终最佳旋律
        if len(best_individuals) > 1:
            from src.visualization import compare_melodies
            initial_melody = best_individuals[0].melody
            final_melody = best_individuals[-1].melody
            compare_melodies(initial_melody, final_melody, "初始最佳旋律", "最终最佳旋律")
    
    print("完成！")

if __name__ == "__main__":
    main()