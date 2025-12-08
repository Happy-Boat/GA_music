# README.md

# 遗传算法机器作曲系统

## 项目简介

本项目实现了一个基于遗传算法的自动作曲系统。系统能够生成和进化旋律，通过适应度函数指导旋律向更符合音乐美学规则的方向发展。

## 功能特性

1. **多种初始种群生成方式**
   - 从MIDI文件导入片段
   - 随机生成旋律

2. **完整的遗传算法流程**
   - 选择、交叉、变异操作
   - 精英保留策略
   - 多代进化

3. **丰富的音乐变换操作**
   - 移调变换
   - 倒影变换
   - 逆行变换
   - 节奏变化

4. **多维适应度评估**
   - 音域范围适应度
   - 节奏多样性适应度
   - 和谐度适应度
   - 旋律轮廓适应度
   - 重复度适应度

5. **可视化与分析**
   - 适应度进化曲线
   - 旋律可视化（钢琴卷帘）
   - 进化过程报告

## 安装与运行

### 环境要求

- Python 3.8+
- 依赖包（见requirements.txt）

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd genetic-algorithm-music

# 安装依赖
conda create -n GA_music python=3.10 -y
conda activate GA_music
pip install -r requirements.txt
```

## 运行程序

```bash
# 使用随机生成的初始种群（默认）
python main.py

# 从MIDI文件生成初始种群
python main.py --method midi

# 自定义参数
python main.py --generations 200 --population 30 --crossover-rate 0.85 --mutation-rate 0.15 --visualize

# 查看所有参数
python main.py --help
```

## 项目结构
```text
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
├── config.py                 # 全局配置参数
├── main.py                   # 主程序入口
│
├── data/                     # 数据目录
│   ├── input_midi/           # 原始MIDI文件（用于初始种群）
│   ├── population/           # 保存的种群数据
│   └── results/              # 生成的结果文件
│
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── initialization.py     # 初始种群生成
│   ├── representation.py     # 音乐表示和数据结构
│   ├── genetic_operations.py # 遗传操作
│   ├── fitness.py           # 适应度函数
│   ├── evolution.py         # 进化过程控制
│   ├── music_operations.py  # 音乐变换操作
│   ├── utils.py             # 工具函数
│   └── visualization.py     # 可视化功能
```

## 配置说明

主要配置参数在`config.py`中：

- `POPULATION_SIZE`: 种群大小
- `GENERATIONS`: 进化代数
- `CROSSOVER_RATE`: 交叉率
- `MUTATION_RATE`: 变异率
- `TIME_SIGNATURE`: 拍号
- `NUM_MEASURES`: 小节数
- `NOTES`: 可用音符集合
- `FITNESS_WEIGHTS`: 适应度函数权重

## 使用示例

1. **基本使用**
   ```bash
   python main.py --generations 50 --visualize
   ```

2. **从MIDI文件初始化**
   ```bash
   # 将MIDI文件放入data/input_midi/目录
   python main.py --method midi --generations 100
   ```

3. **调整适应度权重**
   ```python
   # 在config.py中修改
   FITNESS_WEIGHTS = {
       'pitch_range': 0.15,
       'rhythm_variety': 0.25,
       'consonance': 0.35,
       'contour': 0.15,
       'repetition': 0.1
   }
   ```

## 输出文件

程序运行后会在`results/`目录生成：
- `best_melody.mid`: 最佳旋律的MIDI文件
- `best_melody.txt`: 旋律的文本表示
- `fitness_progress.png`: 适应度进化曲线图
- `evolution_report.txt`: 进化过程报告

## 算法原理

1. **表示方式**: 旋律表示为音符序列，每个音符包含音高、时值和开始时间
2. **遗传操作**: 采用锦标赛选择、均匀交叉和多种变异方式
3. **适应度函数**: 综合考虑音域、节奏、和谐度、轮廓和重复度
4. **进化策略**: 使用精英保留策略，确保优秀个体不丢失

## 作者

2400013094 吴竞航
```