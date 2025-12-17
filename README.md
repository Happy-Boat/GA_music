# 遗传算法机器作曲系统

## 项目简介

本项目实现了一个基于遗传算法的自动作曲系统，能够生成和进化旋律，通过多维适应度函数指导旋律向更符合音乐美学规则的方向发展。该系统结合了遗传算法的进化机制与音乐理论知识，实现从随机旋律到高质量音乐作品的自动生成过程。

## 核心特色

### 🎵 智能音乐生成
- **遗传算法驱动**：采用经典遗传算法框架，包括选择、交叉、变异等操作
- **精英保留策略**：确保优秀个体不会丢失，每代保留最优个体
- **动态适应度调整**：进化过程中权重动态调整，早期注重多样性，后期注重和谐度

### 🎼 多维适应度评估系统
系统采用9个关键音乐指标进行综合评估：
- **和谐度评估**：检测不和谐音程、二度音程占比，惩罚超八度跳跃
- **节奏多样性**：分析时值分布、强拍起拍比例，确保节奏丰富性
- **音域合理性**：控制音高范围，避免过高或过低，保持适中音域
- **旋律轮廓**：评估音高变化趋势，避免单调重复
- **调性一致性**：基于首音构建大小调，惩罚偏离调性的音符

### 🎹 专业音乐变换操作
集成三种经典音乐变换技术：
- **移调变换**：随机半音数移调，扩展音域可能性
- **倒影变换**：以随机轴音进行倒影，创造对称旋律结构
- **逆行变换**：时间顺序反转，生成逆向旋律变体

### 📊 全面可视化分析
- **适应度进化曲线**：实时监控种群适应度变化趋势
- **钢琴卷帘可视化**：直观展示旋律在钢琴键盘上的分布
- **进化过程报告**：详细记录每代统计信息和最佳旋律
- **旋律对比分析**：比较初始与最终旋律的差异

### 💾 智能保存与恢复
- **检查点机制**：每10代自动保存最佳个体，支持中断恢复
- **多格式输出**：同时生成MIDI音频文件和文本旋律描述
- **种群快照**：保存完整种群状态，便于后续分析

### 🔧 灵活配置系统
- **命令行参数**：支持自定义种群大小、进化代数、交叉变异率
- **模块化设计**：各组件独立，便于扩展和定制
- **实时验证**：旋律生成过程中严格验证合法性，确保音乐合理性

## 功能特性

1. **多种初始种群生成方式**
   - 随机生成旋律（默认）
   - 从MIDI文件导入片段（计划中）

2. **完整的遗传算法流程**
   - 轮盘赌选择策略
   - 单点交叉操作
   - 多点变异机制
   - 精英保留策略
   - 多代进化迭代

3. **丰富的音乐变换操作**
   - 移调变换（transpose）
   - 倒影变换（inversion）
   - 逆行变换（retrograde）
   - 节奏变化（计划中）

4. **多维适应度评估**
   - 和谐度评估（consonance）
   - 节奏多样性（rhythm_variety）
   - 音域范围（pitch_range）
   - 旋律轮廓（contour）
   - 重复度控制（repetition）

5. **可视化与分析**
   - 适应度进化曲线图
   - 旋律钢琴卷帘可视化
   - 进化过程详细报告
   - 种群多样性分析

## 技术创新点

### 动态权重调整机制
```python
def dynamic_fitness_adjustment(generation, base_weights):
    progress = min(generation / 50, 1.0)
    adjusted_weights['consonance'] *= (1 + progress * 0.5)
    adjusted_weights['rhythm_variety'] *= (1 - progress * 0.3)
```

随着进化代数增加，系统逐渐提高对和谐度的要求，降低对节奏多样性的权重，实现从"探索"到"优化"的平滑过渡。

### 旋律合法性验证
系统内置严格的旋律验证机制：
- 长度一致性检查
- 休止符使用规范
- 延音时长限制
- 音符范围约束

### 专业音乐表示
采用专门设计的音乐数据结构：
- Melody类：旋律核心表示
- Individual类：遗传算法个体
- Population类：种群管理
- 支持MIDI导入导出

## 安装与运行

### 环境要求

- Python 3.8+
- 依赖包（见requirements.txt）

### 安装步骤

```bash
# 克隆项目
git clone git@github.com:Happy-Boat/GA_music.git

# 安装依赖
conda create -n GA_music python=3.10 -y
conda activate GA_music
pip install -r requirements.txt
```

## 运行程序

```bash
# 使用随机生成的初始种群（默认）
python main.py

# 自定义参数运行
python main.py --generations 200 --population 30 --crossover-rate 0.85 --mutation-rate 0.15 --visualize

# 查看所有参数
python main.py --help
```

## 项目结构
```text
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
├── main.py                   # 主程序入口
│
├── data/                     # 数据目录
│   ├── input_midi/           # 原始MIDI文件（用于初始种群）
│   ├── input_musicxml/       # MusicXML格式输入文件
│   ├── population/           # 保存的种群数据和检查点
│   └── outputs/              # 生成的结果文件
│
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── config.py             # 全局配置参数
│   ├── representation.py     # 音乐表示和数据结构
│   ├── genetic_operations.py # 遗传操作（选择、交叉、变异）
│   ├── fitness.py           # 多维适应度函数
│   ├── evolution.py         # 进化过程控制
│   ├── music_operations.py  # 音乐变换操作
│   ├── utils.py             # 工具函数和验证
│   └── visualization.py     # 可视化功能
```

## 配置说明

主要配置参数在`config.py`中：

- **遗传算法参数**：种群大小、进化代数、交叉率、变异率
- **音乐参数**：节奏、拍号、小节数、音符范围
- **适应度权重**：各评估维度的权重配置
- **音符定义**：可用音符列表和概率分布

## 输出结果

运行完成后将在`data/outputs/`目录生成：
- `best_melody.mid`：最佳旋律的MIDI文件
- `best_melody.txt`：旋律的文本表示
- `evolution_report.txt`：完整的进化过程报告
- `fitness_progress.png`：适应度进化曲线图

## 扩展计划

- [ ] 支持从MIDI文件导入初始种群
- [ ] 添加更多音乐变换操作（如节奏变奏）
- [ ] 实现多声部和声生成
- [ ] 集成深度学习评估模型
- [ ] 支持实时交互式作曲

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License

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


```