# src/initialization.py
"""
初始种群生成
"""

import os
import random
from src.representation import Individual, Population
from src.utils import generate_random_melody
from src.config import NOTES

# 初始化很短，全写evolution.py里了