import pygame
import numpy as np
import json
import random
import heapq
import math

# Initialize pygame
pygame.init()

# Define grid world constants
GRID_SIZE = 15
NUM_ACTIONS = 4
ACTIONS = ['up', 'down', 'left', 'right']
REWARDS = {'empty': -0.9, 'key': 50, 'chest': 100, 'hole': -50, 'spike': -30, 'guard': -100, 'finish': 120}
GUARD_SYMBOL = 'G'
CHEST_SYMBOL = 'C'
KEY_SYMBOL = 'K'
AGENT_SYMBOL = 'A'
HOLE_SYMBOL = 'H'
SPIKE_SYMBOL = 'S'
EMPTY_SYMBOL = ' '
