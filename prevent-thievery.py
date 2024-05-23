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

# Load grid world images and scale them to 50x50
agent_img = pygame.transform.scale(pygame.image.load('agent.png'), (60, 60))
guard_img = pygame.transform.scale(pygame.image.load('guard.png'), (50, 50))
chest_img = pygame.transform.scale(pygame.image.load('chest.png'), (50, 50))
key_img = pygame.transform.scale(pygame.image.load('key.png'), (50, 50))
spike_img = pygame.transform.scale(pygame.image.load('spike.png'), (50, 50))
hole_img = pygame.transform.scale(pygame.image.load('hole.png'), (50, 50))

# Initialize Q-table
Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Load experience from JSON file
def load_experience(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Save experience to JSON file
def save_experience(filename, experience):
    with open(filename, 'w') as file:
        json.dump(experience, file)
