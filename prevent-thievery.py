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

# Define agent class
class Agent:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.has_key = False

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            return np.argmax(Q[state[0], state[1]])

    def update_position(self, action):
        if action == 0 and self.position[0] > 0:
            self.position = (self.position[0] - 1, self.position[1])
        elif action == 1 and self.position[0] < GRID_SIZE - 1:
            self.position = (self.position[0] + 1, self.position[1])
        elif action == 2 and self.position[1] > 0:
            self.position = (self.position[0], self.position[1] - 1)
        elif action == 3 and self.position[1] < GRID_SIZE - 1:
            self.position = (self.position[0], self.position[1] + 1)

# Define guard class
class Guard:
    def __init__(self, position=(GRID_SIZE - 1, GRID_SIZE - 1), hole_positions=None, spike_positions=None):
        self.position = position
        self.last_agent_position = None
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        self.hole_positions = hole_positions  # Store hole_positions
        self.spike_positions = spike_positions  # Store spike_positions

    def move(self, agent_position):
        self.last_agent_position = agent_position
        self.astar_move(agent_position)

        # Ensure the guard doesn't step on holes or spikes while moving
        while self.position in self.hole_positions or self.position in self.spike_positions:
            self.astar_move(agent_position)

    
