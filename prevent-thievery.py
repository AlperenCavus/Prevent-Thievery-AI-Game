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
REWARDS = {'empty': -50, 'key': 500, 'chest': 2000, 'hole': -600, 'spike': -500, 'guard': -900, 'finish': 1200}
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

    def astar_move(self, agent_position):
        # Introduce a probability to make random moves
        if random.random() < 0.3:  # Adjust the probability as needed
            self.random_move()
            return

        # Otherwise, follow the A* path
        open_list = []
        heapq.heappush(open_list, (0, self.position))
        came_from = {}
        cost_so_far = {}
        came_from[self.position] = None
        cost_so_far[self.position] = 0

        while open_list:
            _, current_node = heapq.heappop(open_list)

            if current_node == agent_position:
                break

            for next_direction in self.directions:
                next_node = (current_node[0] + next_direction[0], current_node[1] + next_direction[1])

                if not (0 <= next_node[0] < GRID_SIZE and 0 <= next_node[1] < GRID_SIZE):
                    continue

                if next_node in self.hole_positions or next_node in self.spike_positions:
                    # Avoid holes and spikes
                    continue

                new_cost = cost_so_far[current_node] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(agent_position, next_node)
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current_node

        # Reconstruct path here
        path = []
        current = agent_position
        while current != self.position:
            path.append(current)
            current = came_from[current]

        # Move the guard
        if path:
            next_position = path[-1]
            self.position = next_position

    def heuristic(self, a, b):
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def is_valid_position(self, position):
        return 0 <= position[0] < GRID_SIZE and 0 <= position[1] < GRID_SIZE

    def random_move(self):
        # Choose a random direction
        dx, dy = random.choice(self.directions)
        new_position = (self.position[0] + dx, self.position[1] + dy)
        
        # Ensure the new position is valid and doesn't step on holes or spikes
        if self.is_valid_position(new_position) and new_position not in self.hole_positions and new_position not in self.spike_positions:
            self.position = new_position

# Define grid world environment
class GridWorld:
    def __init__(self, agent, guard, chest_position, key_position, spike_positions, hole_positions):
        self.agent = agent
        self.guard = guard
        self.chest_position = chest_position
        self.key_position = key_position
        self.spike_positions = spike_positions
        self.hole_positions = hole_positions
        self.grid = np.full((GRID_SIZE, GRID_SIZE), EMPTY_SYMBOL)
        self.grid[self.agent.position] = AGENT_SYMBOL
        self.grid[self.guard.position] = GUARD_SYMBOL
        self.grid[self.chest_position] = CHEST_SYMBOL
        self.grid[self.key_position] = KEY_SYMBOL
        self.previous_chest_distance = abs(chest_position[0] - agent.position[0]) + abs(chest_position[1] - agent.position[1])
        self.previous_key_distance = abs(key_position[0] - agent.position[0]) + abs(key_position[1] - agent.position[1])
        for spike_position in self.spike_positions:
            self.grid[spike_position] = SPIKE_SYMBOL
        for hole_position in self.hole_positions:
            self.grid[hole_position] = HOLE_SYMBOL
        self.reward_points = 0
        self.key_collected = False
        self.chest_collected = False
    def reset(self):
        self.agent.position = (0, 0)
        self.agent.has_key = False
        self.guard.position = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.guard.last_agent_position = None
        self.grid = np.full((GRID_SIZE, GRID_SIZE), EMPTY_SYMBOL)
        self.grid[self.agent.position] = AGENT_SYMBOL
        self.grid[self.guard.position] = GUARD_SYMBOL
        self.grid[self.chest_position] = CHEST_SYMBOL
        self.grid[self.key_position] = KEY_SYMBOL
        for spike_position in self.spike_positions:
            self.grid[spike_position] = SPIKE_SYMBOL
        for hole_position in self.hole_positions:
            self.grid[hole_position] = HOLE_SYMBOL
        self.reward_points = 0     
        self.key_collected = False
        self.chest_collected = False  

    def is_valid_position(self, position):
        return 0 <= position[0] < GRID_SIZE and 0 <= position[1] < GRID_SIZE

    def get_state(self):
        # Define the bounds of the observation area
        observation_area = np.full((5, 5), -1)  # Initialize with -1 for empty cells

        # Calculate the top-left corner of the observation area
        top_left_x = max(0, self.agent.position[0] - 2)
        top_left_y = max(0, self.agent.position[1] - 2)

        # Calculate the bottom-right corner of the observation area
        bottom_right_x = min(GRID_SIZE, self.agent.position[0] + 3)
        bottom_right_y = min(GRID_SIZE, self.agent.position[1] + 3)

        # Iterate over the observation area and update the state based on grid contents
        for i in range(top_left_x, bottom_right_x):
            for j in range(top_left_y, bottom_right_y):
                if self.grid[i, j] == SPIKE_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 0
                elif self.grid[i, j] == HOLE_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 1
                elif self.grid[i, j] == GUARD_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 2
                elif self.grid[i, j] == CHEST_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 3
                elif self.grid[i, j] == KEY_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 4
                elif self.grid[i, j] == AGENT_SYMBOL:
                    observation_area[i - top_left_x, j - top_left_y] = 5

        # Return the flattened observation area
        return observation_area.flatten()

    def check_game_over(self):
        if self.agent.position == self.guard.position:
            return True, REWARDS['guard']
        if self.agent.position in self.spike_positions:
            return True, REWARDS['spike']
        if self.agent.position in self.hole_positions:
            return True, REWARDS['hole']
        if self.agent.position == self.chest_position and not self.agent.has_key:
            return False, REWARDS['empty']
        if self.agent.position == self.chest_position and self.agent.has_key:
            return False, REWARDS['chest']
        return False, 0

    def step(self, action):
        # Move the agent based on the action taken
        self.agent.update_position(action)
    
        # Initialize reward
        reward = 0
    
        # Check if the agent collects the key
        if self.agent.position == self.key_position and not self.agent.has_key:
            self.agent.has_key = True
            self.grid[self.key_position] = EMPTY_SYMBOL
            reward += REWARDS['key']  # Add key reward
    
       # Check if the agent collects the chest
        if self.agent.position == self.chest_position:
            if self.agent.has_key:
                self.agent.has_key = False
                self.grid[self.chest_position] = EMPTY_SYMBOL
                reward += REWARDS['chest'] + REWARDS['finish']
            else:
                reward += REWARDS['empty']
    
        # Check if game over due to spike or hole
        if self.agent.position in self.spike_positions:
            reward += REWARDS['spike']  # Add spike reward
            return self.agent.position, reward, True, False
        if self.agent.position in self.hole_positions:
            reward += REWARDS['hole']  # Add hole reward
            return self.agent.position, reward, True, False
    
        # Move guard
        self.guard.move(self.agent.position)
    
        # Check if the agent collides with the guard
        if self.agent.position == self.guard.position:
            reward += REWARDS['guard']  # Add guard reward
            return self.agent.position, reward, True, False
            
        # Calculate the distance to the key
        key_distance = abs(self.key_position[0] - self.agent.position[0]) + abs(self.key_position[1] - self.agent.position[1])  

        # Give positive feedback for moving closer to the key
        if key_distance < self.previous_key_distance:
            reward += 100

        # Update the previous key distance
        self.previous_key_distance = key_distance

        # Calculate the distance to the chest
        chest_distance = abs(self.chest_position[0] - self.agent.position[0]) + abs(self.chest_position[1] - self.agent.position[1])

        # Give positive feedback for moving closer to the chest with the key
        if chest_distance < self.previous_chest_distance and self.agent.has_key:
            reward += 200

        # Update the previous chest distance
        self.previous_chest_distance = chest_distance
    
        # Apply walking cost
        self.reward_points += reward  # Add the current step reward to total reward points
    
        # Otherwise, return the current state, reward, and game-over status
        return self.agent.position, reward, False, False



    

    def render(self):
        screen = pygame.display.get_surface()  # Get the screen surface
        screen.fill((255, 255, 255))  # Fill the screen with white color
        block_size = 50

        # Draw grid lines
        for x in range(GRID_SIZE):
            pygame.draw.line(screen, (0, 0, 0), (x * block_size, 0), (x * block_size, screen.get_height()))
        for y in range(GRID_SIZE):
            pygame.draw.line(screen, (0, 0, 0), (0, y * block_size), (screen.get_width(), y * block_size))

        # Render objects
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == CHEST_SYMBOL:
                    screen.blit(chest_img, (j * block_size, i * block_size))
                elif self.grid[i, j] == KEY_SYMBOL:
                    screen.blit(key_img, (j * block_size, i * block_size))
                elif self.grid[i, j] == SPIKE_SYMBOL:
                    screen.blit(spike_img, (j * block_size, i * block_size))
                elif self.grid[i, j] == HOLE_SYMBOL:
                    screen.blit(hole_img, (j * block_size, i * block_size))

        # Render agent
        agent_position = self.agent.position
        screen.blit(agent_img, (agent_position[1] * block_size, agent_position[0] * block_size))

        # Render guard
        guard_position = self.guard.position
        screen.blit(guard_img, (guard_position[1] * block_size, guard_position[0] * block_size))

        # Render reward points
        font = pygame.font.Font(None, 36)
        reward_color = (0, 128, 0) if self.reward_points >= 0 else (255, 0, 0)
        reward_text = "{:.2f}".format(self.reward_points)
        text = font.render("Reward: " + reward_text, True, reward_color)
        screen.blit(text, (10, 10))
        pygame.display.flip()  # Update the display

def generate_easy_map():
    chest_position = (4, 1)
    key_position = (2, 4)

    # Generate spike positions
    spike_positions = [(7, 2), (4, 7), (9, 9)]

    # Generate hole positions
    hole_positions = [(1, 9), (3, 1), (5, 3)]

    return chest_position, key_position, spike_positions, hole_positions

def generate_moderate_map():
    # Generate chest and key positions
    chest_position = (4, 1)
    key_position = (2, 4)

    # Generate spike positions
    spike_positions = [(7, 2), (4, 7), (9, 9), (3, 3)]

    # Generate hole positions
    hole_positions = [(1, 9), (3, 1), (5, 3), (10, 5)]

    return chest_position, key_position, spike_positions, hole_positions

def generate_hard_map():
    # Generate chest and key positions
    chest_position = (4, 1)
    key_position = (2, 4)

    # Generate spike positions
    spike_positions = [(1, 7), (7, 2), (5, 5), (1,3), (7, 7)]

    # Generate hole positions
    hole_positions = [(6, 3), (3, 7), (4, 4), (1, 4), (7, 5)]

    return chest_position, key_position, spike_positions, hole_positions

def main(difficulty_level):
    agent = Agent()
    if difficulty_level == "easy":
        chest_position, key_position, spike_positions, hole_positions = generate_easy_map()
    elif difficulty_level == "moderate":
        chest_position, key_position, spike_positions, hole_positions = generate_moderate_map()
    elif difficulty_level == "hard":
        chest_position, key_position, spike_positions, hole_positions = generate_hard_map()
    else:
        raise ValueError("Invalid difficulty level")

    guard = Guard(position=(GRID_SIZE - 2, GRID_SIZE - 2), hole_positions=hole_positions, spike_positions=spike_positions)
    grid_world = GridWorld(agent, guard, chest_position, key_position, spike_positions, hole_positions)
    epsilon = 0.1

    # Initialize Q-table
    Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))
    experience_filename = 'experience.json'

    # Load Q-table from file if it exists
    if os.path.exists(experience_filename):
        experience = load_experience(experience_filename)
        Q = np.array(experience['Q'])

    # Q-learning parameters
    epsilon = 1.0  # Initial epsilon value
    epsilon_min = 0.1  # Minimum epsilon value
    epsilon_decay = 0.001  # Decay rate for epsilon 
    alpha = 0.1  # Learning rate 
    discount_factor = 0.9  # Discount factor

    num_episodes = 2500  # Number of episodes for training
    max_steps_per_episode = 500  # Increased episode length to allow for more exploration

    screen_width = GRID_SIZE * 50
    screen_height = GRID_SIZE * 50
    screen = pygame.display.set_mode((screen_width, screen_height))

    pygame.mixer.music.load('ost.mp3')
    pygame.mixer.music.set_volume(0.0)
    pygame.mixer.music.play(-1)

    clock = pygame.time.Clock()
    FPS = 3000
    
    rewards_epsilon = []
    episode_rewards_epsilon = []
    for episode in range(num_episodes):
        state = grid_world.get_state()
        total_reward = 0
        step = 0
        game_over = False
        

        # Run episode until game over or maximum steps reached
        while not game_over and step < max_steps_per_episode:
            # Handle events and user inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Choose action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.choice(NUM_ACTIONS)  # Explore: choose a random action
            else:
                action = np.argmax(Q[state[0], state[1]])  # Exploit: choose the action with the highest Q-value

            # Take action and observe next state and reward
            next_state, reward, done, chest_collected = grid_world.step(action)

            # Update Q-value
            current_q_value = Q[state[0], state[1], action]
            if done:
                next_max_q_value = 0  # Set next_max_q_value to 0 when episode ends
            else:
                next_max_q_value = np.amax(Q[next_state[0], next_state[1]])
            new_q_value = current_q_value + alpha * (reward + discount_factor * next_max_q_value - current_q_value)
            Q[state[0], state[1], action] = new_q_value

            state = next_state
            total_reward += reward
            step += 1

            screen.fill((255, 255, 255))
            grid_world.render()

            if done:
                game_over = True
                grid_world.reset()
                if chest_collected:
                    success = True

            clock.tick(FPS)  # Limit frame rate

        # Decrement epsilon
        epsilon -= epsilon * epsilon_decay
        epsilon = max(epsilon, epsilon_min)

        # Print total reward and steps for the episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}, Epsilon = {epsilon}")

        # Append total reward to the rewards list
        rewards_epsilon.append(total_reward)
        episode_rewards_epsilon.append(reward)  # Store reward for each episode
        # Save experience at the end of each episode
        experience = {'Q': Q.tolist()}
        save_experience(experience_filename, experience)

        grid_world.reset()
        screen.fill((255, 255, 255))
        grid_world.render()


        # Plot rewards against epsilon values
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_epsilon)  # Plot total reward for each episode with changing epsilon
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode (Epsilon)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    difficulty_level = "hard"  # Specify the difficulty level here (easy, moderate, or hard)
    main(difficulty_level)
    
    

    
        


    
