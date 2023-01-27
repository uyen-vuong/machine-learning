import gym
import numpy as np
import random
from IPython.display import clear_output

# Init Taxi-V2 Env
env = gym.make("Taxi-v3",mode="rgb_array").env

# Init arbitary values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


all_epochs = []
all_penalties = []

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

def strategy(taxirow, taxicolumn, passengerindex, destinationindex):
 
 for i in range(0, 3000):
    state = env.encode(taxirow, taxicolumn, passengerindex, destinationindex) 
    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info,human= env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
         
        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
        env.render()

 print("Training finished.\n")
 
taxirow, taxicolumn, passengerindex, destinationindex=[int(x) for x in input().split()]
strategy(taxirow, taxicolumn, passengerindex, destinationindex)