import torch 
import random 
import numpy as np
from collections import deque
from game import BlackJack
from model import QNet, QTrainer
from helper import plot
import pandas as pd

MAX_MEMORY = 300000
BATCH_SIZE = 3000
LR = 1e-3

class Agent ():

    def __init__(self):
        self.n_games = 0
        self.epsilon = None # defined later in get_action
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = QNet (3, 600, 100, 25, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def remember (self, state, action, reward, next_state, done):
        self.memory.append ((state, action, reward, next_state, done))

    def train_long_memory (self):
        if (len(self.memory)) > BATCH_SIZE:
            mini_sample = random.sample (self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip (*mini_sample)
        self.trainer.train_step (states, actions, rewards, next_states, dones)

    def train_short_memory (self, state, action, reward, next_state, done):
        self.trainer.train_step (state, action, reward, next_state, done)

    def get_action (self, state):

        self.epsilon = 1000 - self.n_games
        move = None

        if (random.randint (0, 500) < self.epsilon):
            move = random.randint (0, 1)

        else:
            state0 = torch.tensor (state, dtype = torch.float)
            prediction = self.model (state0)
            move = torch.argmax (prediction).item () # index is correct
        
        return move

def train ():

    agent = Agent ()
    blackjack = BlackJack ()
    state_old = blackjack.initial_state 
    total_games = 1000
    own_sums = [state_old[0]]
    dealer_cards = [state_old[1]]
    useable_ace = [state_old[2]]
    game_over = [False]
    reward_hist = [0]
    moves = []
    count = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    while (count < total_games):

        final_move = agent.get_action (state_old)
        moves.append (final_move)
        observation, reward, done = blackjack.play_action (final_move)
        state_new = observation
        own_sums.append (state_new[0])
        dealer_cards.append (state_new[1])
        useable_ace.append (state_new[2])
        reward_hist.append (reward)
        game_over.append (done)
        agent.train_short_memory (state_old, final_move, reward, state_new, done)
        agent.remember (state_old, final_move, reward, state_new, done)
        state_old = state_new 

        if done:

            moves.append (-1)
            count +=1
            state_old = blackjack.reset () # reset the game -> returns initial state

            own_sums.append (state_old [0])
            dealer_cards.append (state_old [1])
            useable_ace.append (state_old [2])
            reward_hist.append (0)
            game_over.append (False)

            agent.n_games += 1
            agent.train_long_memory ()

            if (agent.n_games%(int(total_games/100+1))==0):
                print (int(agent.n_games/total_games*100), "%", end = " ")
            
            plot_scores.append (reward)
            total_score += (reward)
            mean_score = total_score / agent.n_games
            plot_mean_scores.append (mean_score)
            plot (plot_scores, plot_mean_scores)
    
    own_sums.pop ()
    dealer_cards.pop ()
    useable_ace.pop ()
    game_over.pop ()
    reward_hist.pop ()
        
    df = pd.DataFrame ({
        "own_sums": own_sums,
        "useable ace": useable_ace,
        "dealer_card": dealer_cards,
        "move": moves,
        "reward": reward_hist,
        "game over": game_over
    })

    # df.to_csv ("game_play.csv", index = False)

if __name__ == '__main__':
    train ()