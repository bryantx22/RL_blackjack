import gym 

class BlackJack:

    def __init__ (self):
        self.env = gym.make('Blackjack-v1', natural=False, sab=False)
        self.initial_state, info = self.env.reset(seed=42, return_info=True)
        self.initial_state = list (self.initial_state)
        if (self.initial_state [2]):
            self.initial_state [2] = 1
        else:
            self.initial_state [2] = 0
    
    def reset (self): # can call to start
        observation = self.env.reset ()
        observation = list (observation)
        if (observation [2]):
            observation [2] = 1
        else:
            observation [1] = 0
        return list (observation)
    
    def play_action (self, action):
        observation, reward, done, info = self.env.step(action)
        observation = list (observation)
        if (observation [2]):
            observation [2] = 1
        else:
            observation [1] = 0
        return observation, reward, done
    
    def close (self):
        self.env.close ()

