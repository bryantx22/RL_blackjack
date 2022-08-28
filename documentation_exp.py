import gym
env = gym.make("Blackjack-v1", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(5):
    print (observation)
    observation, reward, done, info = env.step(1)
    print (list(observation))
    print (info)

    if done:
        observation, info = env.reset(return_info=True)

env.close()