import time
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

# Getting State size and Action size
action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)


def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []

    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps_):
            action = np.argmax(qtable_[state, :])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state

    env.close()

    avg_reward = sum(rewards) / num_of_episodes_

    return avg_reward


def Q_learning(q_table, total_episodes, max_steps, gamma):
    episodes = []
    rewards = []

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            action = random.randint(0, 3)
            new_state, reward, done, info = env.step(action)
            q_table[state, action] = reward + (gamma * np.amax(q_table[new_state, :]))
            state = new_state
            if done == True:
                break

        if episode % 10 == 0:
            avg_reward = eval_policy(q_table, 10, 100)
            episodes.append(episode + 1)
            rewards.append(avg_reward)
    return episodes, rewards, q_table


def Q_learning_Non_Deterministic(q_table, total_episodes, max_steps, gamma, alpha):
    episodes = []
    rewards = []

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        reward_total = 0

        for step in range(max_steps):
            action = random.randint(0, 3)
            new_state, reward, done, info = env.step(action)
            q_table[state, action] = q_table[state, action] + alpha * (
                        reward + (gamma * np.amax(q_table[new_state, :]) - q_table[state, action]))
            state = new_state
            if done == True:
                break
        if episode % 10 == 0:
            avg_reward = eval_policy(q_table, 10, 100)
            episodes.append(episode + 1)
            rewards.append(avg_reward)
    return episodes, rewards, q_table


def plot_figure(episode_list, reward_list, slipery=False):
    plt.figure(figsize=(25, 5))
    plt.plot(episode_list[0], reward_list[0])
    plt.plot(episode_list[1], reward_list[1])
    plt.plot(episode_list[2], reward_list[2])
    plt.plot(episode_list[3], reward_list[3])
    plt.plot(episode_list[4], reward_list[4])
    plt.plot(episode_list[5], reward_list[5])
    plt.plot(episode_list[6], reward_list[6])
    plt.plot(episode_list[7], reward_list[7])
    plt.plot(episode_list[8], reward_list[8])
    plt.plot(episode_list[9], reward_list[9])
    plt.legend(['Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5', 'Plot 6', 'Plot 7', 'Plot 8', 'Plot 9', 'Plot 10'],
               loc=4)
    plt.xlabel("Episodes")
    plt.ylabel("Avg Rewards")
    plt.xlim(0, 1000)
    plt.ylim(0, 1.1)
    if slipery == False:
        plt.title("Slipery = False, Deterministic Rule")
    else:
        plt.title("Slipery = True, Deterministic Rule")

    plt.show()


episode_list = []
reward_list = []
re_run = 10

for i in range(re_run):
    qtable_initial = np.random.rand(state_size, action_size)
    episodes, rewards, optimal_q_table = Q_learning(qtable_initial, 1000, 100, 0.9)
    episode_list.append(episodes)
    reward_list.append(rewards)

plot_figure(episode_list, reward_list)

env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()
env.render()

episode_list.clear()
reward_list.clear()

for i in range(re_run):
    qtable_initial = np.random.rand(state_size, action_size)
    episodes, rewards, optimal_q_table = Q_learning(qtable_initial, 1000, 100, 0.9)
    episode_list.append(episodes)
    reward_list.append(rewards)

plot_figure(episode_list, reward_list, slipery=True)

env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()
env.render()

episode_list.clear()
reward_list.clear()
re_run = 10

for i in range(re_run):
    qtable_initial = np.random.rand(state_size, action_size)
    episodes, rewards, optimal_q_table = Q_learning_Non_Deterministic(qtable_initial, 1000, 100, 0.9, 0.5)
    episode_list.append(episodes)
    reward_list.append(rewards)

plt.figure(figsize=(25, 5))
plt.plot(episode_list[0], reward_list[0])
plt.plot(episode_list[1], reward_list[1])
plt.plot(episode_list[2], reward_list[2])
plt.plot(episode_list[3], reward_list[3])
plt.plot(episode_list[4], reward_list[4])
plt.plot(episode_list[5], reward_list[5])
plt.plot(episode_list[6], reward_list[6])
plt.plot(episode_list[7], reward_list[7])
plt.plot(episode_list[8], reward_list[8])
plt.plot(episode_list[9], reward_list[9])
plt.legend(['Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5', 'Plot 6', 'Plot 7', 'Plot 8', 'Plot 9', 'Plot 10'], loc=1)
plt.xlabel("Episodes")
plt.ylabel("Avg Rewards")
plt.xlim(0, 1000)
plt.ylim(0, 1.1)
plt.title("Slippery = True, Non-Detrministic Rule, alpha = 0.5")
plt.show()
