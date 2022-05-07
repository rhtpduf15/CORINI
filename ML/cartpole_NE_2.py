import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
from time import time


class ML_NE_2:
    def __init__(self, generations, pop_size, top_limit, cartpole_version='CartPole-v1', max_step_per_epi=200):
        self.cartpole_version = cartpole_version
        self.max_step_per_epi = max_step_per_epi
        self.game_actions = 2  # 2 actions possible: left or right
        self.agent_eval_num = 15 # 인자로 받아서 커스텀 할 수 있을 듯
        self.mutation_power = 0.02 # 인자로 받아서 커스텀 할 수 있을 듯

        torch.set_grad_enabled(False)

        # 중간에 있던 애들 끌올
        self.TRAINED_AGENT = {}

        #need to be checked
        self.generations = generations
        self.top_limit = top_limit
        self.population_size = pop_size


    # ## Neuroevolution Setup
    class CartPoleAI(nn.Module):
        '''The brain of the agent'''
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 2))

        def forward(self, inputs):
            x = self.fc(inputs)
            return F.softmax(x, dim=1)

    def initialize_population(self, pop_size=2):
        '''Randomly initialize a bunch of agents'''
        population = [self.CartPoleAI() for _ in range(pop_size)]

        return population

    # 기존의 인자인 'max_episode_length'는 최대 스텝은 기본으로 200으로 설정할 것이므로 삭제함.
    # 필요할 경우엔 ML_NE2의 self.step 값을 사용.
    def evaluate_agent(self, agent, episodes=15, max_step_per_epi=200):
        '''Run an agent for a given number episodes and get the rewards'''
        # 한 에이전트 당 한 번의 시뮬레이션을 하는 게 아니라 인자로 받은 episodes수만큼 함.
        # 그 평균을 가지고 한 에이전트의 성능 평가가 이루어짐
        # return -> 한 에이전트의 episodes 수만큼의 수행 reward의 평균

        env = gym.make(self.cartpole_version)
        agent.eval()

        total_rewards = []

        for ep in range(episodes): #self.agent_eval_num
            observation = env.reset()
            # Modify the maximum steps that can be taken in a single episode
            env._max_episode_steps = max_step_per_epi

            episodic_reward = 0
            # Start episode
            for step in range(max_step_per_epi):
                input_obs = torch.Tensor(observation).unsqueeze(0)
                observation, reward, done, info = env.step(agent(input_obs).argmax(dim=1).item())

                episodic_reward += reward
                if done:
                    break

            total_rewards.append(episodic_reward)

        return np.array(total_rewards).mean()

    # 인구 전체에 대하여 각 agent들의 성능을 평가 (위에서 정의한 self.evaluate_agent 사용)
    def evaluate_population(self, population, episodes=15, max_step_per_epi=200):
        '''Evaluate the population'''
        pop_fitness = []
        for agent in population:
            pop_fitness.append(self.evaluate_agent(agent, episodes, max_step_per_epi))
        return pop_fitness

    # mutation_power default는 기존 NE모델과 마찬가지로 0.02
    def mutate(self, parent_agent, mutation_power=0.02):
        '''Creates a mutated copy of the parent agent
        by adding a weighted gaussian noise to the params'''
        child_agent = copy.deepcopy(parent_agent)

        for param in child_agent.parameters():
            param.data = param.data + (torch.randn(param.shape) * mutation_power)

        return child_agent

    def repopulate(self, top_agents, pop_size, mutation_power):
        '''Repopulate the population from the top agents by mutation'''
        new_population = []

        n = 0
        while(n < pop_size):
            for parent in top_agents:
                child = self.mutate(parent, mutation_power)
                new_population.append(child)
                n += 1

        return new_population[:pop_size - 1]

    def evolve(self, generations,
               pop_size,
               topK,
               episodes,
               max_step_per_epi,
               mutation_power):
        '''Start the process of evolution'''

        population = self.initialize_population(pop_size)
        global_best = {}

        t1 = time()
        for g in range(generations):
            # Evaluate the population
            pop_fitness = self.evaluate_population(population, episodes, max_step_per_epi)
            mean_pop_reward = np.array(pop_fitness).mean()

            # Rank the agents in descending order
            topK_idx = np.argsort(pop_fitness)[::-1][:topK]
            topK_agents = [population[i] for i in topK_idx]

            # Get Best Agent
            best_agent = population[topK_idx[0]]
            best_reward = pop_fitness[topK_idx[0]]

            # Check with global best
            if g == 0:
                global_best['reward'] = best_reward
                global_best['agent'] = best_agent
            else:
                if best_reward >= global_best['reward']:
                    global_best['reward'] = best_reward
                    global_best['agent'] = best_agent

            print('Generation', g)
            print('Mean Reward of Population', mean_pop_reward)
            print('Best Agent Reward (mean)', best_reward)
            print('Global Best Reward (mean)', global_best['reward'], '\n')

            # Mutate and Repopulate
            new_population = self.repopulate(topK_agents, pop_size, mutation_power)
            # take the best agent of generation forward without cloning as well
            new_population.append(best_agent)

            population = new_population

            self.TRAINED_AGENT = global_best

    # main train func
    def model_train(self):
        self.evolve(generations=self.generations,
                pop_size=self.population_size,
                topK=self.top_limit,
                episodes=self.agent_eval_num,
                max_step_per_epi=self.max_step_per_epi,
                mutation_power=self.mutation_power)

    # ## Test the Trained Agent
    def play_agent(self, agent, episodes, max_step_per_epi, render=False):
        env = gym.make("CartPole-v1")

        agent.eval()

        total_rewards = []

        for ep in range(episodes):
            observation = env.reset()
            env._max_episode_steps = max_step_per_epi

            episodic_reward = 0

            for step in range(max_step_per_epi):
                if render:
                    env.render()

                input_obs = torch.Tensor(observation).unsqueeze(0)
                observation, reward, done, info = env.step(agent(input_obs).argmax(dim=1).item())

                episodic_reward += reward
                if done:
                    break
            total_rewards.append(episodic_reward)

        env.close()
        print('Mean Rewards across all episodes', np.array(total_rewards).mean())
        print('Best Reward in any single episode', max(total_rewards))

    def test(self, itr):
        self.play_agent(self.TRAINED_AGENT['agent'], episodes=itr, max_step_per_epi=self.max_step_per_epi, render=True)
        torch.save(self.TRAINED_AGENT['agent'].state_dict(), 'model-200.pth')

##