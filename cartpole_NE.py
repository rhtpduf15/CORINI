import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.optim as optim

import math
import copy
import ray

@ray.remote
class ML_NE:
    def __init__(self, generations, num_agents, top_limit):
        self.game_actions = 2  # 2 actions possible: left or right

        # disable gradients as we will not use them
        torch.set_grad_enabled(False)

        # initialize N number of agents
        self.num_agents = num_agents
        agents = self.return_random_agents(num_agents)

        # How many top agents to consider as parents
        self.top_limit = top_limit

        # run evolution until X generations
        self.generations = generations

        self.elite_index = None
        self.agents = None

        self.results = []

    class CartPoleAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(4, 128, bias=True),
                nn.ReLU(),
                nn.Linear(128, 2, bias=True),
                nn.Softmax(dim=1)
            )

        def forward(self, inputs):
            x = self.fc(inputs)
            return x

    def init_weights(self, m):
        # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
        # nn.Conv2d bias is of shape [16] i.e. # number of filters

        # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
        # nn.Linear bias is of shape [32] i.e. # number of output features

        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)


    def return_random_agents(self, num_agents):
        agents = []
        for _ in range(num_agents):

            agent = self.CartPoleAI()

            for param in agent.parameters():
                param.requires_grad = False

            self.init_weights(agent)
            agents.append(agent)

        return agents


    def run_agents(self, agents):
        reward_agents = []
        env = gym.make("CartPole-v1")

        for agent in agents:
            agent.eval()

            observation = env.reset()

            r = 0
            s = 0

            for _ in range(250):
                inp = torch.tensor(observation).type('torch.FloatTensor').view(1, -1)
                output_probabilities = agent(inp).detach().numpy()[0]
                action = np.random.choice(range(self.game_actions), 1, p=output_probabilities).item()
                new_observation, reward, done, info = env.step(action)
                r = r + reward

                s = s + 1
                observation = new_observation

                if (done):
                    break

            reward_agents.append(r)
            # reward_agents.append(s)

        return reward_agents

    def return_average_score(self, agent, runs):
        score = 0.
        for i in range(runs):
            score += self.run_agents([agent])[0]
        return score/runs


    def run_agents_n_times(self, agents, runs):
        avg_score = []
        for agent in agents:
            avg_score.append(self.return_average_score(agent,runs))
        return avg_score


    def mutate(self, agent):
        child_agent = copy.deepcopy(agent)

        mutation_power = 0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

        for param in child_agent.parameters():

            if (len(param.shape) == 4):  # weights of Conv2D

                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                param[i0][i1][i2][i3] += mutation_power * np.random.randn()



            elif (len(param.shape) == 2):  # weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] += mutation_power * np.random.randn()


            elif (len(param.shape) == 1):  # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    param[i0] += mutation_power * np.random.randn()

        return child_agent


    def return_children(self, agents, sorted_parent_indexes, elite_index):
        children_agents = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(len(agents) - 1):
            selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            children_agents.append(self.mutate(agents[selected_agent_index]))

        # now add one elite
        elite_child = self.add_elite(agents, sorted_parent_indexes, elite_index)
        children_agents.append(elite_child)
        elite_index = len(children_agents) - 1  # it is the last one

        return children_agents, elite_index


    def add_elite(self, agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
        candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

        if (elite_index is not None):
            candidate_elite_index = np.append(candidate_elite_index, [elite_index])

        top_score = None
        top_elite_index = None

        for i in candidate_elite_index:
            score = self.return_average_score(agents[i], runs=5)
            #print("Score for elite i ", i, " is ", score)

            if (top_score is None):
                top_score = score
                top_elite_index = i
            elif (score > top_score):
                top_score = score
                top_elite_index = i

        #print("Elite selected with index ", top_elite_index, " and score", top_score)

        child_agent = copy.deepcopy(agents[top_elite_index])
        return child_agent


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def test_name_run(self):
        self.agents = self.return_random_agents(self.num_agents)
        for generation in range(self.generations):

            # return rewards of agents
            rewards = self.run_agents_n_times(self.agents, 3)  # return average of 3 runs

            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1][
                                    :self.top_limit]  # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            # print("")
            # print("")

            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])

            self.results.append(f"Generation {generation}, | Mean rewards: {np.mean(rewards)}, | Mean of top 5: {np.mean(top_rewards[:5])}")
            # print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",
            #       np.mean(top_rewards[:5]))
            # print(rewards)
            # print("Top ", self.top_limit, " scores", sorted_parent_indexes)
            # print("Rewards for top: ", top_rewards)

            # setup an empty list for containing children agents
            self.children_agents, self.elite_index = self.return_children(self.agents, sorted_parent_indexes, self.elite_index)

            # kill all agents, and replace them with their children
            self.agents = self.children_agents

    def play_agent(self):
        try:  # try and exception block because, render hangs if an erorr occurs, we must do env.close to continue working
            env = gym.make("CartPole-v0")

            # env_record = Monitor(env, './video', force=True)
            observation = env.reset()
            last_observation = observation
            r = 0
            for _ in range(250):
                env.render()
                inp = torch.tensor(observation).type('torch.FloatTensor').view(1, -1)
                output_probabilities = self.agents[-1](inp).detach().numpy()[0]
                action = np.random.choice(range(self.game_actions), 1, p=output_probabilities).item()
                new_observation, reward, done, info = env.step(action)
                r = r + reward
                observation = new_observation

                if (done):
                    break

            env.close()
            print("Rewards: ", r)

        except Exception as e:
            env.close()
            print(e.__doc__)
            print(e.message)