import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from copy import deepcopy
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt

name = 'HalfCheetah-v2'
env = gym.make(name)
global_seed = 333
np.random.seed(global_seed)
env.seed(global_seed)
torch.manual_seed(global_seed)
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
torch.set_default_tensor_type('torch.DoubleTensor')
max_timesteps = 50000
start_value = 0

def start():
    global start_value
    start_value = time.time()


def end():
    end = time.time()
    print(end - start_value)

class Value(nn.Module):
    def __init__(self, num_inputs,num_outputs):
        super(Value, self).__init__()
        self.xlayer = nn.Linear(num_inputs, 24)
        self.hlayer = nn.Linear(24, 24)
        self.ylayer = nn.Linear(24, num_outputs)

    def forward(self, x):
        x = F.tanh(self.xlayer(x))
        x = F.tanh(self.hlayer(x))
        x = F.tanh(self.ylayer(x))
        return x

def select_action(state,model):
    state = torch.from_numpy(state).unsqueeze(0)
    x = model(Variable(state))
    return x

def discover_fitness(render, model):
    state = env.reset()
    total_reward = 0
    steps = 0
    for t in range(max_timesteps):
        #if render:
        #    env.render()
        action = select_action(state,model)
        action = action.data[0].numpy()
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    return total_reward,steps

def worker(procnum):
    model = procnum[1]
    model_dict = model.state_dict()
    xl_weights = model_dict['xlayer.weight']
    hl_weights = model_dict['hlayer.weight']
    yl_weights = model_dict['ylayer.weight']
    np.random.seed(procnum[0])
    x = torch.from_numpy(np.random.randn(len(xl_weights), len(xl_weights[0])))
    h = torch.from_numpy(np.random.randn(len(hl_weights), len(hl_weights[0])))
    y = torch.from_numpy(np.random.randn(len(yl_weights), len(yl_weights[0])))
    model_dict['xlayer.weight'] += noise_density * x
    model_dict['hlayer.weight'] += noise_density * h
    model_dict['ylayer.weight'] += noise_density * y
    model.load_state_dict({name: model_dict[name] for name in model_dict})
    reward,steps = discover_fitness(False, model)
    return [reward, procnum[0],steps]

noise_density = 0.01
population_size = 10
plot_data = []
plot_data_1 = []
plot_data_2 = []
plot_data_3 = []
plot_data_4 = []
plot_data_5 = []
n_densities = [0.005,0.01,0.025,0.05,0.1]
n_populations = [2,5,10,30,50]
for i in range(5):
    plot_data.append([])
pool = multiprocessing.Pool(processes=7)
net = Value(num_inputs, num_outputs)
for p in range(5):
    #noise_density = n_densities[p]
    population_size = n_populations[p]
    print("retraining with population size:", population_size)
    total_time_steps = 0
    v = deepcopy(net)
    for i in range(200):
        network_return,_ = discover_fitness(True, v)
        print(network_return)

        seeds = []
        for i in range(population_size):
            tupled = (np.random.randint(999999999),deepcopy(v))
            seeds.append(tupled)
        returns_and_seeds = pool.map(worker, seeds)

        seeds = np.asarray(returns_and_seeds)
        means = np.mean(seeds[:,0])
        std = np.std(seeds[:,0])
        for seed in seeds:
            seed[0] = (seed[0] - means) / std

        total_time_steps += np.sum(seeds[:,2])
        plot_data[p].append([network_return, total_time_steps])
        u = 0
        for seed in seeds:
            model_dict = v.state_dict()
            xl_weights = model_dict['xlayer.weight']
            hl_weights = model_dict['hlayer.weight']
            yl_weights = model_dict['ylayer.weight']
            np.random.seed(int(seed[1]))
            x = torch.from_numpy(np.random.randn(len(xl_weights), len(xl_weights[0])))
            h = torch.from_numpy(np.random.randn(len(hl_weights), len(hl_weights[0])))
            y = torch.from_numpy(np.random.randn(len(yl_weights), len(yl_weights[0])))
            model_dict['xlayer.weight'] += x * seed[0] * noise_density / population_size
            model_dict['hlayer.weight'] += h * seed[0] * noise_density / population_size
            model_dict['ylayer.weight'] += y * seed[0] * noise_density / population_size
            u +=1
        v.load_state_dict({name: model_dict[name] for name in model_dict})


plot_data_1 = plot_data[0]
plot_data_2 = plot_data[1]
plot_data_3 = plot_data[2]
plot_data_4 = plot_data[3]
plot_data_5 = plot_data[4]
plot_data_1 = np.transpose(plot_data_1)
plot_data_2 = np.transpose(plot_data_2)
plot_data_3 = np.transpose(plot_data_3)
plot_data_4 = np.transpose(plot_data_4)
plot_data_5 = np.transpose(plot_data_5)
plt.plot(plot_data_1[1], plot_data_1[0])
plt.plot(plot_data_2[1], plot_data_2[0])
plt.plot(plot_data_3[1], plot_data_3[0])
plt.plot(plot_data_4[1], plot_data_4[0])
plt.plot(plot_data_5[1], plot_data_5[0])
plt.legend(['population size: ' + str(n_populations[0]),
            'population size: ' + str(n_populations[1]),
            'population size: ' + str(n_populations[2]),
            'population size: ' + str(n_populations[3]),
            'population size: ' + str(n_populations[4])], loc='upper left')
plt.xlabel('Timesteps')
plt.title('Evolutionary Strategies , environment = ' + name)
plt.ylabel('Enviroment Rewards')
plt.grid(True)
plt.show()


