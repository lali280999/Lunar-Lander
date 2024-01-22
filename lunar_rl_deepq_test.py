import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
observation, info = env.reset()
import torch
import torch.nn as nn
import numpy as np

import gymnasium.utils.save_video as save_video

import matplotlib



# create four layer neural network, input size is 9, output size is 1
# eight arguments of inpput correspond to eight features of the environment and the ninth is the action

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(8, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 16)
        self.layer4 = torch.nn.Linear(16, 4)


    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    

# load model
model = NeuralNetwork()
target_network = NeuralNetwork()

# load weights
model.load_state_dict(torch.load("model_new7.pt"))

model.eval()

# weights of model_fixed are the same as model

target_network.load_state_dict(model.state_dict())

target_network.eval()

# epsilon greedy policy

epsilon = 0.0

# discount factor
gamma = 0.99

#Replay buffer of length 1000
replay_buffer = []

#action
action = 0

#target
target = 0

#total reward

total_reward = 0

# episodes:

episode = 0

max_episodes = 5


# total rewards of last ten episodes



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_step = 0

model.to(device)
target_network.to(device)



while episode < max_episodes:
        
    try:

        t_step += 1
        

        # with probability epsilon, take random action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            # keep the model in no grad mode
            with torch.no_grad():
            # get the prediction from the model for different actions and choose the action with the highest prediction
                q_state_values = model(torch.FloatTensor(observation).to(device))
                action = q_state_values.argmax().item()
                

        # take the action and get the reward, observation, terminated, truncated and info
            
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        reward = torch.tensor(reward).float().detach()



        if terminated or truncated:
            

            if total_reward > 200:
                # save video
                save_video.save_video(env.render(), '.', fps=env.metadata["render_fps"], step_starting_index=0, episode_index=episode)
                
            observation, info = env.reset()
            
            print(episode, total_reward)
            total_reward = 0
            episode += 1





    except KeyboardInterrupt:
        print("Keyboard interrupt")
        break

# save list of total rewards

env.close()