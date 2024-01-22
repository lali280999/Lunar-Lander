import gymnasium as gym
env = gym.make("LunarLander-v2")
observation, info = env.reset()
import torch
import torch.nn as nn
import numpy as np

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
#model.load_state_dict(torch.load("model_new3.pt"))

# weights of model_fixed are the same as model

target_network.load_state_dict(model.state_dict())

target_network.eval()

# loss is MSE loss
loss_fn = nn.MSELoss()

# optimizer is Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# epsilon greedy policy

epsilon = 1

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

# update after c steps

c_steps = 4

# soft update parameter
tau  = 0.01

# episodes:

episode = 0

max_episodes = 4000


# total rewards of last ten episodes

total_rewards = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_step = 0

model.to(device)
target_network.to(device)

mean_total_rewards = []

list_of_total_rewards = []

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
            
        observation_next, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        reward = torch.tensor(reward).float().detach()

        # define the target

        # if terminated:
        #     target = reward.to(device)
        #     # print(target)
    

        # else:
        #     with torch.no_grad():
        #         q_state_values = target_network(torch.FloatTensor(observation_next).to(device))
        #         target = reward + gamma * torch.max(q_state_values).detach()
        
            # print(target)
        
        with torch.no_grad():
            q_state_values = target_network(torch.FloatTensor(observation_next).to(device))
            target = reward + gamma * torch.max(q_state_values).detach()*(1-terminated)


        # add the observation, action, reward, observation_next and target to the replay buffer

        replay_buffer.append((observation, action, reward, observation_next, target))

        #print(i, action, observation, reward, target)

        observation = observation_next

        # if the replay buffer is full, remove the first element
        if len(replay_buffer) > 10000:
            replay_buffer.pop(0)

        # if the replay buffer has more than 100 elements, train the model
        if len(replay_buffer) > 100:
            # sample 100 elements from the replay buffer
            replay_sample = np.random.choice(len(replay_buffer), 32, replace=False)
            # get the observations, actions, rewards, observations_next and targets
            observations = np.array([replay_buffer[j][0] for j in replay_sample])
            actions = np.array([replay_buffer[j][1] for j in replay_sample])
            rewards = np.array([replay_buffer[j][2] for j in replay_sample])
            targets = [replay_buffer[j][4] for j in replay_sample]
            
            #convert list to tensor
            targets_tensor = torch.tensor(targets).float().unsqueeze(1).to(device)


            observations_tensor = torch.tensor(observations).float().to(device)
            actions_tensor = torch.tensor(actions).float().unsqueeze(1).to(device)

            # print(observations_tensor)
            # print(actions_tensor)
            # targets_tensor = torch.tensor(targets).float().unsqueeze(1)
            # targets_tensor = targets_tensor.detach()

            #print(observations_tensor.shape, actions_tensor.shape, targets_tensor.shape)

        

            # get the predictions for the observations and actions
            predictions = model(observations_tensor).gather(1, actions_tensor.long())

            # print(predictions)
            # print(targets_tensor)
            # calculate the loss
            loss = loss_fn(predictions, targets_tensor)

            # zero the gradients
            optimizer.zero_grad()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()


        if t_step % c_steps == 0:
            for source_parameters, targets_parameters in zip(model.parameters(), target_network.parameters()):
                targets_parameters.data.copy_(tau*source_parameters.data + (1-tau)*targets_parameters.data)


        if terminated or truncated:
            observation, info = env.reset()
        # print("episode:", episode)
            total_rewards.append(total_reward)
            if len(total_rewards) > 20:
                total_rewards.pop(0)
                mean_total_rewards.append(np.mean(total_rewards))
                list_of_total_rewards.append(total_rewards.copy())
            if np.mean(total_rewards) > 225:
                print("solved")
                break

            print(episode, total_reward, np.mean(total_rewards))
            total_reward = 0
            episode += 1
            #print("Resetting env")
            #print(observation, info)

        if epsilon > 0.01:
            epsilon *= 0.995

        if t_step % 100 == 0:
            #epsilon *= 0.95
        
            #print("Saving model")
            torch.save(model.state_dict(), "model_new7.pt")
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        break

# save list of total rewards
np.save("list_of_total_rewards_new7npy", list_of_total_rewards)
np.save("mean_total_rewards_new7.npy", mean_total_rewards)

env.close()