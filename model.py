import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQnet(nn.Module):
    """
    A simple neural network with three fully connected layers.
    """
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LinearQnet, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, fileName='model.pth', additional_info=None):
        """
        Saves the model state dictionary and additional information.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        fileName = os.path.join(model_folder_path, fileName)

        save_dict = {
            'model_state_dict': self.state_dict(),
        }
        if additional_info is not None:
            save_dict['additional_info'] = additional_info

        torch.save(save_dict, fileName)

class Qtrainer:
    """
    Handles the training of the Q-network.
    """
    def __init__(self, model, learningRate, gamma, device):
        self.device = device
        self.model = model.to(self.device)
        self.learningRate = learningRate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        """
        Performs a single training step.
        """
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        nextState = torch.tensor(nextState, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Qnew = reward[idx]
            if not done[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))

            target[idx][action[idx].item()] = Qnew

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()