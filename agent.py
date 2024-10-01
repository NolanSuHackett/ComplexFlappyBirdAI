import torch
import random
from collections import deque
from FlappyComplexAI import Game, Pipe, screenWidth, screenHeight
from model import LinearQnet, Qtrainer
from helper import plot

maxMemory = 100000
batchSize = 128
learningRate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.numberGames = 0
        self.epsilon = 1.0  # Start with exploration
        self.epsilon_min = 0.005  # Minimum epsilon value
        self.epsilon_decay = 0.999  # Decay rate for epsilon
        self.gamma = 0.99
        self.memory = deque(maxlen=maxMemory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearQnet(7, 256, 2).to(self.device)
        self.trainer = Qtrainer(self.model, learningRate, self.gamma, self.device)
        self.record = 0  # Initialize record

    def save_model_with_epsilon(self, fileName='model_with_epsilon.pth'):
        additional_info = {
            'epsilon': self.epsilon,
            'record': self.record,
            'numberGames': self.numberGames
        }
        self.model.save(fileName, additional_info)

    def load_model_with_epsilon(self, fileName='model_with_epsilon.pth'):
        checkpoint = torch.load(fileName)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint['additional_info']['epsilon']
        self.record = checkpoint['additional_info'].get('record', 0)
        self.numberGames = checkpoint['additional_info'].get('numberGames', 0)
        print(f"Loaded model with epsilon: {self.epsilon}, record: {self.record}, and number of games: {self.numberGames}")

    def getState(self, game):
        bird = game.bird
        pipes = game.pipes

        if len(pipes) > 0:
            nearestPipe = pipes[0]
            if bird.x > nearestPipe.x + nearestPipe.topImage.get_width():
                if len(pipes) > 1:
                    nearestPipe = pipes[1]
        else:
            nearestPipe = Pipe(screenWidth, screenHeight // 2, 0)

        distanceToNextPipe = nearestPipe.x - bird.x
        topPipeY = nearestPipe.y - (nearestPipe.pipeGap / 2)
        bottomPipeY = nearestPipe.y + (nearestPipe.pipeGap / 2)
        gapCenterY = (topPipeY + bottomPipeY) / 2
        heightDifference = bird.y - gapCenterY

        state = [
            bird.y / screenHeight,
            bird.velocity / 10,
            distanceToNextPipe / screenWidth,
            topPipeY / screenHeight,
            bottomPipeY / screenHeight,
            nearestPipe.velocityY / 2,
            heightDifference / screenHeight
        ]
        return torch.tensor(state, dtype=torch.float).to(self.device)

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainLongMemory(self):
        if len(self.memory) > batchSize:
            miniSample = random.sample(self.memory, batchSize)
        else:
            miniSample = self.memory

        states, actions, rewards, nextStates, dones = zip(*miniSample)
        states = torch.stack(states).to(self.device)
        nextStates = torch.stack(nextStates).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)

        self.trainer.trainStep(states, actions, rewards, nextStates, dones)

    def trainShortMemory(self, state, action, reward, nextState, done):
        state = state.to(self.device)
        nextState = nextState.to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        self.trainer.trainStep(state, action, reward, nextState, done)

    def getAction(self, state):

        if random.random() < self.epsilon:
            finalMove = random.randint(0, 1)
        else:
            state0 = state.to(self.device)
            prediction = self.model(state0)
            finalMove = torch.argmax(prediction).item()
        return finalMove


def train(continue_training=False, model_path='model_with_epsilon.pth'):
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    agent = Agent()

    if continue_training:
        agent.load_model_with_epsilon(model_path)
        record = agent.record
    else:
        record = 0

    game = Game(render=False)
    num_games_to_train = 500000
    plot_frequency = 100
    train_frequency = 4
    step_count = 0

    for i in range(num_games_to_train):
        while True:
            oldState = agent.getState(game)
            finalMove = agent.getAction(oldState)
            reward, done, score = game.playStep(finalMove)
            newState = agent.getState(game)

            agent.remember(oldState, finalMove, reward, newState, done)

            step_count += 1
            if step_count % train_frequency == 0:
                agent.trainLongMemory()

            if done:
                break

        game.reset()
        agent.numberGames += 1



        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if score > record:
            record = score
            agent.record = record
            agent.save_model_with_epsilon(f'model_{agent.numberGames}_with_epsilon.pth')
        elif agent.numberGames % 500 == 0:
            agent.save_model_with_epsilon(f'model_{agent.numberGames}_with_epsilon.pth')


        print('Game', agent.numberGames, 'Score', score, 'Record:', record)

        plotScores.append(score)
        totalScore += score
        meanScore = totalScore / agent.numberGames
        plotMeanScores.append(meanScore)

        if agent.numberGames % plot_frequency == 0:
            plot(plotScores, plotMeanScores)

    plot(plotScores, plotMeanScores)

# Call the train function with continue_training set to True
train(continue_training=True, model_path='C:\\Users\\nolan\\PycharmProjects\\ComplexFlappyBirdAI\\model\\model_47500_with_epsilon.pth')