# coding=utf-8

"""
ref:https://github.com/JannesKlaas/sometimes_deep_sometimes_learning
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD

from experience_replay import ExperienceReplay
from catch_game import Catch
from catch_game import num_actions
from catch_game import grid_size

from training_testing import train
from training_testing import test



# parameters
epsilon = 0.1  # exploration
max_memory = 500  # Maximum number of experiences we are storing
hidden_size = 100  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
epoch = 50


def baseline_model(grid_size, num_actions, hidden_size):
    # seting up the model with keras
    model = Sequential()
    model.add(
        Dense(hidden_size, input_shape=(grid_size ** 2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(SGD(lr=.1), "mse")
    return model


# Define environment/game
env = Catch()

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

model = baseline_model(grid_size, num_actions, hidden_size)
train(env, model, exp_replay, epoch, epsilon, num_actions, batch_size)
test(model)
