from es import EvolutionStrategy
from model import NeuralNetwork
from training_manager import Training
from agent import SMBAgent

# Game parameters
new_training = False
render = False
training_name = "session/training48"

# Training parameters
pop_size = 50
sigma = 0.4
learning_rate = 0.2


if new_training:
    training = Training.create(training_name, pop_size, sigma, learning_rate)
else:
    training = Training.load(training_name)

model = training.model.copy()
def get_reward(weights):
    model.set_weights(weights)

    agent = SMBAgent("Level1-1")
    fitness1, _ = agent.play(model, render)
    agent.change_env("Level1-2")
    fitness2, _ = agent.play(model, render)

    fitness = fitness1 + fitness2

    return fitness


es = EvolutionStrategy(training.model.get_weights(), get_reward, training.population_size, training.sigma, training.learning_rate)

while True:
    (main_weights, main_reward), (population_weights, population_rewards) = es.run_generation()

    training.save(main_weights, main_reward, population_weights, population_rewards)