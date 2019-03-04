import numpy as np

class EvolutionStrategy(object):

    def __init__(self, weights, get_fitness_func, population_size, sigma, learning_rate):
        self.weights = weights
        self.get_fitness = get_fitness_func
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_population(self):
        population = []
        for i in range(self.population_size):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.sigma * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def _update_weights(self, population_rewards, population):
        population_rewards = np.array(population_rewards)
        std = population_rewards.std()
        if std == 0:
            return
        rewards = (population_rewards - population_rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.population_size * self.sigma)
            self.weights[index] = w + update_factor * np.dot(layer_population.T, rewards).T

    def run_generation(self):
        population = self._get_population()

        population_weights = []
        population_rewards = []
        for p in population:
            weights_try = self._get_weights_try(self.weights, p)
            population_weights.append(weights_try)
            population_rewards.append(self.get_fitness(weights_try))

        self._update_weights(population_rewards, population)
        main_reward = self.get_fitness(self.weights)

        return (self.weights, main_reward), (population_weights, population_rewards)