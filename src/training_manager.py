import os
from model import NeuralNetwork
import numpy as np
from stats_saver import add_csv_row

class Training:

    def __init__(self, path, iterration=0, population_size=50, sigma=0.3, learning_rate=0.1, best_main=-1, best_all=-1, model=None):
        self.path = path
        self.iterration = iterration
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.model = model
        self.best_main = best_main
        self.best_all = best_all

    def save(self, main_weights, main_fitness, population_weights, population_fitness):
        print("Saving", end="\r")

        self.iterration += 1

        max_f = np.amax(population_fitness)
        min_f = np.amin(population_fitness)
        all_fitness = population_fitness.copy()
        all_fitness.insert(0, min_f)
        all_fitness.insert(0, max_f)
        all_fitness.insert(0, main_fitness)
        add_csv_row(f"{self.path}/stats.csv", all_fitness)
        
        self.model.set_weights(main_weights)
        self.model.save(f"{self.path}/model.h5")
        
        if main_fitness > self.best_main:
            self.best_main = main_fitness
            
        best_i = np.argmax(population_fitness)
        best_f = population_fitness[best_i]
        best_w = population_weights[best_i]
        
        if best_f > self.best_all:
            self.best_all = best_f
            self.model.set_weights(best_w)
            self.model.save(f"{self.path}/best.h5")

        if main_fitness > self.best_all:
            self.best_all = main_fitness
            self.model.set_weights(main_weights)
            self.model.save(f"{self.path}/best.h5")
        
        with open(f"{self.path}/training_info.txt", "r") as info_file:
            data = info_file.readlines()
            data[0] = f"{self.iterration}\n"
            data[3] = f"{self.learning_rate}\n"
            data[4] = f"{self.best_main}\n"
            data[5] = f"{self.best_all}"

        with open(f"{self.path}/training_info.txt", "w") as info_file:
            info_file.writelines(data)

        print(f"Iterration {self.iterration}: {main_fitness}")

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            print("Training can not be continued, cause it does not exist!")
            exit()

        with open(f"{path}/training_info.txt", "r") as info_file:
            lines = info_file.readlines()
            iterration    = int(lines[0])
            pop_size      = int(lines[1])
            sigma         = float(lines[2])
            learning_rate = float(lines[3])
            best_main     = float(lines[4])
            best_all      = float(lines[5])
        model = NeuralNetwork.load(f"{path}/model.h5")

        return Training(path, iterration, pop_size, sigma, learning_rate, model=model, best_main=best_main, best_all=best_all)

    @staticmethod
    def create(path, population_size=50, sigma=0.3, learning_rate=0.1, input_shape=(75, 80), output_dim=4):
        if os.path.exists(path):
            print("Training already exists! Did you maybe wanna continue this training?")
            exit()
        
        model = NeuralNetwork(input_shape, output_dim)
        training = Training(path, population_size=population_size, sigma=sigma, learning_rate=learning_rate, model=model)
            
        os.makedirs(path)
        with open(f"{path}/training_info.txt", "w+") as info_file:
            info_file.write(f"{training.iterration}\n")
            info_file.write(f"{training.population_size}\n")
            info_file.write(f"{training.sigma}\n")
            info_file.write(f"{training.learning_rate}\n")
            info_file.write(f"{training.best_main}\n")
            info_file.write(f"{training.best_all}")
        training.model.save(f"{path}/model.h5")
        with open(f"{path}/stats.csv", "w+") as _:
            pass

        return training