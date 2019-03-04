from retrowrapper import RetroWrapper
import numpy as np
import time

class SMBAgent():

    def __init__(self, level="Level1-1"):
        self.env = RetroWrapper("SuperMarioBros-Nes", level)

    def change_env(self, level):
        self.env.close()
        self.env = RetroWrapper("SuperMarioBros-Nes", level)

    def play(self, model, render=False, realtime=False):
        fitness = 0
        images = []
        
        obs = self.env.reset()
        done = False
        score = 0
        timer = 0

        while not done:

            images.append(obs)

            if render:
                self.env.render()

            try:
                nnOutput = model.predict(obs)[0]
                action = np.array([False] * 9, dtype=np.int64)
                action[0] = 1 if nnOutput[0] >= 0 else 0
                action[6] = 1 if nnOutput[1] >= 0 else 0
                action[7] = 1 if nnOutput[2] >= 0 else 0
                action[8] = 1 if nnOutput[3] >= 0 else 0
            except IndexError:
                print("Coulnd't convert NN output to action!")
                exit()    

            obs, reward, done, info = self.env.step(action)

            if reward > 0:
                score += reward
                timer = 0
            else:
                timer += 1

            if timer >= 60*5:
                done = True

            if realtime and render:
                time.sleep(1/60)

        fitness = score

        return fitness, images

class SMBAgent_old():

    def __init__(self, level="Level1-1"):
        self.env = RetroWrapper("SuperMarioBros-Nes", level)

    def change_env(self, level):
        self.env.close()
        self.env = RetroWrapper("SuperMarioBros-Nes", level)

    def play(self, model, render=False, realtime=False):
        fitness = 0
        images = []
        
        obs = self.env.reset()
        done = False
        score = 0
        timer = 0

        while not done:

            images.append(obs)

            if render:
                self.env.render()

            try:
                nnOutput = model.predict(obs)[0]
                action = np.array([False] * 9, dtype=np.int64)
                action[6] = 1 if nnOutput[0] >= 0 else 0
                action[7] = 1 if nnOutput[1] >= 0 else 0
                action[8] = 1 if nnOutput[2] >= 0 else 0
            except IndexError:
                print("Coulnd't convert NN output to action!")
                exit()    

            obs, reward, done, info = self.env.step(action)

            if reward > 0:
                score += reward
                timer = 0
            else:
                timer += 1

            if timer >= 60:
                done = True

            if realtime and render:
                time.sleep(1/60)

        fitness = score

        return fitness, images