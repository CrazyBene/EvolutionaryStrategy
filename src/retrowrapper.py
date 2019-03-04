import retro

class RetroWrapper:

    def __init__(self, game, state):
        self.env = retro.make(game, state)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        sumReward = reward
        obs, reward, done, info = self.env.step(action)
        sumReward += reward
        return obs, sumReward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()