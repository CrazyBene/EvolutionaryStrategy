from model import NeuralNetwork
from agent import SMBAgent, SMBAgent_old
import imageio

model = NeuralNetwork.load(f"session/training48/best.h5")

agent = SMBAgent("Level1-1")
f1, i = agent.play(model, realtime=True, render=True)
agent.change_env("Level1-2")
f2, i = agent.play(model, realtime=True, render=True)


print(f1)
print(f2)

#imageio.mimsave("src/Mario-ES/training45/gifs/winner.gif", i, fps=30)