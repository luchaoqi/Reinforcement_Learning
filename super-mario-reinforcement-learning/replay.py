from agent import DQNAgent
# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper

# from IPython import display
import matplotlib
import matplotlib.pyplot as plt


# Build env
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
#env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = JoypadSpace(env,RIGHT_ONLY)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent:q
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Replay
# model_path = '/home/luchao-qi/Downloads/models'
# model_path = './models'
model_path = './models/temp_models'
# './models/final-vm-1'
agent.replay(env=env, model_path=model_path, n_replay=1, plot=False)
