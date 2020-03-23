
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines import PPO2
import aida_env.aida_gym_env as e
import pybullet as p
import argparse
import imageio
import time
import os
import json

workDirectory = "."
parser = argparse.ArgumentParser(description='Aida making gif from model')
parser.add_argument('--name', default=None, required = True,
				help='Name of the model (required)')
parser.add_argument('--normalize', type=bool, default=False, 
                    help='Normalize the environement for training (default: False)')
args = parser.parse_args()
name_resume = args.name
normalize   = args.normalize
commands = [[1,0],[2,0],[3,0]]
if name_resume!=None:

	model = PPO2.load(   workDirectory+"/resultats/"+name_resume+"/"+name_resume+".zip")

	
env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
										  render  = False, 
										  on_rack = False,
										  )
				])
if normalize:
	env = VecNormalize(env, clip_obs=1000.0, clip_reward=1000.0, training = False)
	env.load_running_average(workDirectory+"/resultats/"+name_resume+"/normalizeData")

images = []
obs = env.reset()
img = env.render(mode='rgb_array')
for i in range(15*2*10):
	images.append(img)
	action, _ = model.predict(obs)
	obs, _, _ ,_ = env.step(action)
	img = env.render(mode='rgb_array')

imageio.mimsave(workDirectory+"/resultats/"+name_resume+"/video/"+name_resume+".gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=20)