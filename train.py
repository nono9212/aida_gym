import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy as MlpPolicyPPO2
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines import PPO2,SAC
import aida_env.aida_gym_env as e
import pybullet as p
import argparse
import imageio
import time
import os
import json
import tensorflow as tf
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy as MlpPolicyTD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import sqlite3



if __name__ == '__main__':

	workDirectory = "."
	


	parser = argparse.ArgumentParser(description='Aida traning script')

	parser.add_argument('--resume_from', default=None,
                    help='Name of the model to start from')				
	parser.add_argument('--normalize', type=bool, default=False, 
                    help='Normalize the environement for training (default: False)')
	parser.add_argument('--name', default=None, required = True,
                    help='Name of the model (required)')
	parser.add_argument('--total_steps', default=10000000, type=int,
                    help='Total number of steps to train the model (default: 10 000 000)')
	parser.add_argument('--save_every', default=500000, type=int,
                    help='The number of step to train the model before saving (default: 500 000)')
	parser.add_argument('--algo', default="ppo2", type=str,
                    help='The algorythme to be used, ppo2 or sac (default: ppo2)')	
	
	parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor (default: 0.99)')
	parser.add_argument('--n_steps', type=int, default=128,
                    help='The number of steps to run for each environment per update (default: 128)')
	parser.add_argument('--ent_coefppo2', type=float, default=0.01,
                    help='Entropy coefficient for the loss calculation (default: 0.01)')
	parser.add_argument('--learning_rate', type=float, default=0.00025,
                    help='The learning rate (default: 0.00025)')
	parser.add_argument('--vf_coef', type=float, default=0.5,
                    help='Value function coefficient for the loss calculation (default: 0.5)')
	parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='The maximum value for the gradient clipping (default: 0.5)')
	parser.add_argument('--lam', type=float, default=0.95,
                    help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator (default: 0.95)')
	parser.add_argument('--nminibatches', type=int, default=4,
                    help='Number of training minibatches per update (default: 4)')
	parser.add_argument('--noptepochs', type=int, default=4,
                    help='Number of epoch when optimizing the surrogate (default: 4)')
	parser.add_argument('--cliprange', type=float, default=0.2,
                    help='Clipping parameter (default: 0.2)')
	parser.add_argument('--cliprange_vf', type=float, default=None,
                    help='Clipping parameter for the value function, it can be a function. This is a parameter specific to the OpenAI implementation. If None is passed (default), then cliprange (that is used for the policy) will be used. IMPORTANT: this clipping depends on the reward scaling. To deactivate value function clipping (and recover the original PPO implementation), you have to pass a negative value (e.g. -1). (default: None)')		
	parser.add_argument('--layers', default=[100,100],nargs='+', type=int,
					help='Architecture of the neural network (default: [100,100])')

					
	parser.add_argument('--default_reward', default=2.0, type=float,
					help='Reward aida gets for staying alive at each step (default: 2.0)')	
	parser.add_argument('--height_weight', default=2.0, type=float,
					help='Multiplicator of the height reward (default: 4.0)')		
	parser.add_argument('--orientation_weight', default=1.0, type=float,
					help='Multiplicator of the reward telling if aida stands straight (default: 1.0)')	
	parser.add_argument('--direction_weight', default=2.0, type=float,
					help='Multiplicator of the reward telling if aida faces its objective (default: 1.0)')	
	parser.add_argument('--speed_weight', default=10.0, type=float,
					help='Multiplicator of the speed reward (default: 0.0)')	
	parser.add_argument('--mimic_weight', default=20.0, type=float,
					help='Multiplicator of the speed reward (default: 0.0)')
			
	parser.add_argument('--consistancy_weight', default=10.0, type=float,
					help='Multiplicator of the speed reward (default: 0.0)')
			
	parser.add_argument('--batch_size', default=64, type=int,
					help=' (int) Minibatch size for each gradient update (default: 64)')	
	parser.add_argument('--buffer_size', default=50000, type=int,
					help=' (int) size of the replay buffer (default: 50000)')	
	parser.add_argument('--learning_starts', default=100, type=int,
					help=' (int) how many steps of the model to collect transitions for before learning starts(default: 100)')	
	parser.add_argument('--train_freq', default=1, type=int,
					help='(int) Update the model every train_freq steps. (default: 1)')	
	parser.add_argument('--gradient_steps', default=1, type=int,
					help='(int) How many gradient update after each step (default: 1)')	
	parser.add_argument('--target_entropy', default='auto',
					help=' (str or float) target entropy when learning ent_coef (ent_coef = ‘auto’) (default: 0.0)')						
	parser.add_argument('--ent_coefsac', default='auto',
                    help='(str or float) Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.) Controlling exploration/exploitation trade-off. Set it to ‘auto’ to learn it automatically (and ‘auto_0.1’ for using 0.1 as initial value)')

	args = parser.parse_args()
	
	model_name  = args.name
	name_resume = args.resume_from
	normalize   = args.normalize


	"""
	workDirectory/resultats/name/name.zip
							    /normalizeData/ret_rms.pkl
											  /obs_rms.pkl
								/video
				 /log
	"""
	
	try:
		os.mkdir(workDirectory+"/resultats")
	except FileExistsError:
		print("Directory already exists")
	try:
		os.mkdir(workDirectory+"/resultats/"+model_name)
	except FileExistsError:
		print("Directory already exists")
	if normalize:
		try:
			os.mkdir(workDirectory+"/resultats/"+model_name+"/normalizeData")
		except FileExistsError:
			print("Directory already exists")
	try:
		os.mkdir(workDirectory+"/resultats/"+model_name+"/video")
	except FileExistsError:
		print("Directory already exists")
	try:
		os.mkdir(workDirectory+"/log")
	except FileExistsError:
		print("Directory already exists")
	try:
		os.mkdir("./server/assets/video/"+model_name)
	except FileExistsError:
		print("Directory already exists")
		
		
	with open(workDirectory+"/resultats/"+model_name+'/data.txt', 'w') as outfile:
		json.dump(vars(args), outfile,sort_keys=True,indent=4)
	
	commands = [[-1,0],[-2,0],[-3,-1],[-3.5,-2],[-3.5,-3],[-3.5,-5],[-2,-6],[0,-7],[2,-6]]
	for i in range(5):
		commands += [[commands[-1][0]+np.random.rand(),commands[-1][0]+np.random.rand()]]

	if(args.algo=="td3" or args.algo == "sac"):

		env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
													  render  = False, 
													  on_rack = False,
													  default_reward     = args.default_reward,
													  height_weight      = args.height_weight,
													  orientation_weight = args.orientation_weight,
													  direction_weight   = args.direction_weight,

													  speed_weight       = args.speed_weight,
													  mimic_weight       = args.mimic_weight,
													  consistancy_weight = args.consistancy_weight
													  )
							])
	elif(args.algo == "ppo2"):
		env = SubprocVecEnv([lambda:  e.AidaBulletEnv(commands,
												  render  = False, 
												  on_rack = False,
												  default_reward     = args.default_reward,
												  height_weight      = args.height_weight,
												  orientation_weight = args.orientation_weight,
												  direction_weight   = args.direction_weight,
												  speed_weight       = args.speed_weight,
												  consistancy_weight = args.consistancy_weight
												  )
						for i in range(32)])
						

	if normalize:
		env = VecNormalize(env, gamma=args.gamma)

	if(args.algo == "ppo2"):
		model = PPO2(MlpPolicyPPO2, 
					 env, 

					 gamma           = args.gamma,
					 n_steps         = args.n_steps,
					 ent_coef        = args.ent_coefppo2,
					 learning_rate   = args.learning_rate,
					 vf_coef         = args.vf_coef,
					 max_grad_norm   = args.max_grad_norm,
					 lam             = args.lam,
					 nminibatches    = args.nminibatches,
					 noptepochs      = args.noptepochs,
					 cliprange       = args.cliprange,
					 cliprange_vf    = args.cliprange_vf,
					 verbose         = 0,
					 policy_kwargs   = dict(layers=args.layers),
					 tensorboard_log = workDirectory+"/log"
				   )
		
		if name_resume!=None:

			model = PPO2.load(   workDirectory+"/resultats/"+name_resume+"/"+name_resume+".zip",
								 env=env,
								 gamma           = args.gamma,
								 n_steps         = args.n_steps,
								 ent_coef        = args.ent_coefppo2,
								 learning_rate   = args.learning_rate,
								 vf_coef         = args.vf_coef,
								 max_grad_norm   = args.max_grad_norm,
								 lam             = args.lam,
								 nminibatches    = args.nminibatches,
								 noptepochs      = args.noptepochs,
								 cliprange       = args.cliprange,
								 cliprange_vf    = args.cliprange_vf,
								 verbose         = 0,
								 policy_kwargs   = dict(layers=args.layers),
								 tensorboard_log = workDirectory+"/log")
		if normalize:
			env.load_running_average(workDirectory+"/resultats/"+name_resume+"/normalizeData")	
							
							
							
	elif(args.algo == "sac"):
		model = SAC(MlpPolicySAC, 
			env, 
			gamma=args.gamma,
			learning_rate= args.learning_rate,

			batch_size=args.batch_size,
			buffer_size=args.buffer_size,
			learning_starts=args.learning_starts,
			train_freq=args.train_freq,
			gradient_steps=args.gradient_steps,
			ent_coef=args.ent_coefsac,
			target_entropy=args.target_entropy,

			policy_kwargs   = dict(layers=args.layers),
			tensorboard_log = workDirectory+"/log"
			)

	elif(args.algo=="td3"):
		n_actions = env.action_space.shape[-1]
		#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions))
		action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions), theta=0.15, dt=0.01, initial_noise=None)
		model = TD3(MlpPolicyTD3, 
				env, 
				action_noise=action_noise,
				verbose=1,
				policy_kwargs   = dict(layers=[400,300]) ,
				tensorboard_log = workDirectory+"/log"
				)

	model.test_env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
													  render  = False, 
													  on_rack = False,
													  default_reward     = args.default_reward,
													  height_weight      = args.height_weight,
													  orientation_weight = args.orientation_weight,
													  direction_weight   = args.direction_weight,

													  speed_weight       = args.speed_weight,
													  mimic_weight       = args.mimic_weight,
													  consistancy_weight = args.consistancy_weight,
													  logReward = True
													  )
							])
	if normalize:
		model.test_env = VecNormalize(model.test_env, gamma=args.gamma)
	
					

	def callback(_locals, _globals):
		"""
		Callback for monitoring learning progress.
		:param _locals: (dict)
		:param _globals: (dict)
		:return: (bool) If False: stop training
		"""
		self_ = _locals['self']


		# Initialize variables

		if not hasattr(self_, 'started'):
			self_.started = False
			self_.last_time_evaluated = 0

		if (self_.num_timesteps - self_.last_time_evaluated) < 20000:
			return True
			

		
		
		sql = ''' SELECT id, type, value FROM parameters WHERE simu="'''+self_.dbName+'''" AND step=-1 '''
		conn = sqlite3.connect("./server/database.db")		
		cur = conn.cursor()
		cur.execute(sql)
		dff = list(cur.fetchall())
		for d in dff:
		   self_.env.set_attr(d[1],d[2],indices=range(32))
		   self_.test_env.set_attr(d[1],d[2],indices=range(1))
		cur.execute("UPDATE parameters SET step={0} WHERE step=-1 AND simu='{1}'".format(self_.num_timesteps,self_.dbName))
		conn.commit()
		cur.close()
		conn.close()

		self_.last_time_evaluated = self_.num_timesteps
		



		# Evaluate the trained agent on the test env
		rewards = []
		n_steps_done, reward_sum = 0, 0.0


		# Sync the obs rms if using vecnormalize
		# NOTE: this does not cover all the possible cases
		if isinstance(self_.test_env, VecNormalize):
			self_.test_env.obs_rms = deepcopy(self_.env.obs_rms)
			self_.test_env.ret_rms = deepcopy(self_.env.ret_rms)
			# Do not normalize reward
			self_.test_env.norm_reward = False


		obs = self_.test_env.reset()
		rewardLog = []
		while n_steps_done < 1500:
			# Use default value for deterministic

			action, _ = self_.predict(obs, deterministic = True)
			obs, reward, done, _ = self_.test_env.step(action)
			reward_sum += reward
			n_steps_done += 1
			if(not(done)):
				rewardLog+=[self_.test_env.env_method("_get_LastrewardLog")[0]]
			if done:
				rewards.append(reward_sum)
				reward_sum = 0.0
				n_steps_done = 1500
				
				break
		rewards.append(reward_sum)
		mean_reward = np.mean(rewards)
		summary = tf.Summary(value=[tf.Summary.Value(tag='evaluation', simple_value=mean_reward)])
		data = np.mean(rewardLog, axis=0)
		names = self_.test_env.env_method("_get_rewardLogNames")[0]
		sql = ''' INSERT INTO output(simu, type, step, value)
              VALUES(?,?,?,?) '''
		val = (self_.dbName, "total_reward", self_.num_timesteps , float(mean_reward))
		conn = sqlite3.connect("./server/database.db")		
		cur = conn.cursor()
		cur.execute(sql,val)
		for i in range(len(data)):
		   val = (self_.dbName, names[i], self_.num_timesteps , float(data[i]))
		   cur.execute(sql,val)
		conn.commit()
		cur.close()
		conn.close()
		_locals['writer'].add_summary(summary, self_.num_timesteps)
		self_.last_mean_test_reward = mean_reward
		return True
			
	model.dbName = model_name
	conn = sqlite3.connect("./server/database.db")        
	cur = conn.cursor()
	val = model.env.env_method("_send_config")[0]
	names, values= val[0],val[1]
	for i in range(len(values)):
		sql = ''' INSERT INTO parameters(simu, type, step, value)
			VALUES(?,?,?,?) '''
		val = (model_name, names[i], 0, float(values[i]))
		cur.execute(sql,val)
		conn.commit()
	cur.close()
	conn.close()
	
	
	for i in range(args.total_steps//args.save_every):
		model.learn(total_timesteps=args.save_every, tb_log_name=model_name, reset_num_timesteps=False, callback=callback)
		if normalize:
			env.save_running_average(workDirectory+"/resultats/"+model_name+"/normalizeData")
		model.save(workDirectory+"/resultats/"+model_name+"/"+model_name)
		os.system("python3 makegif.py --algo "+args.algo+" --dir ./server/assets/"+model_name+"_"+str((i+1)*args.save_every)+"_steps.gif --name "+model_name)
		print("\n saved at "+str((i+1)*args.save_every))
	model.save(workDirectory+"/resultats/"+model_name+"/"+model_name)	
	if normalize:
		env.save_running_average(workDirectory+"/resultats/"+model_name+"/normalizeData")
	env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
													  render  = False, 
													  on_rack = False,
													  default_reward     = args.default_reward,
													  height_weight      = args.height_weight,
													  orientation_weight = args.orientation_weight,
													  direction_weight   = args.direction_weight,

													  speed_weight       = args.speed_weight,
													  mimic_weight       = args.mimic_weight,
													  consistancy_weight = args.consistancy_weight
													  )
							])
	if normalize:
		env = VecNormalize(env, gamma=args.gamma, training = False)
		env.load_running_average(workDirectory+"/resultats/"+model_name+"/normalizeData")

	images = []
	obs = env.reset()
	img = env.render(mode='rgb_array')
	for i in range(15*2*10):
		images.append(img)
		action, _ = model.predict(obs)
		obs, _, _ ,_ = env.step(action)
		img = env.render(mode='rgb_array')

	imageio.mimsave(workDirectory+"/resultats/"+model_name+"/video/vid.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=20)
