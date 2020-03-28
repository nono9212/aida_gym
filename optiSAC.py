from copy import deepcopy

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from stable_baselines import SAC, DDPG, TD3
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines.common.base_class import _UnvecWrapper
import gym

import tensorflow as tf
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines import PPO2
import aida_env.aida_gym_env as e
import pybullet as p


import time
import os



def hyperparam_optimization(   n_trials=20, n_timesteps=100000, hyperparams=None,
                            n_jobs=1, sampler_method='random', pruner_method='halving',
                            seed=1, verbose=1):
    """
    :param algo: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict)
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    if hyperparams is None:
        hyperparams = {}

    # test during 3000 steps
    n_test_steps = 1500
    # evaluate every 20th of the maximum budget per iteration
    n_evaluations = 20
    evaluate_interval = int(n_timesteps / n_evaluations)

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.



    #sampler = RandomSampler(seed=seed)

    #sampler = TPESampler(n_startup_trials=5, seed=seed)

    sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})


    #pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=n_evaluations // 3)


    study = optuna.create_study(study_name="optimisation_PPO2", sampler = sampler , pruner=pruner, storage='sqlite:///optimizationSAC.db',load_if_exists=True)


    def objective(trial):

        kwargs = hyperparams.copy()

        trial.model_class = None

        kwargs.update(sample_sac_params(trial))

        def callback(_locals, _globals):
            """
            Callback for monitoring learning progress.
            :param _locals: (dict)
            :param _globals: (dict)
            :return: (bool) If False: stop training
            """
            self_ = _locals['self']
            trial = self_.trial

            # Initialize variables
            if not hasattr(self_, 'is_pruned'):
                self_.is_pruned = False
                self_.last_mean_test_reward = -np.inf
                self_.last_time_evaluated = 0
                self_.eval_idx = 0

            if (self_.num_timesteps - self_.last_time_evaluated) < evaluate_interval:
                return True

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
            while n_steps_done < n_test_steps:
                # Use default value for deterministic
                action, _ = self_.predict(obs,)
                obs, reward, done, _ = self_.test_env.step(action)
                reward_sum += reward
                n_steps_done += 1

                if done:
                    rewards.append(reward_sum)
                    reward_sum = 0.0
                    obs = self_.test_env.reset()
                    n_steps_done = n_test_steps
            rewards.append(reward_sum)
            mean_reward = np.mean(rewards)
            summary = tf.Summary(value=[tf.Summary.Value(tag='evaluation', simple_value=mean_reward)])
            _locals['writer'].add_summary(summary, self_.num_timesteps)
            self_.last_mean_test_reward = mean_reward
            self_.eval_idx += 1

            # report best or report current ?
            # report num_timesteps or elasped time ?
            trial.report(-1 * mean_reward, self_.eval_idx)
            # Prune trial if need
            if trial.should_prune(self_.eval_idx):
                self_.is_pruned = True
                return False

            return True
        commands = [[1,0],[2,0],[3,0]]
        env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
                                                  render  = False, 
                                                  on_rack = False,
                                                  default_reward     = 2,
                                                  height_weight      = 5,
                                                  orientation_weight = 3,
                                                  direction_weight   = 2,
                                                  speed_weight       = 4
                                                  )
                        ])


        model = SAC(MlpPolicy, 
                 env, 
                  gamma=kwargs['gamma'],
                  learning_rate= kwargs['learning_rate'],
                  batch_size=kwargs['batch_size'],
                  buffer_size=kwargs['buffer_size'],
                  learning_starts=kwargs['learning_starts'],
                  train_freq=kwargs['train_freq'],
                  gradient_steps=kwargs['gradient_steps'],
                  ent_coef=kwargs['ent_coef'],
                  target_entropy=kwargs['target_entropy'],
                  policy_kwargs=kwargs['policy_kwargs'],
                 tensorboard_log = "./optimisationSAC/logOPTI"
               )

        model.test_env = DummyVecEnv([lambda:  e.AidaBulletEnv(commands,
                                                  render  = False, 
                                                  on_rack = False,
                                                  default_reward     = 2,
                                                  height_weight      = 5,
                                                  orientation_weight = 3,
                                                  direction_weight   = 2,
                                                  speed_weight       = 2
                                                  )
                        ])


        model.trial = trial
       
        try:
            model.learn(n_timesteps, callback=callback)
            # Free memory
            model.env.close()
            model.test_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            model.test_env.close()
            raise
        is_pruned = False
        cost = np.inf
        if hasattr(model, 'is_pruned'):
            is_pruned = model.is_pruned
            cost = -1 * model.last_mean_test_reward
        try:
            os.mkdir("./optimisationSAC/resultats/"+str(trial.number))
        except FileExistsError:
            print("Directory already exists")
            


        model.save("./optimisation/resultats/"+str(trial.number)+"/"+str(trial.number))    


        del model.env, model.test_env
        del model

        if is_pruned:
            try:
                # Optuna >= 0.19.0
                raise optuna.exceptions.TrialPruned()
            except AttributeError:
                raise optuna.structs.TrialPruned()

        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical('learning_starts', [0, 1000, 10000, 20000])
    train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])

    net_arch = {
        'small': [64, 64],
        'medium': [256, 256],
        'big': [400, 300],
    }[net_arch]

    target_entropy = 'auto'
    if ent_coef == 'auto':
        target_entropy = trial.suggest_categorical('target_entropy', ['auto', -1, -10, -20, -50, -100])

    return {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'ent_coef': ent_coef,
        'target_entropy': target_entropy,
        'policy_kwargs': dict(layers=net_arch)
    }


if __name__ == '__main__':
    hyperparam_optimization()
