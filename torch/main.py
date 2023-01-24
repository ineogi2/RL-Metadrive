from logger import Logger
from agent import Agent
from graph import Graph
from env import Env

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import copy
import time
import gym
import sys
from collections import deque
import os
import matplotlib.pyplot as plt

sys.path.append("/home/ineogi2/RL-Lab/metadrive")
from metadrive import SafeMetaDriveEnv

from utils import *

###### train ######

def train(main_args, model_args):
    max_ep_len = 1500
    max_episodes = 10
    epochs = 2500
    save_freq = 10
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    """ for random seed """
    seed = model_args['algo_idx'] + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    """ env = Env(env_name, seed, max_ep_len) """
    env=SafeMetaDriveEnv(dict(use_render=True if main_args.render else False,
                    manual_control=True if main_args.expert else False,
                    random_lane_width=True,
                    random_lane_num=True,
                    traffic_density=model_args['traffic_density'],
                    start_seed=random.randint(0, 1000)))
    
    """ for CPO """
    agent = Agent(env, device, model_args)

    """ for wandb """
    if main_args.wandb:
        wandb.init(project='[torch] CPO', entity='ineogi2', name=f"{model_args['agent_name']}-{model_args['algo_idx']}-train")
    if main_args.graph: graph = Graph(10, "TRPO", ['score', 'cv', 'policy objective', 'value loss', 'kl divergence', 'entropy'])

    for epoch in range(epochs):
        trajectories = []
        driving_reward = []; speed_reward = []; lane_deviation_reward = []
        ep = 0
        scores = []
        cvs = []
        fails = 0
        out_of_road = 0; crash_vehicle = 0; crash_object = 0; broken_line = 0

        while ep < max_episodes:
            state = env.reset()
            if main_args.expert: env.vehicle.expert_takeover=True
            env.vehicle.config.update({'max_speed':30}, allow_add_new_key=True)

            ep += 1
            score = 0
            cv = 0
            step = 0
            broken_step = 0
            past_steering = 0

            state, reward, done, info = env.step([0, 0])
            while True:                
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                # print(action)

                if main_args.expert:
                    next_state, reward, done, info = env.step([0, 0])
                else:
                    next_state, reward, done, info = env.step(action)

                """ reward for broken line cross """
                cost = info['cost']
                if cost == 0:
                    broken_step = 0
                else:
                    if info["cost_reason"] == "out_of_road_cost": out_of_road+=1; fails+=1; broken_step = 0
                    elif info["cost_reason"] == "crash_vehicle_cost": crash_vehicle+=1; broken_step = 0
                    elif info["cost_reason"] == "crash_object_cost": crash_object+=1; broken_step = 0
                    elif info["cost_reason"] == "on_broken_line":
                        if broken_step < 20:
                            broken_step+=1
                            cost = 0
                        else:
                            broken_line+=1
                            reward -= cost
                            cost = 0
                """ ---------------------------- """

                """ reward for steering angle """
                current_steering = env.vehicle.steering
                steering_diff = abs(current_steering - past_steering)
                reward -= steering_diff
                past_steering = current_steering
                """ ------------------------- """

                done = True if step >= max_ep_len else done
                fail = True if step < max_ep_len and done else False
                trajectories.append([state, action, reward, cost, done, fail, next_state])
                (driving_r, speed_r, lane_deviation_r) = info['reward_info']
                driving_reward.append(driving_r); speed_reward.append(speed_r); lane_deviation_reward.append(lane_deviation_r)

                state = next_state
                score += reward
                cv += info['num_cv']

                if done or step >= max_ep_len:
                    break

            scores.append(score)
            cvs.append(cv)

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
        driving_reward = np.mean(driving_reward); speed_reward = np.mean(speed_reward); lane_deviation_reward = np.mean(lane_deviation_reward)
        score = np.mean(scores)
        cvs = np.mean(cvs)
        log_data = {"score":score, "objective":objective, "broken line":broken_line, 'driving_reward' : driving_reward, 'speed_reward' : speed_reward,
                    'lane_deviation' : lane_deviation_reward, "out of road":out_of_road, "crash vehicle":crash_vehicle, "crash object":crash_object, 
                    "success_rate (%)":100-100*fails/max_episodes, "cv":cvs, "value loss":v_loss, "cost value loss":cost_v_loss, 
                    "cost surrogate":cost_surrogate, "kl":kl, "entropy":entropy}
        print(f'epoch : {epoch+1}')
        print(log_data,"\n")

        if main_args.graph: graph.update([score, objective, v_loss, kl, entropy])
        if main_args.wandb: wandb.log(log_data)

        if (epoch + 1)%save_freq == 0:
            agent.save()

    if main_args.graph: graph.update(None, finished=True)


###### Behavior cloning ######

def imitaion_learning(main_args, model_args):
    max_ep_len = 1500
    epochs = 300
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')


    """ env = Env(env_name, seed, max_ep_len) """
    env=SafeMetaDriveEnv(dict(use_render=True if main_args.render else False,
                    manual_control=True,
                    random_lane_width=True,
                    random_lane_num=True,
                    traffic_density=model_args['traffic_density'],
                    start_seed=random.randint(0,1000)))

    """ for CPO """
    agent = Agent(env, device, model_args)

    """ for random seed """
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    pred_length = model_args['pred_length']

    if main_args.wandb:
        wandb.init(project='[torch] CPO', entity='ineogi2', name=f"{model_args['agent_name']}-{model_args['algo_idx']}-imitation learning")

    for epoch in range(epochs):
        trajectories = []
        driving_reward = []; speed_reward = []; lane_deviation_reward = []
        env.reset()
        env.vehicle.expert_takeover=True
        env.vehicle.config.update({'max_speed':30}, allow_add_new_key=True)
        step = 0
        broken_step = 0
        score = 0
        past_steering = 0

        state, reward, done, info = env.step([0, 0])
        while True:
            step += 1

            next_state, reward, done, info = env.step([0, 0])
            action = [env.vehicle.steering, env.vehicle.throttle_brake]

            """ reward for broken line cross """
            cost = info['cost']
            if cost == 0:
                broken_step = 0
            else:
                if info["cost_reason"] == "on_broken_line":
                    if broken_step < 20:
                        broken_step += 1
                        cost = 0
                    else:
                        reward -= cost
                        cost = 0
            """ ---------------------------- """

            # """ reward for steering angle """
            # current_steering = env.vehicle.steering
            # steering_diff = abs(current_steering - past_steering)
            # reward -= steering_diff/10
            # past_steering = current_steering
            # """ ------------------------- """

            done = True if step >= max_ep_len else done
            fail = True if step < max_ep_len and done else False

            trajectories.append([state, action, reward, cost, done, fail, next_state])
            # lateral_distance = info['lane_deviation']         # negative : left lateral / positive : right lateral
            (driving_r, speed_r, lane_deviation_r) = info['reward_info']
            driving_reward.append(driving_r); speed_reward.append(speed_r); lane_deviation_reward.append(lane_deviation_r)

            state = next_state
            score += reward

            if done: break

        """ for CPO """
        print(f'\nepoch : {epoch+1}')
        v_loss, cost_v_loss, objective, _, kl, _ = agent.train(trajs=trajectories)
        driving_reward = np.mean(driving_reward); speed_reward = np.mean(speed_reward); lane_deviation_reward = np.mean(lane_deviation_reward)
        log_data = {'score' : score, 'objective' : objective, 'driving_reward' : driving_reward, 'speed_reward' : speed_reward,
                    'lane_deviation' : lane_deviation_reward, 'value loss' : v_loss, 'cost value loss' : cost_v_loss, 'kl' : kl}
        if main_args.wandb: wandb.log(log_data)
        print(log_data)
        if (epoch+1) % 3 == 0:
            agent.save()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--imitation', action='store_true', help='For imitation learning.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    parser.add_argument('--wandb', action='store_true', help='For Wandb.')
    parser.add_argument('--expert', action='store_true', help='For expert takeover')
    parser.add_argument('--render', action='store_true', help='For rendering')
    main_args = parser.parse_args()

    algo_idx = 1
    agent_name = '0123'
    env_name = "Safe-metadrive-env"
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)

    model_args = {
        'algo_idx':algo_idx,
        'agent_name':agent_name,
        'env_name':env_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':256,
        'hidden2':256,
        'v_lr':1e-3,
        'cost_v_lr':1e-3,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.01,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':1.0/1000.0,
        'pred_length':1,
        'traffic_density':0.05
    }

    if main_args.imitation:
        imitaion_learning(main_args, model_args)
    else:
        train(main_args, model_args)
