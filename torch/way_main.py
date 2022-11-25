from logger import Logger
from way_agent import Agent
from graph import Graph
from env import Env

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
# import safety_gym
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

sys.path.append("/home/ineogi2/RL-Lab/metadrive")
from metadrive import SafeMetaDriveEnv

sys.path.append("/home/ineogi2/RL-Lab")
from PID.PID_controller_v6 import Controller, State

def norm(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def yaw(pt1, pt2):
    return np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])

def waypoint_to_action_space(positions):
    positions = np.array(positions)
    waypoint = deque()
    for i in range(len(positions)-1):
        wp = positions[i+1] - positions[0]
        waypoint.append(wp[0]/2)
        waypoint.append(-wp[1]/2)
    return waypoint

def train(main_args):
    algo_idx = 1
    agent_name = '1124'
    env_name = "Safe-metadrive-env"
    max_ep_len = 500
    max_episodes = 10
    epochs = 2500
    save_freq = 10
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':256,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.01,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':10.0/1000.0,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    # env = Env(env_name, seed, max_ep_len)
    env=SafeMetaDriveEnv(dict(use_render=False,
                    # manual_control=True,
                    random_lane_width=True,
                    random_lane_num=True,
                    traffic_density=0,
                    start_seed=random.randint(0, 1000)))
    agent = Agent(env, device, args)

    state_converter = State()
    controller = Controller()

    # for wandb
    wandb.init(project='[torch] CPO', entity='ineogi2', name='1124-waypoint-gru')
    if main_args.graph: graph = Graph(10, "TRPO", ['score', 'cv', 'policy objective', 'value loss', 'kl divergence', 'entropy'])

    for epoch in range(epochs):
        trajectories = []
        ep = 9
        scores = []
        cvs = []
        fails = 0
        out_of_road = 0; crash_vehicle = 0; crash_object = 0; broken_line = 0

        while ep < max_episodes:
            state = env.reset()
            controller.reset()
            # env.vehicle.expert_takeover=True

            ep += 1
            score = 0
            cv = 0
            step = 0
            broken_step = 0

            # waypoint_num = 5
            # state_deque = deque(maxlen=waypoint_num+1)
            # position_deque = deque(maxlen=waypoint_num+1)
            # heading_deque = deque(maxlen=waypoint_num+1)
            # reward_deque = deque(maxlen=waypoint_num+1)
            # cost_deque = deque(maxlen=waypoint_num+1)
            # done_deque = deque(maxlen=waypoint_num+1)
            # fail_deque = deque(maxlen=waypoint_num+1)

            next_state, reward, done, info = env.step([0,0])
            while True:                
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor = agent.getAction(state_tensor, is_train=True)
                waypoints = action_tensor.detach().cpu().numpy()
                # print(waypoints)

                state_converter.state_update(info, waypoints)
                # print(str(state_converter))
                controller.update_all(state_converter)
                controller.update_controls()
                steer = controller.steer
                acc = (25-info["vehicle_speed"])*0.7

                next_state, reward, done, info = env.step([steer, acc])
                # next_state, reward, done, info = env.step([0, 0])

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

                done = True if step >= max_ep_len else done
                fail = True if step < max_ep_len and done else False
                trajectories.append([state, waypoints, reward, cost, done, fail, next_state])

                # # deque
                # if step%3==1:
                #     state_deque.append(state)
                #     position_deque.append([info["vehicle_position"][0], -info["vehicle_position"][1]])
                #     # heading_deque.append(np.arctan2(-info['vehicle_heading'][1], info['vehicle_heading'][0]))
                #     reward_deque.append(reward)
                #     cost_deque.append(cost)
                #     done_deque.append(done)
                #     fail_deque.append(fail)

                # if len(position_deque) == waypoint_num+1:
                #     train_action = waypoint_to_action_space(position_deque)
                #     # print("train action : ", train_action)
                #     trajectories.append([state_deque[0], train_action, reward_deque[0],
                #                     cost_deque[0], done_deque[0], fail_deque[0], state_deque[1]])
                #     position_deque.popleft()

                state = next_state
                score += reward
                cv += info['num_cv']

                if done or step >= max_ep_len:
                    break

            scores.append(score)
            cvs.append(cv)

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
        score = np.mean(scores)
        cvs = np.mean(cvs)
        log_data = {"score":score, "out of road":out_of_road, "crash vehicle":crash_vehicle, 
                    "crash object":crash_object, "broken line":broken_line, "success_rate (%)":100-100*fails/max_episodes,
                    "cv":cvs, "value loss":v_loss, "cost value loss":cost_v_loss, 
                    "objective":objective, "cost surrogate":cost_surrogate, "kl":kl, "entropy":entropy}
        print(f'epoch : {epoch+1}')
        print(log_data,"\n")
        if main_args.graph: graph.update([score, objective, v_loss, kl, entropy])
        wandb.log(log_data)
        if (epoch + 1)%save_freq == 0:
            agent.save()

    if main_args.graph: graph.update(None, finished=True)

def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        test(args)
    else:
        train(args)
