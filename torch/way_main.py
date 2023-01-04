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
import os
import matplotlib.pyplot as plt

sys.path.append("/home/ineogi2/RL-Lab/metadrive")
from metadrive import SafeMetaDriveEnv

sys.path.append("/home/ineogi2/RL-Lab")
from PID.PID_controller_v6 import Controller, State
# from PID.PID_controller_v7 import Controller, State


############ math tools ###########

def l1_distance(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def yaw(pt1, pt2):
    return np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])

def norm(pt):
    return (pt[0]**2+pt[1]**2)**0.5

# relative coordinate
def position_to_relative_wp(position_list, direction, pred_length):
    wp_list = deque()

    positions = np.array(position_list)
    positions[:,1] = -positions[:,1]                        # y coordinate reset
    positions = positions - positions[0]                    # relative position
    x_direction = np.array([direction[0], -direction[1]])    # y coordinate reset
    y_direction = np.array([direction[1], direction[0]])

    for i in range(1, pred_length+1):
        dx = np.dot(x_direction, positions[i])
        dy = np.dot(y_direction, positions[i])
        wp_list.append(dx)
        wp_list.append(dy)

    return wp_list

# absolute coordinate
def position_to_absolute_wp(position_list):
    positions = np.array(position_list)
    wp_list = deque()

    for i in range(len(positions)-1):
        wp = positions[i+1] - positions[0]
        wp_list.append(wp[0])
        wp_list.append(-wp[1])

    return wp_list

# modify waypoint to lane midpoint
def modify_waypoint(wp_list, cur_lane, info):
    cur_position = np.array(info['vehicle_position']); cur_position[1] *= -1
    lateral, _, lane_width, _, _ = info['vehicle_heading_sine']
    lateral = lateral/norm(lateral)
    lane_heading = np.array([lateral[1], -lateral[0]])
    pred_length = len(wp_list)//2
    
    modified_wp = []
    for i in range(pred_length):
        dx, dy = wp_list[2*i], wp_list[2*i+1]
        wp = cur_position+np.array([dx, dy])                    # real position of waypoint
        wp_to_vehicle_dist = l1_distance(wp, cur_position)
        wp_to_lane_dist = cur_lane.distance([wp[0], -wp[1]])    # distance from waypoint to current lane
        wp_to_lane_sign = np.dot(lateral, [dx, dy])     # positive : left wp / negative : right wp

        if wp_to_lane_dist <= lane_width/2:
            new_dx_dy = lane_heading*wp_to_vehicle_dist
        else:
            if wp_to_lane_sign >= 0:
                new_dx_dy = lateral*lane_width+lane_heading*wp_to_vehicle_dist
            else:
                new_dx_dy = lateral*(-lane_width)+lane_heading*wp_to_vehicle_dist
        # print(new_dx_dy)
        modified_wp.append(new_dx_dy[0])
        modified_wp.append(new_dx_dy[1])
    
    return modified_wp


###################################

import torch.nn as nn
class GRUNet(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.predict_length = args['pred_length']
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = 32
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)

        self.gru_way = nn.GRUCell(2, self.action_dim)
        self.fc_way = nn.Linear(self.action_dim, 2)

        self.act_fn = torch.relu
        self.output_act_fn = torch.tanh

    def forward(self, x):
        output_wp = []

        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        z = self.fc_mean(x)

        if z.dim() == 1:
            wp = torch.zeros(2).to("cuda")
        else:
            wp = torch.zeros(z.shape[0],2).to("cuda")

        for _ in range(self.predict_length):
            z = self.gru_way(wp, z)
            d_wp = self.fc_way(z)
            wp = wp + d_wp
            output_wp.append(wp)

        if z.dim() == 1:
            pred_wp = torch.stack(output_wp, dim=0).reshape(-1)
        else:
            pred_wp = torch.stack(output_wp, dim=1).reshape(z.shape[0],-1)

        return pred_wp


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
                    traffic_density=0,
                    start_seed=random.randint(0, 1000)))
    
    """ for CPO """
    agent = Agent(env, device, model_args)
    """ for Neural Network """
    # agent = GRUNet(model_args, env).to(device)
    # if os.path.isfile(f"{model_args['agent_name']}-GRUNet.pt"):
    #     agent = torch.load(f"{model_args['agent_name']}-GRUNet.pt")
    #     print('[Load] success.')
    # else:
    #     print('[New] model')

    state_converter = State()
    controller = Controller()

    """ for wandb """
    if main_args.wandb:
        wandb.init(project='[torch] CPO', entity='ineogi2', name=f"{model_args['agent_name']}-{model_args['algo_idx']}-train")
    if main_args.graph: graph = Graph(10, "TRPO", ['score', 'cv', 'policy objective', 'value loss', 'kl divergence', 'entropy'])

    for epoch in range(epochs):
        trajectories = []
        ep = 0
        scores = []
        cvs = []
        fails = 0
        out_of_road = 0; crash_vehicle = 0; crash_object = 0; broken_line = 0

        while ep < max_episodes:
            state = env.reset()
            controller.reset()
            if main_args.expert: env.vehicle.expert_takeover=True

            ep += 1
            score = 0
            cv = 0
            step = 0
            broken_step = 0

            state, reward, done, info = env.step([0, 0])
            while True:                
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor = agent.getAction(state_tensor, is_train=True)
                # action_tensor = agent(state_tensor)
                waypoints = action_tensor.detach().cpu().numpy()
                # waypoints = modify_waypoint(waypoints, env.vehicle.lane, info)
                # print(waypoints)
                # waypoints = [1,0,2,1,3,2,4,3,5,4]

                state_converter.state_update(info, waypoints)
                print(str(state_converter))
                controller.update_all(state_converter)
                controller.update_controls()
                steer = controller.steer
                acc = controller.acc

                if main_args.expert:
                    next_state, reward, done, info = env.step([0, 0])
                else:
                    next_state, reward, done, info = env.step([steer, acc])

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
        if main_args.wandb: wandb.log(log_data)

        if (epoch + 1)%save_freq == 0:
            agent.save()

    if main_args.graph: graph.update(None, finished=True)


def imitaion_learning(main_args, model_args):
    max_ep_len = 1000
    epochs = 500
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')


    """ env = Env(env_name, seed, max_ep_len) """
    env=SafeMetaDriveEnv(dict(use_render=True if main_args.render else False,
                    manual_control=True,
                    # random_lane_width=True,
                    # random_lane_num=True,
                    traffic_density=0.0,
                    start_seed=random.randint(0,1000)))

    """ for CPO """
    agent = Agent(env, device, model_args)
    """ for Neural Network """
    # agent = GRUNet(model_args, env).to(device)
    # if os.path.isfile(f"{model_args['agent_name']}-GRUNet.pt"):
    #     agent = torch.load(f"{model_args['agent_name']}-GRUNet.pt")
    #     print('[Load] success.')
    # else:
    #     print('[New] model')

    """ for random seed """
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    pred_length = model_args['pred_length']

    if main_args.wandb:
        wandb.init(project='[torch] CPO', entity='ineogi2', name=f"{model_args['agent_name']}-{model_args['algo_idx']}-imitation learning")

    """ for imitation learning """
    # import torch.optim as optim
    # optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    # loss_func = torch.nn.MSELoss()
    # loss_list = []

    for epoch in range(epochs):

        trajectories = []; loss_mean = 0
        state_list = deque(maxlen=pred_length+1)
        position_list = deque(maxlen=pred_length+1)
        reward_list = deque(maxlen=pred_length+1)
        cost_list = deque(maxlen=pred_length+1)
        done_list = deque(maxlen=pred_length+1)
        fail_list = deque(maxlen=pred_length+1)

        env.reset()
        env.vehicle.expert_takeover=True
        step = 0
        broken_step = 0

        while True:
            step += 1
            state, reward, done, info = env.step([0, 0])

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

            done = True if step >= max_ep_len else done
            fail = True if step < max_ep_len and done else False


            state_list.append(state)
            position_list.append(info["vehicle_position"])
            # direction_list.append(info['vehicle_heading'])
            reward_list.append(reward)
            cost_list.append(cost)
            done_list.append(done)
            fail_list.append(fail)

            if len(state_list) == pred_length+1:
                # waypoints = position_to_relative_wp(position_list, direction_list[0], pred_length)
                waypoints = position_to_absolute_wp(position_list)
                print(waypoints)
                trajectories.append([state_list[0], waypoints, reward_list[0], cost_list[0], done_list[0], fail_list[0], state_list[1]])

            if done: break

        """ for Neural Network """
        # batch_size = len(trajectories)
        # for idx in range(batch_size):
        #     batch_state = torch.tensor(trajectories[idx][0], device=device)
        #     wp_data = torch.tensor(trajectories[idx][1], device=device).float()
        #     wp_pred = agent(batch_state).float()
        #     # wp_pred = wp_pred.clone().detach().requires_grad_(True).type(torch.float64)
        #     # print(wp_data.data, wp_pred.data)

        #     loss = loss_func(wp_data, wp_pred)
        #     loss_mean += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        # loss_mean /= batch_size
        # print(f'\nepoch : {epoch+1}')
        # print(f'loss : {loss_mean}'); loss_list.append(loss_mean)
        # if (epoch+1) % 3 == 0:
        #     torch.save(agent, f"{model_args['agent_name']}-GRUNet.pt")
        #     print('Model saved.')
        # if len(loss_list) % 10 == 0:
        #     plt.figure()
        #     plt.plot(loss_list)
        #     plt.xlabel('Epochs'); plt.ylabel('Loss')
        #     plt.savefig(f"{model_args['agent_name']}-{model_args['algo_idx']}-GRUNet-loss.png")
        #     print('Loss figure saved.')

        """ for CPO """
        print(f'\nepoch : {epoch+1}')
        v_loss, cost_v_loss, objective, _, kl, _ = agent.train(trajs=trajectories)
        log_data = {'objective' : objective, 'value loss' : v_loss, 'cost value loss' : cost_v_loss, 'kl' : kl}
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
    agent_name = '0102'
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
        'hidden2':128,
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
        'pred_length':2
    }

    if main_args.imitation:
        imitaion_learning(main_args, model_args)
    else:
        train(main_args, model_args)
