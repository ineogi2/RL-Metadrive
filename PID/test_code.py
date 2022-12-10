import sys, random, csv
import numpy as np
import pickle
from collections import deque

from PID_controller_v6 import Controller, State

sys.path.append("/home/ineogi2/RL-Lab/metadrive")
from metadrive import MetaDriveEnv

epochs = 1
step_max = 10000

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

def position_to_absolute_wp(position_list):
    positions = np.array(position_list)
    wp_list = deque()

    for i in range(len(positions)-1):
        wp = positions[i+1] - positions[0]
        wp_list.append(wp[0])
        wp_list.append(-wp[1])

    return wp_list

for epoch in range(epochs):

    # map_num = random.randint(1, 10)
    # start_seed_num = random.randint(1, 1000)
    map_num, start_seed_num = 1,1

    env=MetaDriveEnv(dict(use_render=True,
                        manual_control=False,
                        # random_lane_width=True,
                        # random_lane_num=True,
                        map = map_num,
                        traffic_density=0.01,
                        start_seed=start_seed_num
                        ))
    state_converter = State()
    controller = Controller()   

    with open('1_1.pkl', 'rb') as f:
        waypoints_list = pickle.load(f)
    # print(waypoints_list)
    print(f"\nepoch : {epoch}")
    
    state = env.reset()
    # env.vehicle.expert_takeover = True
    # controller.reset()
    state, reward, done, info = env.step([0,0])
    # step = 1
    acc_list = []
    steering_list = []
    state_list = []
    pos_list = []

    position_list = deque(maxlen=6)
    direction_list = deque(maxlen=6)

    while not done:

        waypoints = waypoints_list.pop(0)
        print(waypoints)

        state_converter.state_update(info, waypoints)
        print(str(state_converter))
        controller.update_all(state_converter)
        controller.update_controls()
        steer = controller.steer
        acc = controller.acc

        state, reward, done, info = env.step([steer, acc])
        # state, reward, done, info = env.step([0, 0])

        # position = [info['vehicle_position'][0], -info['vehicle_position'][1]]
        # direction = [info['vehicle_heading'][0], -info['vehicle_heading'][1]]

        position_list.append(info['vehicle_position'])
        # direction_list.append(direction)

        # state_list.append(state)
        # acc_list.append(env.vehicle.throttle_brake)
        # steering_list.append(env.vehicle.steering)

        # if len(position_list)==6:
        #     # waypoints = position_to_relative_wp(position_list, direction_list[0], 5)
        #     waypoints = position_to_absolute_wp(position_list)
        #     pos_list.append(waypoints)

        if done:
            break

    file_name = f"{map_num}_{start_seed_num}.pkl"

    # file = {"state_list" : state_list, "acc_list" : acc_list, "steering_list" : steering_list}
    file = pos_list

    # with open(file_name, 'wb') as f:
    #     pickle.dump(file, f)

    env.close()


