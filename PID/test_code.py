import sys, random, csv
import numpy as np
import pickle
from collections import deque

sys.path.append("/home/ineogi2/RL-Lab/PID/past")
from PID_controller_v6 import Controller, State

sys.path.append("/home/ineogi2/RL-Lab/metadrive")
from metadrive import MetaDriveEnv

epochs = 1
step_max = 10000

def l1_distance(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def yaw(pt1, pt2):
    return np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])

def norm(pt):
    return (pt[0]**2+pt[1]**2)**0.5

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


for epoch in range(epochs):

    # map_num = random.randint(1, 10)
    # start_seed_num = random.randint(1, 1000)
    map_num, start_seed_num = 1,1

    env=MetaDriveEnv(dict(use_render=True,
                        manual_control=True,
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
    controller.reset()
    state, reward, done, info = env.step([0,0])
    acc_list = []
    steering_list = []
    state_list = []
    pos_list = []

    position_list = deque(maxlen=6)
    direction_list = deque(maxlen=6)

    while not done:

        waypoints = waypoints_list.pop(0)
        waypoints = modify_waypoint(waypoints, env.vehicle.lane, info)
        # print(waypoints)

        state_converter.state_update(info, waypoints)
        # print(str(state_converter))
        controller.update_all(state_converter)
        controller.update_controls()
        steer = controller.steer
        acc = controller.acc

        # state, reward, done, info = env.step([steer, acc])
        # print(state.shape)
        # print(state[-3:])
        state, reward, done, info = env.step([0, 0])
        print(env.vehicle.throttle_brake)
        # print(info['vehicle_heading_sine'][0]/l1_dist(info['vehicle_heading_sine'][0]), info['vehicle_heading'])

        # position = [info['vehicle_position'][0], -info['vehicle_position'][1]]
        # direction = [info['vehicle_heading'][0], -info['vehicle_heading'][1]]

        # position_list.append(info['vehicle_position'])
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


