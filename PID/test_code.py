import sys, random, csv
import numpy as np
from PID_controller_v5 import Controller, State

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import MetaDriveEnv

env=MetaDriveEnv(dict(use_render=True,
                # manual_control=True,
                # random_lane_width=True,
                # random_lane_num=True,
                map = 3,
                traffic_density=0.05,
                start_seed=5
                ))



with open('waypoints.csv') as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints = np.array(waypoints)
state = State(waypoints)
controller = Controller()

epochs = 1
step_max = 700

for epoch in range(epochs):
    print(f"\nepoch : {epoch}")
    env.reset()
    # env.vehicle.expert_takeover = True
    controller.reset()
    obs, reward, done, info = env.step([0,0])
    step = 1
    pos_list = []
    while step < step_max:
        step+=1
        state.state_update(info)

        print(str(state))
        controller.update_all(state)
        controller.update_controls()

        steer = controller.steer
        acc = (25-info["vehicle_speed"])*0.7
        print(steer)
        obs, reward, done, info = env.step([steer, acc])
        # obs, reward, done, info = env.step([0, 0])
        if step % 5 == 1:
            pos = [info["vehicle_position"][0], -info["vehicle_position"][1]]
            pos_list.append(pos)

        env.render()

        if done:
            break

    pos_list = np.array(pos_list)
    # np.savetxt('waypoints.csv', pos_list, fmt='%f', delimiter=',')

