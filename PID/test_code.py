import sys, random, csv
import numpy as np
import pickle
# from PID_controller_v5 import Controller, State

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import MetaDriveEnv

# with open('waypoints.csv') as waypoints_file_handle:
#         waypoints = list(csv.reader(waypoints_file_handle,
#                                     delimiter=',',
#                                     quoting=csv.QUOTE_NONNUMERIC))
        # waypoints = np.array(waypoints)
# state = State(waypoints)
# controller = Controller()

epochs = 10
step_max = 10000

for epoch in range(epochs):

    map_num = random.randint(1, 10)
    start_seed_num = random.randint(1, 1000)

    env=MetaDriveEnv(dict(use_render=False,
                        manual_control=True,
                        # random_lane_width=True,
                        # random_lane_num=True,
                        map = map_num,
                        traffic_density=0.01,
                        start_seed=start_seed_num
                        ))

    # map_num = random.randint(1, 100)
    # start_seed_num = random.randint(1, 10)

    print(f"\nepoch : {epoch}")
    state = env.reset()
    env.vehicle.expert_takeover = True
    # controller.reset()
    state, reward, done, info = env.step([0,0])
    # step = 1
    acc_list = []
    steering_list = []
    state_list = []
    while not done:
        # step+=1
        # state.state_update(info)

        # print(str(state))
        # controller.update_all(state)
        # controller.update_controls()

        # steer = controller.steer
        # acc = (25-info["vehicle_speed"])*0.7
        # print(steer)
        # obs, reward, done, info = env.step([steer, acc])

        state, reward, done, info = env.step([0, 0])

        state_list.append(state)
        acc_list.append(env.vehicle.throttle_brake)
        steering_list.append(env.vehicle.steering)

        # if step % 5 == 1:
        #     pos = [info["vehicle_position"][0], -info["vehicle_position"][1]]
        #     pos_list.append(pos)

        # env.render()

        if done:
            break

    file_name = f"{map_num}_{start_seed_num}.pkl"

    file = {"state_list" : state_list, "acc_list" : acc_list, "steering_list" : steering_list}

    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

    env.close()
    # pos_list = np.array(pos_list)
    # np.savetxt('waypoints.csv', pos_list, fmt='%f', delimiter=',')

