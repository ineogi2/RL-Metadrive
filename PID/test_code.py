import sys, random
from PID_controller import PID_controller

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

env=SafeMetaDriveEnv(dict(use_render=True,
                    random_lane_width=True,
                    random_lane_num=True,
                    map=2,
                    traffic_density=0.25,
                    start_seed=random.randint(0, 1000)))
env.reset()

obs, reward, done, info = env.step([0,0])
controller = PID_controller(info)

while not done:
    for _ in range(500):
        # print(obs[19])
        if obs[19] < 0.4:
            if controller.aim_lane_num == controller.cur_lane_num:
                if info['vehicle_heading_sine'][1]>0: controller.go_left()
                else: controller.go_right()
    
        input = controller.vehicle_control()

        obs, reward, done, info = env.step(input)
        controller._update(info)
        env.render()

    controller._reset(); print('\n')
    env.reset(); done=False; obs, reward, done, info = env.step([0,0])
    controller._update(info)

    ## density 업해서 다시