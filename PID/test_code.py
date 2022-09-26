import sys, random
from PID_controller_v4 import PID_controller

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

env=SafeMetaDriveEnv(dict(use_render=True,
                    random_lane_width=True,
                    random_lane_num=True,
                    map=2,
                    traffic_density=0.25,
                    start_seed=random.randint(0, 1000)))
env.reset()
waypoint = (0,0)
obs, reward, done, info = env.step([0,0])
controller = PID_controller(info)

while not done:
    for _ in range(500):
    
        input = controller.lane_keeping()

        obs, reward, done, info = env.step(input)
        controller.update(info,[0,1])
        env.render()

    controller._reset(); print('\n')
    env.reset(); done=False; obs, reward, done, info = env.step([0,0])
    controller.update(info,[0,1])

    ## density 업해서 다시