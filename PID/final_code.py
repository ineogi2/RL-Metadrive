import sys, random
from PID_controller import PID_controller

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

env=SafeMetaDriveEnv(dict(use_render=True,
                    random_lane_width=True,
                    random_lane_num=True,
                    map=2,
                    start_seed=random.randint(0, 1000)))
env.reset()

obs, reward, done, info = env.step([0,0])
controller = PID_controller(info)

while True:
    for _ in range(40):
        input = controller.vehicle_control()

        obs, reward, done, info = env.step(input)
        controller._update(info)
        env.render()

    if info['vehicle_heading_sine'][1]>0: controller.go_left()
    else: controller.go_right()
    
    for _ in range(100):
        input = controller.vehicle_control()

        obs, reward, done, info = env.step(input)
        controller._update(info)
        env.render()

    controller._reset(); print('\n')
    env.reset(); done=False; obs, reward, done, info = env.step([0,0])
    controller._update(info)