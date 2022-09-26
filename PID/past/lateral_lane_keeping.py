import sys
from PID_controller import PID_controller

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

straight_line_speed = 25
controller = PID_controller()

env=SafeMetaDriveEnv(dict(start_seed=0, use_render=True)); env.reset()

obs, reward, done, info = env.step([0,0])
cur_lane = 0; prv_lane = 1

while True:
    while (not done):
        # print(info['vehicle_heading_sine'])
        
        cur_speed, heading, (cur_lane, cur_lateral) = info['vehicle_speed'], info['vehicle_heading'], info['vehicle_heading_sine']
        if prv_lane != cur_lane: controller._reset()

        speed_aim = straight_line_speed if (cur_lane==1) or (abs(cur_lateral)>0.015) else straight_line_speed
        speed_cur_err = speed_aim-cur_speed; dir_cur_err = cur_lateral

        controller._update(speed_cur_err, dir_cur_err, heading)
        input = controller.lane_keeping()

        obs, reward, done, info = env.step(input)
        env.render()
        prv_lane = cur_lane

    controller._reset()
    env.reset(); done=False; obs, reward, done, info = env.step([0,0])

env.close()