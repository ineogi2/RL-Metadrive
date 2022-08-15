import time, sys, math
import matplotlib.pyplot as plt

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

def speed_control(speed_err, dt, speed_gain=[1, 0, 0.001]):

    [Kp, Ki, Kd] = speed_gain
    [cur_err, prv_err, err_sum] = speed_err
    acc = Kp*cur_err + Ki*dt*err_sum + Kd*(cur_err-prv_err)/dt

    return acc

def dir_control(dir_err, dt, dir_gain=[1, 0.001, 0.1]):

    [Kp, Ki, Kd] = dir_gain
    [cur_err, prv_err, err_sum] = dir_err
    steering_angle = Kp*cur_err + Ki*dt*err_sum + Kd*(cur_err-prv_err)/dt

    return steering_angle

straight_line_speed = 25
speed_err_sum = 0
speed_cur_err = 0
speed_prv_err = 0
init_speed = 0

dir_err_sum = 0
dir_cur_err = 0
dir_prv_err = 0

Kp_list = [1, 10, 100, 0.1]
Ki_list = [0, 0.1, 0.01, 1, 10]
Kd_list = [0, 0.1, 0.01, 1, 10]

env=SafeMetaDriveEnv(dict(start_seed=0, use_render=True)); env.reset()

prv_time = time.time()
dir_err_list = []

obs, reward, done, info = env.step([0,0])

for Kp in Kp_list:
    for Ki in Ki_list:
        while True:
            while (not done) & (len(dir_err_list)<=500):
                cur_time = time.time()
                # print(info['vehicle_heading_sine'])
                cur_speed, (cur_lane, cur_lateral) = info['vehicle_speed'], info['vehicle_heading_sine']

                speed_aim = straight_line_speed - 5 if (cur_lane==1) or (abs(cur_lateral)>0.015) else straight_line_speed

                dt = cur_time - prv_time

                speed_cur_err = speed_aim - cur_speed; speed_err_sum += speed_cur_err
                speed_err = (speed_cur_err, speed_prv_err, speed_err_sum)

                dir_cur_err = cur_lateral; dir_err_sum += dir_cur_err
                dir_err = (dir_cur_err, dir_prv_err, dir_err_sum)
                
                print(speed_cur_err, dir_cur_err)
                # dir_err_list.append(dir_cur_err)

                acc = speed_control(speed_err, dt)
                steering_angle = dir_control(dir_err, dt)

                input = [steering_angle, acc]#; print(input)
                obs, reward, done, info = env.step(input)

                prv_time = cur_time
                speed_prv_err = speed_cur_err; dir_prv_err = dir_cur_err

                env.render()

            # plt.plot(dir_err_list[:200],'b') # x_range=plt.xlim()
            # plt.hlines(0,0,len(dir_err_list),colors='red')
            # file_name = 'Kp='+str(Kp)+" & Ki="+str(Ki)+" & Kd="+str(Kd)+'.png'
            # plt.savefig(file_name) # plt.show()
            # plt.close()

            # input("Press Enter to continue...")
            dir_err_list = []
            speed_err_sum = 0; speed_cur_err = 0; speed_prv_err = 0; cur_speed = 0
            dir_err_sum = 0; dir_cur_err = 0; dir_prv_err = 0

            env.reset(); done=False; obs, reward, done, info = env.step([0,0])

env.close()