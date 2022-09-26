import time, sys
import matplotlib.pyplot as plt

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

Kp_list = [1]
Ki_list = [0.002]
Kd_list = [0.001]

def speed_control(cur_err, prv_err, err_sum, dt, Kp, Ki, Kd):
    # hyperparameter


    # constraint
    upper_acc = 0.3
    lower_acc = -0.3

    acc = Kp*cur_err + Ki*dt*err_sum + Kd*(cur_err-prv_err)/dt

    # if abs(cur_err) < 3:
    #     if acc < lower_acc: acc = lower_acc
        # elif acc > lower_acc: acc = upper_acc

    return acc

env=SafeMetaDriveEnv(dict(
    # environment_num=1,
    start_seed=0,
    use_render=True
))
env.reset()

speed_aim = 40
err_sum = 0
cur_err = 0
prv_err = 0
init_speed = 0
prv_time = time.time()

speed_list = []

obs, reward, done, info = env.step([0,0])

for Kp in Kp_list:
    for Ki in Ki_list:
        for Kd in Kd_list:
            while (not done) & (len(speed_list)<=100):
                cur_time = time.time()
                cur_speed = info['vehicle_speed']
                speed_list.append(cur_speed)

                # print("\ncur_speed : ",cur_speed)
                print(info)

                dt = cur_time - prv_time
                cur_err = speed_aim - cur_speed
                err_sum += cur_err
                # print("cur_err : ",cur_err," / prv_err : ",prv_err," / err_sum : ",err_sum)

                acc = speed_control(cur_err, prv_err, err_sum, dt, Kp, Ki, Kd)
                # print("acc : ",acc)
                obs, reward, done, info = env.step([0,acc])

                prv_time = cur_time
                prv_err = cur_err
                env.render()

            plt.plot(speed_list[:100],'b') # x_range=plt.xlim()
            plt.hlines(speed_aim,0,99,colors='red')
            file_name = 'Kp='+str(Kp)+" & Ki="+str(Ki)+" & Kd="+str(Kd)+'.png'
            plt.savefig(file_name) # plt.show()
            plt.close()

            # input("Press Enter to continue...")
            speed_list = []; err_sum = 0; cur_err = 0; prv_err = 0; cur_speed = 0

            env.reset(); done=False; obs, reward, done, info = env.step([0,0])
        
# action : [steer, acc]

env.close()