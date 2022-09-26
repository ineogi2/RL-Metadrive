import time, sys
import matplotlib.pyplot as plt

sys.path.append("/home/ineogi2/RL-Lab/metadrive")

from metadrive import SafeMetaDriveEnv

Kp_list = [1, 2, 3, 4, 5]
Ki_list = [0.001, 0.01, 0.1, 1]
Kd_list = [0.001, 0.01, 0.1, 1]

def steering_control(cur_err, prv_err, err_sum, dt, Kp, Ki, Kd):
    # hyperparameter


    # constraint
    upper_acc = 0.3
    lower_acc = -0.3

    steering = Kp*cur_err + Ki*dt*err_sum + Kd*(cur_err-prv_err)/dt

    # if abs(cur_err) < 3:
    #     if acc < lower_acc: acc = lower_acc
        # elif acc > lower_acc: acc = upper_acc

    return steering

env=SafeMetaDriveEnv(dict(
    # environment_num=1,
    start_seed=0,
    use_render=True
))
env.reset()

steering_aim = 0.5
err_sum = 0
cur_err = 0
prv_err = 0
init_steering = 0
prv_time = time.time()

steering_list = []

obs, reward, done, info = env.step([0,0])

for Kp in Kp_list:
    for Ki in Ki_list:
        for Kd in Kd_list:
            while (not done) & (len(steering_list)<=100):
                cur_time = time.time()
                cur_steering = info['steering']
                steering_list.append(cur_steering)

                print("\ncur_steering : ",cur_steering)
                # print(info)

                dt = cur_time - prv_time
                cur_err = steering_aim - cur_steering
                err_sum += cur_err
                # print("cur_err : ",cur_err," / prv_err : ",prv_err," / err_sum : ",err_sum)

                steering = steering_control(cur_err, prv_err, err_sum, dt, Kp, Ki, Kd)
                # print("acc : ",acc)
                obs, reward, done, info = env.step([steering,0.1])

                prv_time = cur_time
                prv_err = cur_err
                env.render()

            plt.plot(steering_list[:100],'b') # x_range=plt.xlim()
            plt.hlines(steering_aim,0,99,colors='red')
            file_name = 'Kp='+str(Kp)+" & Ki="+str(Ki)+" & Kd="+str(Kd)+'.png'
            plt.savefig(file_name) # plt.show()
            plt.close()

            # input("Press Enter to continue...")
            steering_list = []; err_sum = 0; cur_err = 0; prv_err = 0; cur_steering = 0

            env.reset(); done=False; obs, reward, done, info = env.step([0,0])
        
# action : [steer, acc]

env.close()