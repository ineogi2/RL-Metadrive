import math
import numpy as np

class PID_controller:
    def __init__(self, info, speed_gain=[1,0,0.01], steering_gain=[0.7,0,0]):
        self.speed_gain=speed_gain
        self.steering_gain=steering_gain
        self.speed_err=(0,0,0)
        self.dir_err=(0,0,0)
        self.speed_aim=25
        self.lane_change_sign=0
        self.lane_change_count=0

        self.speed, self.heading, self.aim_lane_num = 0,0,None
        (self.lane_curved,self.cur_lane_num,self.lane_width,self.lane_to_left)=(0,0,0,0)
        self.vehicle_length=info["vehicle_length"]

        self._update(info)
        # self.lane_num=self.lane_to_left//self.lane_width+1
        # self.heading=math.degrees(math.asin(info['vehicle_heading']))
        # self.speed=info['vehicle_speed']
        # self.speed_aim=25
        # (self.lane_curved,self.lane_width,self.lane_to_left)=info['vehicle_heading_sine']
        # self.lane_num=self.lane_to_left//self.lane_width+1 # (0 for straight 1 for curve, lane width, distance to left side)

    def _update(self, info):
        self.speed = info['vehicle_speed']
        self.heading = math.degrees(math.asin(info['vehicle_heading']))
        (self.lane_curved,self.cur_lane_num,self.lane_width,self.lane_to_left) = info['vehicle_heading_sine']
        _cur_lateral = self.lane_to_left+self.vehicle_length*info['vehicle_heading']/2

        if self.aim_lane_num==None:
            self.aim_lane_num=self.cur_lane_num

        if self.lane_change_sign:
            if self.aim_lane_num == self.cur_lane_num:
                # print(_cur_lateral,self.lane_width)
                self.lane_change_sign=0

        _lateral_aim = (self.aim_lane_num+0.5)*self.lane_width
        # print(_lateral_aim, _cur_lateral)
        self.speed_err = self._update_err(self.speed_err, self.speed, self.speed_aim)
        self.dir_err = self._update_err(self.dir_err, _cur_lateral, _lateral_aim)
        
    def _update_err(self, past_val, cur_val, aim):
        p_err = aim-cur_val
        i_err = past_val[1]+p_err
        d_err = p_err - past_val[0]
        return (p_err, i_err, d_err)

    def _reset(self):
        self.speed_err=(0,0,0)
        self.dir_err=(0,0,0)
        self.cur_lane_num,self.aim_lane_num=None,None

    def _pid_result(self, pid_gain, err):
        input = 0
        for i in range(3):
            input += pid_gain[i]*err[i]
        return input

    def vehicle_control(self):
        if self.lane_change_sign:
            input = self.lane_change()
        else: 
            input = self.lane_keeping()
        # print(input[0])
        return input

    def lane_keeping(self):
        print('\nlane keeping')
        if abs(self.lane_change_count)>0:
            if self.dir_err[0]*self.dir>0:
                print('1')
                steering_angle = self.heading*0.1
                self.lane_change_count-=1
            else:
                print('2')
                steering_angle = -self.heading*0.001
        else:
            steering_angle = -self._pid_result(self.steering_gain, self.dir_err)
        
        acc = self._pid_result(self.speed_gain, self.speed_err)

        input = [np.clip(steering_angle,-0.8,0.8), acc]
        return input

    def lane_change(self):
        print('\nlane changing')
        steering_angle = -self._pid_result(self.steering_gain, self.dir_err)
        # steering_angle = np.clip(steering_angle, self.heading*0.1-0.3,self.heading*0.1+0.3)
        if not self.lane_curved:
            steering_angle=np.clip(steering_angle,-0.09,0.09)
        else:
            steering_angle=np.clip(steering_angle,-0.4,0.4)
        acc = self._pid_result(self.speed_gain, self.speed_err)

        input = [steering_angle, acc]
        return input

    def go_left(self):
        self.lane_change_sign=-1
        self.aim_lane_num-=1
        self.lane_change_count=4
        self.dir=1

    def go_right(self):
        self.lane_change_sign=1
        self.aim_lane_num+=1
        self.lane_change_count=4
        self.dir=-1