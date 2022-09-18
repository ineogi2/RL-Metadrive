import math
import numpy as np

class PID_controller:
    def __init__(self, info, speed_gain=[1,0,0.01], steering_gain=[0.7,0,0]):
        self.speed_gain=speed_gain
        self.steering_gain=steering_gain
        self.speed_err=(0,0,0)
        self.dir_err=(0,0,0)
        self.speed_aim=20
        self.lane_change_sign=0     # activate when policy decides to change land
        self.lane_change_count=0    # step for smoothing lane change action
        self.map_lane=0             # number of lanes on current map

        self.speed, self.heading, self.aim_lane_num = 0,0,None
        (self.lane_curved,self.cur_lane_num,self.lane_width,self.lane_to_left, self.lane_to_right)=[0,0,0,0,0]
        self.vehicle_length=info["vehicle_length"]
        self.vehicle_pos=info["vehicle_position"]

        self._update(info)

    def _update(self, info):
        self.speed = info['vehicle_speed']
        self.heading = math.degrees(math.asin(info['vehicle_heading']))
        (self.lane_curved,self.cur_lane_num,self.lane_width,self.lane_to_left,self.lane_to_right) = info['vehicle_heading_sine']

        self.map_lane = round((self.lane_to_left+self.lane_to_right)/self.lane_width)
        _cur_lateral = self.lane_to_left+self.vehicle_length*info['vehicle_heading']/2

        if self.aim_lane_num==None:
            self.aim_lane_num=self.cur_lane_num

        if self.lane_change_sign:
            if self.aim_lane_num == self.cur_lane_num:
                self.lane_change_sign=0

        _lateral_aim = (self.aim_lane_num+0.5)*self.lane_width
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

    def vehicle_control(self, aim_pos):
        self.check_pos(aim_pos)
        if num==2:
            if self.cur_lane_num == self.aim_lane_num:
                self.go_left()
            else:
                pass
        elif num==3:
            if self.cur_lane_num == self.aim_lane_num:
                self.go_right()
            else:
                pass

        if self.lane_change_sign:
            input = self.lane_change()
        else: 
            input = self.lane_keeping()

        if num == 1: input[1] = -0.5
        return input

    def lane_keeping(self):
        if abs(self.lane_change_count)>0:
            if self.dir_err[0]*self.dir>0:
                steering_angle = self.heading*0.1
                self.lane_change_count-=1
            else:
                steering_angle = -self.heading*0.001
        else:
            steering_angle = -self._pid_result(self.steering_gain, self.dir_err)
        
        acc = self._pid_result(self.speed_gain, self.speed_err)

        input = [np.clip(steering_angle,-0.8,0.8), acc]
        return input

    def lane_change(self):
        steering_angle = -self._pid_result(self.steering_gain, self.dir_err)
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