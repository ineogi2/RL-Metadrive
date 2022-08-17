import math

class PID_controller:
    def __init__(self, speed_gain=[1,0,0.01], steering_gain=[1,0,0.3]):
        self.speed_gain=speed_gain
        self.steering_gain=steering_gain
        self.speed_err=(0,0,0)
        self.dir_err=(0,0,0)
        self.heading=0
        self.lane_change_sign=0

    def _update(self, speed_cur_err, dir_cur_err, heading):
        self.speed_err = (speed_cur_err, self.speed_err[1]+speed_cur_err, speed_cur_err-self.speed_err[0])
        self.dir_err = (dir_cur_err, self.dir_err[1]+dir_cur_err, dir_cur_err-self.dir_err[0])
        self.heading = math.degrees(math.asin(heading))

    def _reset(self):
        self.speed_err=(0,0,0)
        self.dir_err=(0,0,0)

    def _pid_result(self, pid_gain, err):
        input = 0
        for i in range(3):
            input += pid_gain[i]*err[i]
        return input

    def lane_keeping(self):
        steering_angle = self._pid_result(self.steering_gain, self.dir_err) if (abs(self.dir_err[0])>0.3) & (abs(self.heading)<10) else self.heading*0.1
        acc = self._pid_result(self.speed_gain, self.speed_err)
  
        return [steering_angle, acc]

    def lane_change(self, direction, cur_speed):
        steering_angle = direction*0.12
        acc = self._pid_result(self.speed_gain, self.speed_err)
        steps = int(300/cur_speed)

        return ([steering_angle, acc], steps)