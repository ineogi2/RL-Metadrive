import math
import numpy as np

class PID_controller:
    def __init__(self, info, speed_gain=[1,0,0.01]):
        self.speed_gain=speed_gain
        self.speed_err=(0,0,0)
        self.speed_aim=20
        self.is_arrived=True

        self.speed, self.lateral_error, self._heading, self.lateral_error_degree=0,0,0,0
        self.vehicle_length=info["vehicle_length"]
        self.vehicle_pos=info["vehicle_position"]
        self.lateral, self.lateral_distance=[0,0],0
        self.waypoint=self.vehicle_pos
        self.distance=5

        self.update(info, waypoint=0)

# --------------------inner only--------------------

    def _update_err(self, past_val, cur_val, aim):
        p_err = aim-cur_val
        i_err = past_val[1]+p_err
        d_err = p_err - past_val[0]
        return (p_err, i_err, d_err)

    def _reset(self):
        self.speed_err=(0,0,0)
        self.is_arrived=True
        self.lateral, self.lateral_distance=[0,0],0

    def _pid_result(self, pid_gain, err):
        input = 0
        for i in range(3):
            input += pid_gain[i]*err[i]
        return input

    def _get_lateral_error(self):
        if self.lateral_distance == 0: return 0

        lateral_error = -self._heading[0]*self.lateral[1] + self._heading[1]*self.lateral[0]
        return lateral_error

    def _is_arrived(self):
        self.is_arrived = True if self.lateral_distance < 1 else False

    def _process_waypoint(self, waypoint):
        norm = self.norm(waypoint[0], waypoint[1])
        delta_x = self.clip(self.distance*waypoint[0]/norm, self.distance/5, self.distance)       # normal vector direction
        delta_y = (self.distance/5)*waypoint[1]/norm    # lateral direction

        x = (delta_x*self._heading[0], delta_x*self._heading[1])
        y = (-delta_y*self._heading[1], delta_y*self._heading[0])
        # print(self.vehicle_pos, x, y)
        return self.vehicle_pos+x+y

    # math tools
    def norm(self, x,y):
        return math.sqrt(x**2+y**2)

    def clip(self, a, low, high):
        return min(max(a, low), high)

# --------------------inner only--------------------

    def update(self, info, waypoint):
        self.speed = info['vehicle_speed']
        self._heading = info["vehicle_heading"]
        self.vehicle_pos=info["vehicle_position"]

        if isinstance(waypoint, int): pass
        else:
            self.waypoint = self._process_waypoint(waypoint)

        self.lateral = self.waypoint - self.vehicle_pos
        self.lateral_distance = self.norm(self.lateral[0], self.lateral[1])
        self._is_arrived()

        self.lateral_error = self._get_lateral_error()
        self.speed_err = self._update_err(self.speed_err, self.speed, self.speed_aim)

    def lane_keeping(self):
        steering_angle = 0.7*self.lateral_error
        acc = self._pid_result(self.speed_gain, self.speed_err)
        input = [np.clip(steering_angle,-0.6,0.6), acc]
        return input