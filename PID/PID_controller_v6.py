# for absolute waypoint

import numpy as np

def norm_sq(pt1, pt2):
    return (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2

def return_pos(current_pos, dx, dy, view_ahead = 1):
    
    dx_wp = dx*view_ahead
    dy_wp = dy*view_ahead

    return [current_pos[0]+dx_wp, current_pos[1]+dy_wp]


class Controller():
    def __init__(self):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._aim_speed = 0
        self.waypoints = np.array([])
        self.min_dist_sq = 0
        self.steer = 0
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi

    # ==================================
    # UPDATE all values
    # ==================================
    def _update_values(self, state):
        self._current_x = state._current_x
        self._current_y = state._current_y
        self._current_yaw = state._current_yaw
        self._current_speed = state._current_speed
        self._aim_speed = min(20*state._max_dist_sq, 20)
        # self._aim_speed = 20

    def _update_waypoints(self, state):
        self.waypoints = state._waypoints
        self.min_dist_sq = state._min_dist_sq

    def update_all(self, state):
        self._update_values(state)
        self._update_waypoints(state)

    def reset(self):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self.waypoints = np.array([])
        self.steer = 0
        self.acc = 0


    # ==================================
    # LATERAL CONTROLLER, using stanley steering controller for lateral control.
    # ==================================
    def update_controls(self):
        # update status
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        waypoints = self.waypoints

        k_e = 0.3
        k_v = 15

        # 1. calculate heading error
        if len(self.waypoints) >= 2:
            yaw_path = np.arctan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
        else:
            yaw_path = np.arctan2(waypoints[-1][1]-y, waypoints[-1][0]-x)

        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi

        # 2. calculate crosstrack error
        crosstrack_error = self.min_dist_sq**0.5
        waypoint = self.waypoints[0]

        yaw_cross_track = np.arctan2(y-waypoint[1], x-waypoint[0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + np.log(v+1)))

        # 3. control low
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < - np.pi:
            steer_expect += 2 * np.pi
        steer_expect = np.clip(steer_expect, -1.22, 1.22)

        # 4. update
        steer_output = steer_expect     # radian

        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * steer_output
        input_steer = np.sin(input_steer)

        # Clamp the steering command to valid bounds
        self.steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        
        # Acceleration
        self.acc = 0.7*(self._aim_speed - self._current_speed)
        self.acc = np.fmax(np.fmin(self.acc, 1.0), 0.0)

class State():
    def __init__(self):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._waypoints = np.array([])
        self._direction = [0,0]
        self._min_dist_sq = 0
        self._max_dist_sq = 0

    def _update_waypoints(self, rel_waypoints):
        now = np.array([self._current_x, self._current_y])
        waypoints = [now]

        for i in range(len(rel_waypoints)//2):
            waypoint = return_pos(now, rel_waypoints[2*i], rel_waypoints[2*i+1])
            waypoints.append(waypoint)

        # print(waypoints)
        self._waypoints = waypoints[1:]
        self._min_dist_sq = norm_sq(now, self._waypoints[0])
        self._max_dist_sq = norm_sq(now, self._waypoints[-1])

    def state_update(self, info, waypoints):
        xy = info['vehicle_position']

        self._direction = [info['vehicle_heading'][0], -info['vehicle_heading'][1]]
        self._y_direction = [-self._direction[1], self._direction[0]]
        self._current_x = xy[0]
        self._current_y = -xy[1]
        self._current_speed = info['vehicle_speed']
        self._current_yaw = np.arctan2(self._direction[1], self._direction[0])
        self._update_waypoints(waypoints)

    def __str__(self) -> str:
        return f"xy : {(self._current_x, self._current_y)} / cur_speed : {self._current_speed} / " \
            + f"aim_speed : {(self._max_dist_sq)*20}"
