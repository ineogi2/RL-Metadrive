import numpy as np
from collections import deque

class Controller():
    def __init__(self):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self.waypoints = np.array([])
        self.min_dist = 0
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

    def _update_waypoints(self, state):
        self.waypoints = state._waypoints
        self.min_dist = state._min_dist

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
        crosstrack_error = self.min_dist**0.5
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

class State():
    def __init__(self):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._waypoints = np.array([])
        self._heading = [0,0]
        self._min_idx = 0
        self._min_dist = 0

    def _update_waypoints(self, action):
        now = np.array([self._current_x, self._current_y])
        waypoints = [now]
        yaws = [self._current_yaw]
        dist = np.array(action[:5])
        degree = np.array(action[5:])

        for i in range(len(action)//2):
            waypoint, deg = return_pos(waypoints[-1], yaws[-1],dist[i], degree[i])
            waypoints.append(waypoint)
            yaws.append(deg)

        self._waypoints = np.array(waypoints[1:])
        self._min_dist = norm_sq(now, self._waypoints[0])

    def state_update(self, info, waypoints):
        xy = info['vehicle_position']

        self._heading = info['vehicle_heading']
        self._current_x = xy[0]
        self._current_y = -xy[1]
        self._current_speed = info['vehicle_speed']
        self._current_yaw = np.arctan2(-self._heading[1], self._heading[0])
        self._update_waypoints(waypoints)

    def __str__(self) -> str:
        return f"xy : {(self._current_x, self._current_y)} / speed : {self._current_speed} / " \
            + f"yaw : {self._current_yaw} / waypoint : {(self._waypoints[0][0], self._waypoints[0][1])}"



# class State():
#     def __init__(self, waypoints):
#         self._current_x = 0
#         self._current_y = 0
#         self._current_yaw = 0
#         self._current_speed = 0
#         self._waypoints = np.array([])
#         self._heading = [0,0]
#         self._all_waypoints = waypoints
#         self._min_idx = 0
#         self._min_dist = 0

#     def _update_waypoints(self):
#         now = np.array([self._current_x, self._current_y])
#         self._min_dist = norm_sq(now, self._all_waypoints[self._min_idx])

#         for idx in range(max(self._min_idx-10, 0), min(len(self._all_waypoints), self._min_idx+10)):
#             way = self._all_waypoints[idx]
#             dist = norm_sq(way, now)
#             if dist < self._min_dist:
#                 self._min_idx = idx
#                 self._min_dist = dist

#     def state_update(self, info):
#         xy = info['vehicle_position']

#         self._heading = info['vehicle_heading']
#         self._current_x = xy[0]
#         self._current_y = -xy[1]
#         self._current_speed = info['vehicle_speed']
#         self._current_yaw = np.arctan2(-self._heading[1], self._heading[0])
#         self._update_waypoints()
#         self._waypoints = np.array(self._all_waypoints[self._min_idx:min(self._min_idx+10, len(self._all_waypoints)), :])

#     def __str__(self) -> str:
#         return f"xy : {(self._current_x, self._current_y)} / speed : {self._current_speed} / " \
#             + f"yaw : {self._current_yaw} / waypoint : {(self._waypoints[0][0], self._waypoints[0][1])}"

def norm_sq(pt1, pt2):
    return (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2

def return_pos(current_pos, current_yaw, dist, degree, view_ahead = 2):
    deg = current_yaw+(degree-0.5)*np.pi
    if deg > np.pi:
        deg -= 2 * np.pi
    if deg < - np.pi:
        deg += 2 * np.pi
    
    dx = dist*np.cos(deg)*view_ahead
    dy = dist*np.sin(deg)*view_ahead
    return [current_pos[0]+dx, current_pos[1]+dy], 