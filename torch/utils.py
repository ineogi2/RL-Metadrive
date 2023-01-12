from collections import deque
import numpy as np

############ math tools ###########

def l1_distance(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def yaw(pt1, pt2):
    return np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])

def norm(pt):
    return (pt[0]**2+pt[1]**2)**0.5

# relative coordinate
def position_to_relative_wp(position_list, direction, pred_length):
    wp_list = deque()

    positions = np.array(position_list)
    positions[:,1] = -positions[:,1]                        # y coordinate reset
    positions = positions - positions[0]                    # relative position
    x_direction = np.array([direction[0], -direction[1]])    # y coordinate reset
    y_direction = np.array([direction[1], direction[0]])

    for i in range(1, pred_length+1):
        dx = np.dot(x_direction, positions[i])
        dy = np.dot(y_direction, positions[i])
        wp_list.append(dx)
        wp_list.append(dy)

    return wp_list

# absolute coordinate
def position_to_absolute_wp(position_list):
    positions = np.array(position_list)
    wp_list = deque()

    for i in range(len(positions)-1):
        wp = positions[i+1] - positions[0]
        wp_list.append(wp[0])
        wp_list.append(-wp[1])

    return wp_list

# modify waypoint to lane midpoint
def modify_waypoint(wp_list, cur_lane, info):
    cur_position = np.array(info['vehicle_position']); cur_position[1] *= -1
    lateral, _, lane_width, _, _ = info['vehicle_heading_sine']
    lateral = lateral/norm(lateral)
    lane_heading = np.array([lateral[1], -lateral[0]])
    pred_length = len(wp_list)//2
    
    modified_wp = []
    for i in range(pred_length):
        dx, dy = wp_list[2*i], wp_list[2*i+1]
        wp = cur_position+np.array([dx, dy])                    # real position of waypoint
        wp_to_vehicle_dist = l1_distance(wp, cur_position)
        wp_to_lane_dist = cur_lane.distance([wp[0], -wp[1]])    # distance from waypoint to current lane
        wp_to_lane_sign = np.dot(lateral, [dx, dy])     # positive : left wp / negative : right wp

        if wp_to_lane_dist <= lane_width/2:
            new_dx_dy = lane_heading*wp_to_vehicle_dist
        else:
            if wp_to_lane_sign >= 0:
                new_dx_dy = lateral*lane_width+lane_heading*wp_to_vehicle_dist
            else:
                new_dx_dy = lateral*(-lane_width)+lane_heading*wp_to_vehicle_dist
        # print(new_dx_dy)
        modified_wp.append(new_dx_dy[0])
        modified_wp.append(new_dx_dy[1])
    
    return modified_wp


###################################

