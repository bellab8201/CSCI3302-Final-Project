"""grocery controller."""

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # for mapping, path planning
import heapq # for path planning

#Initialization
print("===> Initializing Boulder Dynamics' Grocery Shopper...")

#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

LID_COL_THRESH = 2
LID_COL_THRESH_FRONT = 1.65 # want to prioritize left/right sensors to go down aisles

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# To save and load map files
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

# Helper Functions --------------------------------------------------------------------

def sleep(duration):
    # provided function that waits for DURATION in seconds before returning
    global robot
    end_time = robot.getTime() + duration
    while robot.step(timestep) != -1 and  robot.getTime() < end_time:
        pass
    
def rand_explore_front():
    # back up, spin randomly and try again
    # not frequently called, but helps with corners and
    # moving towards middle of map
    
    # back up 1 sec
    robot_parts["wheel_left_joint"].setVelocity(-MAX_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(-MAX_SPEED)
    sleep(1)
    
    # randomly rotate left or right
    if random.randint(0, 1):
        robot_parts["wheel_left_joint"].setVelocity(0.7*MAX_SPEED)
        robot_parts["wheel_right_joint"].setVelocity(-0.7*MAX_SPEED)
    else:
        robot_parts["wheel_left_joint"].setVelocity(-0.7*MAX_SPEED)
        robot_parts["wheel_right_joint"].setVelocity(0.7*MAX_SPEED)
    sleep(random.random()/1.5) # spin between 0-0.5 sec
    
def rand_explore_right():
    # veer left briefly
    robot_parts["wheel_left_joint"].setVelocity(.5*MAX_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(MAX_SPEED)
    sleep(.08)
    
def rand_explore_left():
    # veer right briefly
    robot_parts["wheel_left_joint"].setVelocity(MAX_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(.5*MAX_SPEED)
    sleep(.08)
    
def cmp(a, b): # not sure why this function was removed from Python 3
    return (a > b) - (a < b)
    
# State/Global Vars --------------------------------------------------------------------

gripper_status="closed"
map = np.zeros(shape=[360,192]) # room is 30x16 -> 30*12=360, 16*12=192

# SELECT MODE HERE:
# mode = 'mapping'
mode = 'planner'
# mode = 'autonomous'

# begin planning block ---------------------------------------------------------------------------
if mode == 'planner':
    # start/end in world coordinate frame
    start_w = (-4.999,-0.088) # (Pose_X, Pose_Y) in meters
    end_w = (8.78, -5.76) # (Pose_X, Pose_Y) in meters
    # start/end in map coordinate frame
    start = (180-int(start_w[0]*12), 96-int(start_w[1]*12)) # (x, y) in 360x192 map
    end = (180-int(end_w[0]*12), 96-int(end_w[1]*12)) # (x, y) in 360x192 map
    
    # This is an A* algorithm taken from Wikipedia, although it does not consider diagonal indicies
    def heuristic(a, b):
        '''
        :param a: A tuple of indices representing a cell in the map
        :param b: A tuple of indices representing a cell in the map
        :return: The Manhattan distance between a and b
        '''
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        
        dir = [(0,1),(0,-1),(1,0),(-1,0)]
        start_sq = (0, start)
        heap = [start_sq]
        cost = {start:0}
        came_from = {}
        
        while heap:
            curr_cost, current = heapq.heappop(heap)

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
                
            for d in dir:
                neighbor = (current[0] + d[0], current[1] + d[1])
                if not (0 <= neighbor[0] < map.shape[0] and 0 <= neighbor[1] < map.shape[1]) or map[neighbor] == 1:
                    continue
                new_cost = cost[current] + 1
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(heap, (priority, neighbor))
                    came_from[neighbor] = current
                    
        return None

    # Load map from disk and visualize it
    map = np.load("map.npy")
    plt.imshow(map)
    plt.show()
    
    # Compute an approximation of the “configuration space”
    test = np.ones((12, 12))
    c_space = convolve2d(map, test, mode="same")
    c_space[c_space > 1] = 1
    plt.imshow(c_space)
    plt.show()
    
    # Call path_planner
    path = path_planner(c_space, start, end)
    
    # Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []
    for point in path:
        x = -(point[0]-180)/12
        y = -(point[1]-96)/12
        waypoints.append((x,y))
    # print(waypoints)
    np.save('path.npy', waypoints)
    
    # Save map/c-space with path overlay
    map_with_path = np.zeros(shape=[360,192])
    c_space_with_path = np.zeros(shape=[360,192])
    for i in range(len(waypoints)):
        c_space_with_path[180-int(waypoints[i][0]*12)][96-int(waypoints[i][1]*12)] = 1
        map_with_path[180-int(waypoints[i][0]*12)][96-int(waypoints[i][1]*12)] = 1
    c_space_with_path = c_space_with_path + c_space
    map_with_path = map_with_path + map
    
    # Visualize overlay
    np.save('c_space_with_path.npy', c_space_with_path)
    np.save('map_with_path.npy', map_with_path)
    plt.imshow(c_space_with_path)
    plt.show()
    plt.imshow(map_with_path)
    plt.show()
    print("===> Path Planning Stage Exited.")
# end planning block ---------------------------------------------------------------------------

# Main Loop --------------------------------------------------------------------------------
while robot.step(timestep) != -1:
    
    # prevents controller from crashing in 'planner' mode
    if mode == 'planner':
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)
        continue
    
    # obtain robot pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[1], n[0]))-1.5708)
    pose_theta = rad
    
    # begin mapping block ---------------------------------------------------------------------------
    if mode == 'mapping':
        
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
        for i, rho in enumerate(lidar_sensor_readings):
            alpha = lidar_offsets[i]
            if rho > LIDAR_SENSOR_MAX_RANGE:
                continue
    
            # The Webots coordinate system doesn't match the robot-centric axes we're used to
            rx = math.cos(alpha)*rho
            ry = -math.sin(alpha)*rho
            t = pose_theta
            # Convert detection from robot coordinates into world coordinates
            wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
            wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y
    
            if wx >= 15:
                wx = 14.999
            if wy >= 8:
                wy = 7.999
            if rho < LIDAR_SENSOR_MAX_RANGE:
                c_x = 180-int(wx * 12) # x and y coordinate values from previous draw pixel function
                c_y = 96-int(wy * 12)
                
                if c_x >= 0 and c_y >= 0 and c_x < 360 and c_y < 360:
                    map[c_x][c_y] += 0.005 #increase by a small amount when reading cell
                    if map[c_x][c_y] >= .25:
                       map[c_x][c_y] = 1.0
                    g = map[c_x][c_y]
                    #visualize map by converting val into hex format
                    color = int((g * 256**2 + g * 256 + g) * 255)
                    display.setColor(int(color))
                    # if g == 1.0: # only white obstacles
                        # display.drawPixel(c_y,c_x)
                    display.drawPixel(c_y,c_x) # show all non-zero lidar readings
    
        # Draw the robot's current pose on the 192x360 display
        display.setColor(int(0xFF0000))
        display.drawPixel(96-int(pose_y*12), 180-int(pose_x*12))
        
        # Collision detection/avoidance            
        if any(val < LID_COL_THRESH for val in lidar_sensor_readings[125:150]):
            rand_explore_left() # left sensors triggered
        if any(val < LID_COL_THRESH for val in lidar_sensor_readings[350:375]):
            rand_explore_right() # right sensors triggered
        if any(val < LID_COL_THRESH_FRONT for val in lidar_sensor_readings[245:255]):
            rand_explore_front() # front sensors triggered
        else:
            vL = MAX_SPEED
            vR = MAX_SPEED
                
        # Manually store map file
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == ord('S'):
            map = map > .5
            map = np.multiply(map,1)
            np.save('map.npy',map)
            print("Map file saved")
    # end mapping block ------------------------------------------------------------------------------------
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
