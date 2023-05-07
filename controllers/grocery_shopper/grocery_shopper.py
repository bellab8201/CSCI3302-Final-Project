"""grocery_shopper.py"""

from controller import Robot, Motor, Camera, CameraRecognitionObject, RangeFinder, Lidar, Keyboard
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # for mapping, path planning
import heapq # for path planning
import cv2 # for computer vision

# Initialization
print("===> Initializing Boulder Dynamics' Grocery Shopper...")

# Provided Constants
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# Defined Constants
LID_COL_THRESH = 2
LID_COL_THRESH_FRONT = 1.65 # want to prioritize left/right sensors to go down aisles

CAM_OFFSET_Y = 0.23 # cam y-coord relative to robot
CAM_OFFSET_Z = 1.05 # cam z-coord relative to robot
# valid y-coords for path planning
AISLE_ONE = 5.7
AISLE_TWO = 2.0
AISLE_THREE = -2.0
AISLE_FOUR = -5.7

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
mode = None

# Helper Functions --------------------------------------------------------------------

def sleep(duration):
    # provided function that waits for DURATION in seconds before returning
    global robot
    end_time = robot.getTime() + duration
    while robot.step(timestep) != -1 and  robot.getTime() < end_time:
        pass
        
def stop_and_wait(duration):
    # cuts power to wheels for a set duration using sleep()
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    sleep(duration)
    
def start_routine():
    robot_parts["wheel_left_joint"].setVelocity(-MAX_SPEED/3)
    robot_parts["wheel_right_joint"].setVelocity(-MAX_SPEED/3)
    sleep(2)
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    sleep(2)
    
def rand_explore():
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
    sleep(random.random()/1.5) # spin between 0-0.666 sec
    
def avoid_right():
    # veer left briefly
    robot_parts["wheel_left_joint"].setVelocity(.5*MAX_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(MAX_SPEED)
    sleep(.08)
    
def avoid_left():
    # veer right briefly
    robot_parts["wheel_left_joint"].setVelocity(MAX_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(.5*MAX_SPEED)
    sleep(.08)
    
def cmp(a, b): # not sure why this function was removed from Python 3
    return (a > b) - (a < b)
    
def sort_x(points):
    # sorts a list of triples based on the x-value, or first index
    points.sort(key = lambda x: x[0]) # from GeeksForGeeks, better than bubble sort
    return points
                
# State/Global Vars --------------------------------------------------------------------

gripper_status="closed"
map = np.zeros(shape=[360,192]) # room is 30x16 -> 30*12=360, 16*12=192

# for storing cube positions
cube_ids = []
cube_positions = []
cube_pos_valid = []

# ugly, but this helps organize a more efficient path
aisle_one_items = []
aisle_two_items = []
aisle_three_items = []
aisle_four_items = []

# final path used in planning stage
complete_path = []

# ------------------------ SELECT CONTROLLER MODE HERE! ---------------------------

mode = 'mapping'
# mode = 'planner'
# mode = 'shopping'

# begin planning block ---------------------------------------------------------------------------
if mode == 'planner':
    # start/end points in world coordinate frame
    cube_locations_w = np.load("cube_pos_valid.npy") # a list of "end points" to retrieve cubes from 
    start_w = [[-5.0, 0.0]] # (Pose_X, Pose_Y) in meters 
    for point in cube_locations_w:
        start_w.append(point)
    start_w.pop()
       
    # start/end points in map coordinate frame
    start = []
    cube_locations = []
    
    for point in start_w:
        start.append((180-int(point[0]*12), 96-int(point[1]*12))) # (x, y) in 360x192 map
    for point in cube_locations_w:
        cube_locations.append((180-int(point[0]*12), 96-int(point[1]*12))) # (x, y) in 360x192 map
        
    for i in range(len(cube_locations)): # path printed in green
        display.setColor(int(0x00FF00))
        display.drawPixel(cube_locations[i][1], cube_locations[i][0])
        
    # This is an A* algorithm adapted from Wikipedia using Euclidian instead of Manhattan Distance
    def heuristic(a, b):
        '''
        :param a: A tuple of indices representing a cell in the map
        :param b: A tuple of indices representing a cell in the map
        :return: The Euclidian distance between a and b
        '''
        return np.linalg.norm(np.array(a)-np.array(b))

    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        
        dir = [(0,1),(0,-1),(1,0),(-1,0)] # direction matrix
        start_sq = (0, start) # start node
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
    # plt.imshow(map)
    # plt.show()
    
    # Compute an approximation of the “configuration space”
    test = np.ones((27, 27)) # upping the size of this creates more padding, useful for Tiago
    c_space = convolve2d(map, test, mode="same")
    c_space[c_space > 1] = 1
    # plt.imshow(c_space)
    # plt.show()
    
    # Call path_planner
    # path = path_planner(c_space, start, end)
    for i in range(len(cube_locations)):
        complete_path.extend(path_planner(c_space, start[i], cube_locations[i])) # concatenate paths
    
    # Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []
    # for point in path:
    for point in complete_path:
        x = -(point[0]-180)/12
        y = -(point[1]-96)/12
        waypoints.append((x,y))
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
    print("===> Path Planning Phase Exited.")   
# end planning block ---------------------------------------------------------------------------

# begin pre-shopping block ---------------------------------------------------------------------------
if mode == 'shopping':
    # waypoints for IK, like labs 3 and 5
    waypoints = np.load("path.npy")
    
    # stop to grab each block (provides comparison to waypoints[current_goal])
    shopping_points_tuple = []
    shopping_points_triple = np.load("cube_pos_valid.npy")
    for point in shopping_points_triple:
        shopping_points_tuple.append([point[0], point[1]])
    
    # state init for shopping module (to index into waypoint array)
    current_goal = 0
    next_goal = current_goal + 1 # for eta
    
    starting = True # perform starting adjustments
    completed = False # flag to check sim completion
    
    WAYPOINTS_SIZE = len(waypoints)
# end pre-shopping block ---------------------------------------------------------------------------

# Main Loop ---------------------------------------------------------------------------------------
while robot.step(timestep) != -1:

    # prevents controller from crashing in 'planner' mode
    # or when mode is unset
    if mode == 'planner' or mode == None:
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)
        continue
    
    # obtain robot pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[1], n[0]))-1.5708)
    pose_theta = rad
    
    # begin shopping block ---------------------------------------------------------------------------
    if mode == 'shopping':
        if starting:
            start_routine()
            starting = False
        if completed: # has finished shopping -> keep updating time but robot will be stopped forever
            continue
        
        #Calculate error with respect to current and goal position
        rho = math.sqrt(((pose_x - waypoints[current_goal][0])**2) + ((pose_y - waypoints[current_goal][1])**2))
        alpha = math.atan2(waypoints[current_goal][1] - pose_y, waypoints[current_goal][0] - pose_x) - pose_theta
        eta = math.atan2(waypoints[next_goal][1] - pose_y, waypoints[next_goal][0] - pose_x) - pose_theta # waypoint heading should have a heading that points to next waypoint
        
        if alpha < -3.1415: alpha += 6.283 #error containing (angular error is circular)
        if eta < -3.1415: eta += 6.283
        
        #Feedback Controller (coefficients from trial/error)
        dX = rho
        dTheta = 5*alpha
        
        #Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
        #Note that vL and vR in code is phi_l and phi_r on the slides/lecture
        vL = dX - (dTheta/2) # Left wheel velocity in rad/s
        vR = dX + (dTheta/2) # Right wheel velocity in rad/s
        
        #Proportional velocities/Clamp wheel speeds
        if min(abs(vL),abs(vR)) == abs(vL):
            vL = cmp(vR,0)*(vL/vR)*MAX_SPEED # sign preservation with cmp(vR,0)
            vR = cmp(vR,0)*MAX_SPEED
        else:
            vR = cmp(vR,0)*(vR/vL)*MAX_SPEED
            vL = cmp(vR,0)*MAX_SPEED
        
        #Bound pose_theta between [-pi, 2pi+pi/2]
        #Important to not allow big fluctuations between timesteps (e.g., going from -pi to pi)
        if pose_theta > 6.28+3.14/2: pose_theta -= 6.28
        if pose_theta < -3.14: pose_theta += 6.28

        #Stopping (goal reached) criteria -> waypoint progression
        if rho <= 1 and eta <= 0.75: # was 0.05 and 0.524 in lab 3, but Tiago is clumsier and less precise
            # --------------------------
            # a routine that stops the robot to shop, including performing the arm manipulation
            # required to move a block from the shelf into the basket would go here. Ideally,
            # a comparison would be made to determine if our current waypoint matches one of the
            # valid block positions found in mapping.
            # --------------------------
            current_goal += 1
            next_goal += 1
            if next_goal >= WAYPOINTS_SIZE: # to not go out of bounds
                next_goal = current_goal
            if current_goal >= WAYPOINTS_SIZE: # reached last checkpoint
                completed = True
                robot_parts["wheel_left_joint"].setVelocity(0)
                robot_parts["wheel_right_joint"].setVelocity(0)
                print("===> Finished Shopping")
                print("===> Boulder Dynamics' Grocery Shopper Has Exited.")
                continue
    # end shopping block ---------------------------------------------------------------------------
    
    # begin mapping block ---------------------------------------------------------------------------
    if mode == 'mapping':
        
        objects = camera.getRecognitionObjects()
        for obj in objects: # look through all recognized objects
            if obj.getColors() == [1.0, 1.0, 0.0] or obj.getColors() == [0.0, 1.0, 0.0]: # only yellow/green cubes
                if not any(id == obj.getId() for id in cube_ids): # haven't stored cube before
                    cube_ids.append(obj.getId()) # store ID
                    
                    # convert from camera frame to world frame (same math as LIDAR)
                    obj_x = math.cos(pose_theta)*obj.getPosition()[0] - math.sin(pose_theta)*obj.getPosition()[1] + pose_x
                    obj_y =  math.sin(pose_theta)*obj.getPosition()[0] + math.cos(pose_theta)*obj.getPosition()[1] + pose_y + CAM_OFFSET_Y
                    obj_z = CAM_OFFSET_Z + obj.getPosition()[2]
                    
                    cube_positions.append([obj_x, obj_y, obj_z]) # store position in world frame
                    
                    # aisle thresholds determined by finding y-value of middle of each shelf
                    if obj_y > 3.9:
                        obj_y = AISLE_ONE
                        aisle_one_items.append([obj_x, obj_y, obj_z])
                        sort_x(aisle_one_items) # sort based on x-value (efficient pathing)
                    elif obj_y > 0 and obj_y < 3.9:
                        obj_y = AISLE_TWO
                        aisle_two_items.append([obj_x, obj_y, obj_z])
                        sort_x(aisle_two_items)
                    elif obj_y < 0 and obj_y > -3.8:
                        obj_y = AISLE_THREE
                        aisle_three_items.append([obj_x, obj_y, obj_z])
                        sort_x(aisle_three_items)
                    elif obj_y < -3.5: # for some reason that one block needs a more aggressive threshold
                        obj_y = AISLE_FOUR
                        aisle_four_items.append([obj_x, obj_y, obj_z])
                        sort_x(aisle_four_items)
                    else:
                        print("Invalid y-coordinate for object, could not create valid path")
                    
                    # concatenate all aisles to form continuous path of waypoints once all cubes are found
                    if len(cube_positions) == 12:
                        cube_pos_valid = aisle_four_items + aisle_three_items + aisle_two_items + aisle_one_items
                        # print("Cube Positions: ", cube_positions)
                    print("Cube Found! Current Total: ", len(cube_positions)) # stop mapping at 12 cubes (10 yellow, 2, green)
        
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
            avoid_left() # left sensors triggered
        if any(val < LID_COL_THRESH for val in lidar_sensor_readings[350:375]):
            avoid_right() # right sensors triggered
        if any(val < LID_COL_THRESH_FRONT for val in lidar_sensor_readings[245:255]):
            rand_explore() # front sensors triggered
        else:
            vL = MAX_SPEED
            vR = MAX_SPEED
                
        # Manually store map file and located cube objects
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == ord('S'):
            map = map > .5
            map = np.multiply(map,1)
            np.save('map.npy',map)
            print("Map file saved")
        elif key == ord('P'):
            np.save('cube_positions.npy', cube_positions)
            np.save('cube_pos_valid.npy', cube_pos_valid)
            print("Positions (true and modified/valid) saved")
    # end mapping block ------------------------------------------------------------------------------------

    # set wheel velocities
    if mode == 'shopping': # reduces wobble significantly, helps IK
        vL = vL*0.6
        vR = vR*0.6
        
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
