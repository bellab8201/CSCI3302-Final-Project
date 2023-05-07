"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, PositionSensor, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
#import keyboard

import ikpy.chain
import ikpy.utils.plot as plot_utils
#my_chain = ikpy.chain.Chain.from_urdf_file("TiagoSteel-Copy2.urdf")
my_chain1 = ikpy.chain.Chain.from_urdf_file("robot_urdf.urdf", base_elements=["torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint"], active_links_mask=[False, False, True, True, True, True, True, True, True, False, False, True])
print("===Printing Chain===")
print(my_chain1)


#Initialization
print("===Initializing Grocery Shopper...===")
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

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.0450000,0.0450000)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    #robot_parts[part_name].getPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

#enable robot arm
torso_lift_joint = robot.getDevice('torso_lift_joint')
arm_1_joint = robot.getDevice('arm_1_joint')
arm_2_joint = robot.getDevice('arm_2_joint')
arm_3_joint = robot.getDevice('arm_3_joint')
arm_4_joint = robot.getDevice('arm_4_joint')
arm_5_joint = robot.getDevice('arm_5_joint')
arm_6_joint = robot.getDevice('arm_6_joint')
arm_7_joint = robot.getDevice('arm_7_joint')

#enable sensors for arm
arm_1_joint_sensor = robot.getDevice('arm_1_joint_sensor')
arm_2_joint_sensor = robot.getDevice('arm_2_joint_sensor')
arm_3_joint_sensor = robot.getDevice('arm_3_joint_sensor')
arm_4_joint_sensor = robot.getDevice('arm_4_joint_sensor')
arm_5_joint_sensor = robot.getDevice('arm_5_joint_sensor')
arm_6_joint_sensor = robot.getDevice('arm_6_joint_sensor')
arm_7_joint_sensor = robot.getDevice('arm_7_joint_sensor')

arm_1_joint_sensor.enable(timestep)
arm_2_joint_sensor.enable(timestep)
arm_3_joint_sensor.enable(timestep) 
arm_4_joint_sensor.enable(timestep)
arm_5_joint_sensor.enable(timestep)
arm_6_joint_sensor.enable(timestep)
arm_7_joint_sensor.enable(timestep)

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

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# ------------------------------------------------------------------
# Helper Functions

mode = 'manual'
#gripper_status="open"
"""
arm_1_joint.setPosition(0.7)
arm_2_joint.setPosition(0.5)
arm_3_joint.setPosition(-3.46)
arm_4_joint.setPosition(0.1)
arm_5_joint.setPosition(0.1)
arm_6_joint.setPosition(0.1)
arm_7_joint.setPosition(0.1)
"""

gripper_status = 0
current_position = 0
"""
#target_position = [-4.34, -0.00468, 0.0238]
target_position = [0.0049, 0.0153, -0.1208]
print("The angles of each joints are : ", my_chain.inverse_kinematics(target_position))
real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position))
print("Computed position vector : %s, original position vector : %s" % (real_frame[:3, 3], target_position))
"""

current_joint_positions = np.zeros(12)
joint_positions = my_chain1.forward_kinematics(current_joint_positions)
print("===Printing Forward Kinematics===")
print(joint_positions)

#target_position_object = [4.4, -3.53, 1.07]
#print("The angles of each joints are : ", my_chain1.inverse_kinematics(target_position_object))
"""
target = [4.4, -3.53, 1.07]
target_orientation = [-6.32976e-16, -1, -6.62413e-16]

frame_target = np.eye(4)
frame_target[:3, 3] = target

ik = my_chain1.inverse_kinematics_frame(frame_target)
"""
# ------------------------------------------------------------------
# Main Loop
while robot.step(timestep) != -1:

# ------------------------------------------------------------------
    #ARM MOVEMENT
    #i = my_chain1.inverse_kinematics([2,2,2])
    #print(i)
    
    #arm_1_joint.setPosition(0.7)   
    
    arm_1_joint_sensor = robot.getDevice('arm_1_joint_sensor').getValue()
    arm_2_joint_sensor = robot.getDevice('arm_2_joint_sensor').getValue()
    arm_3_joint_sensor = robot.getDevice('arm_3_joint_sensor').getValue()
    arm_4_joint_sensor = robot.getDevice('arm_4_joint_sensor').getValue()
    arm_5_joint_sensor = robot.getDevice('arm_5_joint_sensor').getValue()
    arm_6_joint_sensor = robot.getDevice('arm_6_joint_sensor').getValue()
    arm_7_joint_sensor = robot.getDevice('arm_7_joint_sensor').getValue()
    #print(arm_1_joint_sensor,arm_2_joint_sensor,arm_3_joint_sensor,arm_4_joint_sensor,arm_5_joint_sensor,arm_6_joint_sensor, arm_7_joint_sensor)
    

    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    #arm controller
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        # Move the shoulder lift joint right/left
        #if key == ord('1'):
        if key==Keyboard.CONTROL+ord('B'):
        #position to get cubes in basket - DONT USE THIS ONE IF TESTING ON MIDDLE SHELF
            arm_1_joint.setPosition(0.07)
            arm_2_joint.setPosition(0.2)
            arm_3_joint.setPosition(-1.5)
            arm_4_joint.setPosition(2.29)
            arm_5_joint.setPosition(-2.07)
            arm_6_joint.setPosition(1.20)
            arm_7_joint.setPosition(-1.30)
            current_joint_positions = np.zeros(12)
            joint_positions = my_chain1.forward_kinematics(current_joint_positions)
            print(joint_positions)
            pass
            
        # --- BLOCKS MIDDLE SHELF --- #    
        elif key==Keyboard.CONTROL+ord('M'):
        #BLOCKS ON MIDDLE SHELF
        #waypoint 1: 1.35,-3.02, 0.0881 aisle 2 first block on mid shelf
        #waypoint 2: 1.61, -3.02, 0.0884
            arm_1_joint.setPosition(0.07)
            arm_2_joint.setPosition(0.04)
            arm_3_joint.setPosition(-1.4)
            arm_4_joint.setPosition(1.4)
            arm_5_joint.setPosition(1.98)
            arm_6_joint.setPosition(-0.2)
            arm_7_joint.setPosition(-1.92)
            pass
        elif key==Keyboard.CONTROL+ord('N'):
        #BLOCKS ON MIDDLE SHELF TO BASKET
            arm_4_joint.setPosition(2.1)
            arm_5_joint.setPosition(1.5)
            arm_6_joint.setPosition(-1.34)
            pass
            
        # --- GRIPPERS --- #
        elif key==Keyboard.CONTROL+ord('C'):
            # Close gripper, note that this takes multiple time steps...
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            pass
        elif key==Keyboard.CONTROL+ord('O'):
            #open gripper
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            pass
            
        # ----- MOVE ROBOT MANUALLY ----- #
        elif key == keyboard.LEFT :
            vL = -MAX_SPEED/4
            vR = MAX_SPEED/4
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED/4
            vR = -MAX_SPEED/4
        elif key == keyboard.UP:
            vL = MAX_SPEED/4
            vR = MAX_SPEED/4
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED/4
            vR = -MAX_SPEED/4
        elif key == ord(' '):
            vL = 0
            vR = 0
            #i = arm_1_joint.getPositionSensor()
            #print(i)
            
        """
        if key == ord('1'):
        #main arm left
            #arm_1_joint_position += 0.1
            #arm_1_joint.setPosition(0.1)
            if key == ord('1'):
                for i in np.arange(0.07, 2.68, 0.01):
                    arm_1_joint.setPosition(i)
        elif key == ord('2'):
        #main arm right
            #arm_1_joint_position -= 0.1
            #arm_1_joint.setPosition(-0.1)
            if key == ord('2'):
                for i in np.arange(2.68, 0.07, -0.01):
                    arm_1_joint.setPosition(i)
        elif key == ord('3'):
        #main arm up
            for i in np.arange(-1.5, 1.02, 0.01):
                arm_2_joint.setPosition(i)
        elif key == ord('4'):
        #main arm down
            for i in np.arange(1.02, -1.5, -0.01):
                arm_2_joint.setPosition(i) 
        elif key == ord('7'):
        #main arm rotate left
            #arm_3_joint.setPosition(0.1)
            if key == ord('7'):
                for i in np.arange(-3.46, 1.5, 0.01):
                    arm_3_joint.setPosition(i)
                    print(i)
        elif key == ord('8'):
        #main arm rotate right
            if key == ord('8'):
                for i in np.arange(1.5, -3.46, -0.01):
                    arm_3_joint.setPosition(i)
                    print(i)
                #current_position = arm_3_joint.setPosition(i)
                #print(i)
                #print(current_position)
        elif key == ord('9'):
        #main arm rotate right
            if key == ord('9'):
                for i in np.arange(-0.32, 2.29, 0.01):
                    arm_4_joint.setPosition(i)
        elif key == ord('a'):
        #main arm rotate right
            if key == ord('a'):
                for i in np.arange(2.29, -0.32, -0.01):
                    arm_4_joint.setPosition(i)
        elif key==Keyboard.CONTROL+ord('I'):
            current_joint_positions = np.zeros(12)
            joint_positions = my_chain1.forward_kinematics(current_joint_positions)
            print("new", joint_positions)
            
        elif key == ord('5'):
            # Close gripper, note that this takes multiple time steps...
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            pass
        elif key == ord('6'):
            #open gripper
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        """
   
# ------------------------------------------------------------------       
