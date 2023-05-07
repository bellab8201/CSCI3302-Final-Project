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

mode = 'manipulation'

# ------------------------------------------------------------------
# Main Loop
while robot.step(timestep) != -1:

# ------------------------------------------------------------------
    #ARM MOVEMENT
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    #arm controller
    if mode == 'manipulation':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        # Move the shoulder lift joint right/left
        #if key == ord('1'):
        if key==Keyboard.CONTROL+ord('B'):
        #position to get cubes in basket
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
            torso_lift_joint.setPosition(0)
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
            torso_lift_joint.setPosition(0)
            arm_4_joint.setPosition(2.1)
            arm_5_joint.setPosition(1.5)
            arm_6_joint.setPosition(-1.34)
            pass
            
        # --- BLOCKS TOP SHELF --- #    
        elif key==Keyboard.CONTROL+ord('T'):
        #BLOCKS ON TOP SHELF
            torso_lift_joint.setPosition(0.35)
            arm_1_joint.setPosition(0.07)
            arm_2_joint.setPosition(0.04)
            arm_3_joint.setPosition(-1.8)
            arm_4_joint.setPosition(1.4)
            arm_5_joint.setPosition(-0.1)
            arm_6_joint.setPosition(-0.3)
            arm_7_joint.setPosition(0.1)
            pass
        elif key==Keyboard.CONTROL+ord('Y'):
        #BLOCKS ON TOP SHELF TO BASKET
            torso_lift_joint.setPosition(0.2)
            arm_4_joint.setPosition(2.15)
            arm_5_joint.setPosition(-1.9)
            arm_6_joint.setPosition(1.39)
            arm_7_joint.setPosition(-1.5)
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
            vL = -MAX_SPEED/2.5
            vR = MAX_SPEED/2.5
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED/2.5
            vR = -MAX_SPEED/2.5
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
   
# ------------------------------------------------------------------       
