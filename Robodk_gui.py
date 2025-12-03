import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import csv

from robodk import robolink, robomath  # RoboDK API

class RoboDKgui(gym.Env):
    def __init__(self, start_waypoint="Target 1", end_waypoint="Target 2"):
        super(RoboDKgui, self).__init__()
        self.rdk = robolink.Robolink()

        self.robot = self.rdk.Item('UR10')  # Ensure this is the correct robot name

        self.startwaypoint = self.rdk.Item(start_waypoint)  # Ensure this is the correct target name
        #self.startwaypoint = self.rdk.Item('Target 1')  # Ensure this is the correct target name

        self.endwaypoint = self.rdk.Item(end_waypoint)# It was the rectangle from the previuos software.
        #self.endwaypoint = self.rdk.Item('Target 2')# It was the rectangle from the previuos software.


        self.ref_frame = self.rdk.Item('Reference_Frame')

        self.fields = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']

        self.filename = "joint_Target"

        self.data_ = []

        print('-------------------------------------------')
        print('Robot target Joints ', self.startwaypoint.Joints())
        print('-------------------------------------------')
        print('Robot current Joints:', self.robot.Joints())

        if not self.robot.Valid():
            raise ValueError("Robot not found: Ensure the robot name is correct.")

        if not self.ref_frame.Valid():
            raise ValueError('Reference Frame not found: Ensure the reference frame name is correct.')

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.testinitial_joints = np.array([-154.83, -88.50, -117.39, -60.86, 90.14, -54.19], dtype=np.float32)  # Degrees
        self.initial_joints = np.array(self.startwaypoint.Joints().list(), dtype=np.float32)
        print(self.initial_joints)
        print(self.testinitial_joints)
        self.joint_limits = self.get_joint_limits()  # Get joint limits from the robot
        #Writing joints to a file csv:


        
        self.reset()

    def step(self, action):

        current_joints = np.array(self.robot.Joints().list(), dtype=np.float32)

        target_joint = np.array(self.endwaypoint.Joints().list(), dtype=np.float32)
        new_joints = current_joints + action

        
        current_distance_Joints = np.linalg.norm(current_joints - target_joint)
        new_joints_distance =  np.linalg.norm(new_joints - target_joint)
        #Find the position



        action_position = np.concatenate(([action[0]], [action[1]], [action[2]]))

        current_position = np.array(self.robot.Pose().Pos(), dtype=np.float32)
        
        target_position = np.array(self.endwaypoint.Pose().Pos(), dtype=np.float32)
        
        current_distance_Position = np.linalg.norm(current_position - target_position)
        
        new_position = current_position - action_position
        
        new_distance_position = np.linalg.norm(new_position - target_position)

        #self.refrence_frame_restriction(new_position)

        if new_joints_distance <= current_distance_Joints and current_distance_Joints > 0.01:
            self.robot.setJoints(new_joints.tolist())
            
            # writing to csv file
            self.data_.append(new_joints.tolist())
        
        else:
            with open(self.filename + ".csv", 'w') as csvfile:
            # creating a csv writer object
                csvwriter = csv.writer(csvfile)

            # writing the fields
                csvwriter.writerow(self.fields)

            # writing the data rows
                csvwriter.writerows(self.data_)

        #distance = np.linalg.norm(new_position - target_position)
        distance = np.linalg.norm(new_joints - target_joint)

        reward = float(-distance)  # Convert to Python float
        #print(current_distance_Joints)

        # Determine if the task is done
        done = bool(distance < 2)
        if done == True:
            print('Robot has reached the Target position succefully')  # Ensure done is a boolean

        # Set the observation to the new joint positions
        #observation = new_position
        observation = new_joints
        info = {}

        return observation, reward, done, False, info

    def get_joint_limits(self):
        # Retrieve joint limits from the robot
        lower_limits, upper_limits, _ = self.robot.JointLimits()
        return list(zip(lower_limits, upper_limits))

    
    def refrence_frame_restriction(self, newposition):
        refrence_frame = np.array(self.ref_frame.Pose().Pos(), dtype=np.float32)
        print('New position',newposition)
        print(refrence_frame)
        print(np.array(self.startwaypoint.Pose().Pos(), dtype=np.float32))


    def check_joint_limits(self, joints):
        check = True
        jointLimit = self.robot.JointLimits()
        min_limit = jointLimit[0]
        max_limit = jointLimit[1]

        for i in range(len(min_limit.tolist())):
            if joints[i] < min_limit.tolist()[i] or joints[i] > max_limit.tolist()[i]:
                check = False
                break
        return check

    def update_action(self):
        # Placeholder function to update the action
        print("Action updated with new strategy.")
        # Implement the action update logic specific to your application

    def reset(self, *, seed: int = None, options: dict = None):
        #print('...........Moving to start waypoint............')
        self.robot.MoveJ(self.initial_joints.tolist())
        if seed is not None:
            np.random.seed(seed)
        return self.initial_joints, {}

    def radians_todegrees(self, radians):
        return radians * (180 / math.pi)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
