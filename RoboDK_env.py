import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robodk import robolink, robomath  # RoboDK API
from robodk.robolink import ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET, ITEM_TYPE_FRAME

class RoboDKEnv(gym.Env):
    def __init__(self):
        super(RoboDKEnv, self).__init__()
        self.rdk = robolink.Robolink()

        # Automatically get the first robot in the station
        self.robots = self.rdk.ItemList(filter=ITEM_TYPE_ROBOT)
        if not self.robots:
            raise ValueError("No robots found in the RoboDK station.")
        #self.robot = self.robots[0]
        #print(f"Selected robot: {self.robot.Name()}")

        # Automatically get all self.targets
        self.targets = self.rdk.ItemList(filter=ITEM_TYPE_TARGET)
        #print("Available targets:")
        #for target in self.targets:
        #    print(f" - {target.Name()}")

        # Pick the first target as an example
        if not self.targets:
            raise ValueError("No targets found in the RoboDK station.")
        self.target = self.targets[0]

        # Automatically get all reference frames
        self.frames = self.rdk.ItemList(filter=ITEM_TYPE_FRAME)
        #print("Available reference frames:")
        #for frame in frames:
        #    print(f" - {frame.Name()}")

        if not self.frames:
            raise ValueError("No reference frames found.")
        self.ref_frame = self.frames[0]

    def getTargets(self):
        return self.targets

    def getRobots(self):
        return self.robots

    def getFrames(self):
        return self.frames


'''
for getting robot's name and reference name in another class we do the following:
        from RoboDK_env import RoboDKEnv()

        env = RoboDKEnv()
        myframe = env.getFrames()
        myrobots = env.getRobots()
        frames = []
        robots = []
        for f in myframe:
            frames.append(f.Name())
            print(f.Name()) # just for printing.
        
        for r in myrobots:
            robots.append(r.Names())
            print(r.Name()) # just for printing

'''