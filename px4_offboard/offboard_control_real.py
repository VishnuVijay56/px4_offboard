#!/usr/bin/env python

__author__ = "Kartik Anand Pant"
__contact__ = "@purdue.edu"

import argparse

import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import navpy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus, VehicleOdometry, VehicleCommand

class OffboardMission(Node):

    def __init__(self):

        super().__init__("px4_offboard_mission")

        # set publisher and subscriber quality of service profile
        qos_profile_pub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )

        qos_profile_sub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.VOLATILE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )
        self.ns = '/px4_1'
        # define subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            f'{self.ns}/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            f'{self.ns}/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile_sub)
        
        # define publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            f'{self.ns}/fmu/in/offboard_control_mode', 
            qos_profile_pub)

        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint, 
            f'{self.ns}/fmu/in/trajectory_setpoint', 
            qos_profile_pub)
        
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, 
            f'{self.ns}/fmu/in/vehicle_command', 
            qos_profile_pub)   
        self.arm_counter = 0


        # parameters for callback
        self.timer_period   =   0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)
        
        self.wpt_set_ = np.array([[0, 0,-1.2],
                                  [4.0, 0.0,-2.2],
                                  [2.0, 2.0,-0.2],
                                  [1.0, -12.0,-12],
                                  [0.0, 0.0,-2.2]
                                  ])
        self.velocity = 1
        self.wpt_idx_ = np.int8(0)
        self.nav_wpt_reach_rad_ =   np.float32(0.5)     # waypoint reach condition radius
        # variables for subscribers
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.local_pos_ned_     =   None
        self.prev_wpt_ = np.array([0,0,0])
        self.next_wpt_ = self.wpt_set_[self.wpt_idx_]

    # subscriber callback
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def odom_callback(self,msg):
        self.local_pos_ned_      =   np.array([msg.position[0],msg.position[1],msg.position[2]],dtype=np.float64)

    def publish_vehicle_command(self,command,param1=0.0,param2=0.0):            # disable for an experiment
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command  # command ID
        msg.target_system = 0  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher.publish(msg)

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        self.publisher_offboard_mode.publish(offboard_msg)
        
        #SiTL Test
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
        if self.nav_state != VehicleStatus.ARMING_STATE_ARMED and self.arm_counter < 10:
            self.arm_counter += 1
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)


        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            norm = np.linalg.norm(self.next_wpt_ - self.prev_wpt_)
            trajectory_msg = TrajectorySetpoint()
            # Move in the unit vector direction to the next way point with the given velocity. 
            # Clipping to make sure that it remains within the bounds until the drone is reached.
            
            x_min = np.min([self.prev_wpt_[0], self.next_wpt_[0]])
            x_max = np.max([self.prev_wpt_[0], self.next_wpt_[0]])
            x_pos = np.clip(self.local_pos_ned_[0] \
                            + self.velocity * (self.next_wpt_[0] - self.prev_wpt_[0])/norm, \
                                x_min, x_max)
            
            y_min = np.min([self.prev_wpt_[1], self.next_wpt_[1]])
            y_max = np.max([self.prev_wpt_[1], self.next_wpt_[1]])
            y_pos = np.clip(self.local_pos_ned_[1] \
                            + self.velocity * (self.next_wpt_[1] - self.prev_wpt_[1])/norm, \
                                y_min, y_max)
            
            z_min = np.min([-self.prev_wpt_[2], -self.next_wpt_[2]])
            z_max = np.max([-self.prev_wpt_[2], -self.next_wpt_[2]])
            z_pos = np.clip(self.local_pos_ned_[2] \
                            + self.velocity * (self.next_wpt_[2] - self.prev_wpt_[2])/norm, \
                                -z_max, -z_min)

            trajectory_msg.position[0]  = x_pos
            trajectory_msg.position[1]  = y_pos
            trajectory_msg.position[2]  = z_pos
            
            trajectory_msg.yaw   =   0.0
            
            dist_xyz    =   np.sqrt(np.power(self.next_wpt_[0]-self.local_pos_ned_[0],2)+ \
                                    np.power(self.next_wpt_[1]-self.local_pos_ned_[1],2)+ \
                                    np.power(self.next_wpt_[2]-self.local_pos_ned_[2],2))
            
            if (dist_xyz <= self.nav_wpt_reach_rad_):
                if (self.wpt_idx_ == self.wpt_set_.shape[0] - 1):
                    print("Offboard mission finished")
                else:    
                    print("Waypoint: " + str(self.wpt_idx_))
                    self.wpt_idx_ = self.wpt_idx_ + 1
                    self.prev_wpt_ = self.next_wpt_
                    self.next_wpt_ = self.wpt_set_[self.wpt_idx_]
            self.publisher_trajectory.publish(trajectory_msg)


def main():

   
    rclpy.init(args=None)

    offboard_mission = OffboardMission()

    rclpy.spin(offboard_mission)

    offboard_mission.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()