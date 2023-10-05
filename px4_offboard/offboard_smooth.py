#!/usr/bin/env python

__author__ = "Li-Yu Lin"
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
from px4_msgs.msg import VehicleStatus, VehicleLocalPosition, VehicleCommand

from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt8, Bool

class OffboardMission(Node):

    def __init__(self,mode):

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

        # define subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/px4_1/fmu/out/vehicle_local_position',
            self.local_position_callback,
            qos_profile_sub)
        
        # define publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            '/px4_1/fmu/in/offboard_control_mode', 
            qos_profile_pub)

        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint, 
            '/px4_1/fmu/in/trajectory_setpoint', 
            qos_profile_pub)
        
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, 
            '/px4_1/fmu/in/vehicle_command', 
            qos_profile_pub)                                        # disable for an experiment

        # parameters for callback
        self.timer_period   =   0.02  # seconds
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)

        self.flight_phase_ = np.uint8(1)
        self.entry_execute_ = np.uint8(1)

        self.counter = np.uint16(0)                                 # disable for an experiment

        # self.wpt_set_ = np.array([[0, 0, -1.2],
        #                           [0.0,-10.0,-1.2],
        #                           [10,-15,-1.2],
        #                           [20,-15,-1.2]])
        
         # Interesting trajectory origin
        self.lla_ref = np.array([24.484043629238872, 54.36068616768677, 0]) # latlonele -> (deg,deg,m)
        self.waypoint_idx = 0
        self.waypoints_lla = np.array([
            [24.484326113268185, 54.360644616972564, 10],
           [24.48476311664666, 54.3614948536716, 20],
           [24.485097533474377, 54.36197496905472, 20],
           [24.485400216562002, 54.3625570084458, 25], 
           [24.48585179883862, 54.36321951405934, 25], 
           [24.486198417650844, 54.363726451568475, 25], 
           [24.486564563238797, 54.36423338904003, 20], 
           [24.486894093361375, 54.364729597702144, 20], 
           [24.486664642851466, 54.36508096711639, 20],
           [24.486396136401133, 54.365263357350244, 25],
           [24.486066604972933, 54.36541087887424, 10],
           [24.485610141502686, 54.36572201510017,0],
        ])
        self.wpt_set_ = self.next_pos_ned = navpy.lla2ned(self.waypoints_lla[:,0], self.waypoints_lla[:,1],
                    self.waypoints_lla[:,2],self.lla_ref[0], self.lla_ref[1], self.lla_ref[2],
                    latlon_unit='deg', alt_unit='m', model='wgs84')

        self.theta  = np.float64(0.0)
        self.omega  = np.float64(1/10)

        self.cur_wpt_ = np.array([0.0,0.0,0.0],dtype=np.float64)
        self.prev_wpt_ = np.array([0.0,0.0,0.0],dtype=np.float64)
        
        self.wpt_idx_ = np.int8(0)

        self.nav_wpt_reach_rad_ =   np.float32(0.1)     # waypoint reach condition radius

        # variables for subscribers
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.local_pos_ned_     =   None
        self.local_vel_ned_     =   None

        # variables for publishers
        self.offboard_ctrl_position = False
        self.offboard_ctrl_velocity = False
        self.offboard_ctrl_acceleration = False
        self.offboard_ctrl_attitude = False
        self.offboard_ctrl_body_rate = False
        self.offboard_ctrl_actuator = False

        self.trajectory_setpoint_x = np.float64(0.0)
        self.trajectory_setpoint_y = np.float64(0.0)
        self.trajectory_setpoint_z = np.float64(0.0)
        self.trajectory_setpoint_yaw = np.float64(0.0)

    # subscriber callback
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def local_position_callback(self,msg):
        self.local_pos_ned_      =   np.array([msg.x,msg.y,msg.z],dtype=np.float64)
        self.local_vel_ned_      =   np.array([msg.vx,msg.vy,msg.vz],dtype=np.float64)

    # publisher
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(rclpy.clock.Clock().now().nanoseconds/1000) # time in microseconds
        msg.position = self.offboard_ctrl_position
        msg.velocity = self.offboard_ctrl_velocity
        msg.acceleration =self.offboard_ctrl_acceleration
        msg.attitude = self.offboard_ctrl_attitude
        msg.body_rate = self.offboard_ctrl_body_rate
        self.publisher_offboard_mode.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(rclpy.clock.Clock().now().nanoseconds/1000) # time in microseconds
        msg.position[0] = self.trajectory_setpoint_x
        msg.position[1] = self.trajectory_setpoint_y
        msg.position[2] = self.trajectory_setpoint_z
        msg.yaw = self.trajectory_setpoint_yaw
        self.publisher_trajectory.publish(msg)

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

        self.counter += 1   # disable for an experiment
        if self.counter >= 10 and self.counter <= 20:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1.0,6.0)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1.0)
            self.get_logger().info("Armed and dangerous....")

        # publish offboard control modes
        self.offboard_ctrl_position = True
        self.publish_offboard_control_mode()

        # publish offboard position cmd
        # phase 1: engage offboard control - switch to offboard/position mode
        # hold its position at a starting point
        # proceed to offboard wpt mission when the multicoptor reaches a setpoint
        if (self.flight_phase_ == 1) and (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            print("Current Mode: Offboard (Position hold at a starting point)")
            # entry:
            if self.entry_execute_:
                self.entry_execute_ 	    = 	0
                self.cur_wpt_  = self.wpt_set_[0]
                self.prev_wpt_ = self.local_pos_ned_
                self.theta  = np.float64(0.0)
            # during:
            self.trajectory_setpoint_x = self.theta*self.cur_wpt_[0]+(1-self.theta)*self.prev_wpt_[0]
            self.trajectory_setpoint_y = self.theta*self.cur_wpt_[1]+(1-self.theta)*self.prev_wpt_[1]
            self.trajectory_setpoint_z = self.theta*self.cur_wpt_[2]+(1-self.theta)*self.prev_wpt_[2]
            self.trajectory_setpoint_yaw  =   np.float64(-np.pi/2)
            self.publish_trajectory_setpoint()
            self.theta = self.theta+self.omega*self.timer_period
            self.theta = np.clip(self.theta,a_min=0.0,a_max=1.0)
            # transition
            if (self.local_pos_ned_ is not None) and (self.local_vel_ned_ is not None):
                dist_xyz    =   np.sqrt(np.power(self.cur_wpt_[0]-self.local_pos_ned_[0],2)+ \
                                        np.power(self.cur_wpt_[1]-self.local_pos_ned_[1],2)+ \
                                        np.power(self.cur_wpt_[2]-self.local_pos_ned_[2],2))
                if dist_xyz < self.nav_wpt_reach_rad_:
                    self.flight_phase_     =	2
                    self.entry_execute_    =	1

        # phase 2: engage offboard wpt mission - hold offboard/position mode
        # exit when the offboard mode control turns into off
        elif (self.flight_phase_ == 2) and (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            print("Current Mode: Offboard (wpt mission)")
            # entry:
            if self.entry_execute_:
                self.entry_execute_ 	    = 	0
                self.cur_wpt_  = self.wpt_set_[0]
                self.prev_wpt_ = self.local_pos_ned_
                self.theta  = np.float64(0.0)
            # during:
            if (self.local_pos_ned_ is not None) and (self.local_vel_ned_ is not None):
                self.trajectory_setpoint_x = self.theta*self.cur_wpt_[0]+(1-self.theta)*self.prev_wpt_[0]
                self.trajectory_setpoint_y = self.theta*self.cur_wpt_[1]+(1-self.theta)*self.prev_wpt_[1]
                self.trajectory_setpoint_z = self.theta*self.cur_wpt_[2]+(1-self.theta)*self.prev_wpt_[2]
                self.trajectory_setpoint_yaw  =   np.float64(-np.pi/2)
                self.publish_trajectory_setpoint()
                self.theta = self.theta+self.omega*self.timer_period
                self.theta = np.clip(self.theta,a_min=0.0,a_max=1.0)
                # transition
                dist_xyz    =   np.sqrt(np.power(self.cur_wpt_[0]-self.local_pos_ned_[0],2)+ \
                                        np.power(self.cur_wpt_[1]-self.local_pos_ned_[1],2)+ \
                                        np.power(self.cur_wpt_[2]-self.local_pos_ned_[2],2))

                if (dist_xyz <= self.nav_wpt_reach_rad_):
                    # Reset theta to 0 to start the new waypoint
                    self.theta  = np.float64(0.0)
                    self.prev_wpt_ = self.wpt_set_[self.wpt_idx_].flatten()
                    
                    if (self.wpt_idx_ == self.wpt_set_.shape[0] - 1 ):
                        print("Offboard mission finished")
                    else:    
                        self.wpt_idx_ = self.wpt_idx_+1
                        self.cur_wpt_ = self.wpt_set_[self.wpt_idx_]

def main():
    parser = argparse.ArgumentParser(description='Delivering parameters for tests')
    parser.add_argument('--mode','-m',type=int,default=np.uint8(1),help='mode setting')
    argin = parser.parse_args()

    rclpy.init(args=None)

    offboard_mission = OffboardMission(argin.mode)

    rclpy.spin(offboard_mission)

    offboard_mission.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()