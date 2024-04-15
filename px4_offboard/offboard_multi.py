#!/usr/bin/env python

__author__ = "Yifan Guo, Minhyun Cho, Vishnu Vijay"
__contact__ = "xxx, mhcho, xxx (@purdue.edu)"

import argparse

import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import navpy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleStatus, VehicleLocalPosition, VehicleGlobalPosition, VehicleCommand

from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt8, Bool

from functools import partial

class OffboardMission(Node):

    def __init__(self,N,waypoints,formation_config,ref_lla):

        # inheritance from parent class
        super().__init__("px4_offboard_mission")

        # set publisher and subscriber quality of service profile
        qos_profile_pub     =   QoSProfile(
            reliability     =   QoSReliabilityPolicy.BEST_EFFORT,
            durability      =   QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history         =   QoSHistoryPolicy.KEEP_LAST,
            depth           =   1
        )

        qos_profile_sub     =   QoSProfile(
            reliability     =   QoSReliabilityPolicy.BEST_EFFORT,
            durability      =   QoSDurabilityPolicy.VOLATILE,
            history         =   QoSHistoryPolicy.KEEP_LAST,
            depth           =   1
        )

        # define number of drones
        self.N_drone    =   N

        # define subscribers and publishers
        self.trajectory_midpoint_publisher  =   self.create_publisher(          # Centroid publisher
                    TrajectorySetpoint,
                    '/px4_0/fmu/in/trajectory_midpoint',
                    qos_profile_pub)
        
        self.array_publishers     =   [{'offboard_mode_pub':None, 'trajectory_pub':None, 'vehicle_command_pub':None} for _ in range(self.N_drone)]
        self.array_subscribers    =   [{'status_sub':None, 'local_pos_sub':None, 'global_pos_sub':None} for _ in range(self.N_drone)]

        for i in range(self.N_drone):

            self.ns             =   f'/px4_{i+1}'

            self.array_subscribers[i]['status_sub']     =   self.create_subscription(
                VehicleStatus,
                f'{self.ns}/fmu/out/vehicle_status',
                partial(self.vehicle_status_callback,id=i),                             # instead of lambda function lambda msg: self.vehicle_status_callback(msg,id=i), use partial function
                qos_profile_sub)

            self.array_subscribers[i]['local_pos_sub']  =   self.create_subscription(
                VehicleLocalPosition,
                f'{self.ns}/fmu/out/vehicle_local_position',
                partial(self.local_position_callback,id=i),
                qos_profile_sub)

            self.array_subscribers[i]['global_pos_sub'] =   self.create_subscription(
                VehicleGlobalPosition,
                f'{self.ns}/fmu/out/vehicle_global_position',
                partial(self.global_position_callback,id=i),
                qos_profile_sub)

            self.array_publishers[i]['offboard_mode_pub']   =   self.create_publisher(
                OffboardControlMode,
                f'{self.ns}/fmu/in/offboard_control_mode',
                qos_profile_pub)

            self.array_publishers[i]['trajectory_pub']      =   self.create_publisher(
                TrajectorySetpoint,
                f'{self.ns}/fmu/in/trajectory_setpoint',
                qos_profile_pub)

            self.array_publishers[i]['vehicle_command_pub'] =   self.create_publisher(
                VehicleCommand,
                f'{self.ns}/fmu/in/vehicle_command',
                qos_profile_pub)

        # parameters for callback
        self.timer_period   =   0.02            # [sec] callback function frequency
        self.timer          =   self.create_timer(self.timer_period, self.cmdloop_callback)

        # define formation config
        self.formation      =   formation_config

        # set up the origin and waypoints; convert global coordiante to local ones.
        self.ref_lla        =   ref_lla
        self.waypoint_idx   =   [0 for _ in range(self.N_drone)]

        self.waypoints_lla  =   waypoints
        self.waypoints_ned  =   navpy.lla2ned(self.waypoints_lla[:,0], self.waypoints_lla[:,1],self.waypoints_lla[:,2],
                                              self.ref_lla[0], self.ref_lla[1], self.ref_lla[2],
                                              latlon_unit='deg', alt_unit='m', model='wgs84')

        self.wpt_set_list   =   [None for _ in range(self.N_drone)]

        for i in range(self.N_drone):
            self.wpt_set_list[i]        =   np.copy(self.waypoints_ned)
            self.wpt_set_list[i][:,0]   +=  self.formation[i,0]
            self.wpt_set_list[i][:,1]   +=  self.formation[i,1]
            self.wpt_set_list[i][:,2]   +=  self.formation[i,2]

        # formation flight parameter initialization
        self.velocity           =   np.float64(4.0)     # [m/s] velocity of the vehicle
        self.arm_counter_list   =   [0 for i in range(self.N_drone)]
        self.wpt_idx_list       =   np.array([np.int8(0) for _ in range(self.N_drone)])
        self.nav_wpt_reach_rad  =   1.0*self.velocity   # [m] waypoint reach condition radius

        # variables for subscribing navigation information
        self.nav_state_list         =   [VehicleStatus.NAVIGATION_STATE_MAX for _ in range(self.N_drone)]
        self.local_pos_ned_list     =   [None for _ in range(self.N_drone)]
        self.local_pos_ned_valid    =   [None for _ in range(self.N_drone)]
        self.global_ref_lla_list    =   [None for _ in range(self.N_drone)]
        self.global_ref_lla_valid   =   [None for _ in range(self.N_drone)]

        self.prev_wpt_list      =   []
        self.next_wpt_list      =   []
        self.trajectory_set_pt  =   []
        self.yaw_set_pt         =   []
        for i in range(self.N_drone):
            self.prev_wpt_list.append(np.array([0,0,0],dtype=np.float64))
            self.next_wpt_list.append(np.array([0,0,0],dtype=np.float64))
            self.trajectory_set_pt.append(np.array([0,0,0],dtype=np.float64))
            self.yaw_set_pt.append(np.array(0.0,dtype=np.float64))

        # variables for initialization
        self.entry_execute      =   [False for _ in range(self.N_drone)]
        self.ned_spawn_offset   =   [None for _ in range(self.N_drone)]
        self.flight_phase       =   [np.int8(0) for _ in range(self.N_drone)]
        self.next_phase_flag    =   [False for _ in range(self.N_drone)]
        self.wpt_change_flag    =   [False for _ in range(self.N_drone)]
        self.omega              =   np.float64(0.0)
        
        # attack vector
        self.attack_vector      =   []

        for i in range(self.N_drone):
            self.attack_vector.append(np.array([0,0,0],dtype=np.float64))

        self.attack_vector[4]   =   self.formation[3,:]-self.formation[4,:]
        self.attack_vector[2]   =   3*self.formation[2,:]

        self.attack_duration    =   np.float64(10.0)
        self.attack_timer       =   np.float64(0.0)

    # subscriber callback
    def vehicle_status_callback(self,msg,id):
        self.nav_state_list[id] = msg.nav_state

    def local_position_callback(self,msg,id):
        self.local_pos_ned_list[id]     =   np.array([msg.x,msg.y,msg.z],dtype=np.float64)

    def global_position_callback(self,msg,id):
        self.global_ref_lla_list[id]    =   np.array([msg.lat,msg.lon,msg.alt],dtype=np.float64)

    def publish_vehicle_command(self,command,id,param1=0.0,param2=0.0,param3=0.0):
        msg                     =   VehicleCommand()
        msg.param1              =   param1
        msg.param2              =   param2
        msg.param3              =   param3
        msg.command             =   command     # command ID
        msg.target_system       =   0           # system which should execute the command
        msg.target_component    =   1           # component which should execute the command, 0 for all components
        msg.source_system       =   1           # system sending the command
        msg.source_component    =   1           # component sending the command
        msg.from_external       =   True
        msg.timestamp           =   int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.array_publishers[id]['vehicle_command_pub'].publish(msg)

    def publish_offboard_control_mode(self,id):
        msg                     =   OffboardControlMode()
        msg.timestamp           =   int(Clock().now().nanoseconds/1000)
        msg.position            =   True
        msg.velocity            =   False
        msg.acceleration        =   False
        msg.attitude            =   False
        msg.body_rate           =   False
        self.array_publishers[id]['offboard_mode_pub'].publish(msg)

    def publish_trajectory_setpoint(self,id):
        msg                     =   TrajectorySetpoint()
        msg.timestamp           =   int(Clock().now().nanoseconds/1000)
        msg.position            =   np.array(self.trajectory_set_pt[id],dtype=np.float32)
        msg.yaw                 =   float(self.yaw_set_pt[id])
        self.array_publishers[id]['trajectory_pub'].publish(msg)
    
    def publish_trajectory_centroid(self):
        msg                     =   TrajectorySetpoint()
        msg.timestamp           =   int(Clock().now().nanoseconds/1000)
        msg.position            =   np.array(self.trajectory_centroid,dtype=np.float32)
        msg.yaw                 =   float(self.yaw_set_pt[id])
        self.array_publishers[id]['trajectory_pub'].publish(msg)

    def cmdloop_callback(self):

        # Publish Centroid Trajectory
        #TODO

        # Publish Set of Trajectories
        for idx in range(self.N_drone): 

            if self.flight_phase[idx] == 0:

                # entry:
                # compensation of local position coordinate
                if self.global_ref_lla_list[idx] is not None and self.local_pos_ned_list[idx] is not None and self.entry_execute[idx] is False:
                    self.ned_spawn_offset[idx]  =   navpy.lla2ned(self.global_ref_lla_list[idx][0],self.global_ref_lla_list[idx][1],0,
                                                                self.ref_lla[0],self.ref_lla[1],self.ref_lla[2],
                                                                latlon_unit='deg', alt_unit='m', model='wgs84')

                    self.wpt_set_list[idx][:,0] -=  self.ned_spawn_offset[idx][0]
                    self.wpt_set_list[idx][:,1] -=  self.ned_spawn_offset[idx][1]
                    self.wpt_set_list[idx][:,2] -=  self.ned_spawn_offset[idx][2]

                    self.trajectory_set_pt[idx] =   np.array([0.0,0.0,-5.0],dtype=np.float64)
                    self.yaw_set_pt[idx]        =   self.yaw_set_pt[idx]

                    self.prev_wpt_list[idx]     =   np.copy(self.local_pos_ned_list[idx])
                    self.next_wpt_list[idx]     =   np.copy(self.trajectory_set_pt[idx])

                    self.publish_trajectory_setpoint(idx)

                    self.entry_execute[idx]     =   True

                # during:
                if self.nav_state_list[idx] != VehicleStatus.ARMING_STATE_ARMED or self.arm_counter_list[idx] < 20:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,idx,1.0,6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,idx,1.0)
                    self.arm_counter_list[idx]  +=  1
                    # self.get_logger().info('Drone #'+str(idx+1)+' armed and dangerous....')

                if self.entry_execute[idx] is True:
                    self.publish_offboard_control_mode(idx)
                    self.publish_trajectory_setpoint(idx)

                    if np.linalg.norm(self.next_wpt_list[idx]-self.local_pos_ned_list[idx]) < 1.0:
                        self.next_phase_flag[idx]   =   True

            elif self.flight_phase[idx] == 1:
                # entry:
                if self.entry_execute[idx] is False:
                    self.prev_wpt_list[idx]     =   np.copy(self.next_wpt_list[idx])
                    self.next_wpt_list[idx]     =   np.copy(self.wpt_set_list[idx][self.wpt_idx_list[idx],:])
                    self.omega                  =   np.float64(0.0)

                    norm        =   np.linalg.norm(self.next_wpt_list[idx]-self.prev_wpt_list[idx])
                    self.trajectory_set_pt[idx][0]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][0] \
                                                                +self.omega*self.next_wpt_list[idx][0] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][0]-self.prev_wpt_list[idx][0])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][0], self.next_wpt_list[idx][0]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][0], self.next_wpt_list[idx][0]])))                          
                    self.trajectory_set_pt[idx][1]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][1] \
                                                                +self.omega*self.next_wpt_list[idx][1] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][1]-self.prev_wpt_list[idx][1])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][1], self.next_wpt_list[idx][1]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][1], self.next_wpt_list[idx][1]])))
                    self.trajectory_set_pt[idx][2]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][2] \
                                                                +self.omega*self.next_wpt_list[idx][2] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][2]-self.prev_wpt_list[idx][2])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][2], self.next_wpt_list[idx][2]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][2], self.next_wpt_list[idx][2]])))
                    self.yaw_set_pt[idx]        =   self.yaw_set_pt[idx]

                    self.publish_offboard_control_mode(idx)
                    self.publish_trajectory_setpoint(idx)

                    self.entry_execute[idx]     =   True

                if self.entry_execute[idx] is True:
                    norm        =   np.linalg.norm(self.next_wpt_list[idx]-self.prev_wpt_list[idx])
                    self.trajectory_set_pt[idx][0]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][0] \
                                                                +self.omega*self.next_wpt_list[idx][0] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][0]-self.prev_wpt_list[idx][0])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][0], self.next_wpt_list[idx][0]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][0], self.next_wpt_list[idx][0]]))) \
                                                        +self.attack_vector[idx][0]*self.attack_timer

                    self.trajectory_set_pt[idx][1]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][1] \
                                                                +self.omega*self.next_wpt_list[idx][1] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][1]-self.prev_wpt_list[idx][1])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][1], self.next_wpt_list[idx][1]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][1], self.next_wpt_list[idx][1]]))) \
                                                        +self.attack_vector[idx][1]*self.attack_timer    
                    self.trajectory_set_pt[idx][2]  =   np.clip((1-self.omega)*self.prev_wpt_list[idx][2] \
                                                                +self.omega*self.next_wpt_list[idx][2] \
                                                                +0.3*self.velocity*(self.next_wpt_list[idx][2]-self.prev_wpt_list[idx][2])/norm, \
                                                                np.min(np.array([self.prev_wpt_list[idx][2], self.next_wpt_list[idx][2]])), \
                                                                np.max(np.array([self.prev_wpt_list[idx][2], self.next_wpt_list[idx][2]]))) \
                                                        +self.attack_vector[idx][2]*self.attack_timer    
                                 
                    self.yaw_set_pt[idx]        =   self.yaw_set_pt[idx]

                    self.publish_offboard_control_mode(idx)
                    self.publish_trajectory_setpoint(idx)

                    if np.linalg.norm(self.next_wpt_list[idx]-self.local_pos_ned_list[idx]+self.attack_vector[idx]*self.attack_timer) < self.nav_wpt_reach_rad and self.wpt_idx_list[idx] < np.shape(self.wpt_set_list[idx])[0]-1:
                        self.wpt_change_flag[idx]   =   True

                    elif np.linalg.norm(self.next_wpt_list[idx]-self.local_pos_ned_list[idx]+self.attack_vector[idx]*self.attack_timer) < self.nav_wpt_reach_rad and self.wpt_idx_list[idx] == np.shape(self.wpt_set_list[idx])[0]-1:
                        self.next_phase_flag[idx]   =   True

                if idx == 0 and self.wpt_idx_list[idx] >= 6:
                    self.attack_timer   =   np.clip(self.attack_timer+self.timer_period/self.attack_duration,0,1)
                    print('C2 Link Hijacking')

                if idx == 0 and self.wpt_idx_list[idx] >= 7:
                    self.attack_timer   =   np.float64(0.0)

                if idx == 0:
                    norm_delt       =   np.linalg.norm(self.next_wpt_list[idx]-self.prev_wpt_list[idx])/self.velocity
                    self.omega      =   self.omega+self.timer_period/norm_delt
                    self.omega      =   np.clip(self.omega,0,1)

            elif self.flight_phase[idx] == 2:
                   
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,idx,1.0,4.0,5.0)


        if all(self.next_phase_flag):
            self.flight_phase       =   [self.flight_phase[idx]+1 for idx in range(self.N_drone)]
            self.next_phase_flag    =   [False for _ in range(self.N_drone)]
            self.entry_execute      =   [False for _ in range(self.N_drone)]
            print('Next Flight Phase')

        if all(self.wpt_change_flag):

            self.omega                  =   np.float64(0.0)

            for idx in range(self.N_drone):
                self.wpt_idx_list[idx]      =   self.wpt_idx_list[idx]+1
                self.prev_wpt_list[idx]     =   np.copy(self.next_wpt_list[idx])
                self.next_wpt_list[idx]     =   np.copy(self.wpt_set_list[idx][self.wpt_idx_list[idx],:])
                self.wpt_change_flag[idx]   =   False

            print('Waypoint Update: '+str(self.wpt_idx_list[0]))

        

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', type=int)
    # args = parser.parse_args()

    N_drone     =   6
    ref_lla     =   np.array([24.484043629238872, 54.36068616768677, 0])    # (lat,lon,alt) -> (deg,deg,m)
    WPTs        =   np.array([
                    [24.484043629238872,54.36068616768677,40],
                    [24.484326113268185,54.360644616972564,40],
                    [24.48476311664666,54.3614948536716,40],
                    [24.485097533474377,54.36197496905472,40],
                    [24.485400216562002,54.3625570084458,40],
                    [24.48585179883862,54.36321951405934,40],
                    [24.486198417650844,54.363726451568475,40],
                    [24.486564563238797,54.36423338904003,40],
                    ])
    
    WPTs        =   np.insert(WPTs,8,np.flipud(np.copy(WPTs)),axis=0)
    WPTs        =   np.delete(WPTs,8,0)

    Formation   =   np.array([
                    [3.0*np.cos(np.pi/180*0),3.0*np.sin(np.pi/180*0),0],
                    [3.0*np.cos(np.pi/180*60),3.0*np.sin(np.pi/180*60),0],
                    [3.0*np.cos(np.pi/180*120),3.0*np.sin(np.pi/180*120),0],
                    [3.0*np.cos(np.pi/180*180),3.0*np.sin(np.pi/180*180),0],
                    [3.0*np.cos(np.pi/180*240),3.0*np.sin(np.pi/180*240),0],
                    [3.0*np.cos(np.pi/180*300),3.0*np.sin(np.pi/180*300),0]])

    rclpy.init(args=None)

    offboard_mission = OffboardMission(N_drone,WPTs,Formation,ref_lla)

    rclpy.spin(offboard_mission)

    offboard_mission.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()