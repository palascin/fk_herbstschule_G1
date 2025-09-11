"""
Vehicle.py

This module implements a vehicle class for the CarlaTrafficEnvironment module.  
"""

import numpy as np
import os
import torch
import carla
import time
import math
import gc
from numba import njit

# Precompiled fast helper functions

@njit
def coord_transform(vec, anc):
        onm = np.zeros((2, 2))
        det = -anc[0] ** 2 - anc[1] ** 2
        onm[0, 0] = -anc[0]
        onm[1, 1] = anc[0]
        onm[0, 1] = -anc[1]
        onm[1, 0] = -anc[1]
        return onm.dot(vec.transpose()).transpose() / det
        
@njit  
def yaw_to_vector(yaw):
        yaw_radians = math.radians(yaw)
        return np.array([math.cos(yaw_radians), math.sin(yaw_radians)])
        
@njit   
def my_2d_norm(x):
    return math.sqrt(x[0]**2+x[1]**2)
        
class Vehicle():

    def __init__(self, id, world, bps, env, cuda):
        self.id = id
        self.env = env
        self.sensors = []
        self.world = world
        self.cuda = cuda
        self.bps = bps
        self.carla_vehicle = None
        self.route = None
        self.tree = None
        self.yet_to_take_image = False
        self.taking_image = False
        self.max_speed = np.random.uniform(10, 100) / 3.6
        self.steer = 0
        self.acc = 0
        self.location = np.zeros((2,))
        self.location_before = np.zeros((2,))
        self.heading = np.zeros((2,))
        self.velocity = np.zeros((2,))
        self.imu_data = np.zeros((5,))
        self.images_captured = 0
        self.image = None
        self.latest_image = 0
        self.done = False
        self.camera = None
        self.last_wp_idx = 0
        self.imu_sensor = None
        self.episode = 0
    
    def reset(self, max_speed=None):
        '''Assign a random route and reset vehicle'''

        # Reset attributes
        self.steer = 0
        self.acc = 0
        self.velocity = np.zeros((2,))
        self.imu_data = np.zeros((5,))
        self.collided = False
        self.images_captured = 0
        self.image = None
        self.yet_to_take_image = True
        self.collision_reward = 0
        self.done = False
        self.latest_image = 0
        self.max_speed = np.random.uniform(10, 100) / 3.6
        if max_speed is not None:
            self.max_speed = max_speed
        self.last_wp_idx = 0


        # Spawn vehicle and sensors
        while self.carla_vehicle is None:
            index = np.random.choice(range(len(self.env.routes)))
            spawn_point, self.route = self.env.routes[index]
            self.tree = self.env.trees[index]
            self.carla_vehicle = self.world.try_spawn_actor(self.bps[0], spawn_point)

        self.world.tick()

        while self.camera is None:
            self.camera = self.world.try_spawn_actor(self.bps[1],
                                                 carla.Transform(carla.Location(x=-5, z=5.5), carla.Rotation(pitch=-31)), # 3.3, 5, -29
                                                 attach_to=self.carla_vehicle)
        self.camera.listen(lambda image: self.capture_image(image))

        self.world.tick()

        while self.imu_sensor is None:
            self.imu_sensor = self.world.try_spawn_actor(self.bps[2], carla.Transform(carla.Location(x=0, y=0, z=.5)),
                                                attach_to=self.carla_vehicle)

        self.imu_sensor.listen(lambda data: self.imu_listener(data))

    def set_action(self, action):
        '''Set the new steering and acceleration controls'''

        action[1] = float(action[1]) *2
        control = carla.VehicleControl(throttle=0, steer=0, brake=0)

        control.steer = max(min(self.steer + float(action[0]),1),-1)

        if self.acc + float(action[1]) > 0:
            control.throttle = min(self.acc + float(action[1]),1)
        else:
            control.brake = -1*max(self.acc + float(action[1]),-1)

        self.acc = control.throttle - control.brake
        self.steer = control.steer
        
        self.carla_vehicle.apply_control(control)

    def get_location(self):
        return self.location
        
    def get_obs(self):
        # Initialize Observations

        obs = np.zeros((self.env.observation_dim,), dtype=np.float16)
        # observation_dim in CarlaTrafficEnv auf 21 gesetzt

        # Gather Data
        gear = self.carla_vehicle.get_control().gear
        transform = self.carla_vehicle.get_transform()
        self.location_before = self.location
        location = transform.location
        self.location = np.array([location.x, location.y])
        velocity = self.carla_vehicle.get_velocity()
        velocity = np.array([velocity.x, velocity.y])
        self.velocity = velocity
        steer = self.steer
        angular_velocity_x, angular_velocity_y, angular_velocity_z, acceleration_x, acceleration_y = self.imu_data
        throttle = max(self.acc,0)
        brake = - min(self.acc,0)
        self.heading = yaw_to_vector(transform.rotation.yaw)
        _, index, dist = self.estimate_dist_to_route()
        self.env.render_waypoints(self.route[index:min(index + 40, len(self.route) - 1)], carla.Color(r=255,g=0,b=0))
        dist_to_goal = location.distance(self.route[-1].transform.location)
        # Implement observations
        rel_velocity = coord_transform(velocity, self.heading)
        rel_angular_velocity_xy = coord_transform(np.array([angular_velocity_x, angular_velocity_y]), self.heading)
        rel_acceleration_xy = coord_transform(np.array([ acceleration_x, acceleration_y]), self.heading)

        # normalisieren notwendig
        obs[:2] = self.location / 150 # -150 bis 150
        obs[2:4] = velocity / 75 # -27 bis 27
        obs[4] = rel_velocity[0] / 35 - 1 # 0 bis 27
        obs[5] = rel_velocity[1]  # nach trainieren anpassen
        obs[6] = angular_velocity_x * 20 # -1 bis 1, aber eher klein
        obs[7] = angular_velocity_y * 20 # ...
        obs[8] = angular_velocity_z * 2 # ...
        obs[9:11] = rel_angular_velocity_xy # -1 bis 1
        obs[11] = acceleration_x / 10 # -1,5g bis 1g
        obs[12] = acceleration_y / 10 # ...
        obs[13:15] = rel_acceleration_xy / 10 # ...
        obs[15] = steer # -1 bis 1
        obs[16] = 2 * throttle - 1 # 0 bis 1
        obs[17] = 2 * brake - 1 # 0 bis 1
        obs[18:20] = self.heading # normierter Vektor
        obs[20] = self.max_speed / 15 - 1 #2,7 bis 27
        obs[21] = (self.max_speed - my_2d_norm(velocity)) / self.max_speed
        obs[22] = (self.max_speed - my_2d_norm(velocity)) / 15
        obs[23] = gear

        if self.cuda:
            obs = torch.as_tensor(obs, dtype=torch.float16).cuda()
        else:
            obs = torch.as_tensor(obs)

        stats = [0, dist, math.sqrt(velocity[0]**2+velocity[1]**2)]

        self.done = self.done or dist>7 or dist_to_goal<3
        
        return obs, self.done, stats

    def imu_listener(self, imu_data):
        acceleration = imu_data.accelerometer
        angular_velocity = imu_data.gyroscope  # Winkelgeschwindigkeit (rad/s)
        self.imu_data = np.array(
            [angular_velocity.x, angular_velocity.y, angular_velocity.z, acceleration.x, acceleration.y])

    def capture_image(self, image):

        image_width = image.width
        image_height = image.height
        
        # Convert image to numpy array
        image.convert(carla.ColorConverter.Raw)
        img_array = np.array(image.raw_data, dtype=np.uint8)
        img_array = np.reshape(img_array, (image_height, image_width, 4))[:, :, :3]

        array = torch.from_numpy(img_array)
        if self.cuda:
            array = array.half().cuda()
        self.image = array.permute((2, 0, 1))[[2, 1, 0], :, :] / 255.0#array.reshape((image_h, image_w, 4))[:, :, :3].permute((2, 1, 0)) / 255.0
        #time.sleep(0.001)
        self.latest_image = image.frame
        self.yet_to_take_image = False
        self.images_captured += 1
        self.taking_image = False

    def destroy(self):
        if self.camera is not None:  
            cam_id = self.camera.id 
            self.camera.stop()
            self.camera.destroy()
            time.sleep(1)
            for i in range(3):
                self.world.tick()
                if not self.world.get_actor(cam_id).is_alive:
                    print(f"Cam with id {cam_id} destroyed after {i/10}s ")
                    break
                time.sleep(0.1)
            self.camera = None
            gc.collect()
            

        if self.imu_sensor is not None:
            self.imu_sensor.stop()
            self.imu_sensor.destroy()
            self.imu_sensor = None

        if self.carla_vehicle is not None:
            self.carla_vehicle.destroy()

        self.carla_vehicle = None
        self.world.tick()
        gc.collect()

    def estimate_dist_to_route(self):

        #veh = self.carla_vehicle
        #transform = veh.get_transform()
        #location = transform.location
        distances, indices = self.tree.query(self.location, k=3)
        #heading = yaw_to_vector(transform.rotation.yaw)
        filtered_indices = [id for id in indices if abs(id - self.last_wp_idx) <= 10]
        if filtered_indices:
            indices = filtered_indices
        index = indices[0]
        wp = self.route[index]

        vec_dist = 1000
        for id in indices:
            vec = coord_transform(np.array([self.route[id].transform.location.x - self.location[0],
                                                 self.route[id].transform.location.y - self.location[1]]), self.heading)
            loc1 = self.location#np.array([location.x, location.y])
            loc2 = np.array([self.route[id].transform.location.x, self.route[id].transform.location.y])
            this_dist = my_2d_norm(loc1 - loc2)
            if vec[0] >= 0 and this_dist <= vec_dist:
                index = id
                vec_dist = this_dist
        self.last_wp_idx = index
        wp_before = np.array([self.route[max(indices[0] - 1, 0)].transform.location.x,
                              self.route[max(indices[0] - 1, 0)].transform.location.y])
        wp_nearest = np.array([wp.transform.location.x, wp.transform.location.y])
        wp_after = np.array([self.route[min(indices[0] + 1, len(self.route) - 1)].transform.location.x,
                             self.route[min(indices[0] + 1, len(self.route) - 1)].transform.location.y])
        dist = 1000
        for [wp1, wp2] in [[wp_before, wp_nearest], [wp_nearest, wp_after]]:
            # Vector from w1 to w2
            v = wp2 - wp1
            # Vector from w1 to x
            wp1_to_x = self.location - wp1
            if my_2d_norm(wp1_to_x) == 0:
                return self.route[index], index, 0

            if not my_2d_norm(v) == 0:
                t = np.dot(wp1_to_x, v) / np.dot(v, v)
            else:
                t = 0

            # Clamp t to the range [0, 1] to stay within the segment
            t = max(0, min(1, t))

            # Find the closest point on the segment to x
            closest_point = wp1 + t * v

            # Return the distance from x to the closest point
            dist = min(my_2d_norm(self.location - closest_point), dist)

        return self.route[index], index, dist