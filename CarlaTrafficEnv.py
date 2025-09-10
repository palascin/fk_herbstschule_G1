"""
CarlaTrafficEnv.py

This module implements the Traffic environment connected to Carla. It does not spawn Carla itself but connects to it given the correct port. 
"""

import carla
import random
import time
import torch
from torch.amp import autocast
import math
from global_route_planner import GlobalRoutePlanner
import numpy as np
from scipy.spatial import KDTree
from Vehicle import Vehicle
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
    return math.sqrt(x[0] ** 2 + x[1] ** 2)

class Carla_Traffic_Env():

    def __init__(self, port=2000, map='Town01_Opt', CNN=None, num_gpu=0, veh_id=0, cuda=False):

        # Set Space Dimensions
        self.action_dim = 2
        self.observation_dim = 21
        self.CNN = CNN
        self.num_gpu = num_gpu
        self.cam_id = veh_id
        self.num_agents = 1
        self.cuda = cuda

        # Initialize Client
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(60.0)

        # Set realtime simulation
        self.realtime = False

        # Load the world and change settings
        self.world = self.client.load_world(map, map_layers=carla.MapLayer.NONE) 
        self.world.load_map_layer(carla.MapLayer.All)
        self.map = self.world.get_map()
        self.original_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 1.0 / 10.0  # 20 Hz
        self.settings.max_substep_delta_time = 1.0 / 150.0  # 100 Hz für Substeps -> empfohlenes Minimum
        self.settings.max_substeps = int(self.settings.fixed_delta_seconds / self.settings.max_substep_delta_time + 1)
        self.settings.no_rendering_mode = False
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        self.new_settings = self.settings
        self.world.set_weather(carla.WeatherParameters.CloudySunset)

        # Get Actor Blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.find('vehicle.lincoln.mkz_2020')

        # Get Sensor Blueprints
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.imu_bp = self.blueprint_library.find('sensor.other.imu')

        # Configure camera
        self.camera_bp.set_attribute('image_size_x', '512')  # old: 720
        self.camera_bp.set_attribute('image_size_y', '256')  # old: 510
        self.camera_bp.set_attribute('fov', '105')  # old: 95
        self.camera_bp.set_attribute('sensor_tick', '0.001')

        # Compute routing information
        self.spawn_points = random.sample(self.map.get_spawn_points(), min(25, len(self.map.get_spawn_points())))
        self.spawn_point = self.spawn_points[0]
        self.routes = []  # All routes
        self.route = []  # Current route
        self.trees = []  # All KD Trees
        self.tree = None  # Current KD Tree
        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2)
        self.get_routes()

        # Unload unnecessary layers for highest performance
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        #self.world.unload_map_layer(carla.MapLayer.Buildings)
        #self.world.unload_map_layer(carla.MapLayer.Foliage)
        #self.world.unload_map_layer(carla.MapLayer.Walls)

        # Initialize actors
        self.agents = []
        self.spectator = self.world.get_spectator()

        # Environment state
        self.images = torch.rand(self.num_agents, 3, 256, 512, dtype=torch.float16)
        self.vec_obs = torch.rand(self.num_agents, self.observation_dim)
        self.done = [False for _ in range(self.num_agents)]
        self.episode = 0
        t1=time.time()

        # Warm-up
        while time.time()-t1<5:
            self.world.tick()

    def set_route(self, route, tree):
        self.routes = [route]
        self.trees = [tree]
        
    def get_routes(self):
        def calc_route_length():
            length = 0
            for i in range(len(route) - 1):
                length += route[i].transform.location.distance(route[i + 1].transform.location)
            return length

        def delete_doubles(route):
            last_loc = [route[0].transform.location.x, route[0].transform.location.y]
            unique_route = [route[0]]
            for wp in route[1:]:
                loc = [wp.transform.location.x, wp.transform.location.y]
                if my_2d_norm(np.array(loc) - np.array(last_loc)) > .3:
                    unique_route.append(wp)
                    last_loc = loc
            return unique_route

        print('Processing Routing...')
        self.routes = []
        self.trees = []
        for i, spawn_point in enumerate(self.spawn_points):
            for goal in range(len(self.spawn_points)):
                if self.spawn_points[goal].location.distance(spawn_point.location) >= 20:
                    route = self.grp.trace_route(spawn_point.location, self.spawn_points[goal].location)
                    if len(route) <= 1:
                        pass
                    else:
                        rotation = (route[0])[0].transform.rotation
                        route = delete_doubles([elem[0] for elem in route])
                        x, y, z = (route[0]).transform.location.x, (route[0]).transform.location.y, (route[0]).transform.location.z
                        len_route = calc_route_length()
                        if 300 <= len_route:
                            self.routes.append((carla.Transform(carla.Location(x=x, y=y, z=z + .5), rotation), route))
                            coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in route])
                            self.trees.append(KDTree(coords))
        print('Routing completed')


    def reset_sync_clocks(self):
        """Reset synchronous mode to realign clocks"""
        try:
            # Disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            # Wait a moment for async mode to settle
            import time
            time.sleep(0.1)
            
            # Re-enable synchronous mode with fresh timing
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1/10  # Reset to your desired timestep
            self.world.apply_settings(settings)
            
            # Give it a tick to stabilize
            self.world.tick()
            
            print("Synchronous mode reset")
            return True
        
        except Exception as e:
            print(f"Error resetting sync mode: {e}")
            return False
    
    def cleanup(self):
        # Cleanup
        self.destroy_actors()
        del self.agents[:]

        def debug_gc():
            for obj in gc.get_objects():
                if isinstance(obj, Vehicle):
                    print(f"[GC DEBUG] Vehicle object still in memory: {id(obj)}")

        gc.collect()
        self.world.tick()

    def reset(self, max_speed=None):
        '''Reset the environment'''

        # Destroy all agents and synchronize clocks
        self.cleanup()
        self.reset_sync_clocks()
        self.num_agents = 1
        
        # Reinitialize Agents and Environment Information
        self.agents = [
            Vehicle(i, self.world, [self.vehicle_bp, self.camera_bp, self.imu_bp], self, self.cuda) for i
            in range(self.num_agents)]
        self.done = [False for _ in range(self.num_agents)]
        del self.images
        self.images = None
        for agent in self.agents:
            agent.reset(max_speed)
            self.world.tick()
            time.sleep(0.01)

        # Wait for the image
        current_img = self.agents[0].latest_image
        t = 0
        while self.agents[0].latest_image <=current_img:
            self.world.tick()
            time.sleep(.0001)
            t += 1
            if t > 50000:
                print("Sync Clocks")
                self.reset_sync_clocks()

        # Compute observations and auxiliary data
        obs, _, all_stats = self.get_obs()
        self.get_images()
        self.episode += 1
        with autocast('cuda'):
            img_features = self.CNN(self.images)
            if self.cuda:
                return torch.cat((obs.cuda(), img_features.detach()), dim=1), all_stats
        return torch.cat((obs, img_features.detach()), dim=1), all_stats

    def step(self, actions):
        '''Perform a step in the environment'''

        # Save time for realtime fps correction
        start_time = time.time()

        # Set the chosen action
        for i, vehicle in enumerate(self.agents):
            if self.num_agents > 1:
                vehicle.set_action(actions[i, :])
            else:
                vehicle.set_action(actions)

        # Perform a simulation step
        self.world.tick()

        # Wait for the image
        t=0
        current_img = self.agents[0].latest_image
        while self.agents[0].latest_image <=current_img:
            time.sleep(.0001)
            t += 1
            if t > 100000:
                print("Sync Clocks")
                self.reset_sync_clocks()
                return 0,0,0,0,True
                
        # Compute rewards
        rewards = self.compute_rewards(actions)

        # Gather observations
        obs, done, all_stats = self.get_obs()
        self.done = done
        self.get_images()

        # Time correction step for realtime rendering
        if self.realtime:
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.settings.fixed_delta_seconds - elapsed_time)
            time.sleep(sleep_time)

        # Extract image features, concatenate and return data
        with autocast('cuda'):
            img_features = self.CNN(self.images)
            if self.cuda:
                return torch.cat((obs.cuda(), img_features.detach()), dim=1), rewards, done, all_stats, False
        return torch.cat((obs, img_features.detach()), dim=1), rewards, done, all_stats, False

    def get_obs(self):
        ''' Get current observation '''

        # Setting the camera for illustrative purposes
        def set_camera():
            transform = self.agents[self.cam_id].carla_vehicle.get_transform()
            heading = yaw_to_vector(transform.rotation.yaw)
            position = transform.location - carla.Location(x=7 * heading[0], y=7 * heading[1], z=-5)
            rotation = carla.Rotation(yaw=transform.rotation.yaw, pitch=-20)
            self.spectator.set_transform(carla.Transform(position, rotation))
        set_camera()

        # Initialize Observations
        obs = torch.zeros((self.num_agents, self.observation_dim))
        all_stats = []
        done = []

        # Get observation and other information from vehicle
        for i, vehicle in enumerate(self.agents):
            obs[i, :], done_agent, stats = vehicle.get_obs()
            all_stats.append(stats)
            done.append(done_agent)
        all_stats = np.array(all_stats)
        
        return obs, done[0], all_stats

    def get_images(self):
        '''Get the current image from vehicle'''
        self.images = torch.stack([agent.image for agent in self.agents], dim=0)

    def compute_rewards(self, actions):
        '''Compute Rewards (for all agents)'''
        rewards = np.zeros((self.num_agents,))
        for i, _ in enumerate(self.agents):
            if self.num_agents > 1:
                rewards[i] = float(self.compute_reward_for_agent(actions[i, :]))
            else:
                rewards[i] = float(self.compute_reward_for_agent(actions))

        return rewards

    def compute_reward_for_agent(self, action):
        '''Compute reward per agent'''

        reward = 0

        #1. Kriterium: schnell
        # Aufteilen des Geschwindigkeitsvektors in "Wunschrichtung" und orthogonale Richtung

        velocity = self.agents[0].velocity
        _, index, dist = self.agents[0].estimate_dist_to_route()
        nearest_waypoint = self.agents[0].route[index]
        tangential_vector = yaw_to_vector(nearest_waypoint.transform.rotation.yaw)
        tangential_component = np.dot(velocity, tangential_vector)
        orthogonal_component = np.abs(np.cross(velocity, tangential_vector))

        reward += 5*(tangential_component - orthogonal_component) / 30

        #2. Kriterium: genau
        #quadratische Distanz zur Linie
        reward -= 8*dist * (my_2d_norm(velocity)+1) / 30
        #wechselnde Lenkbewegungen bestrafen
        reward -= np.abs(action[1])

        reward /= 10

        return reward

    def destroy_actors(self):
        '''Destroy all vehicles'''
        for agent in self.agents:
            agent.destroy()

    def render_waypoints(self, waypoints, color):
        '''Print the given waypoints vehicles'''
        for wp in waypoints:
            wp_location = wp.transform.location
            self.world.debug.draw_point(wp_location, .09, color=color,
                                        life_time=0.3)  
