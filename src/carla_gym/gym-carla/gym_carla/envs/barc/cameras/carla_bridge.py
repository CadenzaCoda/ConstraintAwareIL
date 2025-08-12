import os
import sys

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
# import cv2

from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState

import carla
from pathlib import Path
import time

import pygame
import skimage
from tqdm import tqdm


DEBUG = False


def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface


class CarlaConnector:
    def __init__(self, track_name, host='localhost', port=2000):
        # self.client = carla.Client('localhost', 2000)
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.obs_size = 224
        self.dt = 0.1

        self.camera_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.world = None
        self.camera_bp = None
        self.track_name = track_name
        self.track_obj = get_track(track_name)

        self.load_opendrive_map()
        self.spawn_camera()
        # self.last_check = time.time()
        # self.check_freq = 10.
        self.env_steps = 0
        
        # Temporary: built-in rendering
        if DEBUG:
            pygame.init()
            self.display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.surface = pygame.Surface((self.obs_size, self.obs_size))
            self.clock = pygame.time.Clock()
            # self.fig, self.ax = plt.subplots()
            # self.im = self.ax.imshow(self.camera_img)
    
    @property
    def height(self):
        return self.camera_img.shape[0]
    
    @property
    def width(self):
        return self.camera_img.shape[1]

    def load_opendrive_map(self):
        xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{self.track_name}.xodr"
        if not os.path.exists(xodr_path):
            raise ValueError(f"The file {xodr_path} does not exist.")
            return
    
        with open(xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be read.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0  # in meters
        max_road_length = 0.1  # in meters
        wall_height = 0.2      # in meters
        extra_width = 0.1       # in meters
        self.world = self.client.generate_opendrive_world(
                        data, carla.OpendriveGenerationParameters(
                            vertex_distance=vertex_distance,
                            max_road_length=max_road_length,
                            wall_height=wall_height,
                            additional_width=extra_width,
                            smooth_junctions=True,
                            enable_mesh_visibility=True))
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=5, y=-5, z=10),
                                                carla.Rotation(pitch=-45, yaw=-45)))
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

    def destroy_camera(self):
        for actor in self.world.get_actors().filter('sensor.camera.rgb'):
            actor.destroy()

    def spawn_camera(self, x=0., y=0., psi=0.):
        # Remove any previous cameras.
        self.destroy_camera()
        # Next, try to spawn a camera at the origin.
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        self.camera_trans = carla.Transform(carla.Location(x=x, y=y, z=0.2),
                                            carla.Rotation(yaw=-np.rad2deg(psi)))
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans)
        self.camera_sensor.listen(get_camera_img)
    
    def query_rgb(self, state):
        self.env_steps += 1
        if self.env_steps % 102_400 == 0:
            self.destroy_camera()
            self.load_opendrive_map()
            self.spawn_camera()
        # if time.time() - self.last_check > self.check_freq:
        #     self.last_check = time.time()
        #     if self.client.get_world().get_map().name != 'Carla/Maps/OpenDriveMap':
        #         self.load_opendrive_map()
        #         self.spawn_camera()
        self.camera_sensor.set_transform(carla.Transform(carla.Location(x=state.x.x, y=-state.x.y, z=0.2), 
                                                         carla.Rotation(yaw=-np.rad2deg(state.e.psi))))
        # attempt = 0
        # while True:
        #     try:
        self.world.tick()
        if DEBUG:
            # surface = rgb_to_display_surface(self.camera_img, 256)
            pygame.surfarray.blit_array(self.surface, self.camera_img.swapaxes(0, 1))
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
            # self.im.set_array(self.camera_img)
            # plt.pause(0.01)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
        return self.camera_img
        # except RuntimeError as e:
        #     logger.error(e)
        #     logger.error("Waiting for CARLA to restart...")
        #     time.sleep(20)
        #     self.load_opendrive_map()
        #     self.spawn_camera(x=state.x.x, y=state.y.y, psi=state.e.psi)
        #     attempt += 1
        #     if attempt > 5:
        #         raise RuntimeError(e)

        # Built-in rendering
        # cv2.imshow('RGB Camera', self.camera_img[:, :, ::-1])
        # if cv2.waitKey(1) == ord('q'):
        #     exit(0)

    
    def test(self):
        # fig, ax = plt.subplots()
        # im = ax.imshow(self.camera_img)
        state = VehicleState()
        state.p.x_tran = 0.55
        
        while True:
            # im.set_array(self.camera_img)
            state.p.s = (state.p.s + 0.1) % self.track_obj.track_length

            self.track_obj.local_to_global_typed(state)
            self.query_rgb(state)
            # plt.pause(0.1)
            # fig.canvas.draw()
            # fig.canvas.flush_events()


class CarlaConnectorScaled:
    def __init__(self, track_name='L_track_barc_scaled_10x', host='localhost', port=2000, upsample=1):
        # self.client = carla.Client('localhost', 2000)
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.obs_size = 224
        self.upsample = upsample
        self.dt = 0.1 / upsample

        self.camera_img = np.empty((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.world = None
        self.camera_bp = None
        self.vehicle_bp = None
        self.track_name = track_name
        self.track_obj = get_track(track_name)  # Use original track for calculations

        self.load_opendrive_map()
        self.spawn_vehicle_with_camera()
        self.setup_spectator()
        # self.last_check = time.time()
        # self.check_freq = 10.
        self.env_steps = 0
        
        # Temporary: built-in rendering
        if DEBUG:
            pygame.init()
            self.display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.surface = pygame.Surface((self.obs_size, self.obs_size))
            self.clock = pygame.time.Clock()
            # self.fig, self.ax = plt.subplots()
            # self.im = self.ax.imshow(self.camera_img)
    
    @property
    def height(self):
        return self.camera_img.shape[0]
    
    @property
    def width(self):
        return self.camera_img.shape[1]

    def load_opendrive_map(self):
        xodr_path = Path(__file__).resolve().parents[1] / 'OpenDrive' / f"{self.track_name}_scaled_10x.xodr"
        if not os.path.exists(xodr_path):
            raise ValueError(f"The file {xodr_path} does not exist.")
            return
    
        with open(xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be read.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0 * 10  # in meters
        max_road_length = 0.1 * 10  # in meters
        wall_height = 0.2 * 10      # in meters
        extra_width = 0.1 * 10       # in meters
        self.world = self.client.generate_opendrive_world(
                        data, carla.OpendriveGenerationParameters(
                            vertex_distance=vertex_distance,
                            max_road_length=max_road_length,
                            wall_height=wall_height,
                            additional_width=extra_width,
                            smooth_junctions=True,
                            enable_mesh_visibility=True))
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

    def setup_spectator(self):
        """Set up spectator at appropriate location to see the entire scaled track"""
        spectator = self.world.get_spectator()
        # Position spectator high above and looking down at the track center
        # Since track is scaled by 10x, we need to position accordingly
        spectator.set_transform(carla.Transform(
            carla.Location(x=-30, y=30, z=50),  # Scaled up from original (5, -5, 10)
            carla.Rotation(yaw=-60, pitch=-40)
        ))

    def destroy_vehicle_and_camera(self):
        """Destroy both vehicle and camera actors"""
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('sensor.camera.rgb'):
            actor.destroy()

    def spawn_vehicle_with_camera(self, x=0., y=0., psi=0.):
        """Spawn a vehicle with camera attached instead of just a camera"""
        # Remove any previous vehicles and cameras
        self.destroy_vehicle_and_camera()
        
        # Get vehicle blueprint (Tesla Model 3)
        self.vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        if not self.vehicle_bp:
            # Fallback to any available vehicle
            self.vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        
        # Set vehicle attributes
        self.vehicle_bp.set_attribute('role_name', 'hero')
        
        # Spawn vehicle
        vehicle_trans = carla.Transform(carla.Location(x=x * 10.0, y=-y * 10.0, z=3.0),  # z=0.5 to place on ground
                                        carla.Rotation(yaw=-np.rad2deg(psi)))
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, vehicle_trans)
        
        # Get camera blueprint and attach to vehicle
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Attach camera to vehicle (slightly above and behind)
        camera_trans = carla.Transform(carla.Location(x=2.5, z=0.5),  # Behind and above vehicle
                                        carla.Rotation(pitch=0))  # Look slightly down
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, camera_trans, attach_to=self.vehicle)
        self.camera_sensor.listen(get_camera_img)
        for _ in range(100):
            self.world.tick()
        logger.debug(f"Vehicle position: {self.vehicle.get_location()}")
        self.vehicle.set_simulate_physics(False)
    
    def query_rgb(self, state):
        """Query RGB image by teleporting the vehicle instead of just the camera"""
        self.env_steps += 1
        if self.env_steps % 102_400 == 0:
            self.destroy_vehicle_and_camera()
            self.load_opendrive_map()
            self.spawn_vehicle_with_camera()
        
        # Teleport the vehicle to the new position (scaled by 10x)
        # Since track is scaled by 10x, we need to scale the coordinates accordingly
        scaled_x = state.x.x * 10.0
        scaled_y = -state.x.y * 10.0  # Note: y-axis is inverted in CARLA
        
        # Set vehicle transform (this teleports the vehicle)
        vehicle_trans = carla.Transform(
            carla.Location(x=scaled_x, y=scaled_y, z=0.0),
            carla.Rotation(yaw=-np.rad2deg(state.e.psi))
        )
        self.vehicle.set_transform(vehicle_trans)
        
        # Tick the world to update
        self.world.tick()
        
        if DEBUG:
            # surface = rgb_to_display_surface(self.camera_img, 256)
            pygame.surfarray.blit_array(self.surface, self.camera_img.swapaxes(0, 1))
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
            # self.im.set_array(self.camera_img)
            # plt.pause(0.01)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
        
        return self.camera_img

    def test(self):
        """Test method for the scaled connector"""
        state = VehicleState()
        state.p.x_tran = 0.
        
        while True:
            state.p.s = (state.p.s + 0.1) % self.track_obj.track_length

            self.track_obj.local_to_global_typed(state)
            self.query_rgb(state)
            time.sleep(0.05)

    def playback(self, trajectory, hz=10):
        state = VehicleState()
        t = np.linspace(0, 1, trajectory.shape[0])
        t_upsampled = np.linspace(0, 1, trajectory.shape[0] * self.upsample)
        trajectory_upsampled = np.stack([np.interp(t_upsampled, t, trajectory[:, i]) for i in range(trajectory.shape[1])], axis=1)
        hz_upsampled = hz * self.upsample
        for x, y, psi in tqdm(trajectory_upsampled, total=trajectory_upsampled.shape[0], desc='Playback'):
            state.x.x, state.x.y, state.e.psi = x, y, psi
            time_s = time.time()
            self.query_rgb(state)
            time.sleep(max(0, 1 / hz_upsampled - (time.time() - time_s)))
    

if __name__ == '__main__':
    # Test original connector
    # print("Testing original connector...")
    # connector = CarlaConnector(track_name='L_track_barc')
    # connector.spawn_camera()
    # connector.test()
    
    # Test scaled connector
    print("Testing scaled connector...")
    scaled_connector = CarlaConnectorScaled(track_name='L_track_barc', upsample=3)
    # trajectory = np.load(Path(__file__).resolve().parents[7] / 'trajectories' / 'VisionSafeAC_L_track_barc_v1.2.3-lam1_230_evaluation_1.npz')['sts']
    trajectory = np.load(Path(__file__).resolve().parents[7] / 'trajectories' / 'VisionSafeAC_L_track_barc_visionSafeAC-v1.1.7-lam0_evaluation_3_evaluation_1.npz')['sts']
    scaled_connector.playback(trajectory, hz=10)
    # scaled_connector.test()
