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


DEBUG = True


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


if __name__ == '__main__':
    connector = CarlaConnector(track_name='L_track_barc')
    connector.spawn_camera()
    connector.test()
