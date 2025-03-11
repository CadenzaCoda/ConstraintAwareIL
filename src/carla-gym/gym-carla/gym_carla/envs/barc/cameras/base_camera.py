import numpy as np
from mpclab_common.pytypes import VehicleState


class BaseCamera:
    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    def query_rgb(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def query_depth(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def query_depth_raw(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError

    def query_lidar(self, state: VehicleState) -> np.ndarray:
        raise NotImplementedError
