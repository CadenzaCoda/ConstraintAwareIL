from abc import abstractmethod

import numpy as np
# from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import VehicleState, VehicleActuation


def obs2state(obs: np.ndarray) -> VehicleState:
    state = VehicleState()
    state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi = obs
    return state


def state2obs(state: VehicleState) -> np.ndarray:
    return np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])


class BaseController:
    def __init__(self):
        self.publishers = {}  # {topic_name: publisher}

    def publish(self, **kwargs):
        """
        Publishers publish the corresponding messages.
        **kwargs: topic_name=message
        """
        for topic_name, msg in kwargs.items():
            if topic_name not in self.publishers:
                self.get_logger().warn(f"Topic '{topic_name}' not registered.")
                continue
            self.publishers[topic_name].publish(msg)

    def step(self, state):
        ac = self._step(state2obs(state))
        self.publish()
        return VehicleActuation(u_a=ac[0], u_steer=ac[1])

    def _step(self, obs) -> np.ndarray:
        raise NotImplementedError
        # ac = ...
        # return ac
