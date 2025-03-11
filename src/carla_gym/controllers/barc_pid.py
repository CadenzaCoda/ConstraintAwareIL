#!/usr/bin/env python3
from loguru import logger
from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.utils.controllerTypes import PIDParams

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, \
    BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.track import get_track

import pdb

import numpy as np


class PIDWrapper:
    def __init__(self, dt=0.1, t0=0., track_obj=None, noise=False, VL=0.37, VW=0.195):
        # Input type: ndarray, local frame
        self.pid_controller = None
        self.t = None

        self.dt = dt
        self.noise = noise
        self.track_obj = track_obj
        self.t0 = t0
        dynamics_config = DynamicBicycleConfig(dt=dt,
                                               model_name='dynamic_bicycle_cl',
                                               noise=False,
                                               discretization_method='rk4',
                                               simple_slip=False,
                                               tire_model='pacejka',
                                               mass=2.2187,
                                               yaw_inertia=0.02723,
                                               wheel_friction=0.9,
                                               pacejka_b_front=5.0,
                                               pacejka_b_rear=5.0,
                                               pacejka_c_front=2.28,
                                               pacejka_c_rear=2.28,
                                               code_gen=True,
                                               jit=True,
                                               opt_flag='O3')
        self.dyn_model = CasadiDynamicCLBicycle(t0, dynamics_config, track=track_obj)
        self.state_input_ub = VehicleState(p=ParametricPose(s=2 * self.track_obj.track_length, x_tran=self.track_obj.half_width - VW / 2, e_psi=100),
                                           v=BodyLinearVelocity(v_long=10, v_tran=10),
                                           w=BodyAngularVelocity(w_psi=10),
                                           u=VehicleActuation(u_a=2.0, u_steer=0.436))
        self.state_input_lb = VehicleState(p=ParametricPose(s=-2 * self.track_obj.track_length, x_tran=-(self.track_obj.half_width - VW / 2), e_psi=-100),
                                           v=BodyLinearVelocity(v_long=-10, v_tran=-10),
                                           w=BodyAngularVelocity(w_psi=-10),
                                           u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
        self.input_rate_ub = VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
        self.input_rate_lb = VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))
        self.lmpc_params = dict(
            N=15,
            n_ss_pts=48,
            n_ss_its=4,
        )
        self.prediction = VehiclePrediction()
        self.safe_set = VehiclePrediction()

    def setup_pid_controller(self):
        pid_steer_params = PIDParams(dt=self.dt,
                                     Kp=0.5,
                                     u_max=self.state_input_ub.u.u_steer,
                                     u_min=self.state_input_lb.u.u_steer,
                                     du_max=self.input_rate_ub.u.u_steer,
                                     du_min=self.input_rate_lb.u.u_steer,
                                     x_ref=0.0,
                                     noise=False,
                                     noise_max=0.2,
                                     noise_min=-0.2)
        pid_speed_params = PIDParams(dt=self.dt,
                                     Kp=1.5,
                                     u_max=self.state_input_ub.u.u_a,
                                     u_min=self.state_input_lb.u.u_a,
                                     du_max=self.input_rate_ub.u.u_a,
                                     du_min=self.input_rate_lb.u.u_a,
                                     x_ref=1.0,
                                     noise=False,
                                     noise_max=0.9,
                                     noise_min=-0.9)
        self.pid_controller = PIDLaneFollower(self.dt, pid_steer_params, pid_speed_params)

    def _step_pid(self, _state: VehicleState):
        self.pid_controller.step(_state)
        self.track_obj.global_to_local_typed(_state)  # Recall that PID uses the global frame.
        _state.p.s = np.mod(_state.p.s, self.track_obj.track_length)
        return {'success': True, 'status': 0}  # PID controller never fails.

    def reset(self, *, seed=None, options=None):
        self.setup_pid_controller()
        self.t = self.t0

    def step(self, vehicle_state, terminated, lap_no, **kwargs):
        """
        Use VehicleState to step directly. Closer to how the simulation script works.
        """
        info = self._step_pid(vehicle_state)
        return np.array([vehicle_state.u.u_a, vehicle_state.u.u_steer]), info

    def get_prediction(self):
        return None

    def get_safe_set(self):
        return None
