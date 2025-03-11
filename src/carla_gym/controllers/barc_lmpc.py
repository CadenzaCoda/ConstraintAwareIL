#!/usr/bin/env python3
from loguru import logger
from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.CA_LMPC import CA_LMPC
from mpclab_controllers.utils.controllerTypes import CALMPCParams, PIDParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, \
    BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.track import get_track
# from mpclab_common.rosbag_utils import rosbagData

import pdb

import numpy as np
from scipy.interpolate import interp1d
import casadi as ca

import time
import copy

from collections import deque

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


class LMPCWrapper:
    def __init__(self, dt=0.1, t0=0., track_obj=None, noise=True, VL=0.37, VW=0.195):
        # Input type: ndarray, local frame
        self.active_controller = None
        self.step_active = None
        self.last_lap_data = None
        self.lap_data = None
        self.lmpc_controller = None
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

    def _process_lap_data(self, lap_data, lap_end=None):
        _t, _q, _u = [], [], []
        for s in lap_data:
            q, u = self.dyn_model.state2qu(s)
            _t.append(s.t)
            _q.append(q)
            _u.append(u)
        _q = np.array(_q)
        _u = np.array(_u)
        _t = np.array(_t)
        if lap_end is None:
            lap_end = _t[-1]
        _c2g = -(_t - lap_end) / self.dt

        q_interp = interp1d(_t, _q, kind='linear', axis=0, assume_sorted=True)
        u_interp = interp1d(_t, _u, kind='linear', axis=0, assume_sorted=True)
        c2g_interp = interp1d(_t, _c2g, kind='linear', axis=0, assume_sorted=True)

        tq = np.arange(_t[0], _t[-1], self.dt)
        tq[-1] = min(_t[-1], tq[-1])
        q = q_interp(tq)
        u = u_interp(tq)
        c2g = c2g_interp(tq)

        return q, u, c2g

    def setup_pid_controller(self):
        pid_steer_params = PIDParams(dt=self.dt,
                                     Kp=0.5,
                                     u_max=self.state_input_ub.u.u_steer,
                                     u_min=self.state_input_lb.u.u_steer,
                                     du_max=self.input_rate_ub.u.u_steer,
                                     du_min=self.input_rate_lb.u.u_steer,
                                     x_ref=0.0,
                                     noise=self.noise,
                                     noise_max=0.2,
                                     noise_min=-0.2)
        pid_speed_params = PIDParams(dt=self.dt,
                                     Kp=1.5,
                                     u_max=self.state_input_ub.u.u_a,
                                     u_min=self.state_input_lb.u.u_a,
                                     du_max=self.input_rate_ub.u.u_a,
                                     du_min=self.input_rate_lb.u.u_a,
                                     x_ref=1.0,
                                     noise=self.noise,
                                     noise_max=0.9,
                                     noise_min=-0.9)
        self.pid_controller = PIDLaneFollower(self.dt, pid_steer_params, pid_speed_params)

    def setup_lmpc_controller(self):
        N = self.lmpc_params['N']
        mpc_params = CALMPCParams(dt=self.dt,
                                  N=N,
                                  state_scaling=[4.0, 3.0, 7.0, 6.283185307179586, 20.0, 1.0],
                                  input_scaling=[2.0, 0.436],
                                  # delay=[1, 1],
                                  # convex_hull_slack_quad=[100, 1, 10, 1, 100, 10],
                                  convex_hull_slack_quad=[500, 500, 500, 500, 500, 500],
                                  # convex_hull_slack_quad=[400, 4, 40, 4, 400, 40],
                                  convex_hull_slack_lin=[0, 0, 0, 0, 0, 0],
                                  soft_state_bound_idxs=[5],
                                  # soft_state_bound_idxs=None,
                                  soft_state_bound_quad=[5],
                                  soft_state_bound_lin=[25],
                                  n_ss_pts=self.lmpc_params['n_ss_pts'],
                                  n_ss_its=self.lmpc_params['n_ss_its'],
                                  regression_regularization=1e-3,
                                  regression_state_out_idxs=[[0], [1], [2]],
                                  regression_state_in_idxs=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                                  regression_input_in_idxs=[[0], [1], [1]],
                                  nearest_neighbor_weights=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                  nearest_neighbor_bw=10.0,
                                  nearest_neighbor_max_points=25,
                                  wrapped_state_idxs=[4],
                                  wrapped_state_periods=[self.track_obj.track_length],
                                  debug_plot=False,
                                  verbose=False,
                                  keep_init_safe_set=False,
                                  qp_interface='casadi',
                                  qp_solver='osqp')
        # mpc_params = CALMPCParams(dt=dt,
        #                             N=N,
        #                             state_scaling=[4.0, 3.0, 7.0, 6.283185307179586, 20.0, 1.0],
        #                             input_scaling=[2.0, 0.436],
        #                             # delay=[1, 1],
        #                             # convex_hull_slack_quad=[100, 1, 10, 1, 100, 10],
        #                             convex_hull_slack_quad=[500, 500, 500, 500, 500, 500],
        #                             # convex_hull_slack_quad=[400, 4, 40, 4, 400, 40],
        #                             convex_hull_slack_lin=[0, 0, 0, 0, 0, 0],
        #                             soft_state_bound_idxs=[5],
        #                             # soft_state_bound_idxs=None,
        #                             soft_state_bound_quad=[5],
        #                             soft_state_bound_lin=[25],
        #                             n_ss_pts=n_ss_pts,
        #                             n_ss_its=n_ss_its,
        #                             regression_regularization=0.001,
        #                             regression_state_out_idxs=[[0], [1], [2]],
        #                             regression_state_in_idxs=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        #                             regression_input_in_idxs=[[0], [1], [1]],
        #                             nearest_neighbor_weights=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        #                             nearest_neighbor_bw=5.0,
        #                             nearest_neighbor_max_points=15,
        #                             wrapped_state_idxs=[4],
        #                             wrapped_state_periods=[L],
        #                             debug_plot=False,
        #                             verbose=False,
        #                             keep_init_safe_set=False)

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', self.dyn_model.n_q)
        sym_u = ca.MX.sym('u', self.dyn_model.n_u)
        sym_du = ca.MX.sym('du', self.dyn_model.n_u)

        ua_idx = 0
        us_idx = 1

        # sym_input_stage = 0.5*(1*(sym_u[ua_idx])**2 + 1*(sym_u[us_idx])**2)
        # sym_input_term = 0.5*(1*(sym_u[ua_idx])**2 + 1*(sym_u[us_idx])**2)
        # sym_rate_stage = 0.5*(5*(sym_du[ua_idx])**2 + 5*(sym_du[us_idx])**2)

        sym_input_stage = 0.5 * (0.5 * (sym_u[ua_idx]) ** 2 + 0.5 * (sym_u[us_idx]) ** 2)
        sym_input_term = 0.5 * (0.5 * (sym_u[ua_idx]) ** 2 + 0.5 * (sym_u[us_idx]) ** 2)
        sym_rate_stage = 0.1 * (1 * (sym_du[ua_idx]) ** 2 + 1 * (sym_du[us_idx]) ** 2)

        # sym_input_stage = 0.5*(0.1*(sym_u[ua_idx])**2 + 0.1*(sym_u[us_idx])**2)
        # sym_input_term = 0.5*(0.1*(sym_u[ua_idx])**2 + 0.1*(sym_u[us_idx])**2)

        # sym_input_stage = 0.5*(0.5*(sym_u[ua_idx])**2 + 0.5*(sym_u[us_idx])**2)
        # sym_input_term = 0.5*(0.5*(sym_u[ua_idx])**2 + 0.5*(sym_u[us_idx])**2)

        # sym_rate_stage = 0.005*(1*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)
        # sym_rate_stage = 0.01*(1*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)
        # sym_rate_stage = 0.05*(1*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)
        # sym_rate_stage = 0.1*(1*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)
        # sym_rate_stage = 1.0*(1*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)

        sym_costs = {'state': [None for _ in range(N + 1)], 'input': [None for _ in range(N + 1)],
                     'rate': [None for _ in range(N)]}
        for k in range(N):
            sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
            sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
        sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

        sym_constrs = {'state_input': [None for _ in range(N + 1)],
                       'rate': [None for _ in range(N)]}

        self.lmpc_controller = CA_LMPC(self.dyn_model,
                                       sym_costs,
                                       sym_constrs,
                                       {
                                           'qu_ub': self.state_input_ub,
                                           'qu_lb': self.state_input_lb,
                                           'du_ub': self.input_rate_ub,
                                           'du_lb': self.input_rate_lb
                                       },
                                       control_params=mpc_params)

        N = self.lmpc_params['N']
        u_ws = np.zeros((N + 1, self.dyn_model.n_u))
        # u_ws = np.tile(dyn_model.input2u(control), (N+1, 1))
        du_ws = np.zeros((N, self.dyn_model.n_u))
        self.lmpc_controller.set_warm_start(u_ws, du_ws)

    def _step_pid(self, _state: VehicleState):
        self.pid_controller.step(_state)
        self.track_obj.global_to_local_typed(_state)  # Recall that PID uses the global frame.
        _state.p.s = np.mod(_state.p.s, self.track_obj.track_length)
        return {'success': True, 'status': 0}  # PID controller never fails.

    def _step_lmpc(self, _state: VehicleState):
        info = self.lmpc_controller.step(_state)
        self.track_obj.local_to_global_typed(_state)  # Recall that LMPC uses the local frame.
        _state.p.s = np.mod(_state.p.s, self.track_obj.track_length)
        return info

    def reset(self, *, seed=None, options=None):
        self.setup_pid_controller()
        self.setup_lmpc_controller()

        self.t = self.t0

        self.lap_data = []
        self.last_lap_data = None
        self.active_controller = 'pid'

    def clear_lap_data(self):
        self.lap_data = []
        N = self.lmpc_params['N']
        u_ws = np.zeros((N + 1, self.dyn_model.n_u))
        # u_ws = np.tile(dyn_model.input2u(control), (N+1, 1))
        du_ws = np.zeros((N, self.dyn_model.n_u))
        self.lmpc_controller.set_warm_start(u_ws, du_ws)

    def step(self, vehicle_state, terminated, lap_no, **kwargs):
        """
        Use VehicleState to step directly. Closer to how the simulation script works.
        """
        if self.active_controller == 'pid':
            info = self._step_pid(vehicle_state)
        elif self.active_controller == 'lmpc':
            info = self._step_lmpc(vehicle_state)
        else:
            raise ValueError('Invalid controller type')

        if terminated:
            # TODO: Refactor this part if necessary.
            q_data, u_data, lap_c2g = self._process_lap_data(self.lap_data)
            self.lmpc_controller.add_iter_data(q_data, u_data)
            self.lmpc_controller.add_safe_set_data(q_data, u_data, lap_c2g)

            if self.last_lap_data is not None:
                # If this is not the first lap, consider the current lap as the extension of the previous lap
                # and append to the safe set as additional data.
                last_lap_end = self.last_lap_data[-1].t
                for ld in copy.deepcopy(self.lap_data):
                    ld.p.s += self.track_obj.track_length
                    self.last_lap_data.append(ld)
                q_data, u_data, lap_c2g = self._process_lap_data(self.last_lap_data, lap_end=last_lap_end)
                self.lmpc_controller.add_safe_set_data(q_data, u_data, lap_c2g, iter_idx=lap_no - 1)
            self.last_lap_data = copy.deepcopy(self.lap_data)
            self.lap_data = []

            if lap_no == self.lmpc_params['n_ss_its']:
                self.active_controller = 'lmpc'
                logger.debug("Initialization laps complete")
        # self._step_pid(state) if lap_no <= self.lmpc_params['n_ss_its'] else self._step_lmpc(state)
        self.lap_data.append(vehicle_state)  # Note that the state also have the action fields filled out.
        return np.array([vehicle_state.u.u_a, vehicle_state.u.u_steer]), info

    def get_prediction(self):
        return self.lmpc_controller.get_prediction() if self.active_controller == 'lmpc' else None

    def get_safe_set(self):
        return self.lmpc_controller.get_safe_set() if self.active_controller == 'lmpc' else None

    def export(self):
        raise NotImplementedError  # TODO: Resume here!
        if self.disable_export:
            logger.debug('LMPC data exportation disabled according to settings.')
            return
        q = self.dynamics.state2q(self.state)
        u = self.dynamics.input2u(self.input)
        q_prev, u_prev = self.dynamics.state2qu(self.state_prev)
        np.savez_compressed(
            f"{os.path.dirname(__file__)}/../model_data/lmpc_data.npz",
            SS_data=np.asarray(self.lmpc_controller.SS_data, dtype=object),
            iter_data=np.asarray(self.lmpc_controller.iter_data, dtype=object),
            controller_mode=self.controller_mode,
            lap_number=self.lap_number,
            q=q, u=u,
            q_prev=q_prev, u_prev=u_prev,
        )

    def load(self):
        raise NotImplementedError  # TODO: Resume here!
        if not os.path.exists(f"{os.path.dirname(__file__)}/../model_data/lmpc_data.npz"):
            logger.warning("LMPC data not found!")
            return
        data = np.load(f"{os.path.dirname(__file__)}/../model_data/lmpc_data.npz", allow_pickle=True)
        self.lmpc_controller.SS_data.extend(data['SS_data'])
        self.lmpc_controller.iter_data.extend(data['iter_data'])
        self.controller_mode = data['controller_mode']
        self.lap_number = data['lap_number']
        self.dynamics.qu2state(self.state, q=data['q'], u=data['u'])
        self.dynamics.qu2state(self.state_prev, q=data['q_prev'], u=data['u_prev'])
        self.state.copy_control(self.input)

        # Warm start LMPC.
        u_ws = np.tile(self.dynamics.input2u(self.input), (self.mpc_params.N + 1, 1))
        q_ws = np.tile(self.dynamics.state2q(self.state), (self.mpc_params.N + 1, 1))

        du_ws = np.zeros((self.mpc_params.N, self.dynamics.n_u))
        self.lmpc_controller.set_warm_start(u_ws, du_ws, q_ws=q_ws)