from typing import Tuple

import numpy as np
import casadi as ca

from mpclab_common.models.dynamics_models import CasadiDynamicBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, Position, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity, \
    VehicleActuation
from mpclab_controllers.CA_MPCC_conv import CA_MPCC_conv
from mpclab_controllers.utils.controllerTypes import CAMPCCParams


class MPCCConvWrapper:
    def __init__(self, dt=0.1, t0=0., track_obj=None):
        self.controller = None
        self.dt = dt
        self.t0 = t0
        self.track_obj = track_obj
        self.dynamics_config = DynamicBicycleConfig(dt=dt,
                                                    model_name='dynamic_bicycle',
                                                    noise=False,
                                                    discretization_method='rk4',
                                                    simple_slip=False,
                                                    tire_model='pacejka',
                                                    mass=2.2187,
                                                    yaw_inertia=0.02723,
                                                    # mass=2.91,
                                                    # yaw_inertia=0.03323,
                                                    wheel_friction=0.9,
                                                    pacejka_b_front=5.0,
                                                    pacejka_b_rear=5.0,
                                                    pacejka_c_front=2.28,
                                                    pacejka_c_rear=2.28)
        self.dyn_model = CasadiDynamicBicycle(t0, self.dynamics_config, track=track_obj)

        self.state_input_ub = VehicleState(x=Position(x=1e9, y=1e9),
                                           e=OrientationEuler(psi=10),
                                           v=BodyLinearVelocity(v_long=4.0, v_tran=2),
                                           w=BodyAngularVelocity(w_psi=7),
                                           u=VehicleActuation(u_a=2.0, u_steer=0.45))
        self.state_input_lb = VehicleState(x=Position(x=-1e9, y=-1e9),
                                           e=OrientationEuler(psi=-10),
                                           v=BodyLinearVelocity(v_long=-4, v_tran=-2),
                                           w=BodyAngularVelocity(w_psi=-7),
                                           u=VehicleActuation(u_a=-2.0, u_steer=-0.45))
        self.input_rate_max = VehicleState(u=VehicleActuation(u_a=40.0, u_steer=4.5))
        self.input_rate_min = VehicleState(u=VehicleActuation(u_a=-40.0, u_steer=-4.5))

        N = 20
        self.mpc_params = CAMPCCParams(dt=self.dt, N=N,
                                       verbose=False,
                                       debug_plot=False,
                                       damping=0.25,
                                       qp_iters=2,
                                       pos_idx=[3, 4],
                                       state_scaling=[4, 2, 7, 6, 6, 2 * np.pi],
                                       input_scaling=[2, 0.436],
                                       delay=None,
                                       parametric_contouring_cost=False,
                                       contouring_cost=0.1,
                                       contouring_cost_N=0.1,
                                       lag_cost=1000.0,
                                       lag_cost_N=1000.0,
                                       performance_cost=0.03,
                                       vs_cost=1e-4,
                                       vs_rate_cost=1e-3,
                                       vs_max=4.5,  # 5.0,
                                       vs_min=0.0,
                                       vs_rate_max=5.0,
                                       vs_rate_min=-5.0,
                                       soft_track=True,
                                       track_slack_quad=10000,
                                       track_slack_lin=0,
                                       track_tightening=0.1,
                                       code_gen=False,
                                       opt_flag='O3',
                                       solver_name='MPCC_conv',
                                       qp_interface='hpipm')

    def reset(self, *, seed=None, options=None):
        self.setup_mpcc_conv_controller(sim_state=options['vehicle_state'])

    def setup_mpcc_conv_controller(self, sim_state=None):
        N = self.mpc_params.N

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', self.dyn_model.n_q)
        sym_u = ca.MX.sym('u', self.dyn_model.n_u)
        sym_du = ca.MX.sym('du', self.dyn_model.n_u)

        wz_idx = 2
        ua_idx = 0
        us_idx = 1

        sym_state_stage = 0.5 * (1e-5 * sym_q[wz_idx] ** 2)
        sym_state_term = 0.5 * (1e-4 * sym_q[wz_idx] ** 2)
        sym_input_stage = 0.5 * (1e-3 * (sym_u[ua_idx]) ** 2 + 1e-3 * (sym_u[us_idx]) ** 2)
        sym_input_term = 0.5 * (1e-3 * (sym_u[ua_idx]) ** 2 + 1e-3 * (sym_u[us_idx]) ** 2)
        sym_rate_stage = 0.5 * (0.01 * (sym_du[ua_idx]) ** 2 + 0.01 * (sym_du[us_idx]) ** 2)

        sym_costs = {'state': [None for _ in range(N + 1)], 'input': [None for _ in range(N + 1)],
                     'rate': [None for _ in range(N)]}
        for k in range(N):
            sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q], [sym_state_stage])
            sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
            sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
        sym_costs['state'][N] = ca.Function('state_term', [sym_q], [sym_state_term])
        sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

        a_max = self.dynamics_config.gravity * self.dynamics_config.wheel_friction
        # sym_ax, sym_ay, _ = dyn_model.f_a(sym_q, sym_u)
        # friction_circle_constraint = ca.Function('friction_circle', [sym_q, sym_u], [sym_ax**2 + sym_ay**2 - a_max**2])

        # sym_constrs = {'state_input': [friction_circle_constraint for _ in range(N+1)],
        #                 'rate': [None for _ in range(N)]}
        sym_constrs = {'state_input': [None for _ in range(N + 1)],
                       'rate': [None for _ in range(N)]}

        self.controller = CA_MPCC_conv(self.dyn_model,
                                       sym_costs,
                                       sym_constrs,
                                       {'qu_ub': self.state_input_ub, 'qu_lb': self.state_input_lb,
                                        'du_ub': self.input_rate_max,
                                        'du_lb': self.input_rate_min},
                                       self.mpc_params,
                                       print_method=None)

        # u_ws = np.zeros((N+1, dyn_model.n_u))
        u_ws = 1e-3 * np.ones((N + 1, self.dyn_model.n_u))
        vs_ws = np.zeros(N + 1)
        du_ws = np.zeros((N, self.dyn_model.n_u))
        dvs_ws = np.zeros(N)

        z = 0.0
        R = np.array([z for _ in range(N + 1)])
        P = np.array([])
        self.controller.set_warm_start(u_ws, vs_ws, du_ws, dvs_ws,
                                       state=sim_state,
                                       reference=R,
                                       parameters=P)

    def _step_mpcc_conv(self, _state: VehicleState):
        z = 0.0
        R = np.array([z for _ in range(self.mpc_params.N + 1)])
        P = np.array([])
        success = self.controller.step(_state, reference=R, parameters=P)
        return np.array([_state.u.u_a, _state.u.u_steer]), {'success': success}

    def step(self, vehicle_state, **kwargs) -> Tuple[np.ndarray, dict]:
        return self._step_mpcc_conv(vehicle_state)

    def get_prediction(self):
        return self.controller.get_prediction()

    def get_safe_set(self):
        return None
