import numpy as np
import matplotlib
import matplotlib.backends.backend_agg as agg
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
from collections import deque

# matplotlib.use('Agg')

BUFFER_MAXLEN = 50

class LMPCVisualizer:
    def __init__(self, track_obj, VL=0.37, VW=0.195):
        self.VL, self.VW = VL, VW
        self.track_obj = track_obj

        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_xy = self.fig.add_subplot(1, 2, 1)
        self.ax_v = self.fig.add_subplot(3, 2, 2)
        self.ax_a = self.fig.add_subplot(3, 2, 4)
        self.ax_d = self.fig.add_subplot(3, 2, 6)
        self.track_obj.plot_map(self.ax_xy)
        self.ax_xy.set_aspect('equal')
        self.l_pred = self.ax_xy.plot([], [], 'b-o', markersize=4)[0]
        self.l_ss = self.ax_xy.plot([], [], 'rs', markersize=4, markerfacecolor='None')[0]
        self.l_pa = self.ax_a.plot([], [], '-go')[0]
        self.l_pd = self.ax_d.plot([], [], '-go')[0]
        self.l_v = self.ax_v.plot([], [], '-bo')[0]
        self.l_a = self.ax_a.plot([], [], '-bo')[0]
        self.l_d = self.ax_d.plot([], [], '-bo')[0]
        self.rect = patches.Rectangle((-0.5 * VL, -0.5 * VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
        self.ax_xy.add_patch(self.rect)
        self.ax_a.set_ylabel('long in')
        self.ax_d.set_ylabel('lat in')
        self.ax_v.set_ylabel('long vel')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.v_data = deque([], maxlen=BUFFER_MAXLEN)
        self.a_data = deque([], maxlen=BUFFER_MAXLEN)
        self.d_data = deque([], maxlen=BUFFER_MAXLEN)
        self.controller = None

    # def step(self, state, pred=None, ss=None):
    #     """
    #     state: (x, y, psi): Vehicle State
    #     """
    #     # import pdb
    #     # pdb.set_trace()

    def bind_controller(self, controller):
        self.controller = controller

    def step(self, state=None, *, q=None, u=None, pred=None, ss=None):
        if state is not None:
            x, y, psi = self.track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
            v_long, v_tran, w_psi = state.v.v_long, state.v.v_tran, state.w_psi
            u_a, u_steer = state.u.u_a, state.u.u_steer
        elif q is not None:
            v_long, v_tran, w_psi, x, y, psi = q
            u_a, u_steer = u
        else:
            raise ValueError("Must provide either state or q")

        if self.controller is not None:
            pred = self.controller.get_prediction()
            ss = self.controller.get_safe_set()

        b_left = x - self.VL / 2
        b_bot = y - self.VW / 2
        r = Affine2D().rotate_around(x, y, psi) + self.ax_xy.transData
        self.rect.set_xy((b_left, b_bot))
        self.rect.set_transform(r)
        pred_x, pred_y, ss_x, ss_y = [], [], [], []

        # Plot the predictions 
        if pred is not None:
            if pred.x is not None:
                pred_x, pred_y = pred.x, pred.y
            else:
                for i in range(len(pred.s)):
                    x, y, psi = self.track_obj.local_to_global((pred.s[i], pred.x_tran[i], pred.e_psi[i]))
                    pred_x.append(x)
                    pred_y.append(y)
            self.l_pred.set_data(pred_x, pred_y)
            self.l_pa.set_data(np.arange(len(self.a_data), len(self.a_data) + len(pred.u_a)), pred.u_a)
            self.l_pd.set_data(np.arange(len(self.d_data), len(self.d_data) + len(pred.u_steer)), pred.u_steer)

        # Plot the safety set (red squares) 
        if ss is not None:
            # for i in range(len(ss.s)):
            #     x, y, psi = self.track_obj.local_to_global((ss.s[i], ss.x_tran[i], ss.e_psi[i]))
            for s_i, x_tran_i, e_psi_i in zip(ss.s, ss.x_tran, ss.e_psi):
                x, y, psi = self.track_obj.local_to_global((s_i, x_tran_i, e_psi_i))
                ss_x.append(x)
                ss_y.append(y)
            self.l_ss.set_data(ss_x, ss_y)

        self.v_data.append(v_long)
        self.a_data.append(u_a)
        self.d_data.append(u_steer)
        self.l_v.set_data(np.arange(len(self.v_data)), self.v_data)
        self.l_a.set_data(np.arange(len(self.a_data)), self.a_data)
        self.l_d.set_data(np.arange(len(self.d_data)), self.d_data)
        self.ax_v.relim()
        self.ax_v.autoscale_view()
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        self.l_pred.set_data([], [])
        self.l_ss.set_data([], [])
        self.l_pa.set_data([], [])
        self.l_pd.set_data([], [])
        self.l_v.set_data([], [])
        self.l_a.set_data([], [])
        self.l_d.set_data([], [])
        self.v_data.clear()
        self.a_data.clear()
        self.d_data.clear()
