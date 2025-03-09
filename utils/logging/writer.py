from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from labml import tracker
from .ntfy import ntfy


class MultiPurposeWriter(SummaryWriter):
    def __init__(self, model_name: str, log_dir: str = None, comment: str = None,
                 print_method: Callable = None, use_labml_tracker: bool = True,
                 ntfy_freq: int = -1,
                 **params):
        self._comment = f'{model_name}' + (f'_{comment}_' if len(comment) else '') + '_'.join(
                f'{k}_{v}' for k, v in params.items())
        super().__init__(
            log_dir=log_dir,
            comment=f'_{model_name}' + (f'_{comment}_' if len(comment) else '') + '_'.join(
                f'{k}_{v}' for k, v in params.items())
        )
        self.use_labml_tracker = use_labml_tracker
        self.ntfy_freq = ntfy_freq
        self.print_method = print_method if print_method is None else lambda *args, **kwargs: None

    def do_logging(self, info, *, global_step, mode):
        for tag, value in info.items():
            self.add_scalar('/'.join([mode, tag]), value, global_step=global_step)
            self.print_method(f"[{mode}] {tag}: {value:.4f}")
        if self.use_labml_tracker:
            tracker.save(global_step, {'/'.join([mode, tag]): value for tag, value in info.items()})
        if self.ntfy_freq > 0 and global_step % self.ntfy_freq == 0:
            data = '\n'.join(f"- **{'/'.join([mode, tag])}**: {value}" for tag, value in info.items())
            ntfy(data=data,
                 title=f"{self._comment} - Step {global_step} Report",
                 priority=2,
                 topic='research',
                 )

    def ntfy(self, message, *, global_step=None):
        if self.ntfy_freq <= 0:
            return
        ntfy(
            data=message,
            title=f"Message: {self._comment}{f' - Step {global_step}' if global_step else ''}",
            priority=4,
            tags=("warning",),
            topic='research',
        )
