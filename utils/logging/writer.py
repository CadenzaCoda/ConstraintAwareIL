from typing import Callable
from torch.utils.tensorboard import SummaryWriter


class MultiPurposeWriter(SummaryWriter):
    def __init__(self, model_name: str, log_dir: str = None, comment: str = None,
                 print_method: Callable = None, 
                 **params):
        self._comment = f'{model_name}' + (f'_{comment}_' if len(comment) else '') + '_'.join(
                f'{k}_{v}' for k, v in params.items())
        super().__init__(
            log_dir=log_dir,
            comment=f'_{model_name}' + (f'_{comment}_' if len(comment) else '') + '_'.join(
                f'{k}_{v}' for k, v in params.items())
        )
        self.print_method = print_method if print_method is None else lambda *args, **kwargs: None

    def do_logging(self, info, *, global_step, mode):
        for tag, value in info.items():
            self.add_scalar('/'.join([mode, tag]), value, global_step=global_step)
            self.print_method(f"[{mode}] {tag}: {value:.4f}")
