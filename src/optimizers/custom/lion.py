import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """ https://raw.githubusercontent.com/lucidrains/lion-pytorch/main/lion_pytorch/lion_pytorch.py """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda pp: pp.grad is not None, group["params"]):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["betas"]
                state = self.state[p]

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # stepweight decay
                p.data.mul_(1 - lr * wd)

                # weight update
                update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
