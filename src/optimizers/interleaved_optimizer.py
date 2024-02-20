import logging

from kappaschedules import object_to_schedule


class InterleavedOptimizer:
    """
    selects an optimizer from a set of optimizers per update step according to a schedule
    the schedule should return the index of the current optimizer
    """

    def __init__(self, model, optim_ctors, schedule, update_counter):
        self.logger = logging.getLogger(type(self).__name__)
        self.optims = [optim_ctor(model, update_counter=update_counter) for optim_ctor in optim_ctors]
        self.update_counter = update_counter
        self.schedule = object_to_schedule(
            schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
        )

    def _get_optim_for_current_step(self):
        index = self.schedule.get_value(
            step=self.update_counter.cur_checkpoint.update,
            total_steps=self.update_counter.end_checkpoint.update,
        )
        return self.optims[int(index)]

    def get_optim_for_previous_step(self):
        index = self.schedule.get_value(
            step=self.update_counter.cur_checkpoint.update - 1,
            total_steps=self.update_counter.end_checkpoint.update,
        )
        return self.optims[int(index)]

    def step(self, grad_scaler):
        self._get_optim_for_current_step().step(grad_scaler)

    def schedule_step(self):
        self._get_optim_for_current_step().schedule_step()

    def zero_grad(self, set_to_none=True):
        for optim in self.optims:
            optim.zero_grad(set_to_none)

    def state_dict(self):
        return {i: optim.state_dict() for i, optim in enumerate(self.optims)}

    def load_state_dict(self, state_dict_to_load):
        for i in range(len(state_dict_to_load)):
            self.optims[i].load_state_dict(state_dict_to_load[i])
