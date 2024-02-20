def get_value_or_default(default, schedule=None, update_counter=None):
    if schedule is not None:
        assert update_counter is not None
        return schedule.get_value(
            step=update_counter.cur_checkpoint.update,
            total_steps=update_counter.end_checkpoint.update,
        )
    return default
