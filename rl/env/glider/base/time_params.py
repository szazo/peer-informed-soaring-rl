from dataclasses import dataclass
import numpy as np


@dataclass
class TimeParameters:

    # duration_s: float
    # simulation step time
    dt_s: float = 1.0
    # determines time interval between decisions
    # simulation will skip decision_dt_s / dt_s frames between simulation steps
    # and return observations (receive actions) with this interval
    decision_dt_s: float = 1.0
    initial_time_s: float = 0


def calculate_frame_skip_number(time_params: TimeParameters):
    decision_dt_s = time_params.decision_dt_s
    dt_s = time_params.dt_s

    assert ((decision_dt_s / dt_s) % 1
            < 1e-3), 'decision_dt_s should be integer multiple of dt_s'

    frame_skip_num = int(np.round(decision_dt_s / dt_s))

    return frame_skip_num
