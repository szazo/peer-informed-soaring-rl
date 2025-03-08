from typing import Self
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from scipy.integrate import odeint
from utils import Vector2


@dataclass
class SystemParams:
    omega_natural_frequency: float
    zeta_damping_ratio: float
    k_process_gain: float


class SecondOrderSystem:

    _params: SystemParams

    def __init__(self, params: SystemParams):
        self._params = params

    def model(self, x: Vector2, t: float, u_input: float):

        theta, theta_dot = x
        omega = self._params.omega_natural_frequency
        zeta = self._params.zeta_damping_ratio
        k = self._params.k_process_gain

        dtheta_dt = theta_dot
        dtheta_dot_dt = -omega**2 * theta - 2 * zeta * omega * theta_dot + omega**2 * k * u_input

        return [dtheta_dt, dtheta_dot_dt]


@dataclass
class ControlParams:
    proportional_gain: float
    derivative_gain: float
    integral_gain: float


@dataclass
class ControlDynamicsParams:
    system: SystemParams
    control: ControlParams


# theta, theta_dot
ControlState = Vector2


class ControlDynamics:

    _system_params: SystemParams
    _control_params: ControlParams

    _target_setpoint: ControlState | None
    _second_order_system: SecondOrderSystem

    def __init__(self,
                 params: ControlDynamicsParams,
                 target_setpoint: ControlState | None = None):

        self._system_params = params.system
        self._control_params = params.control

        self._second_order_system = SecondOrderSystem(params=params.system)

        self._target_setpoint = target_setpoint

    def update_target_setpoint(self, setpoint: ControlState):
        self._target_setpoint = setpoint

    def clone_state_from(self, other: Self):
        self._target_setpoint = other._target_setpoint

    def calculate(self, initial_state: ControlState, times: npt.ArrayLike):

        assert self._target_setpoint is not None, 'please call `update_target_setpoint`'

        try:
            integral_error0 = 0.0
            y0 = [initial_state[0], initial_state[1], integral_error0]
            result = odeint(self._system_dynamics, y0, times)

            return result
        except Exception as e:
            print('initial_state', initial_state)
            print('times', times)
            raise

    def _system_dynamics(self, x: list[float], t: float):

        theta, theta_dot, integral_error = x

        K_p = self._control_params.proportional_gain
        K_d = self._control_params.derivative_gain
        K_i = self._control_params.integral_gain

        # # control
        assert self._target_setpoint is not None, 'missing setpoint'
        theta_ref, theta_dot_ref = self._target_setpoint

        error = theta_ref - theta
        derivative_error = theta_dot_ref - theta_dot

        u = K_p * error + K_i * integral_error + K_d * derivative_error

        dtheta_dt, dtheta_dot_dt = self._second_order_system.model(x=np.array(
            [theta, theta_dot]),
                                                                   t=t,
                                                                   u_input=u)

        dintegral_error_dt = error  # the integral of the error will change with this value

        return [dtheta_dt, dtheta_dot_dt, dintegral_error_dt]
