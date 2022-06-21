'''
Copyright (C) 2021  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import Any, Dict, Optional, Type, Union

import numpy as np

from .object_primitives import PedestrianStateCV, VehicleStateCTRV, VehicleStateCV
from .primitives import Primitives
from .shapes import BoxCorners3D, BoxYaw3D


class Tracklet:

    def __init__(self,
                 state: Optional[Union[Type[Primitives], np.ndarray]] = None,
                 observation: Optional[Union[Type[Primitives], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None) -> None:
        """_summary_

        Parameters
        ----------
        state : Optional[Union[Type[Primitives], np.ndarray]], optional
            _description_, by default None
        observation : Optional[Union[Type[Primitives], np.ndarray]], optional
            _description_, by default None
        x_covar : Optional[np.ndarray], optional
            _description_, by default None
        z_covar : Optional[np.ndarray], optional
            _description_, by default None
        """
        self._state : np.ndarray = state() if isinstance(state, Primitives) else state

        self._observation : np.ndarray = observation() if isinstance(observation, Primitives) else observation

        self._x_covar = x_covar
        self._z_covar = z_covar
        self._Q = Q

    def update(self,
                state : Optional[Union[Type[Primitives], np.ndarray]] = None,
                observation : Optional[Union[Type[Primitives], np.ndarray]] = None,
                x_covar : Optional[np.ndarray] = None,
                z_covar : Optional[np.ndarray] = None,
                Q : Optional[np.ndarray] = None) -> None:
        """_summary_

        Parameters
        ----------
        state : Optional[Union[Type[Primitives], np.ndarray]], optional
            _description_, by default None
        observation : Optional[Union[Type[Primitives], np.ndarray]], optional
            _description_, by default None
        x_covar : Optional[np.ndarray], optional
            _description_, by default None
        z_covar : Optional[np.ndarray], optional
            _description_, by default None
        """
        if (state) : self._state = state
        if (observation) : self._observation = observation
        if (x_covar) : self._x_covar = x_covar
        if (z_covar) : self._z_covar = z_covar
        if (Q) : self._Q = Q

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    @property
    def state_dims(self) -> np.ndarray:
        return len(self._state)

    @property
    def obs_dims(self) -> np.ndarray:
        return len(self._observation)

    @property
    def info(self) -> Dict[str, Any]:
        info = {
            'f': self.f,
            'h': self.h,
            'phi': self.phi,
            'phi_inv': self.phi_inv,
            'Q': self._Q,
            'R': self._z_covar,
            'state0': self._state,
            'P0': self._x_covar
        }
        return info

    @classmethod
    def f(cls, state : Union[Type[Primitives], np.ndarray], omega : np.ndarray, w : np.ndarray, dt : float) -> np.ndarray:
        _new_state = state() if isinstance(state, Primitives) else state # Identity state transition
        return _new_state

    @classmethod
    def h(cls, state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state # Simplest and complete observation model
        return _pred_observation

    @classmethod
    def phi(cls, state : Union[Type[Primitives], np.ndarray], xi : np.ndarray) -> np.ndarray:
        _new_state = state() + xi.diagonal()
        return _new_state

    @classmethod
    def phi_inv(cls, state : Union[Type[Primitives], np.ndarray], hat_state : Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        if isinstance(state, Primitives): state = state()
        if isinstance(hat_state, Primitives): hat_state = hat_state()

        xi = np.diag(np.square(state - hat_state))

        return xi


class TrackletVehicleCV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[VehicleStateCV], np.ndarray]] = None,
                 observation: Optional[Union[Type[BoxYaw3D], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar)

    def update(self,
               state: Optional[Union[Type[Primitives], np.ndarray]] = None,
               observation: Optional[Union[Type[Primitives], np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar)

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        # no force is exerted
        _new_state = state() if isinstance(state, Primitives) else state
        v, v_z, yaw = _new_state[[7, 8, 3]]

        _new_state += np.array([
            v*dt*np.cos(yaw), # Cx
            v*dt*np.sin(yaw), # Cy
            v_z*dt, # Cz
            0, # yaw
            0, # l
            0, # b
            0, # h
            0, # v
            0 # v_z
        ])

        _new_state += w

        return _new_state

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 4, 5, 6]] # x, y, z, l, b, h
        return _pred_observation

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi


class TrackletPedestrianCV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[PedestrianStateCV], np.ndarray]] = None,
                 observation: Optional[Union[Type[BoxYaw3D], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar)

    def update(self,
               state: Optional[Union[Type[Primitives], np.ndarray]] = None,
               observation: Optional[Union[Type[Primitives], np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar)

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        # no force is exerted
        _new_state = state() if isinstance(state, Primitives) else state
        v, v_z, yaw = _new_state[[7, 8, 3]]

        _new_state += np.array([
            v*dt*np.cos(yaw), # Cx
            v*dt*np.sin(yaw), # Cy
            v_z*dt, # Cz
            0, # yaw
            0, # l
            0, # b
            0, # h
            0, # v
            0 # v_z
        ])

        _new_state += w

        return _new_state

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 4, 5, 6]] # x, y, z, l, b, h
        return _pred_observation

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi


class TrackletVehicleCTRV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[VehicleStateCTRV], np.ndarray]] = None,
                 observation: Optional[Union[Type[BoxYaw3D], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar)

    def update(self,
               state: Optional[Union[Type[Primitives], np.ndarray]] = None,
               observation: Optional[Union[Type[Primitives], np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar)

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        # no force is exerted
        _new_state = state() if isinstance(state, Primitives) else state
        v, v_z, yaw, psi_dot = _new_state[[7, 8, 3, 9]]

        _new_state += np.array([
            v*(np.sin(yaw + dt*psi_dot) - np.sin(yaw))/psi_dot, # Cx
            v*(np.cos(yaw) - np.cos(yaw + dt*psi_dot))/psi_dot, # Cy
            v_z*dt, # Cz
            psi_dot*dt, # yaw
            0, # l
            0, # b
            0, # h
            0, # v
            0, # v_z
            0 # psi_dot
        ])

        _new_state += w

        return _new_state

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 4, 5, 6]] # x, y, z, l, b, h
        return _pred_observation

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi
