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

from enum import Enum
from typing import Any, Dict, Optional, Type, Union

import numpy as np

from .object_primitives import PedestrianStateCV, VehicleStateCTRV, VehicleStateCV
from .primitives import Primitives
from .shapes import BoxYaw3D
from .transform_utils import angle_constraint


class TrackletType(Enum):
    Init = 1
    Real = 2
    Dummy = 3
    End = 4

class Tracklet:

    def __init__(self,
                 state: Optional[Union[Type[Primitives], np.ndarray]] = None,
                 observation: Optional[Union[Type[Primitives], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 descriptor: Optional[np.ndarray] = None) -> None:
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
        Q : Optional[np.ndarray], optional
            _description_, by default None
        descriptor : Optional[np.ndarray], optional
            _description_, by default None
        """
        self._state : np.ndarray = state() if isinstance(state, Primitives) else state

        self._observation : np.ndarray = observation() if isinstance(observation, Primitives) else observation

        self._x_covar = x_covar
        self._z_covar = z_covar
        self._Q = Q

        self.dsc = descriptor

        self.status = TrackletType.Init

    def update(self,
                state : Optional[Union[Type[Primitives], np.ndarray]] = None,
                observation : Optional[Union[Type[Primitives], np.ndarray]] = None,
                x_covar : Optional[np.ndarray] = None,
                z_covar : Optional[np.ndarray] = None,
                Q : Optional[np.ndarray] = None,
                descriptor: Optional[np.ndarray] = None) -> None:
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
        Q : Optional[np.ndarray], optional
            _description_, by default None
        descriptor : Optional[np.ndarray], optional
            _description_, by default None
        """
        if (state is not None) : self._state = state
        if (observation is not None) : self._observation = observation
        if (x_covar is not None) : self._x_covar = x_covar
        if (z_covar is not None) : self._z_covar = z_covar
        if (Q is not None) : self._Q = Q
        if (descriptor is not None) : self.dsc = descriptor

    def set_status(self, status : TrackletType):
        self.status = status

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    @property
    def state_dims(self) -> int:
        return len(self._state)

    @property
    def obs_dims(self) -> int:
        return len(self._observation)

    @property
    def info(self) -> Dict[str, Any]:
        info = {
            'f': self.f,
            'fx': self.fx,
            'h': self.h,
            'hx': self.hx,
            'phi': self.phi,
            'phi_inv': self.phi_inv,
            'Q': self._Q,
            'R': self._z_covar,
            'state0': self._state,
            'P0': self._x_covar
        }
        return info

    @property
    def descriptor(self) -> np.ndarray:
        return self.dsc

    @classmethod
    def f(cls, state : Union[Type[Primitives], np.ndarray], omega : Optional[np.ndarray], w : np.ndarray, dt : float) -> np.ndarray:
        _new_state = state() if isinstance(state, Primitives) else state # Identity state transition
        return _new_state

    @classmethod
    def fx(cls, state, dt):
        raise NotImplementedError

    @classmethod
    def h(cls, state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state # Simplest and complete observation model
        return _pred_observation

    @classmethod
    def hx(cls, state):
        raise NotImplementedError

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

    def state_from_observation(self, observation: np.ndarray, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def observation_from_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TrackletVehicleCV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[VehicleStateCV], np.ndarray]] = None,
                 observation: Optional[Union[BoxYaw3D, np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 descriptor: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar, Q, descriptor)

    def update(self,
               state: Optional[Union[Type[VehicleStateCV], np.ndarray]] = None,
               observation: Optional[Union[BoxYaw3D, np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None,
               Q : Optional[np.ndarray] = None,
               descriptor: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar, Q, descriptor)

    @property
    def state_dims(self) -> int:
        return 9

    @property
    def obs_dims(self) -> int:
        return 7

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: Optional[np.ndarray], w: np.ndarray, dt: float) -> np.ndarray:
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
        ], dtype=np.float32)

        _new_state += w
        _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint

        return _new_state

    @classmethod
    def fx(cls, state, dt):
        return cls.f(state, None, np.array([0.]), dt)

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 3, 4, 5, 6]] # x, y, z, yaw, l, b, h
        _pred_observation[3] = angle_constraint(_pred_observation[3])
        return _pred_observation

    @classmethod
    def hx(cls,
           state):
        return cls.h(state)

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi

    def state_from_observation(self, observation: np.ndarray, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        state = np.zeros((self.state_dims, ), dtype=np.float32) if init_state is None else init_state
        state[:self.obs_dims] = observation
        state[3] = angle_constraint(state[3])
        return state

    def observation_from_state(self, state: np.ndarray) -> np.ndarray:
        observation = np.zeros((self.obs_dims, ), dtype=np.float32)
        observation = state[:self.obs_dims]
        observation[3] = angle_constraint(observation[3])
        return observation

class TrackletPedestrianCV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[PedestrianStateCV], np.ndarray]] = None,
                 observation: Optional[Union[BoxYaw3D, np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 descriptor: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar, Q, descriptor)

    def update(self,
               state: Optional[Union[Type[PedestrianStateCV], np.ndarray]] = None,
               observation: Optional[Union[BoxYaw3D, np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None,
               Q : Optional[np.ndarray] = None,
               descriptor: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar, Q, descriptor)

    @property
    def state_dims(self) -> int:
        return 10

    @property
    def obs_dims(self) -> int:
        return 7

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: Optional[np.ndarray], w: np.ndarray, dt: float) -> np.ndarray:
        # no force is exerted
        _new_state = state() if isinstance(state, Primitives) else state
        vx, vy, vz = _new_state[[7, 8, 9]]

        _new_state += np.array([
            vx*dt, # Cx
            vy*dt, # Cy
            vz*dt, # Cz
            0,     # yaw
            0,     # l
            0,     # b
            0,     # h
            0,     # vx
            0,     # vy
            0      # vz
        ], dtype=np.float32)

        _new_state += w
        _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint

        return _new_state

    @classmethod
    def fx(cls, state, dt):
        return cls.f(state, None, np.array([0.]), dt)

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 3, 4, 5, 6]] # x, y, z, yaw, l, b, h
        _pred_observation[3] = angle_constraint(_pred_observation[3])
        return _pred_observation

    @classmethod
    def hx(cls,
           state):
        return cls.h(state)

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        # _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi

    def state_from_observation(self, observation: np.ndarray, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        state = np.zeros((self.state_dims, ), dtype=np.float32) if init_state is None else init_state
        state[:self.obs_dims] = observation
        # state[3] = angle_constraint(state[3])
        return state

    def observation_from_state(self, state: np.ndarray) -> np.ndarray:
        observation = np.zeros((self.obs_dims, ), dtype=np.float32)
        observation = state[:self.obs_dims]
        # observation[3] = angle_constraint(observation[3])
        return observation

class TrackletVehicleCTRV(Tracklet):

    def __init__(self,
                 state: Optional[Union[Type[VehicleStateCTRV], np.ndarray]] = None,
                 observation: Optional[Union[Type[BoxYaw3D], np.ndarray]] = None,
                 x_covar: Optional[np.ndarray] = None,
                 z_covar: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 descriptor: Optional[np.ndarray] = None) -> None:
        super().__init__(state, observation, x_covar, z_covar, Q, descriptor)

    def update(self,
               state: Optional[Union[Type[VehicleStateCTRV], np.ndarray]] = None,
               observation: Optional[Union[Type[BoxYaw3D], np.ndarray]] = None,
               x_covar: Optional[np.ndarray] = None, z_covar: Optional[np.ndarray] = None,
               Q : Optional[np.ndarray] = None,
               descriptor: Optional[np.ndarray] = None) -> None:
        return super().update(state, observation, x_covar, z_covar, Q, descriptor)

    @property
    def state_dims(self) -> int:
        return 10

    @property
    def obs_dims(self) -> int:
        return 7

    @classmethod
    def f(cls,
          state: Union[Type[VehicleStateCV], np.ndarray],
          omega: Optional[np.ndarray], w: np.ndarray, dt: float) -> np.ndarray:
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
        ], dtype=np.float32)

        _new_state += w
        _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint

        return _new_state

    @classmethod
    def fx(cls, state, dt):
        return cls.f(state, None, np.array([0.]), dt)

    @classmethod
    def h(cls,
          state: Union[Type[VehicleStateCV], np.ndarray]) -> np.ndarray:
        _pred_observation = state() if isinstance(state, Primitives) else state
        _pred_observation = _pred_observation[[0, 1, 2, 3, 4, 5, 6]] # x, y, z, yaw, l, b, h
        # _pred_observation[3] = angle_constraint(_pred_observation[3])
        return _pred_observation

    @classmethod
    def hx(cls,
           state):
        return cls.h(state)

    @classmethod
    def phi(cls,
            state: Union[Type[VehicleStateCV], np.ndarray], xi: np.ndarray) -> np.ndarray:
        _new_state = state + xi #TODO: Check for validity @ShivamPR21
        # _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint
        return _new_state

    @classmethod
    def phi_inv(cls,
                state: Union[Type[Primitives], np.ndarray],
                hat_state: Union[Type[Primitives], np.ndarray]) -> np.ndarray:
        xi = state - hat_state #TODO: Check for validity @ShivamPR21
        return xi

    def state_from_observation(self, observation: np.ndarray, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        state = np.zeros((self.state_dims, ), dtype=np.float32) if init_state is None else init_state
        state[:self.obs_dims] = observation
        # state[3] = angle_constraint(state[3])
        return state

    def observation_from_state(self, state: np.ndarray) -> np.ndarray:
        observation = np.zeros((self.obs_dims, ), dtype=np.float32)
        observation = state[:self.obs_dims]
        # observation[3] = angle_constraint(observation[3])
        return observation

class TrackletVehicleAdaptiveCTRV(TrackletVehicleCTRV):

    @classmethod
    def f(cls, state: Union[Type[VehicleStateCV], np.ndarray], omega: Optional[np.ndarray], w: np.ndarray, dt: float) -> np.ndarray:
         # no force is exerted
        _new_state = state() if isinstance(state, Primitives) else state
        v, v_z, yaw, psi_dot = _new_state[[7, 8, 3, 9]]

        _new_state += np.array([
            v*(np.sin(yaw + dt*psi_dot) - np.sin(yaw))/psi_dot if abs(psi_dot) > 1e-3 else v*dt*np.cos(yaw), # Cx
            v*(np.cos(yaw) - np.cos(yaw + dt*psi_dot))/psi_dot if abs(psi_dot) > 1e-3 else v*dt*np.sin(yaw), # Cy
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
        _new_state[3] = angle_constraint(_new_state[3]) # apply angle constraint

        return _new_state
