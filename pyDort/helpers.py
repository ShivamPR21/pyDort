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

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class Queue:

    def __init__(self, maxsize: int = -1, item : Any = None) -> None:
        self.maxsize = maxsize
        self._data = []
        if (item is not None) : self.put(item)

    def put(self, item : Any) -> None:
        if self.full:
            self._data.pop(0)
        self._data.append(item)

    def get(self) -> Any:
        if self.empty:
            return None
        return self._data.pop(0)

    @property
    def empty(self):
        return not len(self._data) > 0

    @property
    def full(self):
        return (not self.maxsize <= 0) and (len(self._data) == self.maxsize)

    def __getitem__(self, key : int) -> Any:
        if key < 0: key = len(self._data) + key

        if (len(self._data) > key and key >= 0):
            return self._data[key]

        return None

    def __len__(self):
        return len(self._data)


class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]
uuid_gen = UUIDGeneration()

def check_mkdir(dirpath):
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)

def read_json_file(fpath: str):
    """
    Args:
        fpath: string, representing file path
    """
    with open(fpath, 'rb') as f:
        json_data = json.load(f)
    return json_data

def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a JSON file.
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)

def wrap_angle(angles: np.ndarray, period: float = np.pi) -> np.ndarray:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π). This function is
        the inverse of `np.unwrap`.

    Returns:
        Angles (in radians) mapped to the interval [0, π).
    """

    # Map angles to [0, ∞].
    angles = np.abs(angles)

    # Calculate floor division and remainder simultaneously.
    divs, mods = np.divmod(angles, period)

    # Select angles which exceed specified period.
    angle_complement_mask = np.nonzero(divs)

    # Take set complement of `mods` w.r.t. the set [0, π].
    # `mods` must be nonzero, thus the image is the interval [0, π).
    angles[angle_complement_mask] = period - mods[angle_complement_mask]
    return angles

if __name__ == "__main__":

    class TestData:
        def __init__(self, it) -> None:
            self.s = it

        def update(self, d) -> None:
            self.s = d

    # General test 1
    # Test 1
    q = Queue(5, 0)
    for i in range(1, 5):
        q.put(i)

    assert(q.full)

    assert(q._data == [i for i in range(5)])

    # test 2
    q.put(6)
    assert(q[-1] == 6)

    #test 3
    assert(q.get() == 1)
    assert(q[0] == 2)

    # Specific test 1
    q = Queue(5, TestData(0))
    for i in range(1, 5):
        q.put(TestData(i))

    q[-1].update(10)
    assert(q[-1].s == 10)
