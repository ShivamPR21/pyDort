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

from typing import Any


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
