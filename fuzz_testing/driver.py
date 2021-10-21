"""
driver abstract base class for gym fuzz testing and others
"""

import abc

class Driver(abc.ABC):
    """abstract class for planner"""

    @abc.abstractmethod
    def plan(self, obs, ego_index):
        """returns speed, steer, obs_list is a list of observations for each car"""
