import unittest

import numpy as np
from frozendict import frozendict
from msdm.core.distributions import DictDistribution
from msdm.algorithms import TabularMultiagentValueIteration
from msdm.domains import TabularGridGame

class TabularMultiagentVITestCase(unittest.TestCase):
    
    
    def test_ma_value_iteration(self):
        two_player = """
        # # # # #
        # G0 . G1 # 
        # . . . #
        # A0 . A1 #
        # # # # #
        """
        gg = TabularGridGame(two_player,agent_symbols=("A0","A1"),goal_symbols=(("G0",("A0",)),("G1",("A1",))),step_cost=-1,collision_cost=-1,goal_reward=100)
        initial_state = gg.initial_state_dist().sample()
        res = TabularMultiagentValueIteration(["A0","A1"],{},show_progress=True,discount_rate=.9).plan_on(gg)
        out = res.policy.run_on(gg)
        assert out.state_traj[0]["A0"]["x"] == 1
        assert out.state_traj[0]["A0"]["y"] == 1
        assert out.state_traj[1]["A0"]["x"] == 1
        assert out.state_traj[1]["A0"]["y"] == 2
        assert out.state_traj[2]["A0"]["x"] == 1
        assert out.state_traj[2]["A0"]["y"] == 3
