import unittest

import numpy as np
from frozendict import frozendict
from msdm.core.distributions import DictDistribution
from msdm.algorithms import TabularMultiagentQLearner, TabularNashQLearner, TabularFriendFoeQLearner, TabularCorrelatedQLearner
from msdm.domains import TabularGridGame

class TabularMultiagentQLearningTestCase(unittest.TestCase):
    
    
    def test_ma_q_learning(self):
        two_player = """
        # # # # #
        # G0 . G1 # 
        # . . . #
        # A0 . A1 #
        # # # # #
        """
        gg = TabularGridGame(two_player,agent_symbols=("A0","A1"),goal_symbols=(("G0",("A0",)),("G1",("A1",))),step_cost=-1,collision_cost=-1,goal_reward=100)
        initial_state = gg.initial_state_dist().sample()
        res = TabularMultiagentQLearner(["A0","A1"],{},show_progress=True,discount_rate=.99,all_actions=False,epsilon=.1,num_episodes=500).train_on(gg)
        out = res.policy.run_on(gg)
        assert out.state_traj[0]["A0"]["x"] == 1
        assert out.state_traj[0]["A0"]["y"] == 1
        assert out.state_traj[1]["A0"]["x"] == 1
        assert out.state_traj[1]["A0"]["y"] == 2
        assert out.state_traj[2]["A0"]["x"] == 1
        assert out.state_traj[2]["A0"]["y"] == 3
        
    def test_ma_nash_q_learning(self):
        two_player = """
        # # # # #
        # G0 . G1 # 
        # . . . #
        # A0 . A1 #
        # # # # #
        """
        gg = TabularGridGame(two_player,agent_symbols=("A0","A1"),goal_symbols=(("G0",("A0",)),("G1",("A1",))),step_cost=-1,collision_cost=-1,goal_reward=100)
        initial_state = gg.initial_state_dist().sample()
        res = TabularNashQLearner(["A0","A1"],{},show_progress=True,discount_rate=.99,epsilon=.1,num_episodes=500).train_on(gg)
        out = res.policy.run_on(gg)
        assert out.state_traj[0]["A0"]["x"] == 1
        assert out.state_traj[0]["A0"]["y"] == 1
        assert out.state_traj[1]["A0"]["x"] == 1
        assert out.state_traj[1]["A0"]["y"] == 2
        assert out.state_traj[2]["A0"]["x"] == 1
        assert out.state_traj[2]["A0"]["y"] == 3
        
    def test_ma_ffq_learning(self):
        two_player = """
        # # # # #
        # G0 . G1 # 
        # . . . #
        # A0 . A1 #
        # # # # #
        """
        gg = TabularGridGame(two_player,agent_symbols=("A0","A1"),goal_symbols=(("G0",("A0",)),("G1",("A1",))),step_cost=-1,collision_cost=-1,goal_reward=100)
        initial_state = gg.initial_state_dist().sample()
        res = TabularFriendFoeQLearner(["A0","A1"],{"A0":["A1"],"A1":["A0"]},{"A0":[],"A1":[]},{},show_progress=True,discount_rate=.99,epsilon=.1,num_episodes=500).train_on(gg)
        out = res.policy.run_on(gg)
        assert out.state_traj[0]["A0"]["x"] == 1
        assert out.state_traj[0]["A0"]["y"] == 1
        assert out.state_traj[1]["A0"]["x"] == 1
        assert out.state_traj[1]["A0"]["y"] == 2
        assert out.state_traj[2]["A0"]["x"] == 1
        assert out.state_traj[2]["A0"]["y"] == 3

    def test_ma_ceq_learning(self):
        two_player = """
        # # # # #
        # G0 . G1 # 
        # . . . #
        # A0 . A1 #
        # # # # #
        """
        gg = TabularGridGame(two_player,agent_symbols=("A0","A1"),goal_symbols=(("G0",("A0",)),("G1",("A1",))),step_cost=-1,collision_cost=-1,goal_reward=100)
        initial_state = gg.initial_state_dist().sample()
        res = TabularCorrelatedQLearner(["A0","A1"],{},show_progress=True,discount_rate=.99,epsilon=.1,num_episodes=500).train_on(gg)
        out = res.policy.run_on(gg)
        assert out.state_traj[0]["A0"]["x"] == 1
        assert out.state_traj[0]["A0"]["y"] == 1
        assert out.state_traj[1]["A0"]["x"] == 1
        assert out.state_traj[1]["A0"]["y"] == 2
        assert out.state_traj[2]["A0"]["x"] == 1
        assert out.state_traj[2]["A0"]["y"] == 3
        
    
        