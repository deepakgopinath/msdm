import unittest

import numpy as np
from msdm.algorithms import VectorizedValueIteration
from msdm.tests.domains import Counter, GNTFig6_6, Geometric


def simple_vi_entreg(mdp, *, iterations=1000):
    rewards = np.zeros(len(mdp.state_list))
    visited = set()
    for s in mdp.state_list:
        for a in mdp.actions(s):
            for ns in mdp.next_state_dist(s, a).support:
                if ns in visited:
                    assert rewards[ns] == mdp.reward(s, a, ns)
                    continue
                visited.add(ns)
                rewards[ns] = mdp.reward(s, a, ns)

    v = np.zeros(len(mdp.state_list))
    for _ in range(iterations):
        for s in mdp.state_list:
            if mdp.is_terminal(s):
                v[s] = 0
                continue
            p_state = np.zeros(len(mdp.state_list))
            actions = mdp.actions(s)
            for a in actions:
                for ns, nsp in mdp.next_state_dist(s, a).items(probs=True):
                    p_state[ns] += 1/len(actions) * nsp
            v[s] = np.log(p_state @ np.exp(rewards + v))
    return v

def todorov2006(mdp, *, iterations=1000):
    # https://papers.nips.cc/paper/2006/hash/d806ca13ca3449af72a1ea5aedbed26a-Abstract.html
    P = np.zeros((len(mdp.state_list), len(mdp.state_list)))
    G = np.zeros(len(mdp.state_list))
    for s in mdp.state_list:
        if mdp.is_terminal(s):
            G[s] = 0
            P[s, s] = 1
            continue
        actions = mdp.actions(s)
        minus_q = None
        for a in actions:
            nsdist = mdp.next_state_dist(s, a)
            for ns, prob in nsdist.items(probs=True):
                P[s, ns] += 1/len(actions) * prob
                if minus_q is None:
                    minus_q = mdp.reward(s, a, ns)
                else:
                    # HACK need to make sure this is right?
                    assert minus_q == mdp.reward(s, a, ns)
        # HACK TODO pretty sure this is wrong
        G[s] = minus_q
    G = np.diag(np.exp(G))
    assert np.allclose(P.sum(1), np.ones(len(mdp.state_list)))

    z = np.ones(len(mdp.state_list))
    for _ in range(iterations):
        z = G @ P @ z
    V = np.log(z)
    return V

class VITestCase(unittest.TestCase):
    def test_value_iteration(self):
        mdp = Counter(3)
        res = VectorizedValueIteration().plan_on(mdp)
        out = res.policy.run_on(mdp)
        assert out.state_traj == (0, 1, 2)
        assert out.action_traj == (1, 1, 1)
        assert res.policy.action(0) == 1
        assert res.policy.action(1) == 1
        assert res.policy.action(2) == 1

    def test_value_iteration_geometric(self):
        mdp = Geometric(p=1/13)
        res = VectorizedValueIteration(iterations=500).plan_on(mdp)
        print(res.policy.action_dist(0))
        assert np.isclose(res.V[0], -13), res.V

    def test_value_iteration_entreg(self):
        # TODO test to make sure differing # of actions are handled ok?
        # TODO need to test that rewards for different states are appropriately handled
        mdp = Geometric(p=2/3)
        print(mdp.transition_matrix)
        for mdp in [
            # This is nice for debugging.
            Counter(2),
            # Makes sure things work for larger problems
            Counter(10),
            # Making sure to test a stochastic problem
            Geometric(),
        ]:
            print(mdp)
            iters = 1000
            res = VectorizedValueIteration(entropy_regularization=True, iterations=iters).plan_on(mdp)
            V = np.array([res.V[s] for s in mdp.state_list])
            assert np.allclose(todorov2006(mdp, iterations=iters), simple_vi_entreg(mdp, iterations=iters))
            assert np.allclose(V, todorov2006(mdp, iterations=iters))
