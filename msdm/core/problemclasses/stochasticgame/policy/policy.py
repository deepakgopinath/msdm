from abc import ABC, abstractmethod
from msdm.core.algorithmclasses import Result
from msdm.core.problemclasses.stochasticgame import StochasticGame
from msdm.core.distributions import Distribution

class MultiagentPolicy(ABC):
    
    @abstractmethod
    def joint_action_dist(self, s) -> Distribution:
        pass

    def run_on(self,
               problem: StochasticGame,
               initialState=None,
               maxSteps=20):
        if initialState is None:
            initialState = problem.initial_state_dist().sample()
        traj = []
        s = initialState
        for t in range(maxSteps):
            a = self.joint_action_dist(s).sample()
            ns = problem.next_state_dist(s, a).sample()
            r = problem.joint_rewards(s, a, ns)
            traj.append((s, a, ns, r))
            if problem.is_terminal(ns):
                break
            s = ns
        states, actions, nextstates, rewards = zip(*traj)
        return Result(**{
            'state_traj': states,
            'action_traj': actions,
            'reward_traj': rewards
        })

class Policy(ABC):
    
    @abstractmethod 
    def action_dist(self,s) -> Distribution:
        pass 
    

