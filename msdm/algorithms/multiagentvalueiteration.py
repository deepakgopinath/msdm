from msdm.core.algorithmclasses import Result
from msdm.core.algorithmclasses import Plans
from msdm.core.problemclasses.stochasticgame.tabularstochasticgame import TabularStochasticGame
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiagentPolicy, SingleAgentPolicy
from msdm.core.assignment.assignmentmap import AssignmentMap
from tqdm import tqdm
from typing import Iterable
import numpy as np 
import itertools 
from scipy.special import softmax
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time

    
class TabularMultiagentValueIteration(Plans):
    
    def __init__(self,planning_agents:Iterable,other_policies:dict,num_iterations=200,
                 discount_rate=1.0,default_q_value=0.0,max_steps=50,all_actions=True,
                 tol=0.0,show_progress=False,alg_name="Value Iteration"):
        """
        Parameters:
            planning_agents (Iterable): List of agents planning in the environment 
            other_policies (Dict[string -> SingleAgentPolicy]): dictionary of behavioral policies for 
                each non-planning agent 
            num_episodes (int): number of episodes to train for 
            discount_rate (float): Discount factor for computing values 
            default_q_value (float): initial value to use in the q-table 
            max_steps (int): maximum number of timesteps in any given episode 
            all_actions (bool): If all_actions is true, then the q-table contains an entry for each 
                agent and each joint action. If not, it only has entries for each agent and individual action. 
                Correlated-Q, FFQ and NashQ must use all_actions (and are set to by default). 
            tol (float): tolerance parameter for convergence of value iteration. Default will never terminate early 
            show_progress (bool): If true, will show progress bars for training
            alg_name (string): string used to identify algorithm. Useful when visualizing a bunch of algorithms 
        """
        self.planning_agents = planning_agents 
        self.other_agents = list(other_policies.keys())
        self.other_policies = other_policies 
        self.all_agents = [] 
        self.all_agents.extend(self.planning_agents)
        self.all_agents.extend(self.other_agents)
        self.num_iterations = num_iterations 
        self.dr = discount_rate 
        self.default_q_value = default_q_value 
        self.show_progress = show_progress
        self.all_actions = all_actions 
        self.alg_name = alg_name
        self.max_steps = max_steps
    
    def step(self,problem:TabularStochasticGame,state,actions):
        """
        Executes one step in the environment, returning the state, actions, rewards and next state
        
        Parameters: 
        ::problem:: the environment object 
        ::state:: Current state object 
        ::actions:: Hashable[agent_name -> Hashable[{'x','y'} -> {1,0,-1}]]
        
        Outputs:
        (state,actions,jr,nxt_st). state and actions are same as those passed in. 
        """
        nxt_st = problem.next_state_dist(state,actions).sample()
        jr = problem.joint_rewards(state,actions,nxt_st)
        return state,actions,jr,nxt_st
    
    def plan_on(self,problem: TabularStochasticGame,delta=.0001):
        # initialize q values for each planning agent 
        res = Result()
        res.Q = {agent_name: AssignmentMap() for agent_name in self.all_agents}
        
        for state in problem.state_list:
            for agent_name in self.all_agents:
                if not self.all_actions:
                    res.Q[agent_name][state] = AssignmentMap()
                    indiv_actions = list(problem.joint_actions(state)[agent_name])
                    for action in indiv_actions:
                        res.Q[agent_name][state][action] = self.default_q_value
                else:
                    res.Q[agent_name][state] = AssignmentMap()
                    joint_actions = problem.joint_actions(state)
                    ja_keys,ja_values = zip(*joint_actions.items())
                    all_joint_actions = [dict(zip(ja_keys, list(v))) for v in itertools.product(*ja_values)]
                    for joint_action in all_joint_actions:
                        res.Q[agent_name][state][joint_action] = self.default_q_value

        if self.show_progress:
            iters = tqdm(range(self.num_iterations))
        else:
            iters = range(self.num_iterations)
        for i in iters:
            for state in problem.state_list:
                for action in problem.joint_action_list:
                    next_vals = {agent:0.0 for agent in self.planning_agents}
                    for next_state,prob in problem.next_state_dist(state,action).items():
                        rewards = problem.joint_rewards(state,action,next_state)
                        for agent in self.planning_agents:
                            new_q = self.update(agent,action,res.Q,rewards,state,next_state,problem)
                            next_vals[agent] += prob*new_q
                    for agent in self.planning_agents:
                        if self.all_actions:
                            res.Q[agent][state][action] = next_vals[agent]
                        else:
                            res.Q[agent][state][action[agent]] = next_vals[agent]
        # Converting to dictionary representation of deterministic policy
        pi = self.compute_deterministic_policy(res.Q,problem)

        # add in non-learning agents 
        for agent in self.other_agents:
            pi[agent] = self.other_policies[agent]
            
        # create result object
        res.problem = problem
        res.policy = {}
        res.policy = res.pi = TabularMultiAgentPolicy(problem, pi,self.dr,show_progress=self.show_progress)
        return res
    
    def compute_deterministic_policy(self,q_values,problem):
        """
        Turns the Q-values for each learning agent into a deterministic policy
        
        inputs: 
        ::q_values:: Hashable[agent_name -> Hashable[state -> Hashable[action -> float]]]
        ::problem:: Environment object 
        
        outputs: 
        ::pi:: Hashable[learning_agent -> SingleAgentPolicy]
        """
        pi = AssignmentMap()
        for agent in q_values:
            pi[agent] = AssignmentMap()
            for state in q_values[agent]:
                pi[agent][state] = AssignmentMap()
                # Picks randomly among maximum actions 
                max_val = max(q_values[agent][state].items(),key=lambda k:k[1])[1]
                max_acts = []                
                for act in problem.joint_action_list:
                    if self.all_actions:
                        if q_values[agent][state][act] == max_val:
                            max_acts.append(act)
                    else:
                        if q_values[agent][state][act[agent]] == max_val:
                            max_acts.append(act[agent])
                max_act = np.random.choice(max_acts)
                if self.all_actions:
                    max_act = max_act[agent]
                    
                for ai,action in enumerate(problem.joint_action_list):
                    if self.all_actions:
                        if action[agent] == max_act:
                            pi[agent][state][action[agent]] = 1.0
                        else:
                            pi[agent][state][action[agent]] = 0.0
                    else:
                        if action[agent] == max_act:
                            pi[agent][state][action[agent]] = 1.0
                        else:
                            pi[agent][state][action[agent]] = 0.0
            pi[agent] = SingleAgentPolicy(agent,problem,pi[agent],q_values[agent],self.all_actions)
        return pi 

    
    def update(self,agent_name,actions,q_values,joint_rewards,curr_state,next_state,problem):
        """
        Uses current q-values and results from one step in the environment to compute 
        the change in q-values. 
        """
        if problem.is_terminal(next_state):
            return joint_rewards[agent_name]
        if problem.is_terminal(curr_state):
            return 0.0
        q_del = joint_rewards[agent_name]
        q_del += self.dr*(max(q_values[agent_name][next_state].items(),key=lambda k:k[1])[1])
        return q_del 
    

        
    
    
    
        
    
