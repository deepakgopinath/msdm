from msdm.core.algorithmclasses import Result
from msdm.core.algorithmclasses import Learns
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

    
class TabularMultiagentQLearner(Learns):
    
    def __init__(self,learning_agents:Iterable,other_policies:dict,num_episodes=200,
                 learning_rate=.1,discount_rate=1.0,epsilon=0.0,
                 epsilon_decay=1.0,default_q_value=0.0,max_steps=50,all_actions=True,
                 show_progress=False,alg_name="Q-Learning"):
        """
        Parameters:
            learning_agents (Iterable): List of agents learning in the environment 
            other_policies (Dict[string -> SingleAgentPolicy]): dictionary of behavioral policies for 
                each non-learning agent 
            num_episodes (int): number of episodes to train for 
            learning_rate (float): number determining size of q-value updates 
            discount_rate (float): Discount factor for computing values 
            epsilon (float): probability of a random move in the environment 
            epislon_decay (float): rate at which epsilon value will decay. Decays after each episode 
            default_q_value (float): initial value to use in the q-table 
            max_steps (int): maximum number of timesteps in any given episode 
            all_actions (bool): If all_actions is true, then the q-table contains an entry for each 
                agent and each joint action. If not, it only has entries for each agent and individual action. 
                Correlated-Q, FFQ and NashQ must use all_actions (and are set to by default). 
            show_progress (bool): If true, will show progress bars for training
            alg_name (string): string used to identify algorithm. Useful when visualizing a bunch of algorithms 
        """
        self.learning_agents = learning_agents 
        self.other_agents = list(other_policies.keys())
        self.other_policies = other_policies 
        self.all_agents = [] 
        self.all_agents.extend(self.learning_agents)
        self.all_agents.extend(self.other_agents)
        self.num_episodes = num_episodes 
        self.lr = learning_rate 
        self.dr = discount_rate 
        self.eps = epsilon 
        self.default_q_value = default_q_value 
        self.show_progress = show_progress
        self.all_actions = all_actions 
        self.alg_name = alg_name
        self.epsilon_decay = epsilon_decay
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
    
    def pick_action(self,curr_state,q_values,problem):
        """
        Picks actions using epsilon-greedy scheme for each agent
        
        Parameters:
        ::curr_state:: current state object 
        ::q_values:: Hashable[agent_name -> Hashable[state -> Hashable[action -> float]]]
        ::problem:: environment object 
        
        Outputs:
        ::actions:: Hashable[agent_name -> Hashable[{'x','y'} -> {1,0,-1}]]
        """
        actions = {agent_name: None for agent_name in self.learning_agents}
        for agent_name in self.learning_agents:
            indiv_actions = list(problem.joint_actions(curr_state)[agent_name])
            # Chooses randomly among maximum actions 
            max_val = max(q_values[agent_name][curr_state].items(),key=lambda k:k[1])[1]
            max_acts = []
            for act in q_values[agent_name][curr_state].items():
                if act[1] == max_val:
                    max_acts.append(act[0])
            max_act = np.random.choice(max_acts)
            if self.all_actions:
                max_act = max_act[agent_name]
            # Choose action using epsilon-greedy policy 
            action = np.random.choice([max_act,indiv_actions[np.random.choice(len(indiv_actions))]],p=[1-self.eps,self.eps])
            actions[agent_name] = action

        # Getting actions for friendly agents 
        for agent in self.other_agents:
            actions[agent] = self.other_policies[agent].action_dist(curr_state).sample()
        return actions 
    
    def train_on(self,problem: TabularStochasticGame) -> Result:
        """
        Trains on the given problem for self.num_episodes. Returns a result object containing 
        the policy and q-values 
        
        Parameters:
            problem (TabularStochasticGame): The environment the algorithm is learning in 
        Output:
            res (Result): Result object containing policy and q-values
        """
        # initialize Q values for each agent using q learning
        res = Result()
        res.Q = {agent_name: AssignmentMap() for agent_name in self.learning_agents}
        for state in problem.state_list:
            for agent_name in self.learning_agents:
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
                        
        # Adds a progress bar 
        if self.show_progress:
            episodes = tqdm(range(self.num_episodes),desc="Training with " + self.alg_name)
        else:
            episodes = range(self.num_episodes)
            
        for i in episodes:
            curr_state = problem.initial_state_dist().sample()
            curr_step = 0
            while not problem.is_terminal(curr_state) and curr_step < self.max_steps:
                # Choose action 
                actions = self.pick_action(curr_state,res.Q,problem)
                curr_state,actions,jr,nxt_st = self.step(problem,curr_state,actions)
                # update q values for each agent 
                for agent_name in self.learning_agents:
                    new_q = self.update(agent_name,actions,res.Q,jr,curr_state,nxt_st,problem)
                    if not self.all_actions:
                        res.Q[agent_name][curr_state][actions[agent_name]] = (1-self.lr)*res.Q[agent_name][curr_state][actions[agent_name]] + self.lr*new_q
                    else:
                        res.Q[agent_name][curr_state][actions] = (1-self.lr)*res.Q[agent_name][curr_state][actions] + self.lr*new_q
                curr_state = nxt_st
                curr_step += 1
            self.eps *= self.epsilon_decay

        # Converting to dictionary representation of deterministic policy
        pi = self.compute_deterministic_policy(res.Q,problem)

        # add in non_learning agents 
        for agent in self.other_agents:
            pi[agent] = self.other_policies[agent]
            
        # create result object
        res.problem = problem
        res.policy = {}
        res.policy = res.pi = TabularMultiagentPolicy(problem, pi,self.dr,show_progress=self.show_progress)
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
        if problem.is_terminal(next_state):
            return joint_rewards[agent_name]
        if problem.is_terminal(curr_state):
            return 0.0
        q_del = joint_rewards[agent_name]
        q_del += self.dr*(max(q_values[agent_name][next_state].items(),key=lambda k:k[1])[1])
        return q_del 
    

        
    
    
    
        
    
