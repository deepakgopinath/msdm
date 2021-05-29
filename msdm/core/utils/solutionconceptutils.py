def minimax_equilibrium():
    pass 

def correlated_equilibrium(indiv_strategies,payoffs,objective_func):
    """
    indiv_strategies: Dict[string -> List]: Mapping from an agent name to a list of actions available to that agent 
    payoffs: Dict[FrozenDict[string -> action] -> FrozenDict[string -> float]]: Mapping from a joint action to dictionary of 
    rewards for each agent
    """
    joint_strategies = list(payoffs.keys())
    player_names = list(indiv_strategies.keys())
    cvxopt.solvers.options['show_progress'] = False
    max_action_length = len(max(indiv_actions,key=lambda x:len(x)))
    # Joint Actions X Number of Agents X Maximum number of individual actions
    constraint_matrix = np.zeros((len(joint_strategies),len(player_names),max_action_length))
    for js_i,joint_strategy in enumerate(joint_strategies):
        for player_i,player in enumerate(player_names):
            orig_action = joint_strategy[player]
            orig_util = payoffs[joint_strategy][player]
            for strat_i in range(max_action_length):
                if strat_i >= len(indiv_strategies[player]):
                    constraint_matrix[js_i][player_i][strat_i] = 0.0 # For case where agents have different numbers of actions 
                else:
                    joint_strategy[agent] = indiv_strategies[player][strat_i]
                    new_q_val = payoffs[joint_strategy][player]
                    diff = (orig_q_val - new_q_val)
                    constraint_matrix[js_i][player_i][strat_i] += diff
            joint_action[player] = orig_action
    value = np.zeros((len(player_names)))
    constraint_matrix = np.reshape(constraint_matrix,(len(joint_strategies),-1))
    positive_const = np.identity(len(joint_strategies))
    G = np.concatenate((constraint_matrix,positive_const),axis=1).T
    h = np.zeros(constraint_matrix.shape[1]+positive_const.shape[1])
    A = np.ones((1,len(joint_strategies)))
    b = np.ones(1)
    c = objective_func(payoffs,joint_strategies)
    
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    c = cvxopt.matrix(c)
    sol = cvxopt.solvers.lp(-c,-G,h,A,b)
    equilibrium = float(sol["primal objective"])
    strategy = np.array(list(sol['x']))
    expected_value = np.dot(payoff_matrix.T,strategy)
    value[:] = expected_value
    return strategy,value 

def nash_equilibrium():
    """
    Only valid for two player games
    """
    pass 


"""
Objective Functions for correlated equilibria
"""

def utilitarian(payoffs,joint_strategies):
    """
    payoffs: Dict[FrozenDict[string -> action] -> FrozenDict[string -> float]]: Mapping from a joint action to dictionary of
    utilities for each player 
    joint_strategies: list: a list of each possible joint strategy 
    """
    social_welfare = np.zeros(len(joint_strategies))
    for js_i,joint_strategy in enumerate(joint_strategies):
        for player in joint_strategy.keys():
            social_welfare[js_i] += payoffs[joint_strategy][player]
    return social_welfare 
