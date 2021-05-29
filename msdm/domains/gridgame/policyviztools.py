from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiagentPolicy
from msdm.core.assignment.assignmentmap import AssignmentMap
from tqdm import tqdm
import numpy as np 

def projected_Q(policy: TabularMultiagentPolicy, agent_name: str,q_matrix: np.array,weight_matrix: np.array,initial_state: dict):
    """
    Computes the Q-values for each position(x,y coordinate an agent can occupy) in the problem. Averages over the q-values for all states including the agent in that position using the weight_matrix to determine weighting(usually the occupancy distribution for the poicy and enviroment).

    inputs: 
    ::policy:: A multiagent policy for a TabularGridGame
    ::agent_name:: string representing the name of the agent 
    ::q_matrix:: numpy matrix of size (num_states,num_actions) representing q_values for each state,action pair(not joint actions).
    ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
    ::initial_state:: Initial state to use when computing expected position occupancy
    
    outputs: 
    ::proj_q_matrix:: a matrix of size (num_positions,num_actions) representing the projected Q values for each position
    """
    indiv_actions = policy.single_agent_policies[agent_name]._actions
    proj_q_matrix = np.zeros((len(policy.problem.position_list),len(indiv_actions)))
    initial_state_occupancy = weight_matrix[policy._states.index(initial_state)]
    proj_q_vals = {position:AssignmentMap() for position in policy.problem.position_list}
    for position in policy.problem.position_list:
        for action in indiv_actions:
            proj_q_vals[position][action] = 0.0

    if policy.show_progress:
        configs = enumerate(tqdm(policy._states,desc="Calculating Projected Q-Values"))
    else:
        configs = enumerate(policy._states)

    for si, state in configs:
        if policy.problem.is_terminal(state):
            continue
        agent_position = (state[agent_name]["x"],state[agent_name]["y"])
        if not policy.single_agent_policies[agent_name].all_actions:
            for ai,action in enumerate(indiv_actions):
                q_val = q_matrix[si][ai]
                projected_val = q_val*initial_state_occupancy[si]
                proj_q_vals[agent_position][action] += projected_val
        else:
            for ai, action in enumerate(policy._joint_actions):
                agent_action = action[agent_name]
                q_val = q_matrix[si][ai]
                projected_val = q_val*initial_state_occupancy[si]
                proj_q_vals[agent_position][agent_action] += projected_val
    for pi, position in enumerate(policy.problem.position_list):
        for ai,action in enumerate(indiv_actions):
            proj_q_matrix[pi][ai] = proj_q_vals[position][action]
    return proj_q_matrix


def projected_V(policy: TabularMultiagentPolicy, agent_name:str, q_matrix:np.array, weight_matrix:np.array, initial_state:dict):
    """
    Uses the projected_Q functions to compute the projected values
    
    inputs:
    ::policy:: A TabularMultiagentPolicy from a TabularGridGame
    ::agent_name: string representing agent's name 
    ::q_matrix:: numpy matrix of size (num_states,num_actions) representing q_values for each state,action pair(not joint actions).
    ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
    ::initial_state:: Initial state to use when computing expected position occupancy
    
    outputs: 
    ::proj_v:: numpy matrix of size (num_positions) representing the expected value for each position for 
    the given agent. 
    """
    proj_q = projected_Q(policy,agent_name,q_matrix,weight_matrix,initial_state)
    proj_v = np.zeros((len(policy.problem.position_list)))
    for i,position in enumerate(proj_q):
        proj_v[i] = np.max(position)
    return proj_v


def positionMapping(policy,agent_name,q_matrix,weight_matrix,initial_state):
    """
    Computes the projected Values, and then turns them into 
    a dictionary from position -> value for visualization 
    
    inputs:
    ::policy:: A TabularMultiagentPolicy from a TabularGridGame
    ::agent_name: string representing agent's name 
    ::q_matrix:: numpy matrix of size (num_states,num_actions) representing q_values for each state,action pair(not joint actions).
    ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
    ::initial_state:: Initial state to use when computing expected position occupancy
    
    outputs:
    ::mapping:: Dictionary from positions to values used to visualize V projection 
    """
    projected_vals = projected_V(policy,agent_name,q_matrix,weight_matrix,initial_state)
    mapping = {}
    for i,position in enumerate(policy.problem.position_list):
        mapping[position] = projected_vals[i]
    return mapping 
    
    
def positionActionMapping(policy,agent_name,q_matrix,weight_matrix,initial_state):
    """
    Computes the projected Q-values, and then turns them into 
    a dictionary from position,action -> value for visualization 
    
    inputs:
    ::policy:: A TabularMultiagentPolicy from a TabularGridGame
    ::agent_name: string representing agent's name 
    ::q_matrix:: numpy matrix of size (num_states,num_actions) representing q_values for each state,action pair(not joint actions).
    ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
    ::initial_state:: Initial state to use when computing expected position occupancy
    
    outputs:
    ::mapping:: A dictionary from position,action pairs to values 
    """
    projected_q_vals = projected_Q(policy,agent_name,q_matrix,weight_matrix,initial_state)
    mapping = AssignmentMap()
    for i,position in enumerate(policy.problem.position_list):
        mapping[position] = AssignmentMap() 
        for j,action in enumerate(policy.single_agent_policies[agent_name]._actions):
            mapping[position][action] = projected_q_vals[i][j]
    return mapping


def weightMapping(policy,agent_name,weight_matrix,initial_state):
    """
    Uses mapping from state -> weight to create a mapping from 
    position -> weight 
    
    inputs:
    ::policy:: A TabularMultiagentPolicy from a TabularGridGame
    ::agent_name: string representing agent's name 
    ::weight_matrix:: matrix of size (num_states,num_states) used to weight the average across states
    ::initial_state:: Initial state to use when computing expected position occupancy
    
    outputs:
    ::positionMap:: A dictionary from positions to weights
    """
    weights = weight_matrix[policy._states.index(initial_state)]
    positionMap = AssignmentMap()
    for si, weight in enumerate(weights):
        state = policy._states[si]
        if policy.problem.is_terminal(state):
            continue
        agent_position = (state[agent_name]["x"],state[agent_name]["y"])
        if agent_position in positionMap:
            positionMap[agent_position] += weight
        else:
            positionMap[agent_position] = weight
    for position in positionMap:
        positionMap[position] = positionMap[position]
    return positionMap


def construct_state(positions):
    """
    Converts Hashable[agent_name -> tuple(x,y)] into 
    a state object
    """
    state = {} 
    for agent in positions:
        state[agent] = {} 
        state[agent]["name"] = agent 
        state[agent]["type"] = "agent"
        state[agent]["x"] = positions[agent][0]
        state[agent]["y"] = positions[agent][1]
    return state