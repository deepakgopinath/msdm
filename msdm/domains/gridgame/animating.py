from msdm.domains.gridgame.tabulargridgame import TabularGridGame
from msdm.domains.gridgame.plotting import GridGamePlotter
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d
import numpy as np


class GridGameAnimator(GridGamePlotter):
    """
    Does basic animations of trajectorys to the TabularGridGame environment 
    """
    
    def __init__(self, gg: TabularGridGame, figure:plt.Figure, ax: plt.Axes):
        super().__init__(gg,ax)
        self.figure = figure
        
    def animate_trajectory(self,trajectory,interval=20,easing=True,interp_factor=20):
        """
        Generates a matplotlib animation object for a given trajectory
        
        inputs:
        ::trajectory:: Result object from the policy's run_on function 
        ::interval:: time between rendering of frames in the animation
        ::easing:: boolean determining whether or not to use easing (interpolating extra frames for smoother movement)
        ::interp_factor:: number of smoothing frames to add between each frame 
        
        output:
        ::anim:: matplotlib FuncAnimation object 
        """
        stateTraj = trajectory.state_traj
        if easing:
            stateTraj = self.state_trajectory_easing(stateTraj,interp_factor)
            
        def init():
            return self.agents.values()
        
        def animate(i):
            currState = stateTraj[i]
            for agent in self.agents:
                x,y = (currState[agent]["x"],currState[agent]["y"])
                self.agents[agent].set_data(x+.5,y+.5)
            return self.agents.values()
        
        anim = animation.FuncAnimation(self.figure,animate,init_func=init,frames=len(stateTraj),interval=interval,blit=True)
        return anim
    
    def state_trajectory_easing(self,stateTraj,interp_factor=20):
        """
        Uses linear interpolation to add frames between states. Makes the animation much smoother 
        
        inputs:
        ::stateTraj:: list of states in the trajectory 
        ::interp_factor:: number of frames to add between each state 
        
        outputs:
        new_frames: new list of states, with smoothed states added in 
        """
        new_frames = []
        for i in range(len(stateTraj)-1):
            init_pos = stateTraj[i]
            end_pos = stateTraj[i+1]
            x_vals = {}
            y_vals = {}
            interp_funcs = {}
            interp_vals = {}
            for agent in init_pos:
                x_vals[agent] = (init_pos[agent]["x"],end_pos[agent]["x"])
                y_vals[agent] = (init_pos[agent]["y"],end_pos[agent]["y"])
                if x_vals[agent][0] != x_vals[agent][1]:
                    interp_funcs[agent] = interp1d(x_vals[agent],y_vals[agent])
                    max_x = max(x_vals[agent])
                    min_x = min(x_vals[agent])
                    interp_vals[agent] = np.linspace(min_x,max_x,num=interp_factor)
                    # Reversing direction so the agent doesn't move backwards
                    if max_x != x_vals[agent][1]:
                        interp_vals[agent] = np.flip(interp_vals[agent])
                else:
                    interp_funcs[agent] = None
                    max_y = max(y_vals[agent])
                    min_y = min(y_vals[agent])
                    interp_vals[agent] = np.linspace(min_y,max_y,num=interp_factor)
                    if max_y != y_vals[agent][1]:
                        interp_vals[agent] = np.flip(interp_vals[agent])
                
            new_frames.append(init_pos)
            for i in range(interp_factor):
                new_frame = {}
                for agent in x_vals:
                    if interp_funcs[agent] != None:
                        interp_x = interp_vals[agent][i]
                        interp_y = interp_funcs[agent](interp_x)
                        new_frame[agent] = {"x":interp_x,"y":interp_y}
                    else:
                        interp_y = interp_vals[agent][i]
                        new_frame[agent] = {"x":x_vals[agent][0],"y":interp_y}
                new_frames.append(new_frame)
            new_frames.append(end_pos)  
        return new_frames
            
            
            
                    
        
                
        