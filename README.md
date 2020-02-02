# Solving FrozenLake using exact methods

Solving FrozenLake-v0 environment using value iteration and Q-value iteration.  
Environment description: https://gym.openai.com/envs/FrozenLake-v0/  

# Results

Value iteration solved MDP at 75 iterations.  
Q-value iteration solved MDP at 58 iterations.  
Both methods found slightly different optimal policies,  
because there were multiple optimal actions in some states.  
It tooks approximately 73 steps to reach a frisbee (when not falling into a hole), and  
chance to fall into a hole is approx. 25%.  

![val_result](/val_result.png)
![q_val_result](/q_val_result.png)

# Reference
I referred code from here:  
https://github.com/realdiganta/solving_openai/tree/master/FrozenLake8x8  
Thanks!  
