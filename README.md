## Deep RL Agent for the RTC of Storm Water System 

Source code and data used in the *Deep Reinforcement Learning for the Real Time Control of Stormwater Systems* paper. 

![RLagent](./data/RL_main_fig_1.png)

## Dependencies
Python dependencies for the project can be installed using **requirements.txt**

Storm water network is simulated using EPA-SWMM and pyswmm/matswmm. Matswmm has been deprecated and we strongly suggest using pyswmm. 
 
pyswmm/matswmm makes function calls to a static c library. Hence, we advice gcc-8.0 for pyswmm and gcc-4.2 for matswmm. 


## Further references 
1. Reinforcement Learning by Sutton and Barto
2. DQN
3. Batch Normalization 
