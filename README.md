# Simulations
## Dependencies
1. Matswmm
2. tensorflow
3. keras


## Single pond
Classic Q learning[1] was used to train a controller to maintain a constant water level in the pond. Controller has the capability to alter the amount of water released from the pond at every timestep.

![singletank](single_tank/pond_height.jpg "Trained controller maintaining height")
![avg_reward](single_tank/mean_rewards.jpg "Improvement of average reward acheived by the agent per each episode")

- [ ] Add the case with reducing exploration value
- [ ] Inflow with disturbances
- [ ] Varying intial water level in the tank
- [ ] Add uncontrolled response

### References
1. Reinforcement Learning - Sutton and Barto

## Tanks in series

![tanks_series](tanks_in_series/series_network.JPG "Tanks in series")

Though the classical q learning agent has encouraging results, it fails to capture the intricaies of a more complex environemnt.
Deep q learning, a variant of the classical q learning algorithm was used to train these controllers.

### Uncontrolled response during a storm event

![uncon_response](tanks_in_series/series_uncontrolled.jpeg "Uncontrolled response during storm event")


