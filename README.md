# Reinforcement Learning
## Problem description:

Jack manages two car parks in a town. Every day, some number of cars are rented from the
two car parks and some number of cars are returned. If the car park has cars available for
renting, Jack can rent one car and receive $10. Otherwise, the business is lost and Jack does
not receive any money. Jack can move cars between two locations overnight, with a cost of $2
per car moved. Cars are available for renting the day after they are returned.

Every day, the number of rental requests at car park 1 and 2 follows Poisson distribution
with λ=3 and 4 respectively. The number of cars returned at car park 1 and 2 follows
Poisson distribution with λ=3 and 2 respectively.

There can be at most 20 cars at each car park. If the number of cars exceeds 20, extra cars
are returned to the company(with no extra cost). A maximum of 5 cars can be moved during
one night.

The discount rate for the reward is γ = 0.9. Take the problem as a finite Markov decision
process, with time steps being a day. The state is the number of cars at two car parks at the
end of the day. The action is number of cars to move over the night. And the reward is the
money Jack earns. Try to find the optimal policy for Jack to earn the most reward.

## Method
Monte Carlo methods: On-policy method with First-time MC visit.

## Result
Optimal policy with 5000 episodes and 20000 days in each episode:
![5000_episode_20000_T_final_policy](https://user-images.githubusercontent.com/62026976/115825790-83227a80-a43c-11eb-8d47-ccda48da3ff6.png)

