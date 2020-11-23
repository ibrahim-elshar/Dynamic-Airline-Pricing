# Dynamic Airline Pricing

### IE 3094 MDP Project

We study the problem of dynamic airline pricing when the customers can return their tickets for a posted price each period.  The goal is to optimize the prices to set for flight tickets $p_t$ and the value of returned tickets $p_{rt}$ at the beginning of period t. 
Note that there is a trade-off in setting the prices for the demand. Setting low fares at the beginning of the planning horizon may result in a full booking but may be less profitable than following a strategic policy that sets the price to a certain threshold and then increase the price in each period.

We formulate the problem as an MDP and solve for the optimal policy.
Our numerical examples show that, in the case where demand is affected by the flexible returned tickets price (even by a very small amount), the added value of having dynamic returned ticket price is significant and the optimal policy is no longer monotone decreasing in the price. Otherwise, it is always optimal to set the sale price higher than the return price.