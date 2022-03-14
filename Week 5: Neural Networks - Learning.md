# Week 5

### Neural network learning
#### Cost function
- The cost function for neural networks is a generalization of the logistic regression cost function
- Essentially, each output class is its own logisitic regression cost function; therefore, summing the costs obtained for each logistic regression classification
- For example, if the classes are:
```
grade 1    grade 2    grade 3    grade 4
  1           0          0          0            
  0           1          0          0
  0           0          1          0
  0           0          0          1
```
and the prediction problem is:
```
target    prediction
  0           0.2       <-- calculate the logisitic cost function, cost1
  0           0.5       <-- calculate the logisitic cost function, cost2
  1           0.9       <-- calculate the logisitic cost function, cost3
  0           0.1       <-- calculate the logisitic cost function, cost4
  
sum all of the costs: total_cost = cost1 + cost2 + cost3 + cost4
```

#### Gradient computation - Backpropagation algorithm
- Compute forword propagation (see Week 4)
- Initialize an empty matrix to accumulate errors
- Compute the error (e.g., delta) for each node going from the output layer backwards through the hidden layers (omitting the input layer)
- The error, delta, for the last layer is simply prediction - truth
- For all other layers, step backwards using this formula to obtain delta(layer):
	- delta(layer) = ((theta(layer))-transpose * delta(layer+1)) .∗ a(layer) .∗ (1−a(layer))
	- where:
		- * is matrix multiplication and .* is element-wise multiplication
		- theta(layer) is a matrix of parameters
		- delta(layer+1) is the prior layer errors
		- a(layer) is the currently layer activation unit values
	- This is only if activation function g(x) is the sigmoid function
	- If the activation function is a different function, then the formula would be:
		-  ((theta(layer))-transpose * delta(layer+1)) .∗ g'(a(layer))
- The partial derivative is then simply the activation unit values times the error terms
- Accumulate the errors in the matrix initialized above
- Multiply the final filled matrix with 1/number_training_examples

#### Backpropagation intuition
- The delta terms are essentially errors for how "far" the predicted value is from the target
- Backpropagation is computing a weighted some of the error terms for each activation unit value
- The deltas, in practice, are the partial derivatives of the cost function
- The partial derivative values are computing how much to change each activation unit value to get the prediction closer to the target

#### Gradient checking
- Bugs are difficult to find when implementing backpropagation
- It may be that it appears that backpropagation is working correctly, but the accuracy of the model will be lower than it could be with these subtle bugs
- Gradient checking ensures that the calculated gradient is correct and eliminates these subtle bugs
- The gradient is checked by adding and subtracting a small amount epsilon from theta and manually calculating the slope of the line
	- This is the *two-sided difference* and is slightly more accurate that the *one-sided difference* (only adding epsilon)
- This manually calculated slope is then compared to that which is calculated by backpropagation
- Set epsilon ~0.0001

##### Implementation note:
- Implement backpropagation to compute the gradient
- Implement numeric gradient checking to complete approximate gradient
- Make sure these values are very similar (using a small epsilon)
- Turn off gradient checking and use backpropagation to completing the learning
	- This is turned off because this is very computationally expensive

#### Random initialization (the problem of symmetric weights)
- For gradient descent or advanced optimization, the initial thetas must be given
- Initializing all the parameters to 0 does not work because:
	- all of the hidden units values will compute the same numbers
	- all of the errors will be the same
	- all of the partial derivatives will be the same
- Random initialization breaks this symmetry and is implemented as such:
```
# randomly initializes the weights of a layer with L_in 
# incoming connections and L_out outgoing  connections. 
# between epsilon and -epsilon

epsilon_initializer = 0.12;

# rand() here generates a matrix with random numbers between
# 0 and 1 of size L_out by L_in + 1
W = rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init;
```

#### Putting it all together
1. Choose a network architecture:  
	- The number of input units is determined by the number of features
	- The number of output units is determined by the number of classess (if it is a classification problem)
		- NB: targets should be rewritten as vectors as noted at [the top](#cost-function)
	- Reasonable default architecture:
		- 1 hidden layer
		- or if >1 hidden layer, same number of hidden units in each layer
		- More hidden units is generally better
		- Number of hidden units should be approx. number of features 
2. Randomly initialize weights
3. Implement forward propagation to get h_theta(x(i)) for any x(i)
4. Implement code to compute the cost function J(theta)
5. Implement backpropagation to compute the partial derivatives
```
# In pseudocode:
# each training example is [x(i), y(i)]

for i = 1:num_training_examples {
    perform forward propagation
    perform back propagaion
}
compute the partial derivatives
```
6. Use gradient checking to ensure backpropagation is implemeted correctly
7. Disable gradient checking
8. Use gradient descent or an advanced optimization method to minimize the cost as a function of parameters
    - these algorithms take as inputs the cost and the calculated partial derivatives (gradient)
- NB: Neural networks cost functions are susceptible to the problem of local optima; however, in practice, this does not cause issues with performance 
