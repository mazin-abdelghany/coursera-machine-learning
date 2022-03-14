# Week 2: 

### Multivariate Linear Regression
#### Typical notation
- n = number of features
- x(i) = features of the i-th training example
	- i.e., x(i) has all the features (e.g., size of house, number of bedrooms) being used to construct the model
- x(i)_j = value of feature j of the i-th training example

#### Hypothesis function for multivariate linear regression
For n features:
- h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
- include x_0 = 1
- THETA = [theta_0 theta_1 ... theta_n]
- X = [x_0 x_1 ... x_n]
	- NB: these are n+1-dimensional vector
- Therefore, the hypothesis can be rewritten as:
	- h_theta(x) = (THETA)-transpose * X

### Gradient descent for multivariate linear regression
#### Feature scaling
- Make sure features are on the same scale
- If there are features that are orders of magnitude larger than each other, this will contort the contour of the cost function and its derivative making gradient descent much less efficient
- Try to get all the features approximately -1 < x(i) < 1
	- Example of well scaled features:
		- 0 < x(i) < 3
		- -2 < x(i) < 1
		- -0.3 < x(i) < 0.3
	- Example of poorly scaled features:
		- -100 < x < 100
		- 0.0001 < x < 0.00003
- Mean normalization is also typically performed
- Typical procedure for scaling and mean normalization:
	- feature - mean(feature) / either range(feature) or std(feature)

#### Debugging gradient descent
- Plot the cost function on the y-axis with number of iterations on the x-axis
	- This assesses if the cost is truly going down after every iteration
	- This also helps to quantify the number of iterations for convergence of gradient descent
		- Number of iterations for convergence cannot be predicted
		- Automatic methods are available, but plotting is more informative

#### Learning rate
- If the cost is rising or oscillating over iterations, consider reducing learning rate
- It is known that for a sufficiently small alpha, cost should decrease on every iteration
	- If this is not occuring, reassess learning rate or algorithm implementation
- If alpha is too small, gradient descent may be very slow to converge
- Try running gradient descent for several learning rates:
	- 0.001 -> 0.003 -> 0.01 -> 0.03 -> 0.1 -> 0.3 -> 1
	- This method tries to find a learning rate that is too smale and a learning rate that is too large to try choosing the largest learning rate (for faster convergence) that still converges when gradient descent runs

#### Polynomial regression
- New features can be created from the given features present (i.e., multiplying two features, making a feature the square of itself)
- By raising a feature to powers, the above methods can be used for polynomial regression
	- e.g., from a feature x_1, another feature can be made x_2 by performing (x_1)^2
	- **ENSURE THAT FEATURES ARE SCALED**
- Other functions can be applied to features and used to fit other curves using linear regression machinery

#### Normal equation
- See the pdf file in this repo for my derivation of the normal equation
- Feature scaling is not required if using the normal equation to obtain the optimal theta values

**Gradient descent vs. normal equation**
|Gradient descent | Normal equation |
|---------------|------------------|
| Need to choose alpha | No need to choose alpha|
| Needs many iterations | No need to iterate |
| Works well even if feature space* is large | Slow if feature space is large  

\*large feature space, n > ~10,000

##### Normal equation and non-invertibility
- What if X'X is non-invertible?
	- There may be redundant features
	- There may be too many features with respect to the number of training examples
		- Can consider regularization here
		- Can consider removing features 
- Calculate the pseudoinverse instead
