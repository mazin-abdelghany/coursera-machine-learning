# Week 3: 

### Classification
#### Binary classification
- Linear regression cannot be used for classification
- Logistic regression is used instead
  - Regression here is a misnomer

#### Hypothesis representation
- Linear regression: h_theta(x) = theta transpose x
- Logistic regression: g( h_theta(x) ) where g() is the sigmoid/logistic function
  - This serves to confine output values between 0 and 1
  - Output is the probability that y = 1 on input x parameterized by theta

#### Decision boundary
- Logisitc regression draws a decision boundary (plane or hyperplace) for prediction
	- Typically defined as the output of h_theta(x) (probability) > 0.5 predicts y = 1
- Nonlinear, complex decision boundaries can be created by adding features modified by functions such as high order polynomials
- NB: The decision boundary is defined by the parameters (theta) **not** the data set
	- In other words, the data is used to fit the parameters thetas and then the thetas define the decision boundary

#### Cost function for logistic regression
- The sum of squared errors cost function, when applied to logistic regression, is non-convex (i.e., has many local optima)
- Instead, the cost function to ensure convexity:
	- if y = 1: -log(h(x))
	- if y = 0: -log(1 - h(x))
- This function captures the intuition that if the model predicts 1 and the true value of y = 1, the cose is 0; if the model predicts 0, but the true value of y = 1, this is penalized with a very high cost (approaching infinity)
- The opposite of the above is true when y = 0
- Since y is always either only 0 or 1:
	- -ylog(h(x)) - (1 - y)log(h(x))
- Minimizing the cost using gradient descent will give the optimal values of theta

#### Advanced optimization
- Gradient descent is not the only algorithm that can be used to compute the optimum thetas
- There are other algorithms that can take the cost and the derivative of the cost:
	- Conjugate gradient
	- BFGS
	- L-BFGS
- Advantages: learning rate does not need to be chosen, often faster than gradient descent
- Disadvantages: complex algorithms
- Look for "good" implementations of these algorithms if using them instead of gradient descent

#### Multiclass Classification
- One vs. all (one v. rest) classification:
	- If there are 3 classes to predict, then turn this into 3 separate binary classification problems
	- Fit 3 classifies and each classifier is assessing the probability that the feature set belongs to each class separately
		- A prediction is made by running all three classifiers and then choose the prediction with the highest probability
		
#### Solving the problem of overfitting
- Underfit - model has high bias
	- does not predict well because the algorithm has too much of a "preconceived" notion of the data
- Overfit - model has high variance
	- predicts very well on the training set, but fails to generalize to new examples
- How to address the problem of overfitting:
	- Plotting the data if there are few features, though usually not possible
	- Reduce the number of features (manual v. feature/model selection algorithms)
		- This may remove important features
	- Regularization (see the pdf in this repo for a derivation of regularization of linear regression)

#### Regularization
- Penalize the parameters to simplify the hypothesis function; less prone to overfitting
- (Do not penalize the intercept term theta_0)
- There is a regularization parameter (lambda) that balances the penalty to the parameters
	- Large lambdas reduce the thetas to zero (akin to fitting a flat line) -- underfitting
	- Small lambdas do not penalize the thetas -- overfitting is not solved
	- **Lambda should be chosen with care**
- If the number of examples is smaller than the number of features, the normal equation becomes non-invertible
	- Regularization can solve this issue
