# Week 7

### Large margin classification
#### Optimization objective
- Support vector machines is powerful and widely used
- Compared to logistic regression and neural networks, the SVM sometimes gives a cleaner and more powerful way of learning complex non-linear functions

For logistic regression:
- z = theta_transpose * x
- If y = 1, we want h_theta(x) approx. 1, z >> 0
- If y = 0, we want h_theta(x) approx. 0, z << 0

Taking the cost function:
- -(y * log(h_theta(x)) + (1-y) * log(1-h_theta(x)))
- h_theta(x) = 1 / (1 + e^-z) (where z is defined above)
- When y = 1, cost_1 = -log( 1 / (1 + e^-z) )
- When y = 0, cost_0 = -log(1 - ( 1 / (1 + e^-z) ) )

The support vector machine takes the logistic cost functions at y = 0 and y = 1 and generates line segments that approximate the graphs of the above equations rather than the smooth curve representations.
- Computational advantage
- Easier optimization problem

SVM cost: C * sum(y(i) * cost_1(theta_transpose * x(i)) + (1-y(i)) * cost_1(theta_transpose * x(i))) + (1/2)*sum(theta^2)
- By convention, the SVM minimization problem omits the usual (1/m) given that a constant does not change the minimum or the values of theta that achieve the minimum
- Also, the regularization of SVMs includes a constant C out front rather than along with the sum added at the end

SVMs predicts 1 or 0, not a probability.

#### Large margin intuition
- SVMs are considered "large margin classifiers"
