# Week 1: 
## Learning algorithms are everywhere

### Introduction
#### Define Machine Learning
- Arthur Samuel (1959): Field of study that gives computers the ability to learn without being explicitly programmed
- Tom Mitchell (1998): A computer program is said to learn from experience E with respect to some task T and some performace measure P if its performance on T, as measured by P, improves with experience E.

#### Supervised Learning
- The "right answers" are given to the algorithm
- Regression: predict a continuous-valued outcome
- Classification: predict a categorical (discrete value) outcome
- SVMs have a mathematical "trick" that can deal with an infinite number of features

#### Unsupervised Learning
- Algorithm is given data and algorithm tries to group or structure the data (e.g., clustering)
	- Computer clustering
	- Market segmentation
	- Astronomical data
	- Genomic analysis
- Cocktail party algorithm to separate voices that are talking over each other or over other noice (e.g., music)

#### Advice
- May be easiest to prototype learning algorithms using `Octave` and then move to another programming environment.

### Model and Cost Function
#### Model Representation
- A train set is fed to the learning algorithm
- Represent a training example as (x, y) and the i-th training example as ( x(i), y(i) )
- A hypothesis function maps x's to y's
- Linear regression represents the hypothesis function as one where y is a linear function of x
	- h_theta(x) = theta_0 + theta_1 * x

#### Cost Function
- The cost is minimized to find the optimum values of theta_0 and theta_1
- Choose values for the parameters (thetas) so that h_theta(x) is close to y given training examples (x,y)
- For most regression problems, the most commonly used cost function is the squared error function
- Intuition
	- The hypothesis function is a function of x for fixed thetas
	- The cost function is a function of theta

#### Gradient Descent
- Start with guesses of thetas
- Calculate the gradient of the cost function
- Change the thetas in the direction of the gradient
- Repeat until the local minimum cost is reached
	- Does not always guarantee the global minimum!
```
# In pseudocode:

for j = 0 through n:
repeat until convergence {
	theta_j = theta_j - alpha * gradient
}

# NB: All thetas must be updated simultaneously!

# where:
# theta_j is a single parameter
# alpha is a learning rate
# gradient is the partial derivative of J(theta) with respect to theta_j
```
##### A little about alpha
- No need to decrease alpha over time because magnitude of derivatives become smaller as gradient descent converges
- If alpha is too small:
	- Steps to update theta are small and this may cause gradient descent to be slow of fail to converge
- If alpha is too large:
	- Steps to update theta are large and this may cause gradient descent to fail to converge or even diverge

##### Batch gradient descent
- Performing gradient descent using all the available training examples

### Linear Algebra Review
