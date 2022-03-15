# Week 6

### Evaluating a learning algorithm
#### Deciding what to try next
What if your algorithm has large error; what should be done next? Start with the concrete example of performing regularized linear regression to predict housing prices. Assume the model is performing poorly; here are some things to try: 
	- Get more training examples
	- Try a smaller set of features
	- Get more features
	- Try adding polynomial features
	- Try decreasing lambda
	- Try increasing lambda
	
Some of these techniques, such as getting more training examples or getting more features for the training examples already present can take time and be costly. How can we assess which of these methods will improve the model performance?  

**Machine learning diagnostics** are tests that are run to gain insight into what is and isn't working with a learning algorithm and gain guidance as to how best to improve its performance.

#### Evaluating a hypothesis
- A small training error may be "good," but may also mean that the model will fail to generalize to a test set, i.e., the hypothesis is overfit
- Plotting the hypothesis can be helpful, but typically the feature space is too large to meaningfully visualize
- One way to evaluate a hypothesis is:
	- **Randomly** Split the data into a training and test set (70% and 30%, respectively)
	- Train the hypothesis on the training data
	- Use the parameters theta to compute the test set error
	- In linear regression, the error is typically the mean squared error
	- In logisitc regression, the logisitic cost function can be used
		- Another metric for logistic regression is the misclassification error:
```
        Error(h_theta(x), y) = 1 if | h_theta(x) >= 0.5 if y = 0
                                    | h_theta(x) < 0.5 if y = 1
                             = 0 otherwise
							 
        Test error: (1/num_test_training_examples) * sum( Error(h_theta(x), y) )
        In other words, the proportion of the test data that was misclassified
```
#### Model selection problems
- How to choose hyperparameters such as lambda as an example if regularization is being used
- Assume that a linear regression model is being fit and there is a choice of what degree polynomial to use for the model
	- It is as though there is another parameter (or hyperparameter) *degree of polynomial* that much also be fit
	- 1 -> 10th order polynomials can be fit and error obtained by calculating the cost on the test set
	- The degree polynomial to use can be chosen based on the lowest test-set error
	- NB: The parameter *degree of polynomial* has been fit to the test set; therefore, reporting the test set error would not be a fair estimate of the generalization error
- Therefore, the data is typically split into training set, crossvalidation (or validation) set, and the test set (e.g., 60%, 20%, 20%, respectively)
- Now, there is a training, crossvalidation, and test error
- Then, the above model selection problem would then fit the parameter *degree of polynomial* to the crossvalidation set and the test set error will be a better estimate of the generalization error

#### Bias v. variance
- The most likely reason for a poorly functioning predictive model is either a bias problem (underfitting) or a variance problem (overfitting)
<div>
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/bias-variance-error" alt="bias-variance-graph" align="left" width = 30%/>
</div>

This graph represents the [model selection](#model-selection-problems) problem defined above. THe degree of polynomial is on the x-axis and the error is on the y-axis. The training error will continue to decrease as the degree of polynomial increases. Though the error becomes near zero at the highest degree polynomials (far right of the graph), this has clearly overfit the data because the crossvalidation error has risen significantly (data is overfit; high variance). On the contrary, lower degree polynomials (far left of the graph) have a high training **and** crossvalidation error (data is underfit; high bias).  

The optimal degree polynomial is a the inflection point of the crossvalidation erorr.
<br/>
<br/>

#### Regularization and bias/variance
- Suppose a linear regression model is fit using regularization:
	- If lambda is large, the parameters are reduced significantly and only the intercept term remains&mdash;there is high bias
	- If lambda is small, the parameters are not reduced&mdash;there is high variance
- Lambda can be chosen based on the above example of choosing the degree of polynomial
	- As an aside, lambda values to assess = 0, 0.01, 0.02, 0.04, ... , 10 (doubling each time)
- Train models using multiple lambdas
- Use each model to test on the crossvalidation set and choose the lambda on the lowest crossvalidation error
- Test the final model on the test set

#### Learning curves
- Plot training error and crossvalidation error (y-axis) by the number of training examples (x-axis)
- If the model has high bias, then:
	- the crossvalidation error does not decrease significantly regardless of the number of training examples
	- the training error approximates the crossvalidation error
	- the training and crossvalidation error are both large at large training examples
	- **NB: getting more training examples will not help!**
- If the model has high variance, then:
	- as the training set size increases, the training error will increase, but remain relatively low
	- the crossvalidation error will decrease, but remain high
	- there will be a gap between the training error and the crossvalidation error
	- **NB: getting more training examples is likely to help!**

#### Deciding what to do next revisited
From [Deciding what to try next](#deciding-what-to-try-next), here are some options:
| What to do next | What it fixes |
|-----------------|---------------|
| Get more training examples | Fixes high variance |
| Try a smaller set of features | Fixes high variance |
| Get more features | Fixes high bias |
| Try adding polynomial features | Fixes high bias|
| Try decreasing lambda | Fixes high bias |
| Try increasing lambda | Fixes high variance |  

#### Neural networks and overfitting
- Small neural networks have fewer parameters and are prone to underfitting
	- Computationally cheap
- Large neural networks have more parameters and are prone to overfitting
	- Computationally expensive
- Using regularization with a large neural network is almost always better than using a smaller neural network
- Can use the above train/crossvalidate/test paradigm above to choose a neural network architecture as well!
