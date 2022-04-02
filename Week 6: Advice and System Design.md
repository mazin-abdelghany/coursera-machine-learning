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
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/images/bias-variance-error.png" alt="bias-variance-graph" align="left" width = 30%/>
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
<div align="center">
	<a href="https://www.dataquest.io/blog/learning-curves-machine-learning/">
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/images/learning-curves.png" alt="learning-curves-graph" width = 70%/>
	</a>
</div>  


- If the model has high bias (left panel), then:
	- the crossvalidation error does not decrease significantly regardless of the number of training examples
	- the training error approximates the crossvalidation error
	- the training and crossvalidation error are both large at large training examples
	- **NB: getting more training examples will not help!**
- If the model has high variance (right panel), then:
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

### Machine learning system design
#### Prioritizing what to work on
- Using the example of building an email spam classifier
	- This is a supervised classification problem
	- The features x can be words that appear in emails
		- Usually choose 10,000 most frequently used words
		- x would be represented as a vector [0 1 1 0 1 0 0 0], 1 for word appears and 0 otherwise
- What should time be spent on?
	- Gathering more training examples?
	- Building more complex features (e.g., email routing info from header)?
	- Should "discounts" and discount" be different words?
	- Should punctuation be taken into account?
	- Should misspelling be considered a feature?

**It is difficult to tell what to choose to work on!**

#### Error analysis
Recommended approach for a machine learning problem:
- Start with a simple algorithm that can be implemented quickly (<24 hours)
- Implement it and test it on the crossvalidation data
- Plot learning curves to decide if more data, more features, etc. are likely to help
	- Helps avoid **premature optimization**
- Perform error analysis:
	- Manually examine the examples (in crossvalidation set) that the algorithm misclassifies
	- See if any systematic trend is present in these errors (e.g., categorize the errors)

**A quick implementation is recommended** because the main goal is to:
- Figure out what the most difficult training examples to classify are
- Different algorithms have typically find similar categories of training examples difficult, so a more complex algorithm will not necessarily be more helpful in this initial stage than a more simple algorithm
- This helps narrow focus on what to work on

#### The importance of numerical evaluation
- It is important to have a metric that is a single real number of how "good" the algorithm is
- This allows us to compare the algorithm with and without a particular optimization and assess if this helped the algorithm classify better quickly
	- In the spam classifier, an example of this would be: should we use punctuation features or not?

#### Error metrics for skewed classes
- A skewed class is a classification problem where there are very few of a particular class
	- e.g., cancer training set with 0.5% patients with cancer
	- If an algorithm of only getting "no cancer", the algorithm would have 99.5% accuracy!

#### Precision/Recall
```
                           Actual class
                 |---------------------------------|
                 |       1        |       0        |
Predicted        |----------------|----------------|
class        | 1 | True positive  | False positive |
             | 0 | False negative | True negative  |
```
**Precision (positive predictive value):**  
true positive / (true positive + false positive)

**Recall (sensitivity):**  
true positive / (true positives + false negative)

- Algorthims cannot "cheat" these metrics
- y = 1 is typically set as the rarer class

#### Trading off precision and recall
- Assume a logistic regression model is being trained&mdash;0 <= h_theta(x) <= 1
	- predict 1 if h_theta(x) >= 0.5
	- predict 1 if h_theta(x) < 0.5
- If a high threshold is set (i.e., 0.7 or 0.9), then the model will have higher precision, but lower recall
- If a lower threshold is set (i.e., 0.3 or 0.1), then the model will have a lower precision, but higher recall

##### How can we compare precision/recall numbers?
- A single real number metric is much easier to interpret
	- An average of precision and recall does not help
	- F1 score (F score):
		- 2 * (precision * recall) / (precision + recall)
		- A number between 0 and 1
	- There are several scores similar to this

#### Using large data sets
- How much data should we train on? There are some conditions that large data sets will improve the performance of an algorithm
- A study from 2001 by Banko and Brill noted that several different algorithms&mdash;if given enough data&mdash;will all improve the accuracy the more training examples that are available
	- Also, surprisingly, algorithms considered "inferior" tended to perform just as well as those considered "superior"

##### Large data rationale
- Assume feature x has sufficient information to predict y accurately
	- Example: For breakfast I ate TWO eggs (predicting TWO)
	- Counter-example: Predict housing price from only size (sqft) and no other features
	- One way to assess if this is true:
		- Can a human expert confidently predict y accurately with the given feature x?
- Use a learning algorithm with many parameters
	- Low bias algorithms (unlikely to underfit)
	- Cost function J_train(theta) will be small
- Use a very large training set 
	- Low variance training set (unlikely to overfit)
	- J_train(theta) and J_test(theta) likely to be approx. equal
	- J_test will be small
	
Simply:
1. A human expert confidently predict y accurately with the given feature x?
2. Use a learning algorithm with many parameters
3. Use a very large data set

This is likely to obtain a high-performing algorithm.
