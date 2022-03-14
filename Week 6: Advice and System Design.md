# Week 6

### Evaluating a learning algorithm
#### Deciding what to try next
What if your algorithm has large error; what should be done next? Start with the concrete example of performing regularized linear regression to predict housing prices. Assume the model is performing poorly; here are some things to try: 
	- Get more training examples
	- Try a smaller set of features
	- Get more features
	- Try adding polynomial features
	- Try decreasing lamda
	
Some of these techniques, such as getting more training examples or getting more features for the training examples already present can take time and be costly. How can we assess which of these methods will improve the model performance?  

**Machine learning diagnostics** are tests that are run to gain insight into what is and isn't working with a learning algorithm and gain guidance as to how best to improve its performance.

#### Evaluating a hypothesis
- A small training error may be "good," but may also mean that the model will fail to generalize to a test set
- Plotting the hypothesis can be helpful, but typically the feature space is too large to meaningfully visualize
