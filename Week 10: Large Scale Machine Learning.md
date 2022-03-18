# Week 10

### Large-scale machine learning
#### Machine learning with "big" data&mdash;i.e., large datasets
- Machine learning is much more successful now than 10 years from now because of the volume of data that is now available to train
- "It's not who has the best algorithms that wins, but who has the most data"
- Learning with large datasets comes with its own computational problems
- Imagine a dataset with m = 100,000,000, gradient descent would have to sum over 100,000,000 training examples

#### Why not use 1,000 examples?
- Perhaps training on 1,000 examples will do just as well
- The easiest way to assess this is to plot the learning curves of the training cost and the crossvalidation cost
<div>
	<a href="https://www.dataquest.io/blog/learning-curves-machine-learning/">
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/learning-curves.png" align="left" alt="learning-curves-graph" width = 70%/>
	</a>
</div>  

If the result of plotting the learning curve on the first 1,000 training examples is the **left panel**, the model likely has high bias and will not benefit from the addition of more training examples to improve its performance. If the result of plotting the learning curves is instead the **right panel**, the model likely has high variance and would indeed benefit from the addition of more training examples.  

#### Stochastic gradient descent
- A modification of gradient descent that optimizes for large datasets
<div align="center">
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/batch-v-stochastic-gradient-descent.png" alt="batch-v-stochastic-gradient-descent" width=100%/>
</div>

#### Mini-batch gradient descent
- Batch gradient descent: use all m examples
- Stochastic gradient descent: use 1 example in each iteration
- Mini-batch gradient descent: use b examples in each iteration
	- b = mini-batch size
	- If b = 10, use the first 10 examples to update parameters
