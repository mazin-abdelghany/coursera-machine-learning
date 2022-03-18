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
<div>
	<img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/mini-batch-gradient-descent.png" alt="mini-batch-gradient-descent" align="left" width = 40%/>
</div>  


- Batch gradient descent: use all m examples
- Stochastic gradient descent: use 1 example in each iteration
- Mini-batch gradient descent: use b examples in each iteration (b = ~2 to ~100)
	- This is another hyperparameter that may need tuning
- Left panel:  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b = mini-batch size  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If b = 10, use the first 10 examples to update parameters  

Mini-batch gradient descsent can work faster than stochastic gradient descent because the sum over the b examples can be vectorized. This vectorization allows for faster computation than stochastic gradient descent.

#### Stochastic gradient descent convergence
- For batch gradient descent:
	- Plot J_train(theta) as a function of the number of iterations of gradient descent
- For stochastic gradient descent:
	- During learning, compute the cost on the training example just before updating theta
	- Every, say, 1000 iterations, plot cost averaged over the last 1000 examples processed by the algorithm
	- These plots can be noisy and averaging over a larger number of training examples (e.g., 3000 or 5000) may smoothen out the curve to help visualize trends better; the disadvantage of averaging over a larger numbers of training examples is that one data point on the graph is plotted on the graph only once every 5000 examples, so feedback on the learning may be slower
	- If the cost increases over the averaged number of training examples, consider decreasing the learning rate alpha

#### Choosing the learning rate alpha
- Typically the learning rate is left constant during stochastic gradient descent
- If the goal is to attempt for convergence to a global or local minimum, alpha can be decreased slowly over time, e.g., alpha = const1 / (iteration# + const2)
	- This requires more hyperparameters to tune (i.e., const1 and const2)

#### Online learning
- If there is a continuous stream of data, the learning algorithm can be constantly updating
- For example,
	- Shipping service website where user comes, specifies origin and destination, you offer to ship their package for some asking price, and users sometimes choose to use the service (y=1) and sometimes don't (y=0)
	- Features x capture properties of user, of origin/destination, and asking price
	- Goal is to learn the probability p(y=1|x;theta) to optimize price
```
Repeat forever {
	Get (x, y) corresponding to user
	Update theta using (x,y)
		theta_j = theta_j - alpha * (h(x) - y) * x_j
		for j = 0,...,n
}
```
- This trains over each training example that comes and then discards this example
- This type of algorithm can also adapt to changing user preferences

#### Map-reduce and parallelism
- Map-reduce is as important as stochastic gradient descent for large datasets
- Map-reduce batch gradient descent:
	- Machine 1: Use a subset of training examples (e.g., first quarter)
		- Compute temp_cost(1) of the first quarter
	- Machine 2: Second quarter
		- Compute temp_cost(2) of the second quarter
	- Machine 3: Third qurater
		- Compute temp_cost(3) of the third quarter
	- Machine 4: Fourth quarter
		- Compute temp_cost(4) of the third quarter
	- Combine all 4 costs at the end to update theta
- Many learning algorithms can be expressed as computing sums of functions over the training set and therefore can take advantage of map-reduce
- Map-reduce can also be used to take advantage of using multiple cores in a single machine
	- If the linear algebra library already takes advantage of multicore computation, map-reduce may not be necessary as long as the learning algorithm is vectorized appropriately
