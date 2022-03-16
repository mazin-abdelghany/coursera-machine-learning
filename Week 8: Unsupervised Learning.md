# Week 8

### Unsupervised learning
#### Clustering
- Supervised learning: given a labeled training set, fit a hypothesis to it
- Unsupervised learning: given unlabeled data (e.g., {x(1), x(2), x(3), ..., x(m)} - note there are no labels y(i)), find patterns in the data
- Examples of unsupervised learning problems (from earlier in the course):
	- Market segmentation
	- Social network analysis
	- Organized computer clusters
	- Astronomical data analysis

#### K-means algorithm
- Most popular and widely used clustering algorithm
- Steps of the K-means algorithm:
1. Randomly initialize n points (n cluster centroids)
2. Assign all of the training examples to the closest of the n cluster centroids based on a measure of distance (e.g., Euclidean)
3. Move the cluster centroids to the average location of each group of assigned training examples
4. Repeat steps 2 and 3 until convergence

Formally:
```
- Choose the number of clusters K
- Input a training set {x(1), x(2), x(3), ... x(m)}
- Randomly initialize cluster centroids &mu;(1), &mu;(2), ..., &mu;(K)
- Repeat {
	# cluster assignment
	for 1 = 1 to m:
		# assign each point to a cluster centroid
		# computed by [min k || x(i) - u(k) ||^2 ]
		c(i) = index (from 1 to K) of cluster centroid closest to x(i)
	
	# move clusters
	for k = 1 to K
		u(k) = average of points assigned to cluster k
}
```

#### K means for non-separated clusters
- Dataset might not have clearly separated clusters
- K means will still separate the data into clusters

#### Optimization objective
- Given,  
	- c(i) = index of cluster (1, 2, ..., K) to which example x(i) is currently assigned
	- &mu;(k) = cluster centroid k
	- &mu;c(i) = cluster centroid of cluster to which example x(i) has been assigned
- Therefore, the cost function is:
	- J(c(1), ..., c(m), &mu;(1), ..., &mu;(K)) = (1/m) sum(  ||x(i) - &mu;c(i)||^2 )
	- "Distortion" function
- The cluster assignment step above is minimizing J(...) above with respect to c(1) to c(m) holding &mu;(1) ... &mu;(2) fixed
- The move centroid step minimizes J(...) with respect to the variables &mu;

#### Random initialization
- There is one recommended method that avoids local optima, the below
- Should have K clusters < m training examples
- Randomly pick K training examples 
- Set &mu;(i) to &mu;(k) (the cluster centroids) to start at these K training examples, concretely for K = 2 clusters,
	- &mu;(1) = x(23)&mdash;the 23 is meant to symbolize a random (in this case the 23rd) training example
	- &mu;(2) = x(98)
- K means can end up at different solutions depending on the random initialization
	- In other words, K means is susceptible to local optima
	- Therefore, can run K means several times to try to find the global optimum
```
To attempt to find the global optimum, run K means 100 times:

# find 100 ways to cluster the data
for i = 1 to 100 {
	Randomly initialize K-means
	Run K-means
		Obtain c(1) to c(m) and u(1) to u(K)
	Compute the cost function
}

# choose the clustering with the lowest cost
Choose the iteration with the lowest cost
```
- When running K-means with K = 2-10, rerunning as above helps to optimize the solution
- When K is > 10 or >> 10, the first random initialization will unlikely change the optimum solution

#### Choosing the number of clusters
- Try to choose manually based on prior experience or visualization
- The number of clusters is geniunely ambiguous, so there isn't necessary a "right" answer
- One method is called the elbow method:
	- Run K means with 2 clusters, plot cost
	- Run K means with 3 clusters, plot cost
	- Etc. until there is a clear inflection point or "elbow"
	- Choose the number of clusters at the elbow
	- Not used often because fairly often the curve is without an elbow
- Evaluate K-means based on a metric for how well it performs for the purpose that it is being run

#### Dimentionality reduction
- There may be redundant features (e.g., two measures of length&mdash;inches and centimeters) in the dataset that are not necessary
- There also may be features that are correlated (e.g., speed and distance) in the dataset 
- As an example, the data can be reduced from 2D to 1D:
	- x(1) is a two-dimensional feature vector [2 3] that can be projected onto a line and be made into z(1) = 1.5&mdash;a single real number feature
	- This creates a new feature
- As an example, the data can also be reduced from 3D to 2D:
	- x(1) is a three-dimensional feature vector [7 2 10] that can be projected onto a plane and be made into z(1) = [2 5]&mdash;a two-dimensional feature vector
- In reality, the data may be 1000D being reduced to 100D
- Advantages of dimensionality reduction:
	- Compresses the data required (e.g., disk space) to contain all the information
	- Allows our machine learning algorithms to run more quickly
	- Improve the visualization of the data

#### Improving the visualization with dimensionality reduction
- Example: statistics collected on countries with 50 features (50-dimensional data)
- Use dimensionality reduction to reduce the data from 50D to 2D
	- NB: the two new numbers that make up the new 2D data are not ascribed meaning
	- Usually, the analyst must figure out what the two new features represent

#### Principle components analysis (PCA)
- Find a lower-dimensional surface onto which to project to minimize the sum of square distance (i.e., to minimize the squared projection error)
- Standard practice to perform feature scaling and mean normalization before performing PCA
- Formally, 
	- **from 2D to 1D:** find a direction (2D vector) onto which to project the data to minimize the sum of square distance
	- **from n-D to k-D:** find k vectors u(1), u(2), ..., u(k) onto which to project the data to minimize the sum of square distance
		- in formal linear algebra: find the vectors u(1) to u(k) and project the data onto the linear subspace that spans the vectors 
		- e.g., from 3D to 2D, K = 2
- PCA is **not** linear regression&mdash;they are very different algorithms
	- PCA minimizes the squared distance
	- Linear regression minimizes the squared error
	- PCA has no "special variable"
	- Linear regression is also looking to predict a target ("distinguished variable")

#### PCA algorithm
- First, preprocess the data: mean normalization and feature scaling
- Compute the covariate matrix sigma
	- sigma = (1/m) * sum(  x(i) * x(i)_transpose  )
- Compute the eigenvectors of matrix sigma
	- use singular value decomposition
- This outputs a matrix U
	- use the first k columns of the matrix U (n x k size)
- z (the new features) = U_transpose * x
- The vectorized implementation of the above is:
```
	- sigma = 1/m * X_transpose * X
	- [U, S, V] = svd(sigma) # compute the eigenvectors using singular value decomposition
	- U_reduce = U(:,1:k)     # get the first k columns of U (number of dimensions to reduce to)
	- z = U_transpose * x    # obtain the new features z
```

#### Reconstruction from compressed representation
- While z (new features) = Ureduce_tranpose * x (old features),
- x_approx (old features) = Ureduce * z (new features)
	- this approximate is minus the projection error

#### Choosing the number of principle components
```
- Average squared projection error: (1/m) * sum(  ||x(i) - x(i)_approx||^2  ) = PE
- Total variation in the data:      (1/m) * sum(  ||x(i)||^2  ) = var

Choose k to be the smallest value so that
PE / var <= 0.01, i.e., 99% of the variance is retained
```
- In most cases, 0.01 to 0.05 is chosen meaning that 95-99% of the variance in the data is retained
	- Sometimes, folks go down to 85-90% variance retained (but this is less common)
- To perform this in practice,
	- [U, S, V] = svd(sigma)
	- The S matrix is a matrix with only values in the diagonals (i.e., values at S(11), S(22), ... S(nn))
	- For a given k, the value PE / var can be computed as:
		- 1 - ( sum(1 to k of Sii) / sum(1 to n of Sii) )
		- also, variance retained = ( sum(1 to k of Sii) / sum(1 to n of Sii) )

#### Advice for applying PCA
- Supervised learning speedup
	- suppose there is a learning set with data (x(1), y(1)), ..., (x(m), y(m))
		- there are m training examples
		- and x(i) is an n-dimensional feature vector
	- extract the inputs:
		- unlabeled dataset x(1), ..., x(m) where x(i) is a vector of length 10,000
		- apply PCA
		- obtain z(1), ..., z(m) where z(i) is a vector of length 1,000
	- now there is a new training set (z(1), y(1)), ..., (z(m), y(m))
		- use this to run the machine learning algorithm
	- **NB: Mapping x(i) -> z(i) should be defined by running PCA only on the training set**
	- **NB: Mapping must then performed on the crossvalidation and test sets before applying the learning algorithm for prediction**
- Poor use of PCA is to prevent overfitting
	- This might work, but is a poor way to address overfitting
	- A better way to address overfitting is using regularization
		- regularization uses the target values "y", which helps to retain some useful information whereas PCA is reducing the dimensions of the data without any target value consideration
- Misuse of PCA:
	- Sometimes a project plan includes using PCA from the start
	- Consider implementing the project plan **without PCA** first

Before implementing PCA, first try running whatever you want to do with the original/raw data x(i). Only if that doesn't do what you want, then implement PCA and consider using z(i).
