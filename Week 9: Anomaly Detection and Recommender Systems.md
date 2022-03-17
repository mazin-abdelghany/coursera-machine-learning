# Week 9

### Anomaly detection
#### Problem motivation
- Imagine you are making aircraft engines and measuring features of these engines:
	- x(1) = heat generated
	- x(2) = vibration intensity
	- etc.
- There is a dataset {x(1), x(2), ..., x(m)} of m examples
- A new engine is considered x(test) and it should be assessed for uniformity or quality assurance that it is similar to other prior make aircraft engines, i.e., ready to ship
- In essence, this is anomaly detection
- Formally,
	- Given a dataset of non-anomalous examples 
	- Build a model that outputs the probability p(x) that x(test) is not anomalous
	- If p(x(test)) < epsilon -> flag anomaly
	- If p(x(test)) >= epsilon -> OK
- Examples of anomaly detection
	- Fraud detection:
		- x(i) = features of user i's activities
		- Model p(x) from data
		- Identify unusual users by checking which have p(x) < epsilon
	- Manufacturing (as above)
	- Monitor computers in a data center
		- x(i) = features of machine i
		- Model p(x) from data
		- Identify unusual machine activity concerning for machine failure

#### Gaussian (normal) distribution
- Assume x is a real number, if x is a distributed Gaussian with mean &mu; and variance &sigma;^2,
	- x ~ N(&mu;, &sigma;^2)
	- p(x; &mu;, &sigma;^2) = 1/(sqrt(2 * pi) * &sigma;) * exp(- (x-&mu;)^2 / (2&sigma;^2))
	  - p(x; &mu;, &sigma;^2) means "the probability of x parameterized by &mu; and &sigma;

#### Parameter estimation
- Dataset {x(1), ..., x(m)}
- Suspect that these examples were pulled from x(i) ~ N(&mu;, &sigma;^2)
- &mu; = (1/m) * sum(x(i)) # the mean
- &sigma; = (1/m) * sum( (x(i) - &mu;)^2 )
- The above &mu; and &sigma; are the maximum likelihood estimators

#### Develop an algorithm for anomaly detection
- Density estimation
	- Training set: {x(1), ..., x(m)}
	- Each example (i) is an n-dimentional vector (i.e., has n features)
	- Model p(x) = p(x1; &mu;1, &sigma;1^2) * p(x2; &mu;2, &sigma;2^2) * ... p(xn; &mu;n, &sigma;n^2)
		- Assume that each feature has a Gaussian distribution, e.g.,, x1 ~ N(&mu;1, &sigma;1^2)
		- Product j=1 to n p(xj; &mu;j, &sigma;j^2)
		- Note that the above are the n features, **not** the m examples x(1) ... x(m)
	- The above model assumes that the independence assumption holds true (though even if it doesn't this algorithm functions well)
Anomaly detection algorithm:
1. Choose features x(i) that you think might be indicative of anomalous examples
2. Fit parameters &mu;1, ..., &mu;n, &sigma;1^2, ..., &sigma;n^2
3. Given new example x, compute p(x)
4. Label anomaly if p(x) < epsilon

#### How to evaluate an anomaly detection algorithm
- As discuss before, it is important to have a single real-number evaluation of the algorithm
- Assume the data are labeled as non-anomalous (y=0) and anomalous (y=1)
- Training set remains unlabeled x(1), ..., x(m)
- Crossvalidation set is labeled (x(1), y(1)), ..., (x(m_cv), y(m_cv))
- Test set is (x(1), y(1)), ..., (x(m_test), y(m_test))
- Aircraft motivating example
	- 10,000 examples of normal engines
	- 20 engines are anomalous
	- How to split the data:
		- **Training set:** 6000 non-anomalous engines, estimate &mu; and &sigma;
		- **Crossvalidation set:** 2000 non-anomalous engines (y=0), 10 anomalous engines (y=1)
		- **Test set:** 2000 non-anomalous engines (y=0), 10 anomalous engines (y=1)
	- Algorithm evaluation:
		- Fit the model p(x) on the training set
		- On a crossvalidation/test example x, predict:
			- y = 1 if p(x) < epsilon
			- y = 0 if p(x) >= epsilon
		- Possible evaluation metrics:
			- True positive, false positive, false negative, true negative
			- Precision/recall
			- F1-score
		- Can also use crossvalidation set to choose parameter epsilon based on the above evaluation metrics

#### Anomaly detection vs. supervised learning
| Anomaly detection                                                                                                                                                                                           | Supervised learning                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Very small number of positive  examples (i.e., 1-20)                                                                                                                                                        | Large number of positive and negative examples                                                                                                     |
| Large number of negative examples                                                                                                                                                                           |                                                                                                                                                    |
| Many different "types" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous  examples seen so far | Enough positive examples for algorithm to get a sense of what positive examples like; future examples likely to be similar to ones in training set |

Applications of anomaly detection vs. supervised learning
| Anomaly detection                      | Supervised learning       |
|----------------------------------------|---------------------------|
| fraud detection                        | email spam classification |
| manufacturing (e.g., aircraft engines) | weather prediction        |
| monitoring machines in data center     | cancer classification     |

If there are a lot of anomalies, any of the above examples could shift to the supervised learning column.  

#### Optimizing available features for anomaly detection
- This has a lot of impact on how well the anomaly detection works
- Plot the data to ensure that it looks vaguely Gaussian
	- Though the algorithm works OK if the data are not Gaussian, data transformations to make the distribution look more Gaussian (e.g., log-transform, square root, cube root)

#### Error analysis for anomaly detection
- Want p(x) large for normal examples and p(x) small for anomalous examples
- Most common problem:
	- p(x) is comparable (say, both large) for normal and anomalous example
- Look at how the wrongly-classified anomaly and try to find a new feature that is clearly different from the non-anomalous examples

#### Choosing features for anomaly detection
- Choose features that might take on unusually large or small values in the event of an anomaly
- As an example, monitoring a computer cluster:
	- x1 = memory use of computer
	- x2 = number of disk accesses/second
	- x3 = CPU load
	- x4 = network traffic
- Suppose CPU load and network traffic grow linearly with each other
- Perhaps one of the failure cases one of the machines will get stuck in an infinite loop
	- CPU load high, but network traffic low
	- Maybe the new feature would be: CPU load/network traffic
		- This will be large in anomaly and small in non-anomaly

Optional material included modeling anomaly detection using a multivariate Gaussian distribution. This is accomplished by modeling the mean of all of the features as a vector and the variance as a covariance matrix. This allows the model to capture correlations within the data that would otherwise be lost if anomaly detection was performed using the above described method. In other words, the above model can only fit Gaussian distributions that are axis-aligned, i.e., unable to model correlations between features. This special case is when the covariate matrix has zeros in all places other than its diagnoal.  

| Original model                                                                                 | Multivariate Gaussian                                                                       |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Manually create features to capture anomalies where x1, x2 take unusual combinations of values | Automatically captures correlations between features                                        |
| Computationally cheaper (scales better to large n such as >10,000)                             | Computationally more expensive                                                              |
| OK even if m (training set size) is small                                                      | Must have m > n or else covariance matrix is non-invertible. It is preferred that m >= 10n. |

Covariance matrix may also be non-invertible of features are redundant.  

#### Recommender systems
Predicting movie ratings:
- Allow users to rate movies using zero to five stars  

| Movie                | Alice (1) | Bob (2) | Carol (3) | Dave (4) |
|----------------------|-----------|---------|-----------|----------|
| Love at last         | 5         | 5       | 0         | 0        |
| Romance forever      | 5         |         |           | 0        |
| Cute puppies of love |           | 4       | 0         |          |
| Nonstop car chases   | 0         | 0       | 5         | 4        |
| Swords v. karate     | 0         | 0       | 5         |          |

Empty cells are movies that have not been rated.  

Terminology:
- nu = no. users
- nm = no. movies
- r(i,j) = 1 if user j has rated movie i
- y(i,j) = rating given by user j to movie i
	- defined only if r(i,j) = 1

- Above, nu = 4, nm = 5
- The recommender system problem is, given the empty cells, predict the likely rating for each user and movie

#### Content-based recommender systems
- Suppose for each movie, there are associated features, e.g.,
| Movie                | Alice (1) | Bob (2) | Carol (3) | Dave (4) | x1 (romance) | x2 (action) |
|----------------------|-----------|---------|-----------|----------|--------------|-------------|
| Love at last         | 5         | 5       | 0         | 0        | 0.9          | 0           |
| Romance forever      | 5         |         |           | 0        | 1            | 0.01        |
| Cute puppies of love |           | 4       | 0         |          | 0.99         | 0           |
| Nonstop car chases   | 0         | 0       | 5         | 4        | 0.1          | 1           |
| Swords v. karate     | 0         | 0       | 5         |          | 0            | 0.9         |

