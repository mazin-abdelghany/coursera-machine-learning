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

SVM cost: C * sum(y(i) * cost_1(theta_transpose * x(i)) + (1-y(i)) * cost_1(theta_transpose * x(i))) + (1/2) * sum(theta^2)
- See below for cleaner representation
- By convention, the SVM minimization problem omits the usual (1/m) given that a constant does not change the minimum or the values of theta that achieve the minimum
- Also, the regularization of SVMs includes a constant C out front rather than along with the sum added at the end

SVMs predicts 1 or 0, not a probability.

#### Large margin intuition
- SVMs are considered "large margin classifiers"
<div align="center">
  <img src="https://github.com/mazin-abdelghany/coursera-machine-learning/blob/main/svm-cost-and-graph.png" alt-text="SVM cost function and associated graphs" width=50%/>
</div>
<p align="center">On the left is the cost_1(z) function and on the right is the cost_0(z) function.</p>  

#### Large margin intuition

- By requiring that theta_transpose * x to be >= 1 or <= -1 before predicting that y = 1 or y = 0, respectively, the SVM model builds in a "safety" margin. y will be predicted as positive or negative when the algorithm is more "certain" that this is the case by setting the threshold at 1 and -1.

The SVM optimization problem can be intuited by considering the situation where C is large. If C is large, to minimize the above cost function, the goal would be to ensure that the term in square brackets is equal to zero. As above, this is the case when (1) theta_transpose * x to be >= 1 if y = 1 or theta_transpose * x to be <= -1 if y = 0.
- This leaves the minimization problem as min[ (1/2) * sum(theta^2) ]
- The above minization problem defines the large margin classification. The SVM works to separate the classes with the largest margin in between them.
- Though the intuiting for the SVM was built considering C is very large, the SVM functions similarly when C is reduced to reasonable regularization numbers. Large Cs cause unusual SVM behavior when there are outliers that is solved by reducing C.

#### Mathematics behing large margin classification
- I watched this optional video, but will not take notes on it here.

#### Kernels
- When building a non-linear decision boundary using linear regression, high order polynomials help to fit a non-linear function to these data
	- This is computationally expensive
- A kernel is a similarity function that computes new features from x with respect to the proximity of feature x(i) to landmarks L(i)
- The similarity assessed by Euclidean distance between x and the landmark (||x - L(1)||) modified by a certain similarity function 

For example,
```
Given a parameter x:
        a new feature f1 = similarity(x1, L(1)) = exp(   ||x - L(1)||^2  )
                                                     (-  --------------  )
                                                     (     2*sigma^2     )
													
This can also be written as K(x, L(1) where K is the Gaussian kernel.
```
- The above example kernel (i.e., similarity function) is the Gaussian kernel
- If the distance between x and L is small, then the numerator in the above Gaussian kernel definition will be small and the exp() will be near 1
- If the distance between x and L is large, then the numerator in the above kernel will be small and the exp() will be near 0
- The sigma term varies how quickly the new feature f1 approaches 0 based on how far x is from the landmark
	-  smaller sigmas cause feature values to approach 0 more quickly
	-  larger sigmas cause feature values to approach 0 more slowly
-  These properties allows the kernels to learn complex non-linear classifiers

##### How are landmarks chosen?
- Landmarks are chosen at exactly the same location as the training examples
- If there are m training examples, there will be m landmarks
- Therefore,
```
Given m training examples: (x(1), y(1)), (x(2), y(2)), ... , (x(m), y(m))
There will be m landmarks: L(1) = x(1) , L(2) = x(2),  ... , L(m) = x(m)
And features will be computed by:

```
