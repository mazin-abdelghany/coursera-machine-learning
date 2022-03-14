# Week 4

### Non-linear hypotheses
#### Motivation
- There are many machine learning problems that require complex non-linear hypotheses to predict accurately
- Thought logistic and linear regression can generate non-linear decision boundaries, this ability decreases as the number of features increases
- Futher, as more squared features are created to generate a non-linear decision boundary, the number of features grows O(n^2); as more cubed features are created, the number of features grows O(n^3)
	- Concretely, if there are n = 100 features to start with a classification hypothesis that includes all the quadratic features would have ~5000 features and if it included all the cubic features as well, it would have ~170,000 features
- Therefore, this is a poor way to create nonlinear classifiers when n is large
- Neural networks are much more suitable for creating nonlinear hypotheses from a large feature space

#### Neural networks
- Computationally expensive, so were out of favor in the 1980s-1990s
- **One learning algorithm" hypothesis:** parts of the brain can be "reprogrammed" to interpret other inputs
	- Neural rewiring experiments (e.g., connect eye to auditory cortex and auditory cortex reprograms to interpret these inputs to "see")

#### Model representation
- A neuron is a computational unit that has input wires that receive signals, processes these inputs, then sends outputs to other neurons
- In a simple artificial neural network with one input layer and one output layer, the model is represented as input units (e.g., x1, x2, x3) that are mapped to an output unit by parameters (or weights). This mapping is performed as matrix-vector multiplication and a function is applied after the matrix-vector multiplication is completed (i.e., g( x * parameters ) where x is a vector and parameters is a matrix).
	- The g(x) is typically the sigmoid/logisitic function and is commonly called the "activation" function
		- The activation function can be a function other than the sigmoid function
	- x0 = 1 is usually not depicted in the neural network; it is called the "bias" unit
	- thetas are sometimes called "weights" rather than "parameters"
	- The neural network can have several layers&mdash;e.g., layer 1, layer 2, layer 3&mdash;which correspond to an input layer, a hidden layer, and an output layer, respectively
		- Term hidden layer comes from not "seeing" the values that are computed in that layer
		- There can be several hidden layers

#### Forward propagation
- The act of stepping through the computations from the input layer through the hidden layer(s) to the output layer is called forward (or feedforward) propagation
- Feedforward propagation is through the following steps:
```
In pseudocode:

for n, where n is the number of layers {
	- Add the bias unit to feature vectors or obtained activations
	- Perform matrix vector multiplication 
		- where the vector is the feature vector or activations and the matrix is the parameters/weights
	- Apply the activation function (in this case sigmoid)--this obtains the activations for the next layer
}
```
- The neural network is using the inputs to "learn" its own more complex features (the hidden layer(s)) which it then uses to draw a more complex decision boundary and output predictions

#### Examples and intuition
- Neural networks can be used to predict logical functions (such as AND, OR, XOR, XNOR)
```
Logical AND

x0 \
x1 --> g(z) --> h_theta(x)
x2 /

g(x) is the logistic function
Assume theta(1) = [-30, 20, 20]

x1  |  x2  |  output
----|------|--------     output calculations for each respective row:
  0 |   0  |    0        h_theta = sigmoid( -30 + 20*0 + 20*0) = 0
  1 |   0  |    0        h_theta = sigmoid( -30 + 20*1 + 20*0) = 0
  0 |   1  |    0        h_theta = sigmoid( -30 + 20*0 + 20*1) = 0
  1 |   1  |    1        h_theta = sigmoid( -30 + 20*1 + 20*1) = 0
```

#### Multiclass classification using neural networks
- Same as one v. all for logistic regression
- If the goal was to clasify grades of cancer, as an example&mdash;grade 1, 2, 3, or 4 the targets (y) would be the following vectors:
```
grade 1    grade 2    grade 3    grade 4
  1           0          0          0
  0           1          0          0
  0           0          1          0
  0           0          0          1
```
and the neural network would output predictions in the form of:
```
prediction
  1
  0 
  0 
  0 
```
