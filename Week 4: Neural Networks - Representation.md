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

#### Forward propogation
