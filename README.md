# Machine Learning - Coursera - Andrew Ng

This repo serves as a set of notes that I have taken while completing the Machine Learning Coursera course by Andrew Ng. There is a markdown file for each week of the course with notes on the content, best practices, common pitfalls, and any other musings on that week's material. The /images/ folder contains all the images used in the Markdown notes. I completed the coding assignments on Ubuntu Desktop using a `miniconda` environment that contained `Octave`. Coding assigments are&mdash;of course&mdash;**not** uploaded to maintain academic integrity.

I have also included four other files in this repo:
1. Derivation of normal equation and regularized linear regression.pdf
2. Implementation of logistic regression in R  
&nbsp;&nbsp;&nbsp;&nbsp;a. Dataset_students.mat  
&nbsp;&nbsp;&nbsp;&nbsp;b. log_reg.Rmd  
&nbsp;&nbsp;&nbsp;&nbsp;c. log_reg_output.pdf

**Document 1** has two sections. The first is a derivation in LaTeX of the normal equation. The second shows that regularized linear regression can be interpreted as linear regression with a Bayesian prior normal distribution over the betas. Please note that theta in the markdown notes corresponds to beta in the included derivations.

**Document 2a** is the dataset used in Document 2b. **Document 2b** is an implementation of logistic regression in R using RMarkdown. **Document 2c** is the output of this RMarkdown file in pdf format.

For reference, below are the 11 weeks of the course, their corresponding content, and the associated `Octave` coding assignment.

## Table of contents:  
[Week 1: Introduction](#week-1-introduction)  
[Week 2: Linear Regression ](#week-2-linear-regression)  
[Week 3: Logistic Regression](#week-3-logistic-regression)  
[Week 4: Neural Networks - Representation](#week-4-neural-networks---representation)  
[Week 5: Neural Networks - Learning](#week-5-neural-networks---learning)  
[Week 6: Advice and System Design](#week-6-advice-and-system-design)  
[Week 7: Support Vector Machines (SVMs)](#week-7-support-vector-machines-svms)  
[Week 8: Unsupervised Learning](#week-8-unsupervised-learning)  
[Week 9: Anomaly Detection and Recommender Systems](#week-9-anomaly-detection-and-recommender-systems)  
[Week 10: Large Scale Machine Learning](#week-10-large-scale-machine-learning)  
[Week 11: Application Example - Photo OCR](#week-11-application-example---photo-ocr)

### Week 1: Introduction
- Welcome
- Introduction
	- What is Machine Learning
	- An overview of supervised and unsupervised learning
- Linear Regression with One Variable
	- Model and Cost Function
	- Parameter Learning
- Linear Algebra Review

### Week 2: Linear Regression
- Linear Regression with Multiple Variables
	- Multivariate Linear Regression
	- Computing Parameters Analytically
- Setting up `Octave` and submitting assignments
- **Programming assignment:** Implement Linear Regression

### Week 3: Logistic Regression
- Logistic Regression
	- Classification and Representation
	- Logistic Regression Model
	- Multiclass Classification
- Regularization
	- Solving the Problem of Overfitting
- **Programming assignment:** Implement Multiclass Logistic Regression

### Week 4: Neural Networks - Representation
- Representation
	- Motivations
	- Neural Networks
	- Applications
- **Programming assignment:** Implement Multiclass Classification and the Feedforward Propogation Algorithm of a Neural Network

### Week 5: Neural Networks - Learning
- Learning
	- Cost Function and Backpropagation
	- Backpropagation in Practice
	- Application of Neural Networks
- **Programming assignment:** Implement the Backpropagation Algorithm of a Neural Network

### Week 6: Advice and System Design
- Advice for Applying Machine Learning
	- Evaluating a Learning Algorithm
	- Bias vs. Variance
- **Programming assignment:** Implement Regularized Linear Regression and use it to understand bias/variance
- Machine Learning System Design
	- Building a Spam Classifier
	- Handling Skewed Data
- Using Large Data Sets

### Week 7: Support Vector Machines (SVMs)
- Support Vector Machines
	- Large Margin Classification
	- Kernels
	- SVMs in Practice
- **Programming assignment:** Use SVMs to Build a Spam Classifier

### Week 8: Unsupervised Learning
- Clustering
- Dimensionality Reduction
	- Movitaion
	- Principle Components Analysis (PCA)
	- Applying PCA
- **Programming assigment:**
	- Implement K-means clustering and compress an image
	- Implement PCA and find a low-dimensional representation of face images

### Week 9: Anomaly Detection and Recommender Systems
- Anomaly Detection
	- Density Estimation
	- Building an Anomaly Detection System
	- Multivariate Gaussian Distribution
- Recommender Systems
	- Predicting Movie Ratings
	- Collaborative Filtering
	- Low Rank Matrix Factorization
- **Programming assignment:**
	- Implement the Anomaly Detection Algorithm
	- Use Collaborative Filtering to Build a Recommender System for Movies

### Week 10: Large Scale Machine Learning
- Gradient Descent with Large Datasets
- Advanced Topics

### Week 11: Application Example - Photo OCR
- Photo OCR
