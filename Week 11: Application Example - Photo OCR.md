# Week 11

### Photo OCR Case Study
#### Problem description
- Stands for photo optical character recognition
- Recognize letters and words in images in order to motivate search or copy/paste  

Steps:
1. Detect where text is
2. Character segmentation
3. Classify the characters 
4. (Spelling correction)

- The above is a machine learning pipeline that requires multiple modules
	- Some modules may be machine learning and others not
	- The design and conception of the pipeline itself can have a major impact on performance of the overall algorithm
	- Each of the above modules may be the task of 1-5 engineers

#### Sliding windows classifier
- Text detection is a more challenging than, but similar task to pedestrian detection
	- The pedestrian detection problem is more straightforward because the aspect ratio (height:width) of pedestrians in an image is relatively fixed while the aspect ratio of parts of an image with text varies
- Therefore, start with the task of pedestrian detection:
	- Assume that a training set of pedestrian detection consists of an 80px by 40px image of pedestrian (y=1) or no pedestrian (y=0)
	- Train an algorithm (such as a neural network) on this training set
	- Apply the algorithm to every part of a test image by sliding an 80px by 40px "window" across the entire image and testing multiple 80px by 40px sections of the image
		- The amount by which the window slides across the image is called the "step size" or "stride"
		- A step size of 1px performs best, but is more computationally expensive
		- A step size of 4-8px is typically chosen
	- Once the image is processed like this in 80px by 40px chunks, then:
		- Increase the size of the image chunk (e.g., 100px x 60px)
		- Resize it to 80px by 40px
		- Run that patch through the algorithm
		- Repeat until the entire image is processed
- Apply this method to text detection:
	- Once the sliding window has found high-probability text areas, another algorithm must process to make large rectangles that combine regions of the image that have text&mdash;"expansion" operator
		- Operationalized by finding areas where nearby pixels comtain text
- To split the characters, slide a window in one-dimension and find the empty spaces
- Once the characters are separated, classify each letter

#### Getting Lots of Data and Artificial Data
- Artificial data synthesis&mdash;either create data from scratch or increase the size of a small dataset
- Artificial data can be synthesized for the photo OCR problem by pasting images from different fonts together
	- This takes quite a bit of work to ensure that the synthetic data appears similar to real data
	- If the synthetic data is a poor representation, this will affect the performance of the model
- Data can also be synthesized by creating artificial distortions to each letter (for example for the photo OCR)
	- **NB:** Distortion introduced should be representative of the type of noise/distortions that will be encountered in the test set
	- Usually does not help to add purely random/meaningless noise to your data
	- This can be more of an art than science
- Important notes to keep in mind:
	- Make sure you have a low bias classifier before expending the effort (i.e., plot learning curves)
		- Consider continuing to increase the number of features
		- Consider increasing the number of hidden units in a neural network
	- "How much work would it be to get 10x as much data as we currently have?"
		- This is a very common question to ask
		- Often, it turns out that this is quite easy
			- Artificial data
			- Collect/label data yourself
				- Calculate the actual amount of time that it would take to collect more data
				- e.g., we have m = 1,000 and it takes 10 seconds/new example to get 10,000 examples, it is not too much work
		- "Crowd-source" the data labeling (may be less reliable labeling)
			- e.g., Amazon Mechanical Turk
	- This can increase the performance considerably

#### Ceiling analysis
- Most valuable resource is engineering/developer time
- Ceiling analysis helps to assess **what parts of the pipeline to focus on** to improve the performance of the model
- Continuing with the photo OCR example:
	- Here is the pipeline again: image -> text detection -> character segmentation -> character recognition
	- Always find a single real-number evaluation of the model
	- Imagine the overall system has an accuracy of 72%
		- Modify the system so that the text detection algorithm has the ground truth labels (i.e., simulate that it has 100% accuracy) and re-assess the accuracy of the system&mdash;assume this increases the accuracy to 89%
		- Modify the character segmentation to 100% accuracy&mdash;assume this increases the accuracy to 90%
		- Modify the character recognition to 100% accuracy&mdash;assume this increases the accuracy to 100%
	- These above numbers mean that:
		- if the text detection algorithm was improved, there could be up to a 17% improvement in the overall system accuracy (i.e., 72% -> 89%)
		- if the character segmentation algorithm was improved, there could only be up to a 1% improvement in the overall system accuracy (i.e., 89% -> 90%)
		- if the character recognition algorithm was improved, there could be up to a 10% improvement in the overall system accuracy (i.e., 90% -> 100%)
