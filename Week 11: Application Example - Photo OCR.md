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
