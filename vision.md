# Describe features of computer vision workloads on Azure (15-20%)


## Identify common types of computer vision solution
* [1. Identify features of image classification solutions](#1-identify-features-of-image-classification-solutions)
* [2. Identify features of object detection solutions](#2-identify-features-of-object-detection-solutions)
* [3. Identify features of semantic segmentation solutions](#3-identify-features-of-semantic-segmentation-solutions)
* [4. Identify features of optical character recognition solutions](#4-identify-features-of-optical-character-recognition-solutions)
* [5. Identify features of facial detection, facial recognition, and facial analysis solutions](#5-identify-features-of-facial-detection-facial-recognition-and-facial-analysis-solutions)

## Identify Azure tools and services for computer vision tasks
* [6. Identify capabilities of the Computer Vision service](#1-identify-capabilities-of-the-computer-vision-service)
* [7. Identify capabilities of the Custom Vision service](#2-identify-capabilities-of-the-custom-vision-service)
* [8. Identify capabilities of the Face service](#3-identify-capabilities-of-the-Face-service)
* [9. Identify capabilities of the Form Recognizer service](#4-identify-capabilities-of-the-Form-Recognizer-service)


### 1. Identify features of image classification solutions

Example: Assessing the damage to a vehicle from a photograph. You train the model by uploading images of vehicles with differing levels of damage, and you label them with the class labels that you want to identify. The model will then be able to place any new image in one of the categories.

Example: Quality control on a production line. Product labels and bottle caps can be verified to be correctly attached by using image classification against a set of trained images of correctly labelled and capped bottles.

Example: Detecting colour scheme in a image. Colours are classified in an image as: the dominant foreground colour, the dominant background colour, and the accent colour.

Image Classification is a ML model that predicts the category(class) that the contents of an image belong to. A set of images is used to train the model. The model can then be used to categorize a new image.


### 2. Identify features of object detection solutions

You should use Object Detection for returning bounding box coordinates for all identified animals on a photo. OD can process the image to identify various animals such as cats and dogs and return their coordinates.

Identifies and tags individual visual features (objects) in a model. Object detection can recognise many different types of objects. Azure computer vision is trained in more than 80 categories.

Object detection will also return the coordinates for a box surrounding a tagged visual feature (object). Object detection is similar to image classification, but object detection also returns the location of a tagged object.

Object detection may be able to identify the make and model but would not be able to assess damage.

### 3. Identify features of semantic segmentation solutions

Example: Driving autonomous vehicles. 

Semantic segmentation is used when an AI-based system needs to understand the context in which it operates.

Semantic segmentation does pixel level classification of image content. As a part of the image processing, pixels which share specific characteristics, such as parts of tissue or bones on Xray images are assigned with the same labels to define the boundaries of the relevant body parts.

### 4. Identify features of optical character recognition solutions

Optical Character Recognition retrieves printed text from scanned documents. OCR is a process of extracting printed or handwritten text from the input images or PDF documents.

### 5. Identify features of facial detection, facial recognition, and facial analysis solutions

Example: Identifying people in an image.
Example: Customer engagement in retail

Facial detection can identify human faces on an image, generate a rectangle for each detected face and provide additional details such as age and gender.

### 6. Identify capabilities of the Computer Vision service

Recognition of famous people is a feature of the domain-specific content where thousands of images of celebrities have been added to the computer vision model.

### 7. Identify capabilities of the Custom Vision service

### 8. Identify capabilities of the Face service

### 9. Identify capabilities of the Form Recognizer service