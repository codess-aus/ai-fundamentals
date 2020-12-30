# Describe features of computer vision workloads on Azure (15-20%)


## Identify common types of computer vision solution
* [1. Identify features of image classification solutions](#1-identify-features-of-image-classification-solutions)
* [2. Identify features of object detection solutions](#2-identify-features-of-object-detection-solutions)
* [3. Identify features of semantic segmentation solutions](#3-identify-features-of-semantic-segmentation-solutions)
* [4. Identify features of optical character recognition solutions](#4-identify-features-of-optical-character-recognition-solutions)
* [5. Identify features of facial detection, facial recognition, and facial analysis solutions](#5-identify-features-of-facial-detection-facial-recognition-and-facial-analysis-solutions)

## Identify Azure tools and services for computer vision tasks
* [6. Identify capabilities of the Computer Vision service](#6-identify-capabilities-of-the-computer-vision-service)
* [7. Identify capabilities of the Custom Vision service](#7-identify-capabilities-of-the-custom-vision-service)
* [8. Identify capabilities of the Face service](#8-identify-capabilities-of-the-Face-service)
* [9. Identify capabilities of the Form Recognizer service](#9-identify-capabilities-of-the-Form-Recognizer-service)


### 1. Identify features of image classification solutions

Example: Assessing the damage to a vehicle from a photograph. You train the model by uploading images of vehicles with differing levels of damage, and you label them with the class labels that you want to identify. The model will then be able to place any new image in one of the categories.

Example: Quality control on a production line. Product labels and bottle caps can be verified to be correctly attached by using image classification against a set of trained images of correctly labelled and capped bottles.

Example: Detecting colour scheme in a image. Colours are classified in an image as: the dominant foreground colour, the dominant background colour, and the accent colour.

Example: Identifying products on a warehouse shelf.

Example: Perform medical diagnosis on MRI scans.

Image Classification is a ML model that predicts the category(class) that the contents of an image belong to. A set of images is used to train the model. The model can then be used to categorize a new image.

Image classification is a process of applying class or category labels to images according to their visual characteristics.

### 2. Identify features of object detection solutions

Example: Evaluating compliance with building safety regulations
Example: Find people wearing masks in a room.
Example: Returning bounding box coordinates for all identified people on a picture.
Example: Tracking seasonal migration of animals from drone camera images.

You should use Object Detection for returning bounding box coordinates for all identified animals on a photo. OD can process the image to identify various animals such as cats and dogs and return their coordinates.

Identifies and tags individual visual features (objects) in a model. Object detection can recognise many different types of objects. Azure computer vision is trained in more than 80 categories.

Object detection will also return the coordinates for a box surrounding a tagged visual feature (object). Object detection is similar to image classification, but object detection also returns the location of a tagged object.

Object detection may be able to identify the make and model but would not be able to assess damage.

### 3. Identify features of semantic segmentation solutions

Example: Driving autonomous vehicles. 

Semantic segmentation is used when an AI-based system needs to understand the context in which it operates.

Semantic segmentation does pixel level classification of image content. As a part of the image processing, pixels which share specific characteristics, such as parts of tissue or bones on Xray images are assigned with the same labels to define the boundaries of the relevant body parts.

### 4. Identify features of optical character recognition solutions

Example: Processing and validating invoices
Example: Handwritten text from a students essay
Example: Extracting handwritten text from scanned copies of cheques.

Optical Character Recognition retrieves printed text from scanned documents. OCR is a process of extracting printed or handwritten text from the input images or PDF documents.

The OCR can only extract simple text strings. Use the form recogniser to visualise data in a table like format.

### 5. Identify features of facial detection, facial recognition, and facial analysis solutions

Example: Identifying people in an image.
Example: Customer engagement in retail
Example: Validating identity for access to business premises
Example: confirm a driver is looking at the road.
Example: Identify human faces on a security cameras video stream.

Facial detection can identify human faces on an image, generate a rectangle for each detected face and provide additional details such as age and gender, if they are wearing glasses, emotion.



### 6. Identify capabilities of the Computer Vision service

Recognition of famous people is a feature of the domain-specific content where thousands of images of celebrities have been added to the computer vision model.

The computer vision service can moderate adult content. There is a separate Content Moderator service that provides additional functionality and review processes.

Commercial brand identification in social media posts.

It can extract but not translate text.

Can identify landmarks from an image.

Identify dominant colours in online images.

Can detect human faces and predict age and gender. Face Service can be used for a more detailed analysis: identify head pose, estimate gender, age and emotion, detect presence of facial hair or glasses and evaluate if two faces belong to the same person.

Azure cognitive service with a rich set of image processing functionalities to detect objects, brands or faces, describe image content, generate thumbnails etc.

### 7. Identify capabilities of the Custom Vision service

Supports 2 Project Types: Classification and Object Detection. You can specify labels to be applied to the image as tags and return them as bounded boxes.

Allows you to specify labels for an image.

Let's you build and deploy image classifier trained on your custom set of images and labels such as butterflies.

### 8. Identify capabilities of the Face service

Can detect the angle a head is posed at. Detect head gestures in real time.

The Verify Option takes a face and determines if it belongs to the same person as another face (twins). You need to detect face(s) in an image using the Detect API. The Verify option can then compare the two faces.

The Find Similar operation takes a face you have detected and extracts faces that look alike from a list of faces that you provide. Find Similar returns a subset of the faces in that list.

The Group operation creates several smaller groups from a list of faces based on the similarities of the faces. 

The Identify operation takes one or more faces and matches them to people. It returns a list of possible matches with a confidence score between 0 and 1.

Face Service can be used for a more detailed analysis: identify head pose, estimate gender, age and emotion, detect presence of facial hair or glasses and evaluate if two faces belong to the same person.

### 9. Identify capabilities of the Form Recognizer service

Example: Automate data extraction from scanned copies of sales receipts minimizing development efforts.
Example: prebuilt business card model can extract info from business cards in English.

Prebuilt models are english only at this time.
Custom model - spanish, chinese, dutch, french, german, italian and portuguese.

The Form Recognizer API extracts data from a document and provides a GUI to visulaize the data in a table-like format.

The Form Recognizer extracts text, key/value pairs, and table data from documents.