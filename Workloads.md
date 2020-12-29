# Describe Artificial Intelligence workloads and considerations (15-20%)

## Identify features of common AI workloads
* [1. Identify prediction forecasting workloads](#1-identify-prediction-forecasting-workloads)
* [2. Identify features of anomaly detection workloads](#2-identify-features-of-anomaly-detection-workloads)
* [3. Identify computer vision workloads](#3-identify-computer-vision-workloads)
* [4. Identify natural language processing or knowledge mining workloads](#4-identify-natural-language-processing-or-knowledge-mining-workloads)
* [5. Identify conversational AI workloads](#5-identify-conversational-ai-workloads)

## Identify guiding principles for responsible AI
* [6. Describe considerations for fairness in an AI solution](#6-Describe-considerations-for-fairness-in-an-AI-solution)
* [7. Describe considerations for reliability and safety in an AI solution](#7-describe-considerations-for-reliability-and-safety-in-an-ai-solution)
* [8. Describe considerations for privacy and security in an AI solution](#8-describe-considerations-for-privacy-and-security-in-an-ai-solution)
* [9. Describe considerations for inclusiveness in an AI solution](#9-describe-considerations-for-inclusiveness-in-an-ai-solution)
* [10. Describe considerations for transparency in an AI solution](#10-describe-considerations-for-transparency-in-an-ai-solution)
* [11. Describe considerations for accountability in an AI solution](#11-describe-considerations-for-accountability-in-an-ai-solution)

### 1. Identify prediction forecasting workloads:
Predicting whether an airplane arrives early, on-time or late. A ML model analyzes patterns in the data: Departure time, weather conditions, air-traffic volumes and associates historical patterns to predict or forecast the possible outcome.

Predicting whether a customer would buy certain items based on their purchase history. ML model analyzes patterns in the previous purchase history to predict the likelihood of a customer buying certain items.

Predicting whether customers should be targeted in a new marketing campaign.

Determining the likely repair costs for an accident involving a vehicle. An ML model finds patterns in the provided information such as the amount of damage, the location of damage, and the parts damaged. This is compared with historical data to predict the amount of time required to repair the damage and the cost of the repair.

Uses historical data to predict or forecast an outcome based on the data input into the model

### 2. Identify features of anomaly detection workloads:
AD is the process of using ML to find unexpected values or events. Analyzes time series data to determine the boundaries of expected values and detect abnormalities that differ from the expected norm.

Anomalies can be detected by AI as they occur in real-time. The ML model can derive the possible boundaries of the norm from previously seen data and then determine whether the latest data point in the time series is an anomaly or not.

Anomaly detection boundaries that are automatically created by AI are not immutable. AI automatically generates anomaly detection boundaries for the data points seen in the streamed or batched data. However you can still manually adjust those boundaries to make your model more or less sensitive to the data anomalies as required.

Detecting abnormal events in a vehicles engine. Anomaly detection detects unusual patterns or events in a stream of data.

Detects unusual patterns or events, enabling pre-emptive action to be taken before a problem occurs. It monitors streams of data from devices and systems and identifies unusual events, patterns, changes that could indicate degradation or future failure. By flagging these issues, action can be taken to resolve the potential problem before it adversely affects the operation. 

Does not predict when a problem will occur or if one will. Just identifies issues that should be investigated. It does not extract insights from the data, but instead alerts when something out of the ordinary disrupts the expected pattern.

Discovering financial system fraud.

Detecting a change in hospital infection rates.

### 3. Identify computer vision workloads:
Detecting predestrians in the real-time video stream of an autonomous vehicle. Returns bounding box coordinates for pedestrians.

Detecting whether people in on-line posted images are celebrities. An ML model training with domain specific content, for example, celebrities, can determine if they are among the people detected in the online-posted images.

Generating automatic descriptions for published images. Analyzes published images and generates human readable sentences describing the image content. Sentences ae ordered by the confidence score that the model assigns to each sentence as per the visual features detected.

Computer Vision can be used to analyse static images such as objects, living things or scenery.

Computer Vision can also be used to analyze live video streams, requires processing the individual frames first but processing can be overlaid in near-real-time.

Detecting the speed limit using roadside signage. Computer vision can takes images or video streams and extract text from the signs on the roadside.

Determining the distance to the vehicle in front can be detected as objects and calculated.

Interpret the contents of the image and classify it, detect objects in it and analyze and describe the image.

Detecting abnormalities in health scans. Computer vision interprets and classifies images. The custom vision service can be used to train a model with images of scans, some of which have abnormalities and some of which do not. CV can then classify new images with a score between 0 and 1 according to the probability of having abnormalities, where 1 indicates the highest probability.

### 4. Identify natural language processing or knowledge mining workloads:
Detecting the language in the provided text document. The ML model evaluates the text input and returns the language with a score between 0 and 1 to reflect it's confidence.

Analyzing customer feedback on an ecommerce website to determine whether it is positive or negative. An ML model evaluates the content of the provided feedback and returns sentiment labels and confidence scores for each sentence and overall content.

Extracting key phrases from student essays. Discards non-essential words and returns single terms or phrases that appear to be the subject or the object of the relevant sentences.

Knowledge Mining: Use the power of AI to explore vast amounts of information to get better insight and uncover hidden relationship and patterns in your data. KM uses a combination of AI services to extract meaning and relationships from large amounts of data. This information can be held in structured and unstructured data sources, documents and databases. KM uncovers hidden insights in your data.

Interpret written text. Determine the language and the sentiment expressed. Extract key phrases, identify key entities and actions.

Translating commands into actions is performed using NLP that extracts key phrases, intents and actions from written and spoken text.

Detecting spam in emails. Analyzes text in email to determine if it contains a spam message.

Language translation (Speech services)

### 5. Identify conversational AI workloads: 
Providing answers to a customer in a chatbot dialog. A chatbot backend processes input for a customer and sends back answers based on a knowledge base.

Chatbot answers common customer questions.

Using graphics and menu's to improve the user experience with an ecommerce websites chatbot. The chatbot's functionality is extended beyond the default text interface with more interactive components such as graphics, menu's and buttons to improve the user experience.

Used to create applications in which AI agents engage with humans in conversations (dialogs). Commonly through web-chat bots.

Answering FAQs.

Making Travel Reservations.

### 6. Describe considerations for fairness in an AI solution:
AI systems should treat all people fairly

For example, suppose you create a machine learning model to support a loan approval application for a bank. The model should make predictions of whether or not the loan should be approved without incorporating any bias based on gender, ethnicity, or other factors that might result in an unfair advantage or disadvantage to specific groups of applicants.

Azure Machine Learning includes the capability to interpret models and quantify the extent to which each feature of the data influences the model's prediction. This capability helps data scientists and developers identify and mitigate bias in the model.

### 7. Describe considerations for reliability and safety in an AI solution:
AI systems should perform reliably and safely.

For example, consider an AI-based software system for an autonomous vehicle; or a machine learning model that diagnoses patient symptoms and recommends prescriptions. Unreliability in these kinds of system can result in substantial risk to human life.

AI-based software application development must be subjected to rigorous testing and deployment management processes to ensure that they work as expected before release.

### 8. Describe considerations for privacy and security in an AI solution:
AI systems should be secure and respect privacy.

The machine learning models on which AI systems are based rely on large volumes of data, which may contain personal details that must be kept private. Even after the models are trained and the system is in production, it uses new data to make predictions or take action that may be subject to privacy or security concerns.

### 9. Describe considerations for inclusiveness in an AI solution:
AI systems should empower everyone and engage people.

AI should bring benefits to all parts of society, regardless of physical ability, gender, sexual orientation, ethnicity, or other factors.

### 10. Describe considerations for transparency in an AI solution:
AI systems should be understandable.

Users should be made fully aware of the purpose of the system, how it works, and what limitations may be expected.

### 11. Describe considerations for accountability in an AI solution:
People should be accountable for AI systems.

Designers and developers of AI-based solution should work within a framework of governance and organizational principles that ensure the solution meets ethical and legal standards that are clearly defined.