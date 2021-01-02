# Describe features of Natural Language Processing (NLP) workloads on Azure (15-20%)

## Identify features of common NLP Workload Scenarios
* [1. Identify features and uses for key phrase extraction](#1-identify-features-and-uses-for-key-phrase-extraction)
* [2. Identify features and uses for entity recognition](#2-identify-features-and-uses-for-entity-recognition)
* [3. Identify features and uses for sentiment analysis](#3-iidentify-features-and-uses-for-sentiment-analysis)
* [4. Identify features and uses for language modeling](#4-identify-features-and-uses-for-language-modeling)
* [5. Identify features and uses for speech recognition and synthesis](#5-identify-features-and-uses-for-speech-recognition-and-synthesis)
* [6. Identify features and uses for translation](#1-identify-features-and-uses-for-translation)

## Identify Azure tools and services for NLP workloads
* [7. Identify capabilities of the Text Analytics service](#7-identify-capabilities-of-the-Text-Analytics-service)
* [8. Identify capabilities of the Language Understanding Intelligence Service (LUIS)](#8-identify-capabilities-of-the-Language-Understanding-Intelligence-Service)
* [9. Identify capabilities of the Speech service](#9-identify-capabilities-of-the-Speech-service)
* [10. Identify capabilities of the Translator Text service](#10-identify-capabilities-of-the-Translator-Text-service)


### 1. Identify features and uses for key phrase extraction
Example: Identifies the main points in a set of blog posts
Example: Creating tags of popular mentions in reviews on a website

Key phrase extraction performs better on larger amounts of text. The more text you provide, the better it will do. Give it an essay.

Evaluates a piece of text and identifies the key talking points contained in the text.

### 2. Identify features and uses for entity recognition
Example: Dates and times of day in a document
Example: Passport number
Example: Extracting brand information from a document

Named entity recognition identifies entities in a provided test and categorizes them into predefined classes or types such as people, products, events etc. 

Detects the use of people, places, organizations and other known items from a piece of text.

### 3. Identify features and uses for sentiment analysis
Example: Analyze social media for a brand
Example: Determine the emotion in a text statement
Example: Mining customer opinions. 

Sentiment Analysis performs better on smaller amounts of text. Less words means less distractors for the sentiment analysis model, and for that reason, it produces a higher-quality result with smaller amounts of text. Give it some tweets.

Sentiment Analysis returns sentiment labels and scores for the entire document.

Sentiment Analysis returns sentiment labels and scores for each sentence within a document.

Confidence scores range from 0 to 1.

Sentiment Analysis evaluates a provided text for detecting positive or negative sentiments. It then returns sentiment labels and confidence scores, which range from 0 to 1, at the sentence and document levels. While it can detect sentiment in blog posts it cannot identify the main points in them.

It can evaluate tweets as positive, neutral or negative.

Sentiment analysis explores customer perception of products or services. Sentiment Analysis functionality from within the set of Text Analytics services can analyze raw text for clues about positive or negative sentiment.

### 4. Identify features and uses for language modeling
Example: Discover the meaning in a text statement
Example: Convert a command into smart actions.

Language modeling can be performed in many languages but only one language at a time.

Language modelling interprets the intent of a text command and turns the command into an intent which can be converted into a smart action for a device.

### 5. Identify features and uses for speech recognition and synthesis
Example: Real-time transcription of podcast dialogs into text.
Example: Convert audio to text
Example: Detecting and interpreting spoken input is an example of speech recognition
Example: Generating spoken output is an example of speech synthesis
Example: Creating a transcript of a phone call. The audio in the call recording is analyzed to find patterns that are mapped into words. Speech recognition interprets audio and turns it into text data.

SSML is based on XML as per the WWWC standard. SSML lets you improve the quality of speech synthesis by fine-tuning the pitch, pronounciation, speaking rates and other parameters of the text-to-speech output.

Speech synthesis can generate human-like synthesized speech based on input text. Speech synthesis is available in several languages and can be customised to adjust pitch, add pauses, improve pronounciation etc. by using speech synthesis markup language (SSML). Speech synthesis assigns phonetic sounds to each word

Speech recognition recognises and transcribes human speech. It is the ability to detect and interpret spoken input and turn it into data so it can be processed as text. Speech is analyzed to find patterns that are mapped to words. An acoustic model is used to convert the audio stream into phonemes, which a representations of specific sounds and a language model maps these phonemes to words using statistical algorithms to predict the probable sequence of words.

Converts text to speech for people with disabilities.

### 6. Identify features and uses for translation
Example: Enhance your chat applications functionality to enable English to Korean translation in near real-time.
Example: Enabling multi-lingual user experience on your coporate website.

Speech translation provides real-time, multi-lingual translation of audio files or streams.

Translation is the conversion of either text or audio speech from one language to another.

Text translation translates the text documents from one language to another.

Speech translation translates spoken audio from one language to another.

### 7. Identify capabilities of the Text Analytics service
Example: An AI solution that can identify and disambiguate entities in your input texts using Wikipedia as it's knowledge base.
Example: Finding out whether customers like your products from their online posts.
Example: Determine whether the occurence of the word Mustang refers to a feral horse or the model of a car

One of it's features is entity linking, which can identify and disambiguate identities of entries found in a provided text. Entity linking uses wikipedia as it's knowledge base and can determine from the content whether for example Mars refers to the planet or the company brand.

Some operations from the Text Analytics API include a confidence score between 0 and 1 but not all. Sentiment analysis, language detection and entity detection do. Key phrases does not, it simply returns a list of phrases extracted from the text.

.Net, C#, Java, Python, Javascript, Ruby and Go and PowerApp Canvas Apps

### 8. Identify capabilities of the Language Understanding Intelligence Service (LUIS)
Example: Enhance a chatbot's functionality to predict the overall meaning from a users conversational text.
Example: The temperature entity extracts the temperature number and temperature scale, such a C or F.

There are two resources in a LUIS app: the authoring resource to be used to build, manage, train, test and publish your LUIS model and a prediction resource to query the model. To use a LUIS model to find the intent of a text statement, you need the ID of the LUIS app and the endpoing and key for the Prediction resource but not the authoring resource.

When building a LUIS app example utterances must be added to intents.

Utterances are user inputs that LUIS needs to interpret. To train LUIS you add example utterances to each intent you have added to your model.

Intents are the required outcome from an utterance and are linked to actions. Entities are data in an utterance. Entities are the information needed to perform the action identified in the intent.

Features provide LUIS with hints, not hard rules, for LUIS to use when finding intents and entities.

Prebuilt Domains contain intents, uterances and entities. The HomeAutomation Domain contains the common utterances for controlling smart devices such as lights and appliances, with intents such as TurnOn and TurnOff, and entities such as light.

PreBuilt intents contain intents and utterances, but not entities. You can add the intents from the prebuilt domains without adding the entire domain model. The ToDo.Confirm intent contains utterances that confirm that a task should be performed.

### 9. Identify capabilities of the Speech service
Example: Verifying and identifying speakers by their unique voice characteristics.
Example: Create a custom and unique voice font for your mobile app.
Example: Translate simultaneously from one language to multiple others for a conference presentation

Speech service requires audio training data for the machine learning model to learn about the unique characteristics of each speaker. Then it checks with the new sample if it is the same person or identifies whether a new voice sample matches a group of enrolled speaker profiles.

Speech service can be utilized to create a custom and unique voice font for your mobile app. It offers the option of training private and custom-tuned models, so that you can produce recognizable, unique voice for your text to speech mobile app.

You do not need to build a custom speech model. If your application uses generic language and works in an environment with little or no background noise, you can utlize a baseline model pretrained on Microsoft-owned data that is already deployed in the cloud. A custom speech model is a better fit when you need to adapt the speech service to specific noise or language.

It can transcribe audio streams and even local files in real-time. It can transcribe audio files asynchronously.

### 10. Identify capabilities of the Translator Text service

Translation is the conversion of either test or audio speech from one language to another

The Translator service uses NMT - Neural Machine Translation, which uses neural networks and deep learning to translate whole sentences.



Language detection can evaluate text input to determine which language is used. It also returns a score that reflects the model's confidence in it's language prediction results.

