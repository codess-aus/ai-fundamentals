# Describe features of conversational AI workloads on Azure (15-20%)
Identify common use cases for conversational AI
 identify features and uses for webchat bots
 identify features and uses for telephone voice menus
 identify features and uses for personal digital assistants
 identify common characteristics of conversational AI solutions

Identify Azure services for conversational AI
 identify capabilities of the QnA Maker service
 identify capabilities of the Bot Framework

## Identify common use cases for conversational AI
* [1. Identify features and uses for webchat bots](#1-identify-features-and-uses-for-webchat-bots)
* [2. Identify features and uses for telephone voice menus](#2-identify-features-and-uses-for-telephone-voice-menus)
* [3. Identify features and uses for personal digital assistants](#3-identify-features-and-uses-for-personal-digital-assistants)
* [4. Identify common characteristics of conversational AI solutions](#4-identify-common-characteristics-of-conversational-AI-solutions)


## Identify Azure services for conversational AI
* [5. Identify capabilities of the QnA Maker service](#5-identify-capabilities-of-the-QnA-Maker-service)
* [6. Identify capabilities of the Bot Framework](#6-identify-capabilities-of-the-Bot-Framework)

### 1. Identify features and uses for webchat bots
Example webchat bot embedded in a travel site to interact with online customer and help with real-time booking of their trips.
Example: webchat bots are conversational AI agents that can use natural language processing to understand questions and find the most appropriate answer from a knowledge base.
Example: Providing first-line automated customer service to customers across multiple channels.
Example: Customer online ordering
Example: Human Resources related questions from an employee
Example: an interactive component on a banking site that understands the clients requirements and provides general answers.

A chatbot is a conversational AI solution that can utilize cognitive services and knowledge bases to conduct a real-time conversation with humans using text, speech and other available communications channels. Webchat bot is a specific type of chatbot that communicates via a web channel and is typically integrated with web-enabled applications.

Web chat responds to customers using the web channel in a web browser 


### 2. Identify features and uses for telephone voice menus
Example: A customer calling a support line number gets an AI-generated voice prompt with options to choose from.
Example: Providing guided customer support over Skype with conversational AI
Example: an interactive response system that transfers calls to required employee numbers

This type of conversational AI can reduce the workload on human operators by providing generic instructions to customers, automatically transferring calls to the relevant teams, or managing the waiting queue, all of which help support business operations even during non-working hours and holidays.


### 3. Identify features and uses for personal digital assistants
Example: Personal Digital Assistants respond to customers using Cortana and 3rd party services.
Example: An AI solution on your smartphone that can understand your voice command and send text messages while you drive a car.
Example: A conversational AI solition that keeps users informed and productive, helping then get things done across devices and platforms.
Example: an intelligent application that checks your calendar to automatically accept e-meeting invitations.

A personal digital assistant is a conversational AI solution that provides manegement, retrieval and update of a users personal or business information to keep them informed and productive. It can run across devices and platforms with access to electronic calendars, e-mail, contact lists and other applications to enable personalised assistance with routine tasks, for example, booking a new meeting with a client.

By default a PDA will respond in the same way it was queried i.e. it will respond to text with text and speech with speech. You can configure it to always respond with speech.

### 4. Identify common characteristics of conversational AI solutions

LUIS determines a users intentions.
QnA Maker uses a KB with QnA pairs to answer users questions.

A skill is a bot.

A skill manifest is a JSON file.

Users can interact with a root bot.

Web Chat is automatically configured when a bot is created with the Framework Bot Service.

Composer is an open source solution.

Composer is a visual authoring tools which allows bot development in a GUI.

Composer can publish bots to Azure Web App and Azure Functions.

### 5. Identify capabilities of the QnA Maker service

You can natively populate a Q&A Maker from a PDF or word document or by manually adding question and answer pairs.

It cannot use a SQL database (only files and URLs) as datasources.

It cannot be a sharepoint list. Files not webpages. If the URL ends with .ASPX it will not import into QnA maker.

App Service and Azure Cognitive Search are the Azure Resources created when a new QnA Maker service is created. It does not create the web app bot, you can create that later if you intend to surface it through the web channel.

TSV files for chitchat personality uploads.

It can use multiple knowledge bases.

It only supports one language.

It consists of question and answer pairs.

Use QnA Maker for authoring and query prediction. It provides access to the authoring and publishing APIs of the QnA Maker service. It also uses NLP capabilities to learn about the specifics of questions in the KB and predict at runtime witch QnA pair matches as the best answer.

You should use Application Insights to query prediction telemetry. It can collect the chatbots logs and telemetry. It can diagnose potential issues and process telemetry data with KQL.

Cognitive Search for data storage and search. It stores the QnA pairs and maintains indexes for all published KBs.

### 6. Identify capabilities of the Bot Framework

The Bot Framework SDK is required to develop bots using code. There are SDKs for C#, JS, TS and Python. The Bot Framework SDK allows developers to send and receive messages with users on the configured channels.

The Azure Bot Framework Emulator is a desktop application that allows developers to test and debug bots on their local computer.

The Bot service framework CLI tools manage bots and related services and are used in the DevOps pipelines when deploying bots in enterprises.

The Bot Framework Composer is a tool to build bots without Code. The Bot Framework Composer uses a visual user interface to create diaslogs and bot logic.

The Azure Bot Framework separates the logic of the bot from the communication with different apps and platforms. When you create a bot, the bot is available for webchat. You can add additional channels to make the bot available for other platforms like facebook, email, Teams, Slack etc.

The Azure Bot can both consume other bots and be consumed itself by another bot. A skill is a a bot that performs tasks for another bot. You add a skill when you want to reuse or extend a bot.

An Azure bot communicates by receiving and sending messages. A turn handles the received message and sends a message back to the users. You add new turns when you want to handle different messages.

You can add LUIS to your bot when you create it or later. You use the Dispatch tool to route messages from the bot to LUIS.

You can integrate QnA Maker knowledge bases. You use the Dispatch tool to route messages from the bot to QnA Maker. Your bot can choose which has the best response for the user.

You can integrate bots created using Power Virtual Agents. You can use the Dispatch tool to configure your bot to work with a Power Virtual Agent Bot.