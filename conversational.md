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


### 5. Identify capabilities of the QnA Maker service

You can natively populate a Q&A Maker from a PDF or word document or by manually adding question and answer pairs.

It cannot use a SQL database (only files and URLs) as datasources.

It cannot be a sharepoint list. Files not webpages. If the URL ends with .ASPX it will not import into QnA maker.

App Service and Azure Cognitive Search are the Azure Resources created when a new QnA Maker service is created. It does not create the web app bot, you can create that later if you intend to surface it through the web channel.

TSV files for chitchat personality uploads.

### 6. Identify capabilities of the Bot Framework

The Bot Framework SDK is required to develop bots using code. There are SDKs for C#, JS, TS and Python. The Bot Framework SDK allows developers to send and receive messages with users on the configured channels.

The Azure Bot Framework Emulator is a desktop application that allows developers to test and debug bots on their local computer.

The Bot service framework CLI tools manage bots and related services and are used in the DevOps pipelines when deploying bots in enterprises.

The Bot Framework Composer is a tool to build bots without Code. The Bot Framework Composer uses a visual user interface to create diaslogs and bot logic.