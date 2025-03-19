# Chatbot-Victor: Argumentative Dialogue System

## Overview
Chatbot-Victor is an AI-powered dialogue system that focuses on argumentative conversations. The project implements various types of conversational agents that can engage in debates and discussions on controversial topics. It's built using OpenAI's LLM APIs and provides a framework for creating, simulating, and evaluating conversational agents with different personas.

## Features
- **Multiple Agent Types**: Implementation of different agent architectures including basic LLM agents and character-based agents
- **Character Simulation**: Predefined character personas with specific traits, opinions, and conversational styles
- **Dialogue Management**: Tools for creating, tracking, and managing multi-turn conversations
- **Evaluation Framework**: Methods to assess the quality and effectiveness of argumentative dialogues
- **Kialo Integration**: Support for argument structures based on the Kialo debate platform

## Project Structure
- `agents.py`: Defines the core agent classes including LLMAgent and CharacterAgent
- `characters.py`: Contains character definitions with specific traits and conversational styles
- `dialogue.py`: Manages conversation structures and turn-taking
- `kialo.py`: Implements argument structures based on Kialo debate platform
- `simulate.py`: Tools for running simulated conversations between agents
- `evaluate.py`: Framework for evaluating dialogue quality and agent performance
- `myArgubots.py`: Custom implementation of argumentative agents
- `argubots.py`: Core implementation of argument-focused conversational agents
- `tracking.py`: Utility for tracking API usage and conversation metrics
- `logging_cm.py`: Custom logging functionality for the project

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv csc306.venv
   source csc306.venv/bin/activate  # On Windows: csc306.venv\Scripts\activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
4. Set up OpenAI API credentials in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Example
```python
from dialogue import Dialogue
from agents import LLMAgent
import os

# Make sure you have set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Create a simple agent
agent = LLMAgent(name="Assistant", temperature=0.7)

# Start a conversation
dialogue = Dialogue()
dialogue = agent.ask(dialogue, "Human", "What do you think about climate change?")
print(dialogue)
```

### Using Character Agents
```python
from agents import CharacterAgent
from characters import bob, cara
from dialogue import Dialogue

# Create a dialogue between two character agents
dialogue = Dialogue()
dialogue = bob.ask(dialogue, "Human", "Do you think it's ok to eat meat?")
dialogue = cara.respond(dialogue)
print(dialogue)
```

## Evaluation
The project includes an evaluation framework to assess dialogue quality:

```python
from evaluate import eval_on_characters
from myArgubots import MyArgubot

# Evaluate your custom argubot against the dev set
results = eval_on_characters(MyArgubot("MyBot"), n_dialogues=5)
print(results)
```

## Requirements
- Python 3.8+
- OpenAI API access
- Additional packages listed in requirements.txt

## Project Background
This project was developed as part of CSC 306 (likely a course on conversational AI or natural language processing). It explores techniques for creating AI systems capable of engaging in meaningful argumentative dialogues on controversial topics.

## License
[Insert appropriate license information here]

## Acknowledgements
- OpenAI for providing the language model APIs
- [Any other acknowledgements]