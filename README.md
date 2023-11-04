# AutoTemp

AutoTemp is a Python tool that enhances language model interactions by intelligently selecting the optimal temperature setting for generating responses. It leverages multiple temperature settings to produce a variety of outputs and then evaluates these outputs to determine which temperature yields the best result for a given prompt.

## Features

- **Multi-Temperature Evaluation**: Tests multiple temperature settings to find the best output for a given prompt.
- **User Feedback Integration**: Allows users to provide feedback on generated outputs, which can be used to inform the selection of the optimal temperature.
- **Automatic or Manual Selection**: Supports both automatic selection of the best output based on scores and manual selection by presenting options to the user.
- **Customizable Temperature Range**: Users can define a custom range of temperatures to be tested for each prompt.
- **Easy Integration**: Designed to work with OpenAI's GPT-3.5 and compatible with other language models that support temperature settings.

## Installation

To install AutoTemp, you can simply clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/AutoTemp.git
cd AutoTemp
pip install -r requirements.txt

Usage

To use AutoTemp, initialize the AutoTempAgent class with your API key and preferred settings. Here's a basic example:

python

from autotemp import AutoTempAgent

api_key = "your-api-key-here"  # Replace with your actual OpenAI API key
agent = AutoTempAgent(api_key=api_key, auto_select=True)

prompt = "Write a creative short story about a purple dragon"
final_output = agent.run(prompt)
print(final_output)

Configuration

You can customize the behavior of AutoTemp by setting the following parameters when initializing AutoTempAgent:

    default_temp: The default temperature to use for initial output.
    alt_temps: A list of alternative temperatures to evaluate.
    auto_select: Whether to automatically select the best output or present options to the user.
