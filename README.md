# AutoTemp

AutoTemp is a sophisticated Python utility designed to streamline the interaction with generative language models. It generates multiple text outputs at different "temperatures" to provide a variety of responses from highly predictable to very creative. AutoTemp enhances the user experience by allowing for an automatic or manual selection of the best output based on the quality scores.

## Key Features

- **Variety of Outputs**: Generates text at different temperatures, offering a range from conservative and reliable to creative and diverse responses.
- **Interactive Feedback**: Users can give immediate feedback on the generated text, ensuring it aligns with their expectations.
- **Automated Selection**: Features an automated selection process that picks the best output based on internal scoring, enhancing efficiency.
- **Manual Override**: Users can choose their preferred output manually, offering greater control over the final selection.
- **Scoring Visibility**: Scores for each temperature's output are displayed for transparency, allowing users to understand the selection process.

## Installation

Before you begin, ensure you have Python 3.6 or later installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/elder-plinius/AutoTemp.git
   cd AutoTemp

    Install the required dependencies:

    bash

    pip install -r requirements.txt

Usage

Set up your language model API key in the environment or directly in the script, then execute the auto_temp.py script with your desired prompt:

bash

python auto_temp.py "Create a simple and innovative browser-based video game."

Configuration

You can customize the default temperature, alternative temperatures, and the auto_select feature in the auto_temp.py script according to your needs.
