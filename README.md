# AutoTemp

AutoTemp is a Python tool that enhances language model interactions by intelligently selecting the optimal temperature setting for generating responses. It leverages multiple temperature settings to produce a variety of outputs and then evaluates these outputs to determine which temperature yields the best result for a given prompt.

## Features

- **Multi-Temperature Evaluation**: Tests multiple temperature settings to find the best output for a given prompt.
- **Multi-Judge Structured Scoring**: Runs N independent judges and aggregates relevance, clarity, utility, creativity, coherence, safety, and overall.
- **Advanced Optimization (UCB Bandit)**: Optional Upper Confidence Bound (UCB1) optimizer iteratively explores/exploits temperatures.
- **External Metrics (Optional)**: BLEU (`sacrebleu`), ROUGE (`rouge-score`), and BERTScore (`bert-score`) with graceful fallback if not installed.
- **Automatic or Manual Selection**: Supports both automatic selection of the best output based on scores and manual selection by presenting options to the user.
- **Customizable Temperature Range**: Users can define a custom range of temperatures to be tested for each prompt.
- **Easy Integration**: Designed to work with OpenAI's GPT-3.5 or GPT-4 and is compatible with other language models that support temperature settings.

## Installation

To install AutoTemp, you can simply clone the repository and install the required dependencies.


    git clone https://github.com/elder-plinius/AutoTemp.git
    cd AutoTemp
    pip install -r requirements.txt

## OpenAI API Key

Before running AutoTemp, you need to set up your API key in an .env file at the root of the project:

    OPENAI_API_KEY='your-api-key-here'

This file should not be committed to your version control system as it contains sensitive information.

## Usage

To use AutoTemp, simply run the autotemp.py script with Python:

    python autotemp.py


You can use the Gradio UI to enter your prompt, select temperatures (comma-separated), configure `top-p`, toggle `Auto Select`, and enable `Advanced Mode (UCB)` with `Rounds`, number of `Judges`, and `Exploration c`.

### Modes
- **Standard Mode**: Generates one output per temperature, scores with multiple judges, and ranks by mean overall.
- **Advanced Mode (UCB)**: Treats each temperature as a bandit arm, pulls arms iteratively for the specified rounds using UCB1, and returns the best observed output with diagnostics.

### Research Notes
- The multi-judge rubric aims to reduce variance and bias of single-judge evaluations and returns mean scores across judges.
- UCB1 balances exploration and exploitation, offering higher sample-efficiency than naive uniform evaluation, especially on longer prompts.

## GitHub Pages (Static Web App)

This repo includes a static SPA in `docs/` suitable for GitHub Pages. It runs entirely in the browser and calls the OpenAI API with a user-provided key.

### Deploy Steps
1. Commit and push the repo to GitHub.
2. In your GitHub repo: Settings → Pages → Build and deployment → Source: Deploy from branch → Branch: `main` and Folder: `/docs`.
3. Save. Your app will be published at `https://<your-username>.github.io/<repo-name>/`.

### Using the Web App
- Open the GitHub Pages URL.
- Paste your OpenAI API key (optionally store it in your browser).
- Enter your prompt, temperatures, `top-p`, judges, and Advanced/UCB options.
- Click "Run AutoTemp" to see results and judge scores.

Security: The API key stays in the browser. Avoid sharing sensitive keys. For shared production, consider a server-side proxy.

## Benchmarking

You can benchmark across a dataset of `{prompt, reference}` pairs and produce summary stats, confidence intervals, and a CSV of item-level results.

Example (in code):

```python
from autotemp import AutoTemp

dataset = [
  {"prompt": "Summarize: ...", "reference": "Expected summary ..."},
  {"prompt": "Translate to French: ...", "reference": "..."},
]

agent = AutoTemp(judges=3, model_version="gpt-3.5-turbo")
summary = agent.benchmark(
  dataset=dataset,
  temperature_string="0.4,0.7,1.0",
  top_p=0.9,
  models=["gpt-3.5-turbo", "gpt-4"],
  advanced=True,
  rounds=8,
  judges=3,
  csv_path="results.csv",
)
print(summary)
```

Summary includes:
- mean_overall and bootstrap CI
- external metric means and CIs (if dependencies installed)
- token usage and estimated USD cost per model


## Configuration

You can customize the behavior of AutoTemp by setting the following parameters when initializing AutoTemp:

    default_temp: The default temperature to use for initial output.
    alt_temps: A list of alternative temperatures to evaluate.
    auto_select: Whether to automatically select the best output or present options to the user.
    max_workers: The maximum number of threads to use for concurrent API calls.
    model_version: Specifies the model version to use, such as "gpt-3.5-turbo" or "gpt-4".
    judges: Number of independent judges to run for scoring (default 3).
    advanced_mode: Toggle UCB bandit optimization in the UI.
    rounds: Number of bandit rounds (>=1) when Advanced Mode is enabled.
    exploration c: UCB exploration coefficient; higher favors exploration.

## Optional Dependencies

If you want external metrics, install:

```
pip install sacrebleu rouge-score bert-score
```
