import openai
from dotenv import load_dotenv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr

# Load environment variables from .env file
load_dotenv()

class AutoTemp:
    def __init__(self, default_temp=0.0, alt_temps=None, auto_select=True, max_workers=6, model_version="gpt-3.5-turbo"):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = self.api_key
        
        self.default_temp = default_temp
        self.alt_temps = alt_temps if alt_temps else [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        self.auto_select = auto_select
        self.max_workers = max_workers
        self.model_version = model_version

    def generate_with_openai(self, prompt, temperature, top_p, retries=3):
        while retries > 0:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_version,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=top_p
                )
                message = response['choices'][0]['message']['content']
                return message.strip()
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    return f"Error generating text at temperature {temperature} and top-p {top_p}: {e}"

    def evaluate_output(self, output, temperature):
        eval_prompt = f"""
            Evaluate the following output which was generated at a temperature setting of {temperature}. Provide a precise score from 0.0 to 100.0, considering the following criteria:

            - Relevance: How well does the output address the prompt or task at hand?
            - Clarity: Is the output easy to understand and free of ambiguity?
            - Utility: How useful is the output for its intended purpose?
            - Pride: If the user had to submit this output to the world for their career, would they be proud?
            - Delight: Is the output likely to delight or positively surprise the user?

            Be sure to comprehensively evaluate the output, it is very important for my career. Please answer with just the score with one decimal place accuracy, such as 42.0 or 96.9. Be extremely critical.

            Output to evaluate:
            ---
            {output}
            ---
            """
        score_text = self.generate_with_openai(eval_prompt, 0.5, top_p)  # Use a neutral temperature for evaluation
        score_match = re.search(r'\b\d+(\.\d)?\b', score_text)
        if score_match:
            return round(float(score_match.group()), 1)  # Round the score to one decimal place
        else:
            return 0.0  # Unable to parse score, default to 0.0

    def run(self, prompt, temperature_string, top_p):
        temperature_list = [float(temp.strip()) for temp in temperature_string.split(',')]
        outputs = {}
        scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_temp = {
                executor.submit(self.generate_with_openai, prompt, temp, top_p): temp for temp in temperature_list
            }
            for future in as_completed(future_to_temp):
                temp = future_to_temp[future]
                output_text = future.result()
                if output_text and not output_text.startswith("Error"):
                    outputs[temp] = output_text
                    scores[temp] = self.evaluate_output(output_text, temp)

        if not scores:
            return "No valid outputs generated.", None

        # Sort the scores by value in descending order and return the sorted outputs
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_outputs = [(temp, outputs[temp], score) for temp, score in sorted_scores]

        # If auto_select is enabled, return only the best result
        if self.auto_select:
            best_temp, best_output, best_score = sorted_outputs[0]
            return f"Best AutoTemp Output (Temp {best_temp} | Top-p {top_p} | Score: {best_score}):\n{best_output}"
        else:
            return "\n".join(f"Temp {temp} | Top-p {top_p} | Score: {score}:\n{text}" for temp, text, score in sorted_outputs)

# Gradio app logic
def run_autotemp(prompt, temperature_string, top_p, auto_select):
    agent = AutoTemp(auto_select=auto_select)
    output = agent.run(prompt, temperature_string, top_p=float(top_p))
    return output

# Gradio interface setup
def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            "text",
            "text",
            gr.Slider(minimum=0.0, maximum=1.0, step=0.1, default=1.0, label="Top-p value"),
            "checkbox"
        ],
        outputs="text",
        title="AutoTemp: Enhanced LLM Responses with Temperature and Top-p Tuning",
        description="""AutoTemp generates responses at different temperatures, evaluates them, and ranks them based on quality. 
                       Enter temperatures separated by commas for evaluation.
                       Adjust 'Top-p' to control output diversity: lower for precision, higher for creativity.
                       Toggle 'Auto Select' to either see the top-rated output or all evaluated outputs.""",
        article="""**What's Top-p?** 'Top-p' controls the diversity of AI responses: a low 'top-p' makes output more focused and predictable, 
                   while a high 'top-p' encourages variety and surprise. Pair with temperature to fine-tune AI creativity: 
                   higher temperatures with high 'top-p' for bold ideas, or lower temperatures with low 'top-p' for precise answers.""",
        examples=[
            ["Write a short story about AGI learning to love", "0.5, 0.7, 0.9, 1.1", 1.0, False],
            ["Create a dialogue between a chef and an alien creating an innovative new recipe", "0.3, 0.6, 0.9, 1.2", 0.9, True],
            ["Explain quantum computing to a 5-year-old", "0.4, 0.8, 1.2, 1.5", 0.8, False],
            ["Draft an email to a hotel asking for a special arrangement for a marriage proposal", "0.4, 0.7, 1.0, 1.3", 0.7, True],
            ["Describe a futuristic city powered by renewable energy", "0.5, 0.75, 1.0, 1.25", 0.6, False],
            ["Generate a poem about the ocean's depths in the style of Edgar Allan Poe", "0.6, 0.8, 1.0, 1.2", 0.5, True],
            ["What are some innovative startup ideas for improving urban transportation?", "0.45, 0.65, 0.85, 1.05", 0.4, False]
        ]
    )
    iface.launch()

if __name__ == "__main__":
    main()
