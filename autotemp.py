import openai
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()

class AutoTemp:
    def __init__(self, default_temp=0.0, alt_temps=None, auto_select=True):
        self.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
        self.default_temp = default_temp
        self.alt_temps = alt_temps if alt_temps else [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        self.auto_select = auto_select

    def ask_user_feedback(self, text):
        print("Generated text:")
        print(text)
        feedback = input("Are you satisfied with this output? (yes/no): ")
        return feedback.lower() == 'yes'

    def present_options_to_user(self, outputs):
        print("Alternative outputs:")
        for temp, output in outputs.items():
            print(f"Temperature {temp}: {output}")
        chosen_temp = float(input("Choose the temperature of the output you like: "))
        return outputs.get(chosen_temp, "Invalid temperature chosen."), chosen_temp

    def generate_with_openai(self, prompt, temperature):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=temperature
            )
            message = response['choices'][0]['message']['content']
            return message.strip()
        except Exception as e:
            print(f"Error generating text at temperature {temperature}: {e}")
            return None

    def run(self, prompt):
        initial_output_text = self.generate_with_openai(prompt, self.default_temp)

        if self.ask_user_feedback(initial_output_text):
            return initial_output_text, self.default_temp

        outputs = {}
        scores = {}
        for temp in self.alt_temps:
            output_text = self.generate_with_openai(prompt, temp)
            if output_text:
                outputs[temp] = output_text
                eval_prompt = f"You are now an expert task evaluator. Rate the quality of the following output on a scale from 1 to 25. Be thorough, thoughtful, and precise and output only a number. The output is: {output_text}"
                score_text = self.generate_with_openai(eval_prompt, 0.123)
                score_match = re.search(r'\d+', score_text)
                if score_match:
                    scores[temp] = int(score_match.group())
                    print(f"Score for temperature {temp}: {scores[temp]}")
                else:
                    print(f"Unable to parse score for temperature {temp}. Received: {score_text}")
                    scores[temp] = 0

        if not scores:  # No scores could be generated
            return "No valid outputs generated.", None

        if self.auto_select:
            best_temp = max(scores, key=scores.get, default=self.default_temp)
            chosen_output = outputs.get(best_temp, "No valid outputs generated.")
            return chosen_output, best_temp
        else:
            chosen_output, chosen_temp = self.present_options_to_user(outputs)
            return chosen_output, chosen_temp

# Set up the AutoTemp agent
if __name__ == "__main__":
    agent = AutoTemp()
    prompt = "Code a simple new innovative video game that I can play in browser in a single file"
    final_output, used_temp = agent.run(prompt)
    if used_temp is not None:
        print(f"Final selected output (Temperature {used_temp}):")
        print(final_output)
    else:
        print("No valid output was generated.")
