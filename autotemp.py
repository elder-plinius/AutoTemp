import openai
from termcolor import colored
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
        print(colored("Generated text:", "green"))
        print(colored(text, "white"))
        feedback = input(colored("Are you satisfied with this output? (yes/no): ", "green"))
        return feedback.lower() == 'yes'

    def present_options_to_user(self, outputs):
        print(colored("Alternative outputs:", "green"))
        for temp, output in outputs.items():
            print(colored(f"Temperature {temp}:", "green") + colored(f" {output}", "blue"))
        chosen_temp = float(input(colored("Choose the temperature of the output you like: ", "green")))
        return outputs.get(chosen_temp, "Invalid temperature chosen.")

    def generate_with_chat(self, prompt, temperature):
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
            print(colored(f"Error generating text at temperature {temperature}: {e}", "red"))
            return None

    def run(self, prompt):
        initial_output_text = self.generate_with_chat(prompt, self.default_temp)

        user_satisfied = self.ask_user_feedback(initial_output_text)

        if user_satisfied:
            return initial_output_text
        else:
            outputs = {}
            scores = {}
            for temp in self.alt_temps:
                output_text = self.generate_with_chat(prompt, temp)
                if output_text:
                    outputs[temp] = output_text
                    eval_prompt = f"You are a task output evaluator. Rate the quality of the following output on a scale from 0 to 1000. Output only an integer. The output is: {output_text}"
                    score_text = self.generate_with_chat(eval_prompt, 0)
                    score_match = re.search(r'\d+', score_text)
                    if score_match:
                        scores[temp] = int(score_match.group())
                        print(colored(f"Score for temperature {temp}: {scores[temp]}", "yellow"))
                    else:
                        print(colored(f"Unable to parse score for temperature {temp}. Received: {score_text}", "red"))
                        scores[temp] = 0

            if self.auto_select:
                best_temp = max(scores, key=scores.get, default=self.default_temp)
                chosen_output = outputs.get(best_temp, "No valid outputs generated.")
                print(colored(f"Automatically selected output from Temperature {best_temp}:", "green"))
                print(colored(chosen_output, "white"))
                return chosen_output
            else:
                chosen_output = self.present_options_to_user(outputs)
                return chosen_output

if __name__ == "__main__":
    agent = AutoTemp(auto_select=True)  # Set auto_select to False if you want manual selection
    prompt = "Code a simple new innovative video game that I can play in browser"
    final_output = agent.run(prompt)
    print(final_output)
