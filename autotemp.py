from swarms.models import OpenAIChat  # Replace with your actual OpenAIChat import
from termcolor import colored

class AutoTemp:
    def __init__(self, api_key, default_temp=0.5, alt_temps=None, auto_select=True):
        self.api_key = api_key
        self.default_temp = default_temp
        self.alt_temps = alt_temps if alt_temps else [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  # Default alternative temperatures
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

    def run(self, prompt):
        try:
            llm = OpenAIChat(openai_api_key=self.api_key, temperature=self.default_temp)
            initial_output = llm(prompt)
        except Exception as e:
            print(colored(f"Error generating initial output: {e}", "red"))
            initial_output = None

        user_satisfied = self.ask_user_feedback(initial_output)

        if user_satisfied:
            return initial_output
        else:
            outputs = {}
            scores = {}
            for temp in self.alt_temps:
                try:
                    llm = OpenAIChat(openai_api_key=self.api_key, temperature=temp)
                    outputs[temp] = llm(prompt)
                    eval_prompt = f"You are a task output evaluator. Rate the quality of the following output on a scale from 0 to 1000. Output only an integer. The output is: {outputs[temp]}"
                    score_str = llm(eval_prompt)
                    scores[temp] = int(score_str.strip())
                    print(colored(f"Score for temperature {temp}: {scores[temp]}", "yellow"))  # Print the score
                except Exception as e:
                    print(colored(f"Error generating text at temperature {temp}: {e}", "red"))
                    outputs[temp] = None
                    scores[temp] = 0

            if self.auto_select:
                best_temp = max(scores, key=scores.get)
                print(colored(f"Automatically selected output from Temperature {best_temp}:", "green"))
                print(colored(outputs[best_temp], "white"))
                return outputs[best_temp]
            else:
                chosen_output = self.present_options_to_user(outputs)
                return chosen_output

if __name__ == "__main__":
    api_key = ""  # Your OpenAI API key here
    agent = AutoTemp(api_key, auto_select=True)  # Set auto_select to False if you want manual selection
    prompt = "code a simple new innovative video game that i can play in browser"
    final_output = agent.run(prompt)
