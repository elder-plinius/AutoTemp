import openai
from dotenv import load_dotenv
import os
import re
import json
import math
import statistics
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Optional external metrics
try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None  # graceful fallback

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    rouge_scorer = None  # graceful fallback

try:
    from bert_score import score as bert_score  # type: ignore
except Exception:
    bert_score = None  # graceful fallback

# Load environment variables from .env file
load_dotenv()

class AutoTemp:
    def __init__(self, default_temp=0.0, alt_temps=None, auto_select=True, max_workers=6, model_version="gpt-3.5-turbo", judges=3):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = self.api_key
        
        self.default_temp = default_temp
        self.alt_temps = alt_temps if alt_temps else [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        self.auto_select = auto_select
        self.max_workers = max_workers
        self.model_version = model_version
        self.judges = max(1, int(judges))
        # Token usage tracking (aggregate)
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.usage_events: List[Dict[str, int]] = []

    def _update_usage(self, usage: Optional[Any]) -> None:
        try:
            if usage is None:
                return
            prompt = int(getattr(usage, 'prompt_tokens', 0) or usage.get('prompt_tokens', 0))
            completion = int(getattr(usage, 'completion_tokens', 0) or usage.get('completion_tokens', 0))
            total = int(getattr(usage, 'total_tokens', 0) or usage.get('total_tokens', 0) or (prompt + completion))
            self.usage_totals["prompt_tokens"] += prompt
            self.usage_totals["completion_tokens"] += completion
            self.usage_totals["total_tokens"] += total
            self.usage_events.append({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total})
        except Exception:
            pass

    def generate_with_openai(self, prompt: str, temperature: float, top_p: float, retries: int = 3) -> Tuple[str, Optional[Dict[str, int]]]:
        while retries > 0:
            try:
                response = openai.chat.completions.create(
                    model=self.model_version,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=top_p
                )
                # Adjusted to use attribute access instead of dictionary access
                message = response.choices[0].message.content
                usage_obj = getattr(response, 'usage', None)
                usage_dict = None
                if usage_obj is not None:
                    usage_dict = {
                        "prompt_tokens": int(getattr(usage_obj, 'prompt_tokens', 0)),
                        "completion_tokens": int(getattr(usage_obj, 'completion_tokens', 0)),
                        "total_tokens": int(getattr(usage_obj, 'total_tokens', 0)),
                    }
                    self._update_usage(usage_dict)
                return message.strip(), usage_dict
            except Exception as e:
                retries -= 1
                print(f"Attempt failed with error: {e}")  # Print the error for debugging
                if retries <= 0:
                    print(f"Final error generating text at temperature {temperature} and top-p {top_p}: {e}")
                    return f"Error generating text at temperature {temperature} and top-p {top_p}: {e}", None


    def _evaluate_output_json(self, output: str, temperature: float, top_p: float, judge_id: int) -> Dict[str, float]:
        fixed_top_p_for_evaluation = 1.0
        eval_prompt = f"""
            You are Judge #{judge_id}. Evaluate the OUTPUT below which was generated at temperature {temperature} and top_p {top_p}.
            Return a STRICT minified JSON object with numeric fields only (no text outside JSON):
            {{"relevance": float0to100, "clarity": float0to100, "utility": float0to100, "creativity": float0to100, "coherence": float0to100, "safety": float0to100, "overall": float0to100}}
            Scoring rubric:
            - relevance: Addresses the prompt directly and completely.
            - clarity: Clear, unambiguous writing.
            - utility: Practical usefulness for the intended task.
            - creativity: Novel, insightful, or delightful content (not at the cost of truth).
            - coherence: Logical structure and consistency.
            - safety: Avoids hallucinations and harmful content; favors factual accuracy.
            - overall: Weighted aggregate you deem most faithful to a careful human judge.
            Output to evaluate between triple dashes:
            ---
            {output}
            ---
        """
        raw, _ = self.generate_with_openai(eval_prompt, 0.2, fixed_top_p_for_evaluation)
        try:
            # Try to extract a JSON object from the response
            json_text_match = re.search(r"\{[\s\S]*\}", raw)
            json_text = json_text_match.group(0) if json_text_match else raw
            data = json.loads(json_text)
            return {
                "relevance": float(data.get("relevance", 0.0)),
                "clarity": float(data.get("clarity", 0.0)),
                "utility": float(data.get("utility", 0.0)),
                "creativity": float(data.get("creativity", 0.0)),
                "coherence": float(data.get("coherence", 0.0)),
                "safety": float(data.get("safety", 0.0)),
                "overall": float(data.get("overall", 0.0)),
            }
        except Exception:
            score_match = re.search(r'\b\d+(?:\.\d+)?\b', raw)
            fallback_overall = float(score_match.group()) if score_match else 0.0
            return {
                "relevance": 0.0,
                "clarity": 0.0,
                "utility": 0.0,
                "creativity": 0.0,
                "coherence": 0.0,
                "safety": 0.0,
                "overall": round(fallback_overall, 1),
            }

    def evaluate_output(self, output: str, temperature: float, top_p: float) -> Dict[str, float]:
        if self.judges <= 1:
            judge_scores = [self._evaluate_output_json(output, temperature, top_p, judge_id=1)]
        else:
            with ThreadPoolExecutor(max_workers=min(self.judges, self.max_workers)) as executor:
                futures = [
                    executor.submit(self._evaluate_output_json, output, temperature, top_p, judge_id=j+1)
                    for j in range(self.judges)
                ]
                judge_scores = [f.result() for f in as_completed(futures)]

        # Aggregate by mean
        def mean(key):
            vals = [js.get(key, 0.0) for js in judge_scores]
            return round(sum(vals) / max(1, len(vals)), 2)

        aggregated = {
            "relevance": mean("relevance"),
            "clarity": mean("clarity"),
            "utility": mean("utility"),
            "creativity": mean("creativity"),
            "coherence": mean("coherence"),
            "safety": mean("safety"),
            "overall": mean("overall"),
        }
        return aggregated

    def run(self, prompt: str, temperature_string: str, top_p: float, advanced: bool = False, rounds: int = 1, exploration_c: float = 1.0) -> str:
        temperature_list = [float(temp.strip()) for temp in temperature_string.split(',') if temp.strip()]
        if not temperature_list:
            return "No temperatures provided."

        if not advanced:
            outputs = {}
            overall_scores = {}
            detailed_scores = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_temp = {
                    executor.submit(self.generate_with_openai, prompt, temp, top_p): temp for temp in temperature_list
                }
                for future in as_completed(future_to_temp):
                    temp = future_to_temp[future]
                    try:
                        output_text, _ = future.result()
                        print(f"Output for temp {temp}: {output_text}")
                        if output_text and not output_text.startswith("Error"):
                            outputs[temp] = output_text
                            score_dict = self.evaluate_output(output_text, temp, top_p)
                            detailed_scores[temp] = score_dict
                            overall_scores[temp] = score_dict.get("overall", 0.0)
                    except Exception as e:
                        print(f"Error while generating or evaluating output for temp {temp}: {e}")

            if not overall_scores:
                return "No valid outputs generated."

            sorted_scores = sorted(overall_scores.items(), key=lambda item: item[1], reverse=True)
            if self.auto_select:
                best_temp, best_overall = sorted_scores[0]
                best_output = outputs[best_temp]
                best_detail = detailed_scores[best_temp]
                return (
                    f"Best AutoTemp Output (Temp {best_temp} | Top-p {top_p} | Overall: {best_overall}):\n"
                    f"{best_output}\n\n"
                    f"Judges (mean scores): {json.dumps(best_detail, ensure_ascii=False)}"
                )
            else:
                lines = []
                for temp, overall in sorted_scores:
                    lines.append(
                        f"Temp {temp} | Top-p {top_p} | Overall: {overall} | Detail: {json.dumps(detailed_scores[temp], ensure_ascii=False)}:\n{outputs[temp]}"
                    )
                return "\n\n".join(lines)
        else:
            # Advanced: UCB1 bandit over temperatures
            num_rounds = max(1, int(rounds))
            c = float(exploration_c)
            pulls = {t: 0 for t in temperature_list}
            sums = {t: 0.0 for t in temperature_list}
            best_outputs = {t: {"overall": -1.0, "text": "", "detail": {}} for t in temperature_list}
            total_pulls = 0

            # Ensure each arm is pulled at least once
            init_order = list(temperature_list)
            for t in init_order:
                out, _ = self.generate_with_openai(prompt, t, top_p)
                if out and not out.startswith("Error"):
                    score_detail = self.evaluate_output(out, t, top_p)
                    score = score_detail.get("overall", 0.0)
                    pulls[t] += 1
                    sums[t] += score
                    total_pulls += 1
                    if score > best_outputs[t]["overall"]:
                        best_outputs[t] = {"overall": score, "text": out, "detail": score_detail}

            for _ in range(num_rounds - 1):
                # Compute UCB
                ucb_values = {}
                for t in temperature_list:
                    if pulls[t] == 0:
                        ucb_values[t] = float("inf")
                    else:
                        mean = sums[t] / pulls[t]
                        bonus = c * math.sqrt(max(1e-9, math.log(max(1, total_pulls)) / pulls[t]))
                        ucb_values[t] = mean + bonus
                # Select best arm
                next_t = max(temperature_list, key=lambda tt: ucb_values[tt])
                out, _ = self.generate_with_openai(prompt, next_t, top_p)
                if out and not out.startswith("Error"):
                    score_detail = self.evaluate_output(out, next_t, top_p)
                    score = score_detail.get("overall", 0.0)
                    pulls[next_t] += 1
                    sums[next_t] += score
                    total_pulls += 1
                    if score > best_outputs[next_t]["overall"]:
                        best_outputs[next_t] = {"overall": score, "text": out, "detail": score_detail}

            # Prepare output
            means = {t: (sums[t] / pulls[t]) if pulls[t] > 0 else 0.0 for t in temperature_list}
            ranked = sorted(temperature_list, key=lambda t: means[t], reverse=True)
            best_t = ranked[0]
            best = best_outputs[best_t]
            header = (
                f"Advanced Mode (UCB) — Best Output (Temp {best_t} | Top-p {top_p} | Mean: {round(means[best_t], 2)} | Best Overall: {round(best['overall'], 2)}):\n"
            )
            summary_lines = [header, best["text"], "", f"Detail: {json.dumps(best['detail'], ensure_ascii=False)}", ""]
            if not self.auto_select:
                for t in ranked:
                    summary_lines.append(
                        f"Temp {t}: pulls={pulls[t]}, mean_overall={round(means[t], 2)}, best_overall={round(best_outputs[t]['overall'], 2)}"
                    )
            return "\n".join(summary_lines)

    # -------------------- Metrics & Benchmarking Utilities --------------------
    @staticmethod
    def _compute_external_metrics(candidate: str, reference: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        try:
            if sacrebleu is not None:
                bleu = sacrebleu.corpus_bleu([candidate], [[reference]])
                metrics["BLEU"] = float(bleu.score)
        except Exception:
            pass
        try:
            if rouge_scorer is not None:
                scorer = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)
                scores = scorer.score(reference, candidate)
                metrics["ROUGE1_F"] = float(scores["rouge1"].fmeasure)
                metrics["ROUGE_Lsum_F"] = float(scores["rougeLsum"].fmeasure)
        except Exception:
            pass
        try:
            if bert_score is not None:
                P, R, F1 = bert_score([candidate], [reference], lang="en", rescale_with_baseline=True)
                metrics["BERTScore_F1"] = float(F1.mean().item())
        except Exception:
            pass
        return metrics

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        values_sorted = sorted(values)
        k = (len(values_sorted) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(values_sorted[int(k)])
        d0 = values_sorted[f] * (c - k)
        d1 = values_sorted[c] * (k - f)
        return float(d0 + d1)

    @staticmethod
    def _bootstrap_ci(values: List[float], num_samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        means = []
        n = len(values)
        for _ in range(num_samples):
            sample = random.choices(values, k=n)
            means.append(sum(sample) / n)
        lower = AutoTemp._percentile(means, alpha / 2)
        upper = AutoTemp._percentile(means, 1 - alpha / 2)
        return float(sum(values) / n), float(lower), float(upper)

    def estimate_cost_usd(self) -> float:
        # Simple estimator; update as needed
        model_costs = {
            "gpt-3.5-turbo": {"prompt_per_1k": 0.50, "completion_per_1k": 1.50},
            "gpt-4": {"prompt_per_1k": 30.00, "completion_per_1k": 60.00},
        }
        cfg = model_costs.get(self.model_version)
        if not cfg:
            return 0.0
        prompt_usd = (self.usage_totals["prompt_tokens"] / 1000.0) * cfg["prompt_per_1k"]
        completion_usd = (self.usage_totals["completion_tokens"] / 1000.0) * cfg["completion_per_1k"]
        return round(prompt_usd + completion_usd, 4)

    @staticmethod
    def _extract_best_output_from_run(run_text: str) -> str:
        # Extract the body text after the first header line until an empty line or 'Judges'
        try:
            lines = run_text.splitlines()
            if not lines:
                return run_text
            # skip header line
            body_lines = []
            for ln in lines[1:]:
                if not ln.strip():
                    break
                if ln.strip().startswith("Judges"):
                    break
                body_lines.append(ln)
            return "\n".join(body_lines).strip() or run_text
        except Exception:
            return run_text

    def benchmark(self, dataset: List[Dict[str, str]], temperature_string: str, top_p: float, models: Optional[List[str]] = None, advanced: bool = False, rounds: int = 1, judges: int = 3, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Benchmark across a dataset of {prompt, reference} items.
        Returns summary with means and (optional) external metrics if available."""
        results: Dict[str, Any] = {}
        model_list = models or [self.model_version]
        for model_name in model_list:
            self.model_version = model_name
            self.judges = judges
            self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            per_item_scores: List[float] = []
            per_item_metrics: Dict[str, List[float]] = {}
            rows_for_csv: List[Dict[str, Any]] = []
            for item in dataset:
                prompt = item.get("prompt", "")
                reference = item.get("reference", "")
                try:
                    run_text = self.run(prompt, temperature_string, top_p, advanced=advanced, rounds=rounds)
                    best_output = self._extract_best_output_from_run(run_text)
                    # We do not have direct overall score; compute via judges again for consistency
                    score_detail = self.evaluate_output(best_output, temperature=float(self.default_temp or 0.7), top_p=float(top_p))
                    per_item_scores.append(float(score_detail.get("overall", 0.0)))
                    if reference:
                        met = self._compute_external_metrics(best_output, reference)
                        for k, v in met.items():
                            per_item_metrics.setdefault(k, []).append(float(v))
                    else:
                        met = {}
                    rows_for_csv.append({
                        "model": model_name,
                        "prompt": prompt,
                        "output": best_output,
                        "overall": float(score_detail.get("overall", 0.0)),
                        **{f"metric_{k}": v for k, v in met.items()}
                    })
                except Exception as e:
                    print(f"Benchmark error on item: {e}")
                    per_item_scores.append(0.0)
            mean_overall = round(sum(per_item_scores) / max(1, len(per_item_scores)), 3)
            mean_o, lower_o, upper_o = AutoTemp._bootstrap_ci(per_item_scores)
            metric_means = {k: round(sum(v) / max(1, len(v)), 4) for k, v in per_item_metrics.items()}
            metric_cis = {k: AutoTemp._bootstrap_ci(v) for k, v in per_item_metrics.items()}
            if csv_path:
                # union of all keys across rows
                fieldnames = sorted({key for row in rows_for_csv for key in row.keys()})
                try:
                    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in rows_for_csv:
                            writer.writerow(row)
                except Exception as e:
                    print(f"CSV export failed: {e}")
            results[model_name] = {
                "mean_overall": mean_overall,
                "mean_overall_ci": [round(lower_o, 3), round(upper_o, 3)],
                "metric_means": metric_means,
                "metric_cis": {k: [round(v[1], 4), round(v[2], 4)] for k, v in metric_cis.items()},
                "num_items": len(dataset),
                "tokens": dict(self.usage_totals),
                "estimated_cost_usd": self.estimate_cost_usd(),
            }
        return results

# Gradio app logic
def run_autotemp(prompt, temperature_string, top_p, auto_select, advanced_mode, rounds, judges, exploration_c):
    agent = AutoTemp(auto_select=auto_select, judges=int(judges))
    output = agent.run(prompt, temperature_string, top_p=float(top_p), advanced=bool(advanced_mode), rounds=int(rounds), exploration_c=float(exploration_c))
    return output

# Gradio interface setup
def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            "text",
            "text",
            gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="top-p value"),
            gr.Checkbox(value=True, label="Auto Select Best"),
            gr.Checkbox(value=False, label="Advanced Mode (UCB)").style(container=True),
            gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Rounds (Advanced)"),
            gr.Slider(minimum=1, maximum=7, step=1, value=3, label="Judges"),
            gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=1.0, label="Exploration c (UCB)")
        ],
        outputs="text",
        title="AutoTemp: Research-Grade Temperature & Top-p Optimization",
        description="""AutoTemp now supports multi-judge structured evaluation and an optional UCB bandit optimizer.
                       Enter temperatures separated by commas for evaluation.
                       Adjust 'Top-p' to control output diversity, and switch to Advanced Mode for iterative optimization.
                       Judges average multiple independent evaluations into robust overall scores.""",
        article="""**FAQs**

**What's Top-p?** 'Top-p' controls the diversity of AI responses: a low 'top-p' makes output more focused and predictable, while a high 'top-p' encourages variety and surprise. Pair with temperature to fine-tune AI creativity: higher temperatures with high 'top-p' for bold ideas, or lower temperatures with low 'top-p' for precise answers. Using top_p=1 disables nucleus sampling.

**How Does Temperature Affect AI Outputs?** Temperature controls the randomness of word selection. Lower temperatures lead to more predictable text, while higher temperatures allow for more novel text generation.

**What is Advanced Mode (UCB)?** Advanced Mode treats each temperature as an arm in a bandit and iteratively selects temperatures using the UCB1 strategy to balance exploration and exploitation, improving sample efficiency while converging to better settings for your prompt.

**Why multiple judges?** Independent judging runs reduce variance and bias in single-evaluator scores. We report mean scores across judges for robustness.""",
        examples=[
            ["Write a short story about AGI learning to love", "0.5, 0.7, 0.9, 1.1", 1.0, True, False, 5, 3, 1.0],
            ["Explain quantum computing to a 5-year-old", "0.4, 0.8, 1.2, 1.5", 0.8, True, True, 8, 3, 1.2],
            ["Draft an email to a hotel asking for a special arrangement for a marriage proposal", "0.4, 0.7, 1.0, 1.3", 0.7, True, True, 10, 5, 0.8]
        ]
    )
    iface.launch()

if __name__ == "__main__":
    main()
