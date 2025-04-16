import logging
import yaml
import argparse
import os
from summarisation.data_loader import DataLoader
from summarisation.model import Summarizer
from summarisation.pipeline import SummarizationPipeline
from summarisation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
path = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Modular Text Summarization Pipeline (Pegasus)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to summarize")
    parser.add_argument("--model_name", type=str, help="Model name or path (overrides config)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config()

    model_name = args.model_name or config["model_name"]
    device = args.device or config["device"]

    logging.info(f"Using model: {model_name} on device: {device}")
    logging.info(f"Summarizing {args.num_samples} samples...")

    loader = DataLoader()
    model = Summarizer(model_name, device)
    pipeline = SummarizationPipeline(loader, model)
    evaluator = Evaluator()

    results = pipeline.run(num_samples=args.num_samples)
    inputs, predictions, references = zip(*results)

    for i, (inp, pred, ref) in enumerate(results):
        print(f"\n--- Sample {i+1} ---")
        print(f"[Dialogue]:\n{inp}\n")
        print(f"[Generated]:\n{pred}\n")
        print(f"[Reference]:\n{ref}\n")

    scores = evaluator.evaluate(predictions, references)
    print("\n🟢 ROUGE Scores:", scores)

if __name__ == "__main__":
    main()
