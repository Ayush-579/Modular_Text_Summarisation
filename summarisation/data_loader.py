import logging
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name="samsum"):
        self.dataset_name = dataset_name
        self.dataset = None
        logging.info(f"Initializing DataLoader with dataset: {dataset_name}")

    def load(self, split="train"):
        self.dataset = load_dataset(self.dataset_name)[split]
        logging.info(f"Loaded {len(self.dataset)} samples from {split} split.")
        return self.dataset
