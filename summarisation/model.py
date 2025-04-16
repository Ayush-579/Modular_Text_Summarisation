import logging
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

class Summarizer:
    def __init__(self, model_name, device="cuda"):
        logging.info(f"Loading model: {model_name}")
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # ✅ Add special tokens for speaker roles
        special_tokens_dict = {"additional_special_tokens": ["[USER]", "[AGENT]"]}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info(f"Added {num_added_toks} special tokens: {special_tokens_dict['additional_special_tokens']}")
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def summarize(self, texts, max_input_length=1024, max_target_length=128):
        inputs = self.tokenizer(
            texts,
            max_length=max_input_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        summaries = self.model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=8,
            early_stopping=True
        )
        return self.tokenizer.batch_decode(summaries, skip_special_tokens=True)
