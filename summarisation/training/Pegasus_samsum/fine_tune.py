import logging
import torch
from datasets import load_dataset
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_function(example, tokenizer):
    inputs = tokenizer(
        example["dialogue"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        example["summary"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def main():
    model_name = "google/pegasus-xsum"
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    logger.info("Applying LoRA")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, peft_config)

    # Freeze all non-LoRA layers
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    model.print_trainable_parameters()

    logger.info("Loading and tokenizing SAMSum dataset")
    dataset = load_dataset("samsum", trust_remote_code=True)
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./lora_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # further reduced for low-VRAM
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training with LoRA...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    trainer.train()

    logger.info("Saving fine-tuned LoRA model...")
    model.save_pretrained("./lora_pegasus_samsum")

if __name__ == "__main__":
    main()

    logger.info("Fine-tuning completed.")
    logger.info("Model saved successfully.")
    logger.info("Training completed.")