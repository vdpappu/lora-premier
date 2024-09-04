# python train.py --dataset "Rahulholla/stock-analysis" --builder_config "default"

import argparse
import os
import torch
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face Dataset ID")
    parser.add_argument("--builder_config", type=str, default="default", help="Hugging Face Dataset ID")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b", help="Name of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--max_steps", type=int, default=800, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_bnb", action="store_true", help="Use BitsAndBytes quantization")
    parser.add_argument("--seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    return parser.parse_args()

# def formatting_prompts_func(examples):
      #formatting funciton for instruction, question, answer format
#     output_texts = []
#     for i in range(len(examples['instruction'])):
#         text = f"### Instruction: {examples['instruction'][i]}\n"\
#                f"### Question: {examples['input'][i]}\n\n"\
#                f"### Answer: {examples['output'][i]}"
#         text = text.strip() + ' <eos>'
#         output_texts.append(text)
#     return output_texts

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Question: {example['input'][i]}\n\n ### Answer: {example['output'][i]}"
        text = text.strip()+' <eos>'
        output_texts.append(text)
    return output_texts

def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, 
                                              trust_remote_code=True,
                                              token=huggingface_token,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        token=huggingface_token,
        device_map="auto",
    )
    model.config.use_cache = False

    return model, tokenizer

def main(args):
    set_seed(args.seed)
    notebook_login()

    # Setup wandb
    if args.use_wandb:
        wandb.login()

    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args)

    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset, args.builder_config, use_auth_token=False, streaming=False)

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=30,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        optim="adamw_torch",
        report_to="wandb" if args.use_wandb else "none",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed. Saving model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)