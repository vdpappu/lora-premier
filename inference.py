import os
import logging
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_NAME = "google/gemma-2b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_TOKEN = '<eos>'

class InferenceEngine:
    def __init__(self, lora_adapter: str):
        self.huggingface_token = os.environ("HUGGINGFACE_TOKEN")
        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN Is required for downloading Gemma2")
        
        self.lora_adapter = lora_adapter
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self._load_model_and_tokenizer()
        self._setup_generation_config()

    def _load_model_and_tokenizer(self):
        logger.info("Loading tokenizer and model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=self.huggingface_token)
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, token=self.huggingface_token)
            if self.lora_adapter:
                self.model = PeftModel.from_pretrained(base_model, self.lora_adapter).to(DEVICE)
            else:
                self.model = base_model
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def _setup_generation_config(self):
        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            min_length=5,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    def generate_prompt(self, input_text: str, instruction: Optional[str] = None) -> str:
        text = f"### Question: {input_text}\n\n### Answer: "
        if instruction:
            text = f"### Instruction: {instruction}\n\n{text}"
        return text

    def generate_response(self, input_text: str, instruction: Optional[str] = None) -> str:
        prompt = self.generate_prompt(input_text, instruction)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=self.generation_config,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_text = generated_text.split(EOS_TOKEN)[0]
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return ""

def main():
    parser = argparse.ArgumentParser(description="Generate inference using a LoRA-adapted language model.")
    parser.add_argument("input_text", type=str, help="The input text for which to generate a response.")
    parser.add_argument("--lora_adapter", type=str, default=None, 
                        help="The LoRA adapter to use. Default is 'vdpappu/lora_medicalqa'.")
    parser.add_argument("--instruction", type=str, default=None, 
                        help="An optional instruction to guide the generation.")

    args = parser.parse_args()

    engine = InferenceEngine(args.lora_adapter)
    response = engine.generate_response(args.input_text, args.instruction)

    print(f"Response: {response}")

if __name__ == "__main__":
    main()