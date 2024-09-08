# lora-premier
Everything LoRA

This repository contains scripts for fine-tuning language models using LoRA (Low-Rank Adaptation), performing inference with fine-tuned models, and converting LoRA-adapted models to GGUF format.

## Scripts

1. `train.py`: Fine-tunes language models using LoRA.
2. `inference.py`: Performs inference using LoRA-adapted models.
3. `lora_hf_gguf_convertor.py`: Converts LoRA-adapted models to GGUF format for use with llama.cpp.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/vdpappu/lora-premier.git
   cd lora-premier
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face token as an environment variable:
   ```
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

### Fine-tuning

To fine-tune a model using LoRA:

## Requirements

See `requirements.txt` for a list of required Python packages. This includes:
- PyTorch
- Transformers
- PEFT
- llama-cpp-python (for GGUF conversion)

Some of these packages may have their own dependencies which will be automatically installed.
