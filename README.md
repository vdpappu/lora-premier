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

To fine-tune the Gemma-2b model using LoRA:

1. Prepare your dataset in the required format (jsonl or csv).
2. Run the training script:
   ```
   python train.py --dataset <path_to_dataset> --output_dir <output_directory>
   ```
   Additional parameters can be specified. Run `python train.py --help` for more options.

### Converting to GGUF (Optional)

To convert the LoRA-adapted Gemma-2b model to GGUF format:

1. Run the conversion script:
   ```
   python lora_hf_gguf_convertor.py --base_model google/gemma-2b --lora_weights <path_to_lora_weights> --output_dir <output_directory>
   ```

### Inference

Multiple inference notebooks are provided to test the trained Gemma-2b model:

1. `inference_basic.ipynb`: Basic inference using the LoRA-adapted Gemma-2b model.
   - Loads the model and tokenizer
   - Generates text based on a given prompt

2. `inference_advanced.ipynb`: Advanced inference with additional parameters.
   - Allows customization of generation parameters (temperature, top_p, etc.)
   - Supports batch inference

3. `inference_gguf.ipynb`: Inference using the GGUF-converted Gemma-2b model with llama.cpp.
   - Demonstrates how to use the converted model for efficient inference

4. `inference_comparison.ipynb`: Compares outputs from the base Gemma-2b model and the LoRA-adapted model.
   - Generates responses from both models for the same prompts
   - Helps evaluate the impact of fine-tuning

To use these notebooks:
1. Open the desired notebook in Jupyter or your preferred environment.
2. Update the model path and other parameters as needed.
3. Run the cells to perform inference and analyze the results.

## Requirements

See `requirements.txt` for a list of required Python packages. This includes:
- PyTorch
- Transformers
- PEFT
- llama-cpp-python (for GGUF conversion)

Some of these packages may have their own dependencies which will be automatically installed.
