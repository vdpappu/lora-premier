import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import subprocess
import os

def main(lora_adapter_id: str, gguf_output_file: str, llamacpp_path: str, hf_local_save_folder: Optional[str] = "./MERGED_LORA_MODEL", model_precision: Optional[str] = "q8_0"):

    BASE_MODEL_ID = "google/gemma-2b"
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, lora_adapter_id)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(hf_local_save_folder)
    tokenizer.save_pretrained(hf_local_save_folder)

    # Expand the tilde to the full home directory path
    llamacpp_path = os.path.expanduser(llamacpp_path)
    convert_script = os.path.join(llamacpp_path, "convert_hf_to_gguf.py")
    print(f"Convert script path: {convert_script}")

    command = [
        "python",
        convert_script,
        os.path.abspath(hf_local_save_folder),
        "--outfile",
        os.path.abspath(gguf_output_file),
        "--outtype",
        model_precision
    ]

    print(f"Command: {' '.join(command)}")

    # Run the command from the llama.cpp directory
    try:
        # Check if the directory exists before running the command
        if not os.path.isdir(llamacpp_path):
            raise FileNotFoundError(f"The directory does not exist: {llamacpp_path}")

        process = subprocess.Popen(command, cwd=llamacpp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("GGUF model generated successfully.")
            print(stdout)
        else:
            print("An error occurred while generating gguf model.")
            print(stderr)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GGUF model from LoRA adapter")
    parser.add_argument("lora_adapter_id", type=str, help="LoRA adapter ID")
    parser.add_argument("gguf_output_file", type=str, help="Output GGUF file name")
    parser.add_argument("--hf_local_save_folder", type=str, default="./MERGED_LORA_MODEL", help="Local folder to save merged model")
    parser.add_argument("--model_precision", type=str, default="q8_0", help="Model precision for quantization")
    parser.add_argument("--llamacpp_path", type=str, help="Path to llama.cpp directory")

    args = parser.parse_args()

    main(args.lora_adapter_id, args.gguf_output_file, args.llamacpp_path, args.hf_local_save_folder, args.model_precision)
