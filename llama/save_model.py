from unsloth import FastLanguageModel
import torch
from rich import print

def save_model(checkpoint_path: str, output_path: str):
    """
    Load a checkpoint and save it in 16-bit format for VLLM inference.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., 'outputs/checkpoint-820')
        output_path: Path where to save the merged model (e.g., 'outputs/model_p16_20e')
    """
    max_len = 128
    dtype = torch.bfloat16
    load_in_4bit = True

    print(f"Loading model from {checkpoint_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_len,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    print(f"Saving merged model to {output_path}")
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Model saved successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Save Llama model in 16-bit format for VLLM inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--output', type=str, required=True, help='Path where to save the merged model')
    
    args = parser.parse_args()
    save_model(args.checkpoint, args.output) 