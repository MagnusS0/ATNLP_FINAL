from unsloth import FastLanguageModel
import torch
from rich import print
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import load_chat_template
import csv

from dataset_llama import load_data
import re

def extract_act_tags(text):
    """
    Extracts the text within <act>...</act> tags from the given text.

    Args:
        text (str): The input text containing <act> tags.

    Returns:
        list: A list of strings containing the text within each <act>...</act> tag.
    """
    # Define the regex pattern to match <act>[some text]</act>
    pattern = r"<action>(.*?)</action>"

    # Use re.findall to extract all matches
    extracted_texts = re.findall(pattern, text)

    return " ".join(extracted_texts)

dataset = load_data("./data/simple_split/size_variations/tasks_test_simple_p64.txt", test=True)

max_len = 128 # Keep it the same as in the seq2seq model
dtype = torch.bfloat16
load_in_4bit = True

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="outputs/checkpoint-820",
#     max_seq_length=max_len,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit
# )

# if True: model.save_pretrained_merged("outputs/model_p16_20e", tokenizer, save_method = "merged_16bit",)

# del model
# del tokenizer


llm = LLM(model="./outputs//model_64p_7e", max_model_len=256, max_num_seqs=512)
custom_chat_template = load_chat_template("./utils/llama_3.1_template.jinja")
print("Loaded chat template:", custom_chat_template)
sampling_params = SamplingParams(temperature=0.1, min_p=0.1, max_tokens=128)

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['output', 'input'])  # Header
    # With VLLM it is possible to generate multiple outputs at once So we can do batches
    all_inputs = []
    for item in dataset:
        messages = item["conversations"]
        all_inputs.append(messages)
    outputs = llm.chat(messages=all_inputs, sampling_params=sampling_params, chat_template=custom_chat_template)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        writer.writerow([extract_act_tags(generated_text), prompt])