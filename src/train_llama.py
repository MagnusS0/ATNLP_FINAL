from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from rich import print
from rich.traceback import install

install()

from .data.dataset_llama import load_data

dataset = load_data("./data/simple_split/size_variations/tasks_train_simple_p32.txt")

max_len = 128  # Keep it the same as in the seq2seq model
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=max_len,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Set up LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Optimized for performance
    bias="none",  # Optimized for performance
    use_gradient_checkpointing=False,  # Change if OOM error
    random_state=2025,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# save chat template to jinja template file


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


pass

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print(dataset[5]["conversations"])
print(dataset[5]["text"])

# Get epochs based on dataset size divided by 100 000 so we train for 100 000 samples.
epochs = int(100000 // dataset.num_rows)
print(f"Training for {epochs} epochs.")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_len,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=8,
    packing=True,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        num_train_epochs=5,  # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",  # Use this for WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

space = tokenizer(" ", add_special_tokens=False).input_ids[0]
print(
    tokenizer.decode(
        [space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]
    )
)

trainer_stats = trainer.train()

print(trainer_stats)
