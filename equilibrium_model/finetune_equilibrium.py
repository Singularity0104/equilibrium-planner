import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import pickle
import gc
from .plw_trainer import PLWTrainer

def merge_model(model_path, lora_dir, save_model_dir):
    print(f'merging {lora_dir} into {model_path} and saving in {save_model_dir}......')
    # Merge and save the fine-tuned model
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )
    llama_model = PeftModel.from_pretrained(base_model, lora_dir)
    llama_model = llama_model.merge_and_unload()

    # Reload tokenizer to save it
    llama_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    # Save the merged model
    llama_model.save_pretrained(save_model_dir)
    llama_tokenizer.save_pretrained(save_model_dir)

def finetune(args, model_path, data_path, save_lora_dir, save_model_dir, cache_dir):
    if not isinstance(data_path, list):
        data_path = [data_path]

    training_data = []
    for p in data_path:
        print(f'loading {p}...')
        with open(p, 'rb') as f:
            training_data += pickle.load(f)['train']

    for item in training_data:
        cur_feedback = '\n'.join(item['feedback']) if 'feedback' in item.keys() and len(item['feedback']) > 0 else 'Null'
        cur_draft_program = item['fix_point'][-1] if 'fix_point' in item.keys() and len(item['fix_point']) > 0 else 'Null'
        item['text'] = item['text_template'].format(
            feedback=cur_feedback,
            draft_program=cur_draft_program,
            refine_program=item['program']
        )

    training_data = Dataset.from_list(training_data)
    print(training_data[0])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    llama_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    llama_base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        attn_implementation='flash_attention_2',
        device_map={"": 0},
        trust_remote_code=True
    )

    llama_base_model = prepare_model_for_kbit_training(llama_base_model)

    llama_base_model.config.use_cache = False
    llama_base_model.config.pretraining_tp = 1

    # LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Params
    training_params = TrainingArguments(
        output_dir=cache_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=args.save_steps,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # Trainer
    if args.prompt_loss_weight is None:
        llama_fine_tuning = SFTTrainer(
            model=llama_base_model,
            train_dataset=training_data,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=llama_tokenizer,
            args=training_params,
            max_seq_length=args.seq_length,
          )
    else:
        llama_fine_tuning = PLWTrainer(
            model=llama_base_model,
            train_dataset=training_data,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=llama_tokenizer,
            args=training_params,
            max_seq_length=args.seq_length,
            prompt_loss_weight=args.prompt_loss_weight,
            sep=llama_tokenizer.convert_tokens_to_ids('<|end_header_id|>'),
        )

      # Training
    llama_fine_tuning.train()

    llama_fine_tuning.model.save_pretrained(save_lora_dir)

    merge_model(model_path, save_lora_dir, save_model_dir)

    del llama_base_model
    del llama_fine_tuning

    gc.collect()
    gc.collect()