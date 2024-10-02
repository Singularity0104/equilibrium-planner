import time

def generate_world_model(args, prompt, pipeline, tokenizer, top_k=2):
    begin_time = time.time()
    prompt_token = tokenizer(prompt, return_tensors="pt").input_ids
    max_length = args.world_model_max_generate_length + prompt_token.shape[1]
    sequence = pipeline(
        prompt,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=1,
        eos_token_id=[tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                      tokenizer.convert_tokens_to_ids('END')],
        max_length=max_length,
        # truncation=True
    )
    end_time = time.time()
    print(f'generate one time: {end_time - begin_time}s')
    answer_llama = sequence[0]['generated_text']
    answer_llama = answer_llama.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1]

    return answer_llama
