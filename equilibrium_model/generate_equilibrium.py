import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import transformers
import torch
import pickle
from tqdm import tqdm
import copy
import time


def generate_one(args, prompt, pipeline, tokenizer, top_k=None,
                       num_return_sequences=1):
    begin_time = time.time()
    prompt_token = tokenizer(prompt, return_tensors="pt").input_ids
    max_length = args.max_generate_length + prompt_token.shape[1]
    sequence = pipeline(
        prompt,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=[tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                      tokenizer.convert_tokens_to_ids('END')],
        max_length=max_length
    )
    end_time = time.time()
    print(f'generate one time: {end_time - begin_time}s')
    answer_llama = sequence[0]['generated_text']
    answer_llama = answer_llama.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1]

    return answer_llama


def generate_prefix_cache(args, prefix_prompt, model, tokenizer):
    begin_time = time.time()
    prefix_id = tokenizer(prefix_prompt, return_tensors="pt").to(f'cuda:{args.gpu}').input_ids
    prefix_outputs = model(prefix_id)
    prefix_kv = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
    end_time = time.time()
    print(f'cache time: {end_time - begin_time}s')
    return prefix_id, prefix_kv


def generate_use_cache(args, prefix_kv, prefix_id, postfix_prompt, pipeline, tokenizer, top_k=None,
                       num_return_sequences=1, **kwargs):
    begin_time = time.time()

    postfix_prompt_token = tokenizer(postfix_prompt, return_tensors="pt").input_ids
    max_length = args.max_generate_length + postfix_prompt_token.shape[1]

    sequence = pipeline(
        postfix_prompt,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=[tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                      tokenizer.convert_tokens_to_ids('END')],
        max_length=max_length,
        prefix_kv_cache=prefix_kv,
        prefix_id_cache=prefix_id,
        **kwargs
    )
    end_time = time.time()
    print(f'generate one time: {end_time - begin_time}s')
    answer_llama = sequence[0]['generated_text']
    answer_llama = answer_llama.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1]

    return answer_llama


def generate_self_refine(args, prompt_template, pipeline, model, tokenizer, beginner=None):
    init_prompt = prompt_template.format(draft_program='Null' if beginner is None else beginner)
    init_action_lists = generate_one(args,
                                     init_prompt, pipeline, tokenizer,
                                     top_k=args.init_top_k if beginner is None else args.refine_top_k,
                                     num_return_sequences=1)
    if init_action_lists.endswith('[END'):
        init_action_lists = init_action_lists + ']'
    return init_action_lists


def generate_equilibrium_cached(args, prompt_template, pipeline, model, tokenizer, mid_res=False, beginner=None):
    prefix_idx = prompt_template.find('The draft sub-task sequence:') + len('The draft sub-task sequence:')
    prefix_prompt = prompt_template[:prefix_idx]
    postfix_prompt_template = prompt_template[prefix_idx:]

    prefix_id, prefix_kv = generate_prefix_cache(args, prefix_prompt, model, tokenizer)

    init_prompt = postfix_prompt_template.format(draft_program='Null' if beginner is None else beginner)

    init_action_list = generate_use_cache(args,
                                          copy_cache(prefix_kv),
                                          copy.deepcopy(prefix_id),
                                          init_prompt, pipeline, tokenizer,
                                          top_k=args.init_top_k, num_return_sequences=1)
    if init_action_list.endswith('[END'):
        init_action_list = init_action_list + ']'

    last_list = init_action_list
    cnt = 0
    mid = []
    mid.append(init_action_list)
    while True:
        postfix_prompt = postfix_prompt_template.format(draft_program=last_list)
        try:
            action_list = generate_use_cache(args,
                                             copy_cache(prefix_kv),
                                             copy.deepcopy(prefix_id),
                                             postfix_prompt, pipeline, tokenizer,
                                             top_k=args.refine_top_k, num_return_sequences=1)
        except Exception as e:
            print(e)
            action_list = last_list[: len(last_list) / 2]

        if action_list.endswith('[END'):
            action_list = action_list + ']'
        mid.append(action_list)
        cnt += 1
        if action_list == last_list:
            break
        elif cnt >= args.convergence:
            break
        else:
            last_list = action_list

    if mid_res:
        return last_list, mid
    else:
        return last_list


def copy_cache(cache):
    legacy_cache = ()
    for layer_idx in range(len(cache)):
        legacy_cache += ((cache.key_cache[layer_idx].detach().clone(), cache.value_cache[layer_idx].detach().clone()),)
    return DynamicCache.from_legacy_cache(legacy_cache)


def compare(action_list_1, action_list_2):
    if len(action_list_1) != len(action_list_2):
        return False
    else:
        for i in range(len(action_list_1)):
            if action_list_1[i] != action_list_2[i]:
                return False
    return True


def split_action(generate):
    generate = generate.split('\n')
    action_list = []
    is_begin = False
    for action in generate:
        if action.startswith('['):
            if not is_begin:
                is_begin = True
            if 'SEP' in action:
                break
            action_list.append(action)
        else:
            if is_begin:
                break
    return action_list


def generate(args, model_path, save_path,
             splits=['test_new_task', 'test_new_scene', 'test_new_task_and_new_scene'],
             raw_data=None, environment=None, world_model=None, generate_iters=None, inner_world_iter=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": args.gpu},
        torch_dtype=torch.float16,
        trust_remote_code=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    if os.path.exists(save_path):
        return

    if raw_data is None:
        with open(args.data_path, 'rb') as f:
            raw_data = pickle.load(f)

    begin_time = time.time()

    if os.path.exists(save_path + '.cache'):
        print('using cache...')
        with open(save_path + '.cache', 'rb') as f:
            cache = pickle.load(f)
        cache_idxs = cache['idx']
        generate_data = cache['data']
        print(cache_idxs)
    else:
        cache_idxs = {}
        generate_data = {}

    for split in splits:
        test_data = raw_data[split]
        if split in cache_idxs.keys():
            cache_idx = cache_idxs[split]
        else:
            cache_idx = -1
            generate_data[split] = []
        for i in tqdm(range(len(test_data))):
            if i <= cache_idx:
                print('continue')
                continue
            data_dict = test_data[i]

            text_template = data_dict['text_template']
            program = data_dict['program']
            meta = data_dict['meta']
            feedback = data_dict['feedback'] if 'feedback' in data_dict else []
            fix_point = data_dict['fix_point'] if 'fix_point' in data_dict else []
            mid_res = data_dict['mid_res'] if 'mid_res' in data_dict else []
            print(meta)
            tmp_feedback = []
            tmp_mid_res = []

            feedback_iter = 0
            cur_inner_world_iter = 0
            last_fix_point = None
            item_begin_time = time.time()
            while True:
                if generate_iters is not None and generate_iters > 0 and feedback_iter >= generate_iters:
                    print('exceeded the maximum number of iters')
                    break
                if '{feedback}' in text_template:
                    tmp_feedback_concat = feedback + tmp_feedback
                    if len(tmp_feedback_concat) == 0:
                        feedback_str = 'Null'
                    else:
                        feedback_str = '\n'.join(tmp_feedback_concat[-args.max_depth:])
                    print('###feedback_str###')
                    print(feedback_str)
                    text = text_template.format(
                        feedback=feedback_str,
                        draft_program='{draft_program}',
                        refine_program='{refine_program}'
                    )
                else:
                    text = text_template

                text_split = text.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
                prompt_template = text_split[0] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

                if args.self_refine_only:  # self-refine baseline
                    print('self-refine generate')
                    beginner = last_fix_point
                    cur_fix_point = generate_self_refine(
                        args, prompt_template, pipeline, model, tokenizer, beginner=beginner)
                    cur_mids = None
                else:  # ours equilibrium model
                    print('equilibrium generate')
                    beginner = last_fix_point
                    print('reuse equilibrium fix-point {}...'.format(beginner[:40] if beginner is not None else None))
                    cur_fix_point, cur_mids = generate_equilibrium_cached(
                        args, prompt_template, pipeline, model, tokenizer, mid_res=True, beginner=beginner)

                last_fix_point = cur_fix_point

                if environment is None and world_model is None:
                    fix_point.append(cur_fix_point)
                    mid_res.append(cur_mids)
                    feedback.append(None)
                    print('no need to get feedback and recursively generate')
                    break
                elif environment is not None and world_model is None:
                    success, cur_feedback = environment(meta, cur_fix_point)
                    fix_point.append(cur_fix_point)
                    mid_res.append(cur_mids)
                    feedback.append(cur_feedback)
                    if success == 1:
                        print('get the task success feedback, no need to continue')
                        break
                    if len(fix_point) >= 2 and fix_point[-1] == fix_point[-2]:
                        print('get the out-loop fix-point, no need to continue')
                        break
                    feedback_iter += 1

                elif environment is None and world_model is not None:
                    success, cur_feedback = world_model(meta, cur_fix_point)
                    fix_point.append(cur_fix_point)
                    mid_res.append(cur_mids)
                    feedback.append(cur_feedback)
                    if success == 1:
                        print('get the task success feedback, no need to continue')
                        break
                    if len(fix_point) >= 2 and fix_point[-1] == fix_point[-2]:
                        print('get the out-loop fix-point, no need to continue')
                        break
                    feedback_iter += 1

                elif environment is not None and world_model is not None:
                    if cur_inner_world_iter < inner_world_iter:
                        print('mixed feedback: world model')
                        cur_inner_world_iter += 1

                        success, cur_feedback = world_model(meta, cur_fix_point)

                        tmp_feedback.append(cur_feedback)
                        tmp_mid_res.append(cur_mids)
                    else:
                        print('mixed feedback: environment')
                        cur_inner_world_iter = 0
                        tmp_feedback = []

                        success, cur_feedback = environment(meta, cur_fix_point)

                        fix_point.append(cur_fix_point)
                        tmp_mid_res.append(cur_mids)
                        mid_res.append(tmp_mid_res)
                        tmp_mid_res = []
                        feedback.append(cur_feedback)

                        if success == 1:
                            print('get the task success feedback, no need to continue')
                            break
                        if len(fix_point) >= 2 and fix_point[-1] == fix_point[-2]:
                            print('get the out-loop fix-point, no need to continue')
                            break
                        feedback_iter += 1
            item_end_time = time.time()
            item_time = item_end_time - item_begin_time
            generate_data[split].append({
                'mid_res': mid_res,
                'fix_point': fix_point,
                'feedback': feedback,
                'text_template': text_template,
                'program': program,
                'meta': meta,
                'time': item_time
            })
            cache_idxs[split] = i

            with open(save_path + '.cache.tmp', 'wb') as f:
                cache = {
                    'data': generate_data,
                    'idx': cache_idxs,
                }
                pickle.dump(cache, f)
            shutil.move(save_path + '.cache.tmp', save_path + '.cache')

    with open(save_path, 'wb') as f:
        pickle.dump(generate_data, f)
    os.remove(save_path + '.cache')

    end_time = time.time()
    print(f'finish time: {end_time - begin_time}s')
