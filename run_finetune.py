import random
from utils.predefine import MODEL_DIR, FINTUNE_DIR, DATASET_DIR
import argparse
import os
from equilibrium_model.finetune_equilibrium import finetune, merge_model
from equilibrium_model.generate_equilibrium import generate
from utils.feedback import virturalhome_feedback
import pickle
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
# path
parser.add_argument('--res_dir', type=str,
                    default='equilibrium_planner')
parser.add_argument('--data_path', type=str,
                    default='virtualhome_env_equilibrium_planner.p')
parser.add_argument('--base_model', type=str,
                    default='Meta-Llama-3-8B-Instruct')
parser.add_argument('--save_llama_model', type=str,
                    default='llama_3_8b_instruct_equilibrium_planner')
parser.add_argument('--save_bast_llama_model', type=str,
                    default=None)
parser.add_argument('--cache_path', type=str,
                    default='llama_finetuning_cache/equilibrium_planner')

# generate
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_generate_length', type=int, default=500)
# self-refine baseline
parser.add_argument('--self_refine_only', action='store_true')
# ours equilibrium model
parser.add_argument('--convergence', type=int, default=10)
parser.add_argument('--init_top_k', type=int, default=10)
parser.add_argument('--refine_top_k', type=int, default=1)

# feedback
parser.add_argument('--begin_iter', type=int, default=0)
parser.add_argument('--feedback_ratio', type=list, nargs='+', default=[0.3, 0.4, 0.5])
parser.add_argument('--max_depth', type=int, default=3)

# replay buffer
parser.add_argument('--replay_sample_ratio', type=float, default=0.5)
parser.add_argument('--weight_type', type=str, default='uniform')

# finetune
parser.add_argument('--iters', type=int, default=6)
parser.add_argument('--seq_length', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--prompt_loss_weight', type=float, default=0)
parser.add_argument('--reconstruct', type=int, default=None)
parser.add_argument('--save_past_model', action='store_true')

args = parser.parse_args()

args.res_dir = os.path.join(FINTUNE_DIR, args.res_dir)
args.data_path = os.path.join(DATASET_DIR, args.data_path)
args.base_model = os.path.join(MODEL_DIR, args.base_model)
args.save_llama_model = os.path.join(MODEL_DIR, args.save_llama_model)
if args.save_bast_llama_model is not None:
    args.save_bast_llama_model = os.path.join(MODEL_DIR, args.save_bast_llama_model)
args.cache_path = os.path.join(MODEL_DIR, args.cache_path)

random.seed(args.seed)

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

if args.self_refine_only:
    save_lora = os.path.join(args.res_dir, 'lora')
    if args.reconstruct is None:
        print(f"========================finetune========================")
        args.epochs = args.epochs * args.iters
        if not os.path.exists(save_lora):
            os.makedirs(save_lora)
        if len(os.listdir(save_lora)) == 0:
            finetune(
                args=args,
                model_path=args.base_model,
                data_path=args.data_path,
                save_lora_dir=save_lora,
                save_model_dir=args.save_llama_model,
                cache_dir=args.cache_path
            )
    else:
        if os.path.exists(save_lora) and len(os.listdir(save_lora)) > 0:
            merge_model(args.base_model, save_lora, args.save_llama_model)
else:
    if args.reconstruct is None:
        last_model_path = args.base_model
        best_sr = 0
        best_iter = -1

        # main loop
        for train_iter in range(args.iters):
            cur_res_dir = os.path.join(args.res_dir, str(train_iter))
            if not os.path.exists(cur_res_dir):
                os.makedirs(cur_res_dir)

            finetune_data_paths = []
            print(f"========================generate_{train_iter}========================")
            cur_data_path = os.path.join(cur_res_dir, 'finetune_data.p')
            finetune_data_paths.append(cur_data_path)

            if not os.path.exists(cur_data_path):
                generate(
                    args=args,
                    model_path=last_model_path,
                    save_path=cur_data_path,
                    splits=['train'],
                    environment=virturalhome_feedback,
                    generate_iters=1
                )

            print(f"========================feedback_{train_iter}========================")
            if train_iter >= args.begin_iter:
                last_feedback_path = cur_data_path
                for depth in range(args.max_depth):
                    cur_feedback_path = os.path.join(cur_res_dir, f'feedback_data_{depth}.p')
                    finetune_data_paths.append(cur_feedback_path)

                    if not os.path.exists(cur_feedback_path):
                        with open(last_feedback_path, 'rb') as f:
                            cur_data = pickle.load(f)['train']
                        feedback_items = {
                            'train': []
                        }
                        for item in cur_data:
                            if item['feedback'][-1] != 'Task success!':
                                feedback_items['train'].append(item)

                        random.shuffle(feedback_items['train'])
                        select_len = int(len(cur_data) * args.feedback_ratio[depth])
                        feedback_len = len(feedback_items['train'])
                        feedback_items['train'] = feedback_items['train'][:select_len if select_len < feedback_len else feedback_len]

                        generate(
                            args=args,
                            model_path=last_model_path,
                            save_path=cur_feedback_path,
                            raw_data=feedback_items,
                            splits=['train'],
                            environment=virturalhome_feedback,
                            generate_iters=1

                        )
                        last_feedback_path = cur_feedback_path

            if args.replay_sample_ratio is not None:
                print(f"========================replay_buffer_{train_iter}========================")
                cur_replay_path = os.path.join(cur_res_dir, 'replay_data.p')
                finetune_data_paths.append(cur_replay_path)

                if not os.path.exists(cur_replay_path):
                    ratio = args.replay_sample_ratio
                    replay_data = {
                        'train': []
                    }
                    for replay_iter in range(train_iter - 1, -1, -1):
                        replay_res_dir = os.path.join(args.res_dir, str(replay_iter))
                        replay_finetune_data_paths = [os.path.join(replay_res_dir, 'finetune_data.p')]
                        for replay_depth in range(args.max_depth):
                            replay_finetune_data_paths.append(os.path.join(replay_res_dir, f'feedback_data_{replay_depth}.p'))

                        # replay finetune data
                        for replay_finetune_data_path in replay_finetune_data_paths:
                            if not os.path.exists(replay_finetune_data_path):
                                print(f'Can\'t find replay data: {replay_finetune_data_path}')
                            else:
                                with open(replay_finetune_data_path, 'rb') as f:
                                    replay_finetune_data = pickle.load(f)['train']
                                random.shuffle(replay_finetune_data)
                                sample_len = int(len(replay_finetune_data) * ratio)
                                replay_data['train'] += replay_finetune_data[:sample_len]
                                print(f'replay data: {replay_finetune_data_path} with length {sample_len}')
                        ratio = ratio * args.replay_sample_ratio

                    with open(cur_replay_path, 'wb') as f:
                        pickle.dump(replay_data, f)


            print(f"========================finetune_{train_iter}========================")
            cur_save_lora = os.path.join(cur_res_dir, 'lora')
            cur_save_model = f'{args.save_llama_model}_iter{train_iter}'
            if not os.path.exists(cur_save_lora):
                os.makedirs(cur_save_lora)
            if len(os.listdir(cur_save_lora)) == 0:
                finetune(
                    args=args,
                    model_path=last_model_path,
                    data_path=finetune_data_paths,
                    save_lora_dir=cur_save_lora,
                    save_model_dir=cur_save_model,
                    cache_dir=args.cache_path
                )
            if not args.save_past_model:
                if str(last_model_path) != str(args.base_model) and os.path.exists(last_model_path):
                    shutil.rmtree(last_model_path)

            last_model_path = cur_save_model

    else:  # reconstruct model from previous lora
        last_model_path = args.base_model
        for reconstruct_iter in range(args.reconstruct):
            cur_lora_dir = os.path.join(args.res_dir, str(reconstruct_iter), 'lora')
            cur_save_model = f'{args.save_llama_model}_iter{reconstruct_iter}'
            if os.path.exists(cur_lora_dir) and len(os.listdir(cur_lora_dir)) > 0:
                merge_model(last_model_path, cur_lora_dir, cur_save_model)
            else:
                print('Overflow')
            if not args.save_past_model:
                if str(last_model_path) != str(args.base_model) and os.path.exists(last_model_path):
                    shutil.rmtree(last_model_path)
            last_model_path = cur_save_model

