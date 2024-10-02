import random
from utils.predefine import MODEL_DIR, FINTUNE_DIR
import argparse
import os
from world_model.finetune_world_model import finetune_world_model, merge_model

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
# path
parser.add_argument('--res_dir', type=str,
                    default='world_model')
parser.add_argument('--data_dir', type=str,
                    default='equilibrium_planner')
parser.add_argument('--base_model', type=str,
                    default='Meta-Llama-3-8B-Instruct')
parser.add_argument('--save_llama_model', type=str,
                    default='llama_3_8b_instruct_world_model')
parser.add_argument('--save_bast_llama_model', type=str,
                    default=None)
parser.add_argument('--cache_path', type=str,
                    default='llama_finetuning_cache/world_model')

# finetune
parser.add_argument('--seq_length', type=int, default=6000)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--prompt_loss_weight', type=float, default=0)
parser.add_argument('--reconstruct', type=int, default=None)
parser.add_argument('--save_past_model', action='store_true')
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()

args.data_dir = os.path.join(FINTUNE_DIR, args.data_dir)
args.res_dir = os.path.join(FINTUNE_DIR, args.res_dir)
args.base_model = os.path.join(MODEL_DIR, args.base_model)
args.save_llama_model = os.path.join(MODEL_DIR, args.save_llama_model)
if args.save_bast_llama_model is not None:
    args.save_bast_llama_model = os.path.join(MODEL_DIR, args.save_bast_llama_model)
args.cache_path = os.path.join(MODEL_DIR, args.cache_path)

random.seed(args.seed)

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

finetune_data = []
for sub_dir in os.listdir(args.data_dir):
    for f in os.listdir(os.path.join(args.data_dir, sub_dir)):
        if 'finetune_data' in f or 'feedback_data' in f:
            finetune_data.append(os.path.join(args.data_dir, sub_dir, f))


save_lora = os.path.join(args.res_dir, 'lora')
if args.reconstruct is None:
    if not os.path.exists(save_lora):
        os.makedirs(save_lora)
    if len(os.listdir(save_lora)) == 0:
        finetune_world_model(
            args=args,
            model_path=args.base_model,
            data_path=finetune_data,
            save_data_path=os.path.join(args.res_dir, 'finetune_data.p'),
            save_lora_dir=save_lora,
            save_model_dir=args.save_llama_model,
            cache_dir=args.cache_path
        )
else:
    if os.path.exists(save_lora) and len(os.listdir(save_lora)) > 0:
        merge_model(args.base_model, save_lora, args.save_llama_model)