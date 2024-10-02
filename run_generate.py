from utils.predefine import MODEL_DIR, DATASET_DIR, TEST_DIR
import os
import argparse
import random
from utils.feedback import virturalhome_feedback, world_model_feedback
from equilibrium_model.generate_equilibrium import generate


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='llama_3_8b_instruct_equilibrium_planner')
parser.add_argument('--data_path', type=str,
                    default='virtualhome_env_equilibrium_planner.p')
parser.add_argument('--save_path', type=str,
                    default='generate_equilibrium_planner')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)

# generate
parser.add_argument('--max_generate_length', type=int, default=350)
# self-refine baseline
parser.add_argument('--self_refine_only', action='store_true')
# ours equilibrium model
parser.add_argument('--convergence', type=int, default=20)
parser.add_argument('--init_top_k', type=int, default=10)
parser.add_argument('--refine_top_k', type=int, default=1)

# feedback
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--world_model_feedback', action='store_true')
parser.add_argument('--mixed_feedback', action='store_true')
parser.add_argument('--world_model_path', type=str,
                    default='llama_3_8b_instruct_world_model')
parser.add_argument('--world_model_gpu', type=int, default=0)
parser.add_argument('--world_model_max_generate_length', type=int, default=200)
parser.add_argument('--inner_world_model', type=int, default=1)

args = parser.parse_args()
random.seed(args.seed)

args.model_path = os.path.join(MODEL_DIR, args.model_path)
args.world_model_path = os.path.join(MODEL_DIR, args.world_model_path)
args.data_path = os.path.join(DATASET_DIR, args.data_path)
args.save_path = os.path.join(TEST_DIR, args.save_path)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mixed_feedback:
    args.world_model_feedback = False

if args.world_model_feedback:
    world_model = world_model_feedback(args)
    environment = None
elif args.mixed_feedback:
    world_model = world_model_feedback(args)
    environment = virturalhome_feedback
else:
    world_model = None
    environment = virturalhome_feedback

last_feedback_path = os.path.join(args.save_path, 'generate_data.p')
generate(args, args.model_path, last_feedback_path,
         splits=['test_new_task', 'test_new_scene', 'test_new_task_and_new_scene'],
         environment=environment, generate_iters=args.max_depth,
         world_model=world_model, inner_world_iter=args.inner_world_model)
