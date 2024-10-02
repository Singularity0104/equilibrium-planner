import os
from utils.predefine import DATASPLITS_DIR, DATASET_DIR, RAW_DATA_DIR
import argparse
from dataset.generate_dataset import all_generate_withrooms_only_obj_equilibrium
from dataset.data_split import data_split

parser = argparse.ArgumentParser()
parser.add_argument('--raw_dir', type=str,
                    default='programs_processed_precond_nograb_morepreconds')
parser.add_argument('--datasplit_path', type=str,
                    default='dataset_splits.json')
parser.add_argument('--dataset_overview_path', type=str,
                    default='dataset_overview.json')
parser.add_argument('--dataset_path', type=str,
                    default='virtualhome_env_equilibrium_planner.p')
args = parser.parse_args()

args.raw_dir = os.path.join(RAW_DATA_DIR, args.raw_dir)
args.datasplit_path = os.path.join(DATASPLITS_DIR, args.datasplit_path)
args.dataset_overview_path = os.path.join(DATASPLITS_DIR, args.dataset_overview_path)
args.dataset_path = os.path.join(DATASET_DIR, args.dataset_path)

data_split(args.raw_dir, args.dataset_overview_path, args.datasplit_path)
all_generate_withrooms_only_obj_equilibrium(args.datasplit_path, args.dataset_path)
