import argparse
import os
from evolving_graph.test import exec_set
from utils.predefine import TEST_DIR


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,
                    default='generate_equilibrium_planner')
parser.add_argument('--save_name', type=str, default='res.json')
parser.add_argument('--detailed_res', action='store_true')
parser.add_argument('--iter', type=int, default=-1)

args = parser.parse_args()

args.generate_path = os.path.join(TEST_DIR, args.test_path, 'generate_data.p')
save_path = os.path.join(TEST_DIR, args.test_path, args.save_name)
if args.detailed_res:
    detailed_res_path = os.path.join(TEST_DIR, args.test_path, 'detailed_res.json')
else:
    detailed_res_path = None

res = exec_set(args.generate_path, save_path, fix_point_iter=args.iter, detailed_res=detailed_res_path)
