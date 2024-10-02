import json
import pickle

import evolving_graph.utils as utils
from evolving_graph.scripts import read_script_from_list_string
from evolving_graph.execution import ScriptExecutor
from evolving_graph.environment import EnvironmentGraph, EnvironmentState


def del_graph(a, b):
    # edges
    node_id_to_class_name = {n['id']:n['class_name'] for n in a['nodes']}
    b_edges = [(edge['from_id'], edge['to_id'], edge['relation_type'], node_id_to_class_name[edge['from_id']], node_id_to_class_name[edge['to_id']]) for edge in b['edges']]
    a_edges = [(edge['from_id'], edge['to_id'], edge['relation_type'], node_id_to_class_name[edge['from_id']], node_id_to_class_name[edge['to_id']]) for edge in a['edges']]
    b_edges, a_edges = set(b_edges), set(a_edges)
    res = list(a_edges - b_edges)
    res = list(filter(lambda x:x[2]!='CLOSE', res))
    def node_set(nodes):
        res = []
        for node in nodes:
            for state in node['states']:
                res.append((node['id'], node['class_name'], state))
        return set(res)
    a_nodes = node_set(a['nodes'])
    b_nodes = node_set(b['nodes'])
    return res, list(a_nodes-b_nodes)


def calc_gcr(init_graph_dict, final_graph_dict, goal_graph_dict):
    edge_list, node_list = del_graph(final_graph_dict, init_graph_dict)
    goal_edge_list, goal_node_list = del_graph(goal_graph_dict, init_graph_dict)
    s = len(goal_edge_list) + len(goal_node_list)
    if s == 0:
        return 1, [], []
    edge_inter = list(set(edge_list) & set(goal_edge_list))
    node_inter = list(set(node_list) & set(goal_node_list))
    edge_diff = list(set(goal_edge_list) - set(edge_list))
    node_diff = list(set(goal_node_list) - set(node_list))
    return (len(edge_inter) + len(node_inter)) / s, edge_diff, node_diff


def split_action(generate, format_warning=0.3):
    generate = generate.split('\n')
    action_list = []
    is_begin = False
    warning = False

    format_error_cnt = 0
    for action in generate:
        if len(action) > 0 and not action.startswith('['):
            format_error_cnt += 1
    if format_error_cnt / len(generate) > format_warning:
        warning = True

    for action in generate:
        if action.startswith('['):
            if not is_begin:
                is_begin = True
            if 'END' in action:
                break
            action_list.append(action)
        else:
            if is_begin:
                break
    return action_list, warning


def exec(init_graph, goal_graph, script):
    graph = EnvironmentGraph(init_graph)
    name_equivalence = utils.load_name_equivalence()
    executor = ScriptExecutor(graph, name_equivalence)

    sr = 0
    success, state, graph_state_list = executor.execute(script)
    gcr, edge_diff, node_diff = calc_gcr(init_graph, graph_state_list[-1], goal_graph)
    if abs(gcr - 1) < 1e-10:
        sr = 1

    return sr, gcr, graph_state_list, edge_diff, node_diff


def exec_string(init_graph, goal_graph, item):
    action_format, format_warning = split_action(item)
    script, error_line = read_script_from_list_string(action_format)

    exec_error = None
    try:
        sr, gcr, _, edge_diff, node_diff = exec(init_graph, goal_graph, script)
    except Exception as e:
        sr, gcr = 0, 0
        edge_diff, node_diff = [], []
        exec_error = e
        print(e)

    return sr, gcr, edge_diff, node_diff, format_warning, error_line, exec_error


def exec_item(meta, action_list):
    graph_path = meta['init_and_final_graphs']
    with open(graph_path) as f:
        graph = json.load(f)
    init = graph['init_graph']
    goal = graph['final_graph']

    sr, gcr, edge_diff, node_diff, format_warning, error_line, exec_error = exec_string(init, goal, action_list)

    return (sr, gcr,
            edge_diff, node_diff,
            format_warning,
            error_line,
            exec_error)


def exec_set(data_path, save_path=None, fix_point_iter=-1, detailed_res=None):
    with open(data_path, 'rb') as f:
        test_datas = pickle.load(f)

    if detailed_res is not None:
        detailed_res_list = []

    res = {}
    total_sr_0 = 0
    total_gcr_0 = 0
    total_sr_fix = 0
    total_gcr_fix = 0
    total_fix_iters = 0
    total_cnt = 0
    total_error_cnt_0 = 0
    total_error_cnt_fix = 0
    for split in test_datas:
        test_data = test_datas[split]

        all_sr_0 = 0
        all_gcr_0 = 0
        all_sr_fix = 0
        all_gcr_fix = 0
        fix_iters = 0
        error_cnt_0 = 0
        error_cnt_fix = 0
        cnt = 0

        for item in test_data:

            if item['mid_res'][0] is None:
                print('self refine test')
                action_list_0 = item['fix_point'][0]
            else:
                print('equilibrium test')
                action_list_0 = item['mid_res'][0][0]
                if isinstance(action_list_0, list):
                    action_list_0 = action_list_0[0]
            sr_0, gcr_0, edge_diff_0, node_diff_0, format_warning_0, error_line_0, exec_error_0 = exec_item(item['meta'], action_list_0)

            fix_idx = fix_point_iter
            if fix_point_iter < 0 or fix_point_iter >= len(item['fix_point']):
                fix_idx = len(item['fix_point']) - 1
            fix_iters += fix_idx

            action_list_fix = item['fix_point'][fix_idx]
            sr_fix, gcr_fix, edge_diff_fix, node_diff_fix, format_warning_fix, error_line_fix, exec_error_fix = exec_item(item['meta'], action_list_fix)

            if sr_0 == 1:
                print('g0 success')
            else:
                print('g0 fail')
            if sr_fix == 1:
                print('fix success')
            else:
                print('fix fail')

            all_sr_0 += sr_0
            all_gcr_0 += gcr_0
            all_sr_fix += sr_fix
            all_gcr_fix += gcr_fix

            if exec_error_0 is not None or error_line_0 is not None:
                error_cnt_0 += 1
            if exec_error_fix is not None or error_line_fix is not None:
                error_cnt_fix += 1

            cnt += 1

            if detailed_res is not None:
                detailed_res_list.append({
                    'sr': sr_fix,
                    'gcr': gcr_fix,
                    'item': item
                })

        total_sr_0 += all_sr_0
        total_gcr_0 += all_gcr_0
        total_sr_fix += all_sr_fix
        total_gcr_fix += all_gcr_fix
        total_fix_iters += fix_iters
        total_cnt += cnt
        total_error_cnt_0 += error_cnt_0
        total_error_cnt_fix += error_cnt_fix

        all_sr_0 = 0 if cnt == 0 else all_sr_0 / cnt
        all_gcr_0 = 0 if cnt == 0 else all_gcr_0 / cnt
        all_sr_fix = 0 if cnt == 0 else all_sr_fix / cnt
        all_gcr_fix = 0 if cnt == 0 else all_gcr_fix / cnt
        fix_iters = 0 if cnt == 0 else fix_iters / cnt
        exec_0 = 0 if cnt == 0 else (cnt - error_cnt_0) / cnt
        exec_fix = 0 if cnt == 0 else (cnt - error_cnt_fix) / cnt

        res[split] = {
            'sr_0': all_sr_0,
            'gcr_0': all_gcr_0,
            'sr_fix': all_sr_fix,
            'gcr_fix': all_gcr_fix,
            'fix_iters': fix_iters,
            'total': cnt,
            'error_0': error_cnt_0,
            'error_fix': error_cnt_fix,
            'exec_0': exec_0,
            'exec_fix': exec_fix,
        }

    res['all_splits'] = {
        'sr_0': 0 if total_cnt == 0 else total_sr_0 / total_cnt,
        'gcr_0': 0 if total_cnt == 0 else total_gcr_0 / total_cnt,
        'sr_fix': 0 if total_cnt == 0 else total_sr_fix / total_cnt,
        'gcr_fix': 0 if total_cnt == 0 else total_gcr_fix / total_cnt,
        'fix_iters': 0 if total_cnt == 0 else total_fix_iters / total_cnt,
        'error_0': total_error_cnt_0,
        'error_fix': total_error_cnt_fix,
        'exec_0': 0 if total_cnt == 0 else (total_cnt - total_error_cnt_0) / total_cnt,
        'exec_fix': 0 if total_cnt == 0 else (total_cnt - total_error_cnt_fix) / total_cnt,
        'total': total_cnt,
    }

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(res, f)

    if detailed_res is not None:
        with open(detailed_res, 'w') as f:
            json.dump(detailed_res_list, f)

    print('all results')
    for k in res.keys():
        for i in res[k].keys():
            print(f'{k} {i}:', str(res[k][i]))
        print('\n')

    return res
