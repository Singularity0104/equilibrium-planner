import json
import pickle
from dataset.prompt_template import prompt_template_for_rooms_only_obj_equilibrium_feedback, prompt_template_each_room_only_obj
from evolving_graph.scripts import read_script
import random


def all_generate_withrooms_only_obj_equilibrium(data_splits, save_path):
    random.seed(42)
    finetune_data = {}
    with (open(data_splits, 'r') as f):
        dataset_splits = json.load(f)
        for split in dataset_splits.keys():
            data_list = dataset_splits[split]
            finetune_data[split] = []
            for data in data_list:
                high_inst = data['high_inst']
                low_inst = data['low_inst']
                program_path = data['program']
                init_and_final_graphs = data['init_and_final_graphs']

                script, error_line = read_script(program_path)
                assert error_line is None

                program = []
                for s in script:
                    str_s = '[{}]'.format(s.action.name) + ''.join([' ' + str(par) for par in s.parameters])
                    program.append(str_s)
                program.append('[END]')
                # program = program * 5
                program = '\n'.join(program)

                with open(init_and_final_graphs, 'r') as f2:
                    graphs = json.load(f2)
                init_graph = graphs['init_graph']

                id2name = {}
                id2state = {}
                room_ids = []
                character_ids = []
                character_rooms = []
                room_map = {}
                obj_map = {}
                room_graph = {}
                for node in init_graph['nodes']:
                    node_id = node['id']
                    node_name = node['class_name']
                    id2name[node_id] = node_name
                    id2state[node_id] = ', '.join(node['states'])
                    if node['category'] == 'Rooms':
                        room_ids.append(node_id)
                        room_map[node_id] = []
                        room_graph[node_id] = {
                            'nodes': [],
                            'edges': []
                        }
                    elif node['category'] == 'Characters':
                        character_ids.append(node_id)

                for edge in init_graph['edges']:
                    from_id = edge['from_id']
                    to_id = edge['to_id']
                    relation = edge['relation_type']
                    if relation == 'INSIDE':
                        if from_id in character_ids and to_id in room_ids:
                            character_rooms.append(to_id)
                        if to_id in room_ids:
                            room_map[to_id].append(from_id)
                            obj_map[from_id] = to_id
                            obj_name = id2name[from_id]
                            obj_state = id2state[from_id]
                            # room_graph[to_id]['nodes'].append(f'{from_id} {obj_name} [{obj_state}]')
                            room_graph[to_id]['nodes'].append(f'{from_id} {obj_name}')

                for edge in init_graph['edges']:
                    from_id = edge['from_id']
                    to_id = edge['to_id']
                    relation = edge['relation_type']
                    if relation == 'CLOSE' or relation == 'FACING':
                    # if relation == 'CLOSE':
                        continue
                    if from_id in obj_map.keys() and to_id in obj_map.keys() \
                            and obj_map[from_id] == obj_map[to_id]:
                        room_id = obj_map[from_id]
                        room_graph[room_id]['edges'].append(f'{from_id} {relation} {to_id}')

                assert len(character_ids) == 1 and len(character_rooms) == 1
                len_all = len(id2name.keys())
                len_room = len(room_map.keys())
                len_obj = 0
                for room_id in room_map.keys():
                    len_obj += len(room_map[room_id])
                print(len_all, len_room, len_obj)
                assert len_all == len_room + len_obj

                room_graph_str = []
                for room_id in room_graph.keys():
                    room_name = id2name[room_id]
                    room_nodes = '\n'.join(room_graph[room_id]['nodes'])
                    room_edges = '\n'.join(room_graph[room_id]['edges'])
                    room_template = prompt_template_each_room_only_obj.format(
                        room_name=room_name,
                        room_id=room_id,
                        nodes=room_nodes,
                    )
                    room_graph_str.append(room_template)
                room_graph_str = '\n'.join(room_graph_str)
                if len(id2state[character_ids[0]]) > 0:
                    character_state = ' and state {}'.format(id2state[character_ids[0]])
                else:
                    character_state = ''

                text = prompt_template_for_rooms_only_obj_equilibrium_feedback.format(
                    high_inst=high_inst,
                    low_inst=low_inst,
                    num_rooms=len_room,
                    character_id=character_ids[0],
                    character_state=character_state,
                    character_room=id2name[character_rooms[0]],
                    character_room_id=character_rooms[0],
                    rooms_graph=room_graph_str,
                    feedback='{feedback}',
                    draft_program='{draft_program}',
                    refine_program='{refine_program}'
                )
                finetune_data[split].append({"text_template": text,
                                             'program': program,
                                             'meta': data})

                print(text)
                print(len(text))

    with open(save_path, 'wb') as f:
        pickle.dump(finetune_data, f)
