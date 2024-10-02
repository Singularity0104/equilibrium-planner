import json
from evolving_graph.scripts import read_script
from dataset.prompt_template import prompt_template_for_rooms_only_obj_world_model, prompt_template_each_room_full_graph

def generate_world_model_text(raw_data, train=True):
    data = raw_data['meta']
    high_inst = data['high_inst']
    low_inst = data['low_inst']
    init_and_final_graphs = data['init_and_final_graphs']

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
                room_graph[to_id]['nodes'].append(f'{from_id} {obj_name} [{obj_state}]')
                # room_graph[to_id]['nodes'].append(f'{from_id} {obj_name}')

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
    # print(len_all, len_room, len_obj)
    assert len_all == len_room + len_obj

    room_graph_str = []
    for room_id in room_graph.keys():
        room_name = id2name[room_id]
        room_nodes = '\n'.join(room_graph[room_id]['nodes'])
        room_edges = '\n'.join(room_graph[room_id]['edges'])
        room_template = prompt_template_each_room_full_graph.format(
            room_name=room_name,
            room_id=room_id,
            nodes=room_nodes,
            edges=room_edges
        )
        room_graph_str.append(room_template)
    room_graph_str = '\n'.join(room_graph_str)
    if len(id2state[character_ids[0]]) > 0:
        character_state = ' and state {}'.format(id2state[character_ids[0]])
    else:
        character_state = ''

    text_template = prompt_template_for_rooms_only_obj_world_model.format(
        high_inst=high_inst,
        low_inst=low_inst,
        num_rooms=len_room,
        character_id=character_ids[0],
        character_state=character_state,
        character_room=id2name[character_rooms[0]],
        character_room_id=character_rooms[0],
        rooms_graph=room_graph_str,
        program='{program}',
        feedback='{feedback}'
    )

    res = []
    if train:
        programs = raw_data['fix_point']
        feedbacks = raw_data['feedback']
        for program, feedback in zip(programs, feedbacks):
            if feedback != 'Task success!':
                feedback = feedback.split('Feedback:\n')[1]
            res.append({'text': text_template.format(program=program, feedback=feedback)})
    else:
        programs = raw_data['fix_point']
        text_template = text_template.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        for program in programs:
            res.append({'text': text_template.format(program=program)})

    return res