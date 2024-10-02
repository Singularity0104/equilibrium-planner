import json
import os
import random


def data_split(raw_dir, save_dataset_overview, save_dataset_splits):
    executable_programs_dir = os.path.join(raw_dir, 'executable_programs')
    init_and_final_graphs_dir = os.path.join(raw_dir, 'init_and_final_graphs')

    dataset = {}

    for scene in os.listdir(executable_programs_dir):
        for subdir in os.listdir(os.path.join(executable_programs_dir, scene)):
            for program in os.listdir(os.path.join(executable_programs_dir, scene, subdir)):
                program_path = os.path.join(executable_programs_dir, scene, subdir, program)
                init_and_final_graphs_path = os.path.join(init_and_final_graphs_dir, scene, subdir, program.replace('txt', 'json'))
                with open(program_path) as f:
                    high_inst = f.readline().strip()
                    low_inst = f.readline().strip()

                if high_inst not in dataset.keys():
                    task_item = {
                        scene: {
                            'low_inst': low_inst,
                            'program': program_path,
                            'init_and_final_graphs': init_and_final_graphs_path
                        }
                    }
                    dataset[high_inst] = task_item
                else:
                    dataset[high_inst][scene] = {
                        'low_inst': low_inst,
                        'program': program_path,
                        'init_and_final_graphs': init_and_final_graphs_path
                    }

    print('all task:', len(dataset.keys()))
    with open(save_dataset_overview, 'w') as f:
        json.dump(dataset, f)

    train_data = []
    test_new_task = []
    test_new_scene = []
    test_new_task_and_new_scene = []

    test_scene = ['TrimmedTestScene6_graph']
    test_split = 0.4

    all_task_list = list(dataset.keys())
    random.seed(42)
    random.shuffle(all_task_list)

    for task in all_task_list[:int(len(all_task_list) * test_split)]:
        data_item = dataset[task]
        for scene in data_item.keys():
            traj_item = data_item[scene]
            traj_item['high_inst'] = task
            if scene not in test_scene:
                test_new_task.append(traj_item)
            else:
                test_new_task_and_new_scene.append(traj_item)

    for task in all_task_list[int(len(all_task_list) * test_split):]:
        data_item = dataset[task]
        for scene in data_item.keys():
            traj_item = data_item[scene]
            traj_item['high_inst'] = task
            if scene not in test_scene:
                train_data.append(traj_item)
            else:
                test_new_scene.append(traj_item)

    random.shuffle(test_new_task)
    random.shuffle(test_new_scene)
    random.shuffle(test_new_task_and_new_scene)
    datasplits = {
        'train': train_data,
        'test_new_task': test_new_task,
        'test_new_scene': test_new_scene,
        'test_new_task_and_new_scene': test_new_task_and_new_scene
    }

    print('train:', len(train_data))
    print('new_task:', len(test_new_task))
    print('new_scene:', len(test_new_scene))
    print('new_task_and_new_scene:', len(test_new_task_and_new_scene))

    with open(save_dataset_splits, 'w') as f:
        json.dump(datasplits, f)

