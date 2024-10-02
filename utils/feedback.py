import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evolving_graph.test import exec_item
from world_model.generate_world_model import generate_world_model
from world_model.world_model_data import generate_world_model_text


def feedback_str(action_list, sr, edge_diff, node_diff, format_warning, error_line, exec_error):
    feedback_string = f'Action sequence:\n{action_list}\nFeedback:\n'
    if format_warning:
        feedback_string += 'Your output does not conform to the required format.\n'
    if error_line is not None:
        feedback_string += f'Your output has an invalid command: {error_line}.\n'
    if exec_error is not None:
        feedback_string += f'Your output is executed incorrectly in the environment.\n'
    if sr < 1:
        feedback_string += 'You have not completed this task.\n'
        objs = []
        for node in node_diff:
            node_id = node[0]
            node_name = node[1]
            node_state = node[2]
            objs.append(f'({node_id}, {node_name}, {node_state})')
        if len(objs) > 0:
            objs = ', '.join(objs)
            feedback_string += f'The following objects and corresponding states do not meet the goals: {objs}.\n'

        edges = []
        for edge in edge_diff:
            node_id0 = edge[0]
            node_id1 = edge[1]
            node_name0 = edge[3]
            node_name1 = edge[4]
            edges.append(f'({node_id0}, {node_name0}) and ({node_id1}, {node_name1})')
        if len(edges) > 0:
            edges = ', '.join(edges)
            feedback_string += f'The following objects have wrong relative position: {edges}.\n'

    return feedback_string


def virturalhome_feedback(meta, action_list):
    success = False
    sr, gcr, edge_diff, node_diff, format_warning, error_line, exec_error \
        = exec_item(meta, action_list)
    if sr < 1 or format_warning or error_line is not None or exec_error is not None:
        item_feedback = feedback_str(action_list, sr, edge_diff, node_diff,
                                     format_warning, error_line, exec_error)
    else:
        success = True
        item_feedback = 'Task success!'
    return success, item_feedback


class world_model_feedback:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.world_model_path,
            trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.world_model_path,
            device_map={"": args.world_model_gpu},
            torch_dtype=torch.float16,
            trust_remote_code=True)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def __call__(self, meta, action_list):
        item = {
            'meta': meta,
            'fix_point': [action_list]
        }

        prompt = generate_world_model_text(item, train=False)[0]['text']

        # print(prompt)

        feedback = generate_world_model(
            args=self.args,
            prompt=prompt,
            pipeline=self.pipeline,
            tokenizer=self.tokenizer
        )

        feedback = feedback.split('\n')
        beginner = ['Your output',
                    'You have not completed this task',
                    'The following objects',
                    'Task success']
        filter_feedback = []
        filter_success = []
        for f in feedback:
            valid = False
            for b in beginner:
                if b in f:
                    valid = True
                    break
            if valid and f not in filter_feedback:
                if 'Task success' in f:
                    filter_success.append(f)
                else:
                    filter_feedback.append(f)

        feedback = '\n'.join(filter_feedback)

        print('world model filter feedback:')
        print(feedback)

        if len(filter_feedback) == 0 and len(filter_success) > 0:
            success = True
            item_feedback = 'Task success!'
        else:
            success = False
            item_feedback = f'Action sequence:\n{action_list}\nFeedback:\n' + feedback

        return success, item_feedback
