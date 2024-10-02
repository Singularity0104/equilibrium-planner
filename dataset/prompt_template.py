prompt_template_each_room_only_obj = """
Room name: {room_name}
Room ID: {room_id}
Object ID and name in this room:
{nodes}"""

prompt_template_each_room_full_graph = """
Room name: {room_name}
Room ID: {room_id}
Object ID, name and states in this room:
{nodes}

Object relations in this room:
{edges}"""

prompt_template_for_rooms_only_obj_equilibrium = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You need to act as a task planner, who first draft an initial sub-task sequence and then refine it in the next few iterations.
When the the draft sub-task sequence is Null, you should output the initial sub-task sequence.
When the the draft sub-task sequence is not Null, You should refine it based on the the draft sub-task sequence.
Each sub-task can be one of the following form: 1. [action_name]; 2. [action_name] <object name 1> (object id 1); 3. [action_name] <object name 1> (object id 1) <object name 2> (object id 2).
The (object id) is used to tell the simulator which object the action should act on.
The number of arguments depends on the action type.
For action type 1, the available actions are: SLEEP, STANDUP, WAKEUP
For action type 2, the available actions are: WALK, FIND, GRAB, WASH, WIPE, PULL, PUSH, POUR, TURNTO, POINTAT, WATCH, TOUCH, OPEN, CLOSE, RUN, SIT, READ, PUTON, PUTOFF, DROP, LIE, SWITCHON, SWITCHOFF, DRINK, LOOKAT, TYPE, CUT, PUTOBJBACK, EAT, RINSE, PLUGIN, PLUGOUT, GREET, SCRUB, SQUEEZE
For action type 3, the available actions are: PUTIN, PUTBACK
All action_name of the sub-tasks must be chosen from the above actions.
You should output the sub-task sequence in succinct form.
You must output END after you have output the entire sub-task sequence.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Task name:
{high_inst}

Instructions:
{low_inst}

There are {num_rooms} rooms, and you are an embodied character with ID {character_id}{character_state} in {character_room} with ID {character_room_id}.
The objects in each room is as follows:
{rooms_graph}

The draft sub-task sequence:
{draft_program}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{refine_program}
<|eot_id|><|end_of_text|>"""



prompt_template_for_rooms_only_obj_equilibrium_feedback = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You need to act as a task planner, who first draft an initial sub-task sequence and then refine it in the next few iterations.
When the the draft sub-task sequence is Null, you should output the initial sub-task sequence.
When the the draft sub-task sequence is not Null, You should refine it based on the the draft sub-task sequence.
If you have previously generated some action sequences and tried to execute them in the environment, their feedback will be provided to you for reference.
Each sub-task can be one of the following form: 1. [action_name]; 2. [action_name] <object name 1> (object id 1); 3. [action_name] <object name 1> (object id 1) <object name 2> (object id 2).
The (object id) is used to tell the simulator which object the action should act on.
The number of arguments depends on the action type.
For action type 1, the available actions are: SLEEP, STANDUP, WAKEUP
For action type 2, the available actions are: WALK, FIND, GRAB, WASH, WIPE, PULL, PUSH, POUR, TURNTO, POINTAT, WATCH, TOUCH, OPEN, CLOSE, RUN, SIT, READ, PUTON, PUTOFF, DROP, LIE, SWITCHON, SWITCHOFF, DRINK, LOOKAT, TYPE, CUT, PUTOBJBACK, EAT, RINSE, PLUGIN, PLUGOUT, GREET, SCRUB, SQUEEZE
For action type 3, the available actions are: PUTIN, PUTBACK
All action_name of the sub-tasks must be chosen from the above actions.
You should output the sub-task sequence in succinct form.
You must output END after you have output the entire sub-task sequence.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Task name:
{high_inst}

Instructions:
{low_inst}

There are {num_rooms} rooms, and you are an embodied character with ID {character_id}{character_state} in {character_room} with ID {character_room_id}.
The objects in each room is as follows:
{rooms_graph}

Feedbacks from past executions:
{feedback}

The draft sub-task sequence:
{draft_program}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{refine_program}
<|eot_id|><|end_of_text|>"""

prompt_template_for_rooms_only_obj_world_model = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You need to act as a world, who evaluate the given sub-task sequence in the given environment and output corresponding feedback.
Each sub-task can be one of the following form: 1. [action_name]; 2. [action_name] <object name 1> (object id 1); 3. [action_name] <object name 1> (object id 1) <object name 2> (object id 2).
The (object id) is used to tell the simulator which object the action should act on.
The number of arguments depends on the action type.
For action type 1, the available actions are: SLEEP, STANDUP, WAKEUP
For action type 2, the available actions are: WALK, FIND, GRAB, WASH, WIPE, PULL, PUSH, POUR, TURNTO, POINTAT, WATCH, TOUCH, OPEN, CLOSE, RUN, SIT, READ, PUTON, PUTOFF, DROP, LIE, SWITCHON, SWITCHOFF, DRINK, LOOKAT, TYPE, CUT, PUTOBJBACK, EAT, RINSE, PLUGIN, PLUGOUT, GREET, SCRUB, SQUEEZE
For action type 3, the available actions are: PUTIN, PUTBACK
All action_name of the sub-tasks must be chosen from the above actions.
You can only output the following types of feedback:
"Task success!", which means the given sub-task sequence can successfully complete the given task.
"Your output does not conform to the required format.", which means the given sub-task sequence does not conform to the required format.
"Your output has an invalid command: the invalid command...", which means the given sub-task sequence has an illegal command line.
"Your output is executed incorrectly in the environment.", which means the given sub-task sequence can not be executed in the environment.
"You have not completed this task.\nThe following objects and corresponding states do not meet the goals: (object id, object name, object state), ...\nThe following objects have wrong relative position: (object id1, object name1) and (object id2, object name2), ...\n", which means the given sub-task sequence can not complete the task and gives the reason for task failure.
You must output END after you have output the entire feedback.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Task name:
{high_inst}

Instructions:
{low_inst}

There are {num_rooms} rooms, and you are an embodied character with ID {character_id}{character_state} in {character_room} with ID {character_room_id}.
The scene graph of each room is as follows:
{rooms_graph}

The sub-task sequence:
{program}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{feedback}
END
<|eot_id|><|end_of_text|>"""
