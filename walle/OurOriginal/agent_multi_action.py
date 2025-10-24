import os
import openai
import json
import time
import copy
import re
import inspect
import textwrap
import argparse

from typing import List, Callable, Tuple, Dict, Optional


# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class LLMAgent_MultiAction:
    """
    OpenAI "gpt-3.5-turbo" を使用するLLMベースのエージェント。
    観測としてJSON形式の状態と過去のアクションを含む辞書を直接受け取り、行動を生成する。
    """
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client 

        if self.client is None:
            print("[LLMAgent] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def generate_action(self, observation_data: Dict, task: str) -> Dict:
        """
        現在の観測データ (state, action, action_result を含む辞書)、
        フィードバック、提案に基づいて行動を生成する。
        行動はJSON形式で「action_type」と「object」フィールドを持ち、
        さらにALFWorld環境用の「command」文字列フィールドも持つことを期待する。
        step_index: 生成された行動を保存する際のファイル名に含めるためのオプションのインデックス。
        """
        if self.client is None:
            print("[LLMAgent] Error: OpenAI client is not available. Cannot generate action.")
        
        # with open("./input/admissible_commands.txt", "r", encoding="utf-8") as f:
        #    admissible_commands = f.read()

        current_obs_json = json.dumps(observation_data, indent=4, ensure_ascii=False)
        task_name = task

        # 英語プロンプト(改良後) ===================================================================
        system_prompt = textwrap.dedent("""
ROLE
You are WALL-E, a high-level planning agent. 
Your task is to generate multiple reasonable next actions (5) that could help achieve the goal,
based on the current observation and past actions.
                                        
DATA INTERPRETATION GUIDE
item_in_hand: { "item_name": null } (the JSON keyword) means your hand is EMPTY.
items_in_locations: This is your memory of visited locations.

ACTION SPACE
Your generated action MUST be one of the following commands. 
The location argument is always keyed by recep.

goto: {"action_name": "goto", "args": {"recep": "..."}}
take: {"action_name": "take", "args": {"obj": "...", "recep": "..."}}
put: {"action_name": "put", "args": {"obj": "...", "recep": "..."}}
open: {"action_name": "open", "args": {"recep": "..."}}
close: {"action_name": "close", "args": {"recep": "..."}}
clean: {"action_name": "clean", "args": {"obj": "...", "recep": "..."}}
heat: {"action_name": "heat", "args": {"obj": "...", "recep": "..."}}
cool: {"action_name": "cool", "args": {"obj": "...", "recep": "..."}}
use: {"action_name": "use", "args": {"tool": "..."}}
  Note: 'use' activates devices like desklamps. The device name includes its instance number (e.g., "desklamp 1").

SPECIAL INSTRUCTION FOR "EXAMINE/LOOK AT X UNDER/WITH LIGHT" GOALS
When the goal involves examining an object "under the desklamp" or similar phrasing:
- This means you need to illuminate the object with a light source to examine it
- Required sequence: [1. find and take the target object, 2. find the desklamp location, 3. go to desklamp location, 4. use the desklamp]
- The 'use' action on a desklamp while holding the object completes the examination task
- You do NOT need a separate "look" action - 'use desklamp' while holding the object accomplishes the goal

CRITICAL LOOP PREVENTION:
- NEVER put down an object you need for the goal unless you have completed the goal
- If you are holding the target object (e.g., "cd 2" for goal "look at cd under desklamp"):
  - DO NOT put it down
  - Your next action should be to find the desklamp (if unknown) or go to the desklamp (if known)
- Once you take an object for the goal, keep it in your hand until the goal is complete

INSTRUCTION:
Analyze all information and generate multiple plausible next actions (5) 
that could help achieve the goal based on the current observation.

Follow this "Goal-Oriented Precondition Analysis" process step-by-step
to reason logically about which actions are possible.

You should not select only one "best" action.
Instead, output a list of several (about 5) reasonable actions
that might help progress toward the goal in different ways.

Each action must follow one of the valid command formats defined above.


CRITICAL DIVERSITY RULE:
You must generate exactly 5 DISTINCT actions. 
- NO two actions in your output can be identical
- NO two actions can have the same action_name AND the same argument values
- Each action must represent a different tactical choice
- If you find yourself wanting to repeat an action, you MUST choose a different alternative instead                                        


Step 1: Deconstruct the Goal into an Ordered Plan.
Based on the main GOAL, mentally create an ordered list of major sub-goals.

Example 1: For "put a cool bowl in cabinet", your mental plan is: 
[1. find and take a bowl, 2. cool the bowl, 3. find a cabinet, 4. put the bowl in the cabinet].

Example 2: For "look at book under the desklamp", your mental plan is:
[1. find and take the book, 2. find the desklamp location, 3. go to desklamp location, 4. use the desklamp].

IMPORTANT: "Find" means you must locate the object/device in items_in_locations by exploring.
- If an object is not in items_in_locations, you haven't found it yet
- You must explore locations until you find all required objects/devices

Step 2: Assess Plan Completion and Identify Current Sub-Goal.
Review your mental plan against the CURRENT OBSERVATION to find the first sub-goal that is NOT yet complete. 
This becomes your current sub-goal.

How to check if a sub-goal is complete:
- "find X": Complete if X appears in items_in_locations. Not complete if items_in_locations doesn't contain X.
- "take X": Complete if item_in_hand.item_name == X.
- "go to Y": Complete if current_position.location_name == Y.
- "use Z": This is typically the final action and completes the entire goal.

Example: For "look at book under the desklamp" mental plan [1. find and take book, 2. find desklamp, 3. go to desklamp, 4. use desklamp]:
- If items_in_locations is empty or doesn't contain "book" → Current sub-goal is 1 (find book by exploring)
- If "book 1" is in items_in_locations but item_in_hand.item_name == null → Current sub-goal is still 1 (take the book)
- If holding "book 1" but "desklamp" not in items_in_locations → Current sub-goal is 2 (find desklamp by exploring)
- If holding "book 1" and know desklamp location but not there → Current sub-goal is 3 (go to desklamp)
- If holding "book 1" and at desklamp location → Current sub-goal is 4 (use desklamp)

Step 3: Analyze Preconditions for the Current Sub-Goal.
List the strict preconditions required to achieve ONLY the current sub-goal.

Example for sub-goal "1. find and take a book":
- Precondition A: The location of 'book' must be known (found in items_in_locations).
- Precondition B: You must be at the location of the 'book'.
- Precondition C: Your hand must be empty.

Example for sub-goal "2. find desklamp":
- Precondition A: The location of 'desklamp' must be known (found in items_in_locations).
  Note: If not known, you need to explore (goto unvisited locations).

Example for sub-goal "4. use desklamp":
- Precondition A: You must be holding the target object (the book).
- Precondition B: You must be at the location where the desklamp is present.
- Precondition C: The desklamp location must be known (in items_in_locations).

Step 4: Find the First Unmet Precondition.
Check your CURRENT OBSERVATION to find the very first precondition from your list that is NOT met.

CRITICAL: How to check if an object's location is known:
- Search through ALL entries in items_in_locations
- The object is "known" ONLY if it appears in the "items" list of any location
- If items_in_locations is empty {}, then NO objects are known
- If the object doesn't appear in any location's items list, its location is UNKNOWN

Example: For sub-goal "1. take book 1" at "bed 1":
- Precondition A: Location known? Search items_in_locations for "book 1" in ANY location's items list
  - If found → Met
  - If NOT found or items_in_locations is empty → NOT Met, must explore
- Precondition B: At location? Check if current_position.location_name == "bed 1" → Met/Not Met
- Precondition C: Hand empty? Check if item_in_hand.item_name == null → Met/Not Met

EXPLORATION RULE:
If an object's location is UNKNOWN (not found in items_in_locations), you MUST explore by:
1. Choose an unexplored location from reachable_locations
2. Use goto to visit that location
3. The environment will update items_in_locations with what you find there

Step 5: Generate Action.
If all preconditions for the current sub-goal are met, your action should execute that sub-goal.
If a precondition is NOT met, your action must resolve that specific unmet precondition first.

Step 6: Ensure Action Diversity.
When generating the list of 5 actions:
- Check that no two actions are identical (same action_name and same args)
- If multiple similar actions are possible (e.g., goto different locations), spread them across your 5 actions
- Consider different strategies: exploration, retrieval, manipulation, etc.
- If you would generate the same action twice, replace one with the next most reasonable alternative                                                                                

EXPLORATION STRATEGY:
When you need to find an object/device whose location is unknown:
1. Check which locations you've already visited (present in items_in_locations)
2. Choose a location from reachable_locations that is NOT yet in items_in_locations
3. Generate: {"action_name": "goto", "args": {"recep": "<unvisited_location>"}}
4. Prioritize likely locations:
   - For desklamps: try "sidetable", "desk", "shelf" first
   - For books/CDs: try "bed", "desk", "shelf", "drawer" first
   - For kitchen items: try "countertop", "cabinet", "fridge" first

OUTPUT ACTIONS (a_t_list):
Return your output as a JSON array (list) of 5 DISTINCT action objects.

MANDATORY DIVERSITY REQUIREMENTS:
1. All 5 actions MUST be completely different from each other
2. No action can have identical action_name AND arguments as another
3. Consider multiple strategies simultaneously:
   - If exploring: visit 5 different unvisited locations
   - If multiple target objects exist: consider different object instances
   - If multiple receptacles are suitable: try different receptacles
4. Before finalizing your list, verify no duplicates exist
5. Each action should represent a meaningfully different approach to the goal

Each element must be a valid action command following the ACTION SPACE format.

Example outputs:
[                                        
    {"action_name": "goto", "args": {"recep": "bed 1"}},
    {"action_name": "take", "args": {"obj": "book 1", "recep": "bed 1"}},
    {"action_name": "use", "args": {"tool": "desklamp 1"}},
    {"action_name": "cool", "args": {"obj": "bowl 1", "recep": "fridge 1"}},
    {"action_name": "open", "args": {"recep": "cabinet 1"}}                                                                                                                            
]                                    

IMPORTANT: Respond ONLY with the JSON array of actions, and nothing else.
Do NOT include any reasoning, explanations, or text outside the JSON.                                                                                
        """)
        
        header = textwrap.dedent(f"""
GOAL
Your final objective is to: "{task_name}"
CURRENT OBSERVATION (o_t)
This JSON object represents your current perception and memory.
        """)

        user_prompt = header + current_obs_json + "\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # プロンプトのログ残す用
        # ========================================================
        log_text = "--- System Prompt ---\n"
        log_text += system_prompt + "\n\n"
        
        for msg in messages[1:]:
            log_text += f"--- {msg['role'].capitalize()} ---\n"
            log_text += msg['content'] + "\n\n"
        # ========================================================

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.5,
            )

            content = response.choices[0].message.content
            print(f"[LLMAgent] Raw response content: {content}")
            action_data = json.loads(content)

            if action_data:
                return action_data, log_text
            else:
                print(f"[LLMAgent] Warning: Missing 'action_type' in LLM response: {action_data}")

        except openai.APITimeoutError:
            print("[LLMAgent] API request timed out.")
          
        except openai.APIConnectionError as e:
            print(f"[LLMAgent] API connection error: {e}")

        except openai.RateLimitError:
            print("[LLMAgent] OpenAI API rate limit exceeded. Waiting 5 seconds...")
            time.sleep(5)
            return self.generate_action(observation_data, task)

        except openai.APIStatusError as e:
            print(f"[LLMAgent] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[LLMAgent] JSON decode error from LLM response: {content}")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    agent = LLMAgent_MultiAction(model="gpt-3.5-turbo")

    with open("./check2/test_obs.json", "r", encoding="utf-8") as f:
        obs = json.load(f)
    
    multi_action = agent.generate_action(obs, "put some knife on sidetable.")

    print("生成したアクション:")
    print(json.dumps(multi_action, indent=4, ensure_ascii=False))
