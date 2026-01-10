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

from .new_scene_graph import *

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class LLMAgent:
    """
    OpenAI "gpt-3.5-turbo" を使用するLLMベースのエージェント。
    観測としてJSON形式の状態と過去のアクションを含む辞書を直接受け取り、行動を生成する。
    """
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client 

        if self.client is None:
            print("[LLMAgent] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def generate_action(self, observation_data: Dict, feedback: str, suggestion: str, step: int, task: str) -> Dict:
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

        print("=== Feedback ===")
        print(feedback)
        print("=== Suggestion ===")
        print(suggestion)
            
        # 英語プロンプト(改良後) ===================================================================
        system_prompt = textwrap.dedent("""
ROLE
You are WALL-E, a high-level planning agent. Your task is to generate the single, optimal next action to achieve a goal by reasoning backwards from the goal's preconditions.

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

INSTRUCTION
Analyze all information and generate the single most logical next action.
Follow this "Goal-Oriented Precondition Analysis" process step-by-step:

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

EXPLORATION STRATEGY:
When you need to find an object/device whose location is unknown:
1. Check which locations you've already visited (present in items_in_locations)
2. Choose a location from reachable_locations that is NOT yet in items_in_locations
3. Generate: {"action_name": "goto", "args": {"recep": "<unvisited_location>"}}
4. Prioritize likely locations:
   - For desklamps: try "sidetable", "desk", "shelf" first
   - For books/CDs: try "bed", "desk", "shelf", "drawer" first
   - For kitchen items: try "countertop", "cabinet", "fridge" first

OUTPUT ACTION (a_t)
Your output must be a single JSON object based on your analysis.

Example outputs:
{"action_name": "goto", "args": {"recep": "bed 1"}}
{"action_name": "take", "args": {"obj": "book 1", "recep": "bed 1"}}
{"action_name": "use", "args": {"tool": "desklamp 1"}}
        """)
        
        header = textwrap.dedent(f"""
GOAL
Your final objective is to: "{task_name}"
CURRENT OBSERVATION (o_t)
This JSON object represents your current perception and memory.
        """)

        footer = textwrap.dedent(f"""
FEEDBACK FROM WORLD MODELFeedback on your MOST RECENT proposed action.
Prioritize this to correct your immediate next step.
feedback: "{feedback}"
suggestion: "{suggestion}"
        """)

        user_prompt = header + current_obs_json + "\n" + footer

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
                response_format={"type": "json_object"},
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
            return self.generate_action(observation_data, feedback, suggestion, step, task)

        except openai.APIStatusError as e:
            print(f"[LLMAgent] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[LLMAgent] JSON decode error from LLM response: {content}")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        


class LLMWorldModel:
    """
    OpenAI を使用する"gpt-3.5-turbo"LLMベースのWorld Model。
    現在の観測と提案された行動から、その行動が「成功」するか「失敗」するかを予測する。
    """
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = client

        if self.client is None:
            print("[LLMWorldModel] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def predict_transition_outcome(self, current_observation_state, proposed_action):
        """
        現在の観測状態と提案された行動に基づいて、その行動が成功するか失敗するかを予測し、
        失敗の場合はその理由と提案も返す。
        戻り値: dict 形式
        {
        "success": bool,
        "feedback": str,
        "suggestion": str
        }
        """
        if self.client is None:
            print("[LLMWorldModel] Error: OpenAI client is not available. Cannot predict transition outcome.")
            return {"success": False, "feedback": "OpenAI client not initialized.", "suggestion": "Initialize the client before calling this method."}

        # with open("./input/admissible_commands.txt", "r", encoding="utf-8") as f:
        #     admissible_commands = f.read()

        current_obs_json = json.dumps(current_observation_state, indent=4, ensure_ascii=False)
        proposed_action_json = json.dumps(proposed_action, indent=4, ensure_ascii=False)

        # =================================================================================

        # 英語プロンプト(改良後) =================================================================
        system_prompt = textwrap.dedent("""
ROLE
You are an All-in-One World Model for WALL-E. Your purpose is to act as a strict, procedural referee. You must predict if a proposed action will succeed or fail by meticulously following a step-by-step evaluation process. Your primary function is to enforce deterministic rules.

CRITICAL DATA INTERPRETATION GUIDE
The JSON keyword null (without quotes) for item_in_hand.item_name means the agent's hand is EMPTY. This is a critical distinction you must not misinterpret.

SPECIAL NOTE ON RECEPTACLE STATUS
The "status" inside current_position refers only to the agent's spatial state (e.g., whether the agent is near or inside an open area).
It should NEVER be used to determine if a receptacle (like a microwave or fridge) is open or closed.
Always check items_in_locations[recep].status for open/close evaluations.

SPECIAL NOTE ON HEAT/COOL ACTIONS
Do not require the object to already be placed inside the device.
If the agent is holding the object and is physically located at the heating or cooling device, the action is valid.

SPECIAL NOTE ON USE ACTIONS
The 'use' action is for activating devices like desklamps.
When evaluating 'use' actions on desklamps, the agent must be holding an object to examine it under the light.

RULE INTERPRETATION PRINCIPLE
If the evaluated condition fully satisfies the rule's success criteria, the action must proceed to the next step without marking it as failed.
Do not describe such a case as a "failure" in the feedback.
Only generate a failure if the state violates a precondition or makes the action impossible.

INSTRUCTION
To generate your prediction, you MUST follow this reasoning process step-by-step without deviation. Report ONLY the FIRST failure you detect.

---

Step 1: Analyze the Action.
Identify the action_name and its arguments from PROPOSED ACTION.

---

Step 2: Evaluate Rule - Location Prerequisite. (Applies to take, put, open, close, clean, heat, cool)
                                        
If action_name == "use", skip this step and proceed directly to Step 2.5.                                        

Question: Is the agent at the correct location to perform this action?
Check: Does the recep value in the action's args exactly match the location_name value in current_position?
Answer: If they do not match, the action fails. Stop and report this failure. Otherwise, proceed.

---

Step 2.5: Evaluate Rule - Use Action Prerequisites. (Applies ONLY to use)
Question: Can the agent use this tool/device?

Check Process:
1. Identify the tool from the action's args (e.g., "desklamp 1").

2. Find the tool's location:
   - Search through items_in_locations to find which location contains this tool in its items list.
   - If the tool is not found in any location's items, the action fails with:
     "Action 'use' with args {'tool': '{tool}'} failed. Reason: The tool '{tool}' is not present in any known location."

3. Verify agent location:
   - If current_position.location_name does NOT match the location where the tool was found, the action fails with:
     "Action 'use' with args {'tool': '{tool}'} failed. Reason: You are at '{current_position.location_name}' but '{tool}' is at '{tool_location}'. You must go to '{tool_location}' first."

4. Special check for desklamp usage (examining objects under light):
   - If the tool name contains "desklamp":
     - If item_in_hand.item_name is null (hand is empty), the action fails with:
       "Action 'use' with args {'tool': '{tool}'} failed. Reason: To examine an object under the desklamp, you must be holding the object first. Your hand is currently empty."
     - If item_in_hand.item_name contains an object, this is valid → proceed to final success.

5. For other tools (non-desklamp):
   - If the agent is at the correct location, the action succeeds → proceed to final success.

Answer: If any check fails, stop and report that specific failure. Otherwise, proceed to the next step.

---

Step 3: Evaluate Rule - Hand State. (Applies ONLY to take and put)
Question: Is the agent's hand in the correct state for this action?
Check for take: Is item_in_hand.item_name null? If not, take fails.
Check for put: Does item_in_hand.item_name match the obj in the action's args? If not, put fails.
Answer: If a check fails, stop and report this failure. Otherwise, proceed.

---

Step 4: Evaluate Rule - Receptacle State. (Applies ONLY to open and close)
Question: Is the target receptacle in the correct state for this action?

Check:
- Locate the target receptacle in items_in_locations using its name (the recep value).
- If the target is not found, the action fails immediately.

For open:
- If items_in_locations[recep].status == "closed", this is the correct precondition → proceed to the next step.
- If items_in_locations[recep].status == "open", the action fails with feedback:
  "The target receptacle '{recep}' is already open, so it cannot be opened again."

For close:
- If items_in_locations[recep].status == "open", this is the correct precondition → proceed to the next step.
- If items_in_locations[recep].status == "closed", the action fails with feedback:
  "The target receptacle '{recep}' is already closed, so it cannot be closed again."

Important:
- Ignore current_position.status completely.
- Only use items_in_locations[recep].status to determine open/close success.
- When the precondition is correct, do not mark it as a failure.

Answer: If a check fails, stop and report this failure. Otherwise, proceed.

---

Step 4.5: Evaluate Rule - Object State for Heat and Cool.
(Applies ONLY to heat and cool)

Question: Is the object available for this action in the correct context?

For heat:
- If the agent is at a heating-capable location (e.g., "microwave", "stoveburner", or "oven"), then check:
    1. If the object (obj) is inside items_in_locations[recep].items → success.
    2. If item_in_hand.item_name == obj AND current_position.location_name == recep → success.
- If either condition is true, this step passes.
- If neither condition is true, this step fails.

For cool:
- If the agent is at a cooling-capable location (e.g., "fridge" or "freezer"), then check:
    1. If the object (obj) is inside items_in_locations[recep].items → success.
    2. If item_in_hand.item_name == obj AND current_position.location_name == recep → success.
- If either condition is true, this step passes.
- If neither condition is true, this step fails.

Answer: If a check fails, stop and report this failure. Otherwise, proceed.

---

FINAL SUCCESS CONDITION
Throughout all steps:
- If no failure has been detected in any step, the action MUST be considered successful.
- Do NOT describe such a case as a failure, partial failure, or uncertainty.
- When all preconditions and checks are satisfied, the final output MUST be:
  {
    "flag": true,
    "feedback": "",
    "suggestion": ""
  }
This rule overrides any prior language that might express uncertainty.
Only if an explicit failure is detected should flag be set to false.

---

Step 5: Evaluate Commonsense Plausibility.
Question: Does this action make sense in the real world? (e.g., can you open a desk?)

Special Case Exception:
- For 'heat', 'cool', and 'use' actions, skip this step entirely.
  These actions are evaluated purely based on physical preconditions, not semantic plausibility.

Answer:
- For all other actions (e.g., open, close, take, put, clean), if the action is implausible, it fails.
- For heat/cool/use, always proceed without failure from this step.

---

Step 6: Formulate Final Output.
If any step resulted in a failure, formulate the feedback and suggestion based on that specific failure.
If no step failed, explicitly output:
{
  "flag": true,
  "feedback": "",
  "suggestion": ""
}

---

OUTPUT FORMAT
Respond only with the JSON object.
flag: true for success, false for failure.
feedback: Explanation of the failure, including the specific reason. Empty if successful.
suggestion: A specific hint for the agent. Empty if successful.

Example for SUCCESS:
{
  "flag": true,
  "feedback": "",
  "suggestion": ""
}

Example for FAILURE (not at correct location):
{
  "flag": false,
  "feedback": "Action 'use' with args {'tool': 'desklamp 1'} failed. Reason: You are at 'desk 1' but 'desklamp 1' is at 'sidetable 2'. You must go to 'sidetable 2' first.",
  "suggestion": "Use 'goto' action to move to 'sidetable 2' where 'desklamp 1' is located."
}

Example for FAILURE (hand empty when using desklamp):
{
  "flag": false,
  "feedback": "Action 'use' with args {'tool': 'desklamp 1'} failed. Reason: To examine an object under the desklamp, you must be holding the object first. Your hand is currently empty.",
  "suggestion": "First, use 'take' action to pick up the object you want to examine, then return to the desklamp location."
}

Example for FAILURE (tool not found):
{
  "flag": false,
  "feedback": "Action 'use' with args {'tool': 'desklamp 1'} failed. Reason: The tool 'desklamp 1' is not present in any known location.",
  "suggestion": "Explore more locations to find 'desklamp 1', or check if the tool name is correct."
}
        """)

        header = textwrap.dedent("""
CURRENT OBSERVATION (o_t)
This JSON object represents the state of the environment.
        """)

        footer = textwrap.dedent("""
PROPOSED ACTION (a_t)
This is the action you must evaluate.
        """)

        user_prompt = header + current_obs_json + "\n" + footer + proposed_action_json

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 追加の処理:ログ残す用
        # ========================================================
        log_text = "--- System Prompt ---\n"
        log_text += system_prompt + "\n\n"
        for msg in messages[1:]:
            log_text += f"--- {msg['role'].capitalize()} ---\n"
            log_text += msg['content'] + "\n\n"
        # ========================================================
        
        print("使用しているモデルの確認:", self.model)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0,
            )
            
            content = response.choices[0].message.content
            print(f"[LLMWorldModel] Raw outcome prediction content: {content}")
            outcome_data = json.loads(content)
            
            return outcome_data, log_text
        
        except Exception as e:
            print(f"[LLMWorldModel] Unexpected error: {e}")
            

# MAPEXECUTE の実装 (Rcode を適用し、次の状態を合成的に構築)
def MAPEXECUTE(Rcode: List[Callable], llm_wm_predicted_success: bool, llm_wm_predicted_feedback: str, llm_wm_predicted_suggestion: str, current_observation_state: Dict, proposed_action: Dict, output_dir: str, t_index: int, scene_graph) -> Tuple[Dict, str, str, bool]:
    """
    Rcode (ルール) を用いてWorld Modelの予測を検証し、フィードバックとサジェスチョン、
    そして行動後の最終的な次の状態を生成する。
    Rcode: NSLEARNINGから得られる実行可能なコードルール群（今回は expected_rule_code を使用）
    llm_wm_predicted_success: LLM World Modelが予測した行動の成功/失敗 (True/False)。
    current_observation_state: 現在の環境状態のJSON辞書。
    proposed_action: LLM Agentが生成した行動のJSON辞書。

    戻り値: (oˆt+1, feedback, sugg, flag)
    oˆt+1: 最終的に決定された次の状態 (Rcodeの適用結果が優先される)
    feedback: LLM Agentに返すフィードバック (なぜ予測が間違っていたかなど)
    sugg: LLM Agentに返す具体的なサジェスチョン (次の行動調整のヒント)
    flag: 行動が最終的に受け入れられたかどうか (True/False) - MPCループを抜ける条件
    """

    # Current Observation =========================================================================
    print("=== current_observation_state ===")
    print(json.dumps(current_observation_state, indent=4, ensure_ascii=False))
    # ====================================================================================================

    # Proposed Action ===================================================================================
    print("=== proposed_action ===")
    print(json.dumps(proposed_action, indent=4, ensure_ascii=False))
    # ====================================================================================================
    
    if not Rcode:
        # Rcodeが提供されない場合のデフォルト動作
        print("***************** Rcodeは渡されませんでした *********************")
        print("***************** WMの予測だけで判定します *****************")
        final_feedback = llm_wm_predicted_feedback
        final_suggestion = llm_wm_predicted_suggestion
        final_flag = llm_wm_predicted_success
    else:
        print("***************** Rcodeが渡されました *********************")
        print("***************** Rcode, WMの予測を比較して判定します *****************")

        ################### LLMの予測 ##########################
        llm_feedback = llm_wm_predicted_feedback
        llm_success = llm_wm_predicted_success
        llm_suggestion = llm_wm_predicted_suggestion
        ########################################################

        ################  コードルールの予測 #####################
        rule_feedback, rule_success, rule_suggestion = "", True, ""
    
        for rule_func in Rcode:
            feedback, success, suggestion = rule_func(current_observation_state["state"], proposed_action, scene_graph)
            
            # Rcodeのルールの一部で失敗判定. すぐに出る.
            if not success:
                rule_feedback = feedback
                rule_success = success
                rule_suggestion = suggestion
                break

            # 成功判定だったら次のコードルールで検証.
            else:
                rule_feedback = feedback
                rule_success = success
                rule_suggestion = suggestion
        ########################################################


        # === LLM WMの予測とRcodeの判断の比較 ===
        if llm_success != rule_success:
            print(f"矛盾が検出されました. [LLM]: {llm_success}, [Rcode]: {rule_success}.")
            if rule_success: # Rcodeは成功と言っているのにLLMは失敗と予測
                print("Rcodeは成功と判定、LLMは失敗と判定:")
                final_feedback = rule_feedback
                final_suggestion = rule_suggestion
                final_flag = rule_success
            else: # Rcodeは失敗と言っているのにLLMは成功と予測
                print("Rcodeは失敗と判定、LLMは成功と判定:")
                final_feedback = rule_feedback
                final_suggestion = rule_suggestion
                final_flag = rule_success
            
        else: # LLMの予測とRcodeの判断が一致する場合
            print(f"LLMとRcodeで一致しました. [LLM]: {llm_success}, [Rcode]: {rule_success}.")
            if rule_success: # 両方成功
                print("どちらも成功と判定:")
                final_feedback = rule_feedback
                final_suggestion = rule_suggestion
                final_flag = rule_success
            else: # 両方失敗
                print("どちらも失敗と判定:")
                final_feedback = rule_feedback
                final_suggestion = rule_suggestion
                final_flag = rule_success
                
    # === 次の状態 (oˆt+1) の構築ロジック ===
    # LLM World Modelが直接ot+1を予測しないため、MAPEXECUTEがその役割を担う

    print(f"[MAPEXECUTE] 最終的な結果: Flag={final_flag}, Feedback='{final_feedback}', Sugg='{final_suggestion}'")
    return final_feedback, final_suggestion, final_flag


# Algorithm 2: Model-Predictive Control (MPC) の実装
def MPC(ot: Dict, Rcode: List[Callable], LLM_AGENT: LLMAgent, LLM_WORLD_MODEL: LLMWorldModel, t_index: int, outdir: str, REPLANLIMIT: int, task_name: str, scene_graph) -> Tuple[Dict, Dict]:
    """
    Model-Predictive Control のアルゴリズム。
    ot: 現在の観測 (traj_real.jsonのステップ全体)
    Rcode: NeuroSymbolic Learningから得られる実行可能なコードルール群
    LLM_AGENT: LLMAgentのインスタンス
    LLM_WORLD_MODEL: LLMWorldModelのインスタンス
    REPLANLIMIT: リプランの最大回数
    """
    print("\n--- Starting MPC Loop ---")
    feedback = ""
    sugg = ""
    replan_count = 0
    action_log = {}

    # 保存処理 (outdir を使う)
    agent_prompt_dir = os.path.join(outdir, "agent_prompts_log")
    wm_prompt_dir = os.path.join(outdir, "wm_prompts_log")
    iteration_dir = os.path.join(outdir, "iteration_log")

    os.makedirs(agent_prompt_dir, exist_ok=True)
    os.makedirs(wm_prompt_dir, exist_ok=True)
    os.makedirs(iteration_dir, exist_ok=True)

    while replan_count < REPLANLIMIT:
        print(f"\n[MPC] イテレーション回数: {replan_count + 1}/{REPLANLIMIT}")

        feedbacks = {k: v for k, v in action_log.items() if k.startswith("Feedback_")}
        merged_feedback = " ".join(feedbacks.values())

        # 4: at ← LLMAGENT(ot, feedback, sugg)
        # LLMAgentが現在の観測、フィードバック、サジェスチョンに基づいて行動を生成
        at, agent_prompt = LLM_AGENT.generate_action(ot, feedback=merged_feedback, suggestion=sugg, step=replan_count, task=task_name)

         # 5: o_t1 ← LLM_WORLD_MODEL.predict_transition_outcome(ot['state'], at)
        o_t1, wm_prompt = LLM_WORLD_MODEL.predict_transition_outcome(ot, at)

        flag = o_t1.get("flag")
        feedback = o_t1.get("feedback")
        sugg = o_t1.get("suggestion")

        print(f"[MPC] LLM World Modelが予測した成功判定: {flag}")
        if not flag:
            print(f"[MPC] Feedback from WM: {feedback}")
            print(f"[MPC] Suggestion from WM: {sugg}")
        
        print("==========================MAPEXECUTE[実行]===========================================")

        # 6: MAPEXECUTEに渡して結果を得る
        feedback, sugg, final_flag = MAPEXECUTE(Rcode, flag, feedback, sugg, ot, at, outdir, t_index, scene_graph)

        print(f"[MPC] MAPEXECUTE関数の結果 → Flag: {final_flag}, Feedback: '{feedback}', Suggestion: '{sugg}'")

        print("==========================MAPEXECUTE[終了]===========================================")


        # リプラン回数付きキーで記録
        idx = replan_count + 1
        action_log[f"Action_{idx}"] = at
        action_log[f"Feedback_{idx}"] = feedback
        action_log[f"Suggestion_{idx}"] = sugg

        # ==================================================================================================
        # エージェントのプロンプト保存

        agent_prompt_file_name = os.path.join(agent_prompt_dir, f"agent_prompt_{t_index}_{replan_count}.txt")
        with open(agent_prompt_file_name, "w", encoding="utf-8") as f:
            f.write(agent_prompt)

        wm_prompt_file_name = os.path.join(wm_prompt_dir, f"wm_prompt_{t_index}_{replan_count}.txt")
        with open(wm_prompt_file_name, "w", encoding="utf-8") as f:
            f.write(wm_prompt)

        # ==================================================================================================
        
        replan_count += 1

        # MAPEXECUTE関数でTrueになったら...
        if final_flag:
            print(f"[MPC] 以下の Action が受け入れられました.")
            print(f"{json.dumps(at, indent=4, ensure_ascii=False)}")

            print(f"{t_index}ステップ目のイテレーション結果を保存します.")
            iteration_file_name = os.path.join(iteration_dir, f"iteration_log_{t_index}.txt")
            with open(iteration_file_name, "w", encoding="utf-8") as f:
                f.write(json.dumps(action_log, indent=4, ensure_ascii=False))

            return at

    # リプラン回数を超過した場合の処理
    print(f"[MPC] REPLANLIMITに到達しました. ({REPLANLIMIT})")

    print(f"{t_index}ステップ目のイテレーション結果を保存します.")
    iteration_file_name = os.path.join(iteration_dir, f"iteration_log_{t_index}.txt")
    with open(iteration_file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(action_log, indent=4, ensure_ascii=False))

    return at
