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

class LLMWorldModel_BestSelect:
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
You are an Action Selector for WALL-E. Your purpose is to evaluate multiple proposed actions and select the single best action that is most likely to succeed and make progress toward the goal, based on the current observation state.

CRITICAL DATA INTERPRETATION GUIDE
The JSON keyword null (without quotes) for item_in_hand.item_name means the agent's hand is EMPTY. This is a critical distinction you must not misinterpret.

SPECIAL NOTE ON RECEPTACLE STATUS
The "status" inside current_position refers only to the agent's spatial state (e.g., whether the agent is near or inside an open area).
It should NEVER be used to determine if a receptacle (like a microwave or fridge) is open or closed.
Always check items_in_locations[recep].status for open/close evaluations.

INSTRUCTION
You will receive:
1. CURRENT OBSERVATION (o_t) - The current state of the environment
2. PROPOSED ACTIONS - A list of candidate actions to choose from

Your task is to:
1. Evaluate EACH action in the list
2. Check if each action satisfies its preconditions
3. Select the SINGLE BEST action that:
   - Has ALL preconditions satisfied
   - Will most likely succeed
   - Makes the most logical progress toward the goal

EVALUATION PROCESS
For each action, check the following based on action type:

For "goto" actions:
- Precondition: Target location must exist in reachable_locations
- Valid if: recep is in reachable_locations
- Priority: Prefer unexplored locations (not in items_in_locations) when searching for objects

For "take" actions:
- Precondition A: Agent's hand must be empty (item_in_hand.item_name == null)
- Precondition B: Object must be known and its location must be in items_in_locations
- Precondition C: Agent must be at the object's location (current_position.location_name == recep)
- Valid if: All three preconditions are met

For "put" actions:
- Precondition A: Agent must be holding the target object (item_in_hand.item_name == obj)
- Precondition B: Agent must be at the target receptacle (current_position.location_name == recep)
- Valid if: Both preconditions are met

For "open" actions:
- Precondition A: Receptacle must exist in items_in_locations
- Precondition B: Receptacle must be closed (items_in_locations[recep].status == "closed")
- Precondition C: Agent must be at the receptacle (current_position.location_name == recep)
- Valid if: All three preconditions are met

For "close" actions:
- Precondition A: Receptacle must exist in items_in_locations
- Precondition B: Receptacle must be open (items_in_locations[recep].status == "open")
- Precondition C: Agent must be at the receptacle (current_position.location_name == recep)
- Valid if: All three preconditions are met

For "clean" actions:
- Precondition A: Agent must be holding the target object (item_in_hand.item_name == obj)
- Precondition B: Agent must be at a cleaning location like sinkbasin (current_position.location_name == recep)
- Valid if: Both preconditions are met

For "heat" actions:
- Precondition A: Agent must be at a heating device (current_position.location_name == recep)
- Precondition B: Either:
  - Object is inside the device (obj in items_in_locations[recep].items), OR
  - Agent is holding the object (item_in_hand.item_name == obj)
- Valid if: Both preconditions are met

For "cool" actions:
- Precondition A: Agent must be at a cooling device like fridge (current_position.location_name == recep)
- Precondition B: Either:
  - Object is inside the device (obj in items_in_locations[recep].items), OR
  - Agent is holding the object (item_in_hand.item_name == obj)
- Valid if: Both preconditions are met

For "use" actions:
- Precondition A: Tool/device must be known (exists in items_in_locations)
- Precondition B: Agent must be at the tool's location
- Precondition C (for desklamps): Agent must be holding an object (item_in_hand.item_name != null)
- Valid if: All applicable preconditions are met

SELECTION STRATEGY
1. First, filter out all actions that FAIL any precondition check
2. Among the VALID actions (all preconditions satisfied), select the one that:
   - Makes the most direct progress toward completing the current sub-goal
   - Avoids redundant exploration (don't revisit already explored locations unless necessary)
   - Is the most logical next step in the task sequence

3. If NO actions are valid, select the action that is CLOSEST to being valid (fewest unmet preconditions)

4. Evaluation Priority Order:
   - Actions that complete the current sub-goal > Actions that prepare for the next sub-goal > Exploration actions

IMPORTANT RULES
- items_in_locations being empty {} means NO locations have been explored yet
- If an object's location is unknown (not in items_in_locations), the agent must explore first
- "goto" to unexplored locations is often the best choice when searching for objects
- Never select actions that clearly violate preconditions

OUTPUT FORMAT
Respond ONLY with a JSON object containing:
{
  "selected_action": <the single best action object>
}

Example Output 1 (when a valid action exists):
{
  "selected_action": {"action_name": "goto", "args": {"recep": "countertop 1"}}
}

Example Output 2 (exploring for an object):
{
  "selected_action": {"action_name": "goto", "args": {"recep": "desk 1"}}
}

Example Output 3 (executing main action):
{
  "selected_action": {"action_name": "take", "args": {"obj": "book 1", "recep": "desk 1"}}
}

IMPORTANT: 
- Respond ONLY with the JSON object
- Do NOT include any text outside the JSON
- The "selected_action" must be one of the actions from the PROPOSED ACTIONS list
        """)

        header = textwrap.dedent("""
CURRENT OBSERVATION (o_t)
This JSON object represents the state of the environment.
        """)

        footer = textwrap.dedent("""
PROPOSED ACTIONS
This is the list of candidate actions. Select the SINGLE BEST action from this list.
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