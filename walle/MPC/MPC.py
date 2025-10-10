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
        current_obs = json.loads(current_obs_json)
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


ACTION SPACEY
our generated action MUST be one of the following commands. 
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

INSTRUCTION
Analyze all information and generate the single most logical next action.
Follow this "Goal-Oriented Precondition Analysis" process step-by-step:

Step 1: Deconstruct the Goal into an Ordered Plan.
Based on the main GOAL, mentally create an ordered list of major sub-goals.
Example: For "put a cool bowl in cabinet", your mental plan is: [1. take a bowl, 2. cool the bowl, 3. put the bowl in a cabinet].

Step 2: Assess Plan Completion and Identify Current Sub-Goal.
Review your mental plan against the CURRENT OBSERVATION to find the first sub-goal that is NOT yet complete. 
This becomes your current sub-goal.

Example: If item_in_hand contains bowl 1 with status: cool, then sub-goals 1 and 2 are complete. 
Your current sub-goal is 3.
put the bowl.... In the observation above, item_in_hand is empty, so your current sub-goal is 1. 
take a bowl.
 
Step 3: Analyze Preconditions for the Current Sub-Goal.
List the strict preconditions required to achieve ONLY the current sub-goal.
Example: If current sub-goal is 1.
take a bowl:Precondition A: The location of the 'bowl' must be known.
Precondition B: You must be at the location of the 'bowl'.
Precondition C: Your hand must be empty.
  
Step 4: Find the First Unmet Precondition.
Check your CURRENT OBSERVATION to find the very first precondition from your list that is NOT met.
  
Example: In the observation above, for sub-goal 1. 
take a bowl:Precondition A: Location is known ('countertop 1' in items_in_locations). -> Met.Precondition B: You are at countertop 1. -> Met.Precondition C: Your hand is empty (null). -> Met.Conclusion: All preconditions are met for the current sub-goal.
  
Step 5: Generate Action.
If all preconditions for the current sub-goal are met (as in the example above), your action should be to execute that sub-goal (e.g., take the bowl).
If a precondition was NOT met in Step 4, your action must be to resolve that specific unmet precondition (e.g., if you were not at the location, your action would be to goto there).
  
OUTPUT ACTION (a_t)Your output must be a single JSON object based on your analysis.
{ "action_name": "take", "args": {   "obj": "bowl 1",   "recep": "countertop 1" }}

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
            return self.generate_action(observation_data, feedback, suggestion, step_index)

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

        current_obs = json.loads(current_obs_json)
        proposed_action = json.loads(proposed_action_json)

        # =================================================================================

        # 英語プロンプト(改良後) =================================================================
        system_prompt = textwrap.dedent("""
ROLE
You are an All-in-One World Model for WALL-E. Your purpose is to act as a strict, procedural referee. You must predict if a proposed action will succeed or fail by meticulously following a step-by-step evaluation process. Your primary function is to enforce deterministic rules.

CRITICAL DATA INTERPRETATION GUIDE
The JSON keyword null (without quotes) for item_in_hand.item_name means the agent's hand is EMPTY. This is a critical distinction you must not misinterpret.

INSTRUCTION
To generate your prediction, you MUST follow this reasoning process step-by-step without deviation. Report ONLY the FIRST failure you detect.
Step 1: Analyze the Action.
Identify the action_name and its arguments from PROPOSED ACTION.
Step 2: Evaluate Rule - Location Prerequisite. (Applies to take, put, open, close, clean, heat, cool)
Question: Is the agent at the correct location to perform this action?
Check: Does the recep value in the action's args exactly match the location_name value in current_position?
Answer: If they do not match, the action fails. Stop and report this failure. Otherwise, proceed.
Step 3: Evaluate Rule - Hand State. (Applies ONLY to take and put)
Question: Is the agent's hand in the correct state for this action?
Check for take: Is item_in_hand.item_name null? If not, take fails.
Check for put: Does item_in_hand.item_name match the obj in the action's args? If not, put fails.
Answer: If a check fails, stop and report this failure. Otherwise, proceed.
Step 4: Evaluate Rule - Receptacle State. (Applies ONLY to open and close)
Question: Is the target receptacle in the correct state for this action?
Check: Find the target receptacle in items_in_locations.
Check for open: Is its status "closed"? If not, open fails.
Check for close: Is its status "open"? If not, close fails.
Answer: If a check fails, stop and report this failure. Otherwise, proceed.
Step 5: Evaluate Commonsense Plausibility.
Question: Does this action make sense in the real world? (e.g., can you open a desk?)
Answer: If the action is implausible, it fails. Stop and report this failure. Otherwise, the action succeeds.
Step 6: Formulate Final Output.
If any step resulted in a failure, formulate the feedback and suggestion based on that specific failure.
If all steps passed, the action succeeds (flag: true).

OUTPUT FORMAT
Respond only with the JSON object.
flag: true for success, false for failure.
feedback: Explanation of the failure, including the specific reason. Empty if successful.
suggestion: A specific hint for the agent. Empty if successful.
Example for FAILURE:
{
 "flag": false,
 "feedback": "Action 'open' with args {'recep': 'desk 1'} failed. Reason: This action is implausible. A 'desk' is typically a surface and cannot be opened.",
 "suggestion": "If you want to open something, look for a container like a 'drawer' or 'cabinet'. Desks are for placing items on."
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
def MAPEXECUTE(Rcode: List[Callable], llm_wm_predicted_success: bool, llm_wm_predicted_feedback: str, llm_wm_predicted_suggestion: str, current_observation_state: Dict, proposed_action: Dict, output_dir: str, t_index: int) -> Tuple[Dict, str, str, bool]:
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
    # Proposed Action ===================================================================================
    print("=== proposed_action ===")
    print(json.dumps(proposed_action, indent=4, ensure_ascii=False))
    # Scene Graph =======================================================================================
    sg = SceneGraph()
    sg.update(current_observation_state.get("state", {}))
    print("=== SceneGraph ===")
    sg.visualize()
    
    SG_file_name = os.path.join(output_dir, f"SceneGraph/scene_graph_{t_index}.json")
    os.makedirs(os.path.dirname(SG_file_name), exist_ok=True)
    sg.save(SG_file_name)
    # ====================================================================================================

    if not Rcode:
        # Rcodeが提供されない場合のデフォルト動作
        print("***************** Rcodeは渡されませんでした *********************")
        print("***************** WMの予測だけで判定します *****************")
        final_feedback = llm_wm_predicted_feedback
        final_success = llm_wm_predicted_success
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
                final_feedback = llm_feedback
                final_suggestion = llm_suggestion
                final_flag = llm_success
            else: # Rcodeは失敗と言っているのにLLMは成功と予測
                print("Rcodeは失敗と判定、LLMは成功と判定:")
                final_feedback = rule_feedback
                final_suggestion = rule_suggestion
                final_flag = rule_success
            
        else: # LLMの予測とRcodeの判断が一致する場合
            print(f"LLMとRcodeで一致しました. [LLM]: {llm_success}, [Rcode]: {rule_success}.")
            if rule_success: # 両方成功
                print("どちらも成功と判定:")
                final_feedback = llm_feedback
                final_suggestion = llm_suggestion
                final_flag = llm_success
            else: # 両方失敗
                print("どちらも失敗と判定:")
                final_feedback = llm_feedback
                final_suggestion = llm_suggestion
                final_flag = llm_success
                
    # === 次の状態 (oˆt+1) の構築ロジック ===
    # LLM World Modelが直接ot+1を予測しないため、MAPEXECUTEがその役割を担う

    print(f"[MAPEXECUTE] 最終的な結果: Flag={final_flag}, Feedback='{final_feedback}', Sugg='{final_suggestion}'")
    return final_feedback, final_suggestion, final_flag


# Algorithm 2: Model-Predictive Control (MPC) の実装
def MPC(ot: Dict, Rcode: List[Callable], LLM_AGENT: LLMAgent, LLM_WORLD_MODEL: LLMWorldModel, t_index: int, outdir: str, REPLANLIMIT: int, task_name: str) -> Tuple[Dict, Dict]:
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
    predicted_o_t1 = {}
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
        feedback, sugg, final_flag = MAPEXECUTE(Rcode, flag, feedback, sugg, ot, at, outdir, t_index)

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

        """
        agent_prompt_dir = "./walle/MPC/agent_prompts_log"
        if not os.path.exists(agent_prompt_dir):
            os.makedirs(agent_prompt_dir)
        agent_prompt_file_name = os.path.join(agent_prompt_dir, f"agent_prompt_{t_index}_{replan_count}.txt")
        with open(agent_prompt_file_name, "w", encoding="utf-8") as f:
            f.write(agent_prompt)
        
        # ワールドモデルのプロンプト保存
        wm_prompt_dir = "./walle/MPC/wm_prompts_log"
        if not os.path.exists(wm_prompt_dir):
            os.makedirs(wm_prompt_dir)
        wm_prompt_file_name = os.path.join(wm_prompt_dir, f"wm_prompt_{t_index}_{replan_count}.txt")
        with open(wm_prompt_file_name, "w", encoding="utf-8") as f:
            f.write(wm_prompt)
        """

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
            """
            debug_dir = "./walle/MPC/iteration_log"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            iteration_file_name = os.path.join(debug_dir, f"iteration_log_{t_index}.txt")

            with open(iteration_file_name, "w", encoding="utf-8") as f:
                f.write(json.dumps(action_log, indent=4, ensure_ascii=False))
            """

            return at

    # リプラン回数を超過した場合の処理
    print(f"[MPC] REPLANLIMITに到達しました. ({REPLANLIMIT})")

    print(f"{t_index}ステップ目のイテレーション結果を保存します.")
    iteration_file_name = os.path.join(iteration_dir, f"iteration_log_{t_index}.txt")
    with open(iteration_file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(action_log, indent=4, ensure_ascii=False))
    """
    debug_dir = "./walle/MPC/iteration_log"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    iteration_file_name = os.path.join(debug_dir, f"iteration_log_{t_index}.txt")
    
    with open(iteration_file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(action_log, indent=4, ensure_ascii=False))
    """

    return at
    

"""
def is_action_admissible(admissible_commands: list[str], proposed_action_json: dict) -> bool:
    # JSON文字列が渡された場合はdictに変換
    proposed_action_json = json.loads(proposed_action_json)
    proposed_command = proposed_action_json.get("command", "").strip()
    # admissible_commands が str の場合（改行区切りテキストなど）をリストに変換
    admissible_commands = [cmd.strip() for cmd in admissible_commands.splitlines() if cmd.strip()]
    # 全部 strip して比較
    admissible_set = {cmd.strip() for cmd in admissible_commands}
    return proposed_command in admissible_set
"""


if __name__ == "__main__":

    # 環境変数 OPENAI_API_KEY が設定されているか確認
    if client is None:
        print("Skipping LLMAgent and LLMWorldModel tests due to OpenAI client initialization failure.")
    elif not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key before running the script.")
    else:
        print("OpenAI API Key is set. Running MPC with LLMAgent and LLMWorldModel...")

        # エージェントとワールドモデルのインスタンス化
        agent = LLMAgent(prompt_log_dir="Demo/MPC/agent_prompts_log", action_log_dir="Demo/MPC/agent_actions_log")
        world_model = LLMWorldModel(prompt_log_dir="Demo/MPC/wm_prompts_log", outcome_log_dir="Demo/MPC/wm_outcomes_log")

        # MPCの結果を保存するディレクトリ
        mpc_results_dir = "Demo/MPC/mpc_results_log"
        if not os.path.exists(mpc_results_dir):
            os.makedirs(mpc_results_dir)
            print(f"Created directory: {mpc_results_dir}")

        traj_real_file_path = "Demo/buffer_fact/traj_s0.json"

        try:
            # 1. 実環境データの読み込み．観測値Stの取得．==============================================================================
            with open(traj_real_file_path, 'r', encoding='utf-8') as f:
                traj_real_data = json.load(f)
            
            # 'state'キーが存在することを確認し、それを ot に設定します
            if "state" not in traj_real_data:
                raise ValueError(f"'{traj_real_file_path}' does not contain a 'state' key at the top level.")
            
            ot = traj_real_data["state"]
            print(ot)

            planned_action = {"action_type":" "}
            t_index = 0
            
            while planned_action.get("action_type", "").lower() not in ["done", "do_nothing"] and t_index < 10:
                Rcode_t = []
                print(f"\n--- Running MPC for Step {t_index} ---")

                # MPCを実行し、計画されたアクションと予測された次の状態を取得
                current_planned_action, predicted_next_state = MPC(ot, Rcode_t, agent, world_model)

                print(f"\n--- MPC Result for Step {t_index} ---")
                print(f"Planned Action: {planned_action}")
                print(f"Predicted Outcome (Next State): {predicted_next_state}")
                
                # 計画されたアクションの保存
                planned_action_filename = os.path.join(mpc_results_dir, f"mpc_planned_action_step_{t_index}.json")
                try:
                    with open(planned_action_filename, "w", encoding="utf-8") as f:
                        json.dump(current_planned_action, f, indent=4, ensure_ascii=False)
                    print(f"MPC planned action saved to {planned_action_filename}")
                except Exception as e:
                    print(f"Error saving planned action to file: {e}")

                # 予測された次の状態の保存
                predicted_state_filename = os.path.join(mpc_results_dir, f"mpc_predicted_state_step_{t_index}.json")
                try:
                    with open(predicted_state_filename, "w", encoding="utf-8") as f:
                        json.dump(predicted_next_state, f, indent=4, ensure_ascii=False)
                    print(f"MPC predicted next state saved to {predicted_state_filename}")
                except Exception as e:
                    print(f"Error saving predicted next state to file: {e}")
                
                ot = predicted_next_state
                planned_action = current_planned_action
                t_index += 1
            
                
        except FileNotFoundError:
            print(f"Error: The file '{traj_real_file_path}' was not found. Please ensure it's in the same directory.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{traj_real_file_path}'. Check file format.")
        except (IndexError, ValueError) as e:
            print(f"Error processing traj_real.json: {e}. It might be empty or in an unexpected format (e.g., no task name found).")
        except Exception as e:
            print(f"An unexpected error occurred during file processing or action generation: {e}")