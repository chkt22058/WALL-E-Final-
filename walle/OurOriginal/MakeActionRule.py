import os
import openai
import json
import time
import copy
import re
import inspect
import textwrap
from typing import List, Callable, Tuple, Dict, Optional

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class ActionRules:

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client 

        if self.client is None:
            print("[LLMAgent] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def generate_ActionRules(self, transitions_data, input_dir, All_AR_dir):
       
        if self.client is None:
            print("[ActionRules] Error: OpenAI client is not available. Cannot generate action.")
        
        # 前回まで作成していたアクションルールがあればセット!
        existing_action_rules = []
        improve_path = os.path.join(All_AR_dir, "all_action_rules.json")
        if os.path.exists(improve_path):
            with open(improve_path, "r", encoding="utf-8") as f:
                existing_action_rules = json.load(f)
                existing_rules_json_str = json.dumps(existing_action_rules.get("final_rules", []), indent=4, ensure_ascii=False)
        else:
            existing_action_rules = []
            existing_rules_json_str = json.dumps(existing_action_rules, indent=4, ensure_ascii=False)
        """
        # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
        new_transitions = copy.deepcopy(transitions_data)

        # action_result_X 全てを走査
        for key, value in new_transitions.items():
            if key.startswith("action_result_") and isinstance(value, dict):
            # feedback と suggestion を削除
                for remove_key in ["feedback", "suggestion"]:
                    value.pop(remove_key, None)
        """
        transitions_json_str = json.dumps(transitions_data, indent=4, ensure_ascii=False)
            
        # プロンプト ===================================================================
        system_prompt = textwrap.dedent("""
You are responsible for mining new rules from the given transitions, ensuring that
these rules differ from the ones already provided. Focus on generating general and
universal rules that are not tied to any specific item or tool. Your goal is to
generalize across different objects, creating flexible rules that can be applied
broadly to diverse contexts and situations.

You should only respond in the format as described below:
RESPONSE FORMAT:
{
    "new_rules":[
        "Rule ...: For action ...,...; Checking Method: ...",
        "Rule ...: For action ...,...; Checking Method: ...",
    ...
    ]
}

Instructions:
- Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing
commas, **no single quotes**, etc.
- Please use you knowledge in <ENV>, do inductive reasoning. You need to dig up
as many rules as possible that satisfy all transitions.
- Extract and utilize only the features that influence the outcome of the action.
- Please generate general and universal rules; the rules should not reference
any specific item or tool! You need to generalize across various items or tools.
- Generate only the rules under what conditions the action will fail.
- While generating a rule, you also need to state how to check if a transition
satisfies this rule. Please be specific as to which and how 'features' need to
be checked

        """)
        
        header = textwrap.dedent(f"""
I will give you an array of transitions:
        """)

        footer = textwrap.dedent(f"""
and an array of rules:
        """)

        user_prompt = header + transitions_json_str + "\n" + footer + existing_rules_json_str

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

        AR_llm_prompt_file_name = os.path.join(input_dir, "AR_llm_prompt.txt")
        with open(AR_llm_prompt_file_name, "w", encoding="utf-8") as f:
            f.write(log_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0,
            )

            generate_response = response.choices[0].message.content
            action_rules_data = json.loads(generate_response) 
            return action_rules_data

        except openai.APITimeoutError:
            print("[ActionRules] API request timed out.")
          
        except openai.APIConnectionError as e:
            print(f"[ActionRules] API connection error: {e}")

        except openai.RateLimitError:
            print("[ActionRules] OpenAI API rate limit exceeded. Waiting 5 seconds...")
            time.sleep(5)
            return self.generate_ActionRules(transitions_data)

        except openai.APIStatusError as e:
            print(f"[ActionRules] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[ActionRules] JSON decode error from LLM response")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    

    def generate_ActionRulesImprove(self, transitions_data, existing_rules, input_dir):
       
        if self.client is None:
            print("[ActionRules] Error: OpenAI client is not available. Cannot generate action.")
        
        # 前回まで作成していたアクションルールがあればセット!
        if existing_rules is None:
            existing_rules = []

        # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
        # 元の辞書をコピー（元データを保持する場合）
        new_transitions = copy.deepcopy(transitions_data)

        # action_result_X 全てを走査
        for key, value in new_transitions.items():
            if key.startswith("action_result_") and isinstance(value, dict):
            # feedback と suggestion を削除
                for remove_key in ["feedback", "suggestion"]:
                    value.pop(remove_key, None)

        transitions_json_str = json.dumps(new_transitions, indent=4, ensure_ascii=False)
        
        # 既存のルールをJSON文字列に変換
        existing_rules_json_str = json.dumps(existing_rules, indent=4, ensure_ascii=False)
            
        # プロンプト ===================================================================
        system_prompt = textwrap.dedent("""
You are a rule miner tasked with extracting meaningful rules, improve rules with a collection of transitions in Alfworld. Each transition consists of an inital state, an action, and the action result. 
The rules are for mapping the inputs, 'state' and 'action' to their corresponding 'action_result'. 
You need to verify that the given rules satisfy all the transitions, find any conflicting rules and modify them. 
Additionally, try to mine new additional rules from these transitions, new rules must be different from the given rules, and only rules for under what conditions an action will fail need to be generated.

The actions in the transitions are introduced as follows:
go to [location/object]: Move to a specified location or object. ​​
open [object]: Open a specified object like a cabinet or drawer. ​​
close [object]: Close an opened object. ​​
take [object] from [location]: Pick up an item from a specified location. ​​
put [object] in/on [location]: Place an item in or on a specified location. ​​
clean [object] with [location/tool]: Clean an object using a specific location or tool, like cleaning lettuce at the sink basin. ​​
heat [object] with [tool]: Use an appliance, such as a microwave, to heat an item. ​​
cool [object] with [tool]: Use a cooling tool or appliance, such as a fridge, to cool an item. ​​
use [tool]: Activate or use a tool, such as a desklamp. ​​


I will give you an array of transitions:
[
    {
        'state_1': '...', 
        'action_1': '...', 
        'action_result_1': "Whether the action is executed successfully, give 'True' or 'False' only"
    },
    {
        'state_2': '...', 
        'action_2': '...', 
        'action_result_2': "Whether the action is executed successfully, give 'True' or 'False' only"
    },
    ...
]
and an array of rules：
[
    "Rule 1: For action ..., if..., the action will fail; Checking Method: ...",
    "Rule 2: For action ..., if..., the action will fail; Checking Method: ...",
    "Rule 3: For action ..., if..., the action will fail; Checking Method: ...",
    "Rule 4: For action ..., if..., the action will fail; Checking Method: ...",
    ...
]

You should only respond in the format as described below:
RESPONSE FORMAT:
{
    "verified_rules":[
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        ...
    ],
    "conflicting_rules":[
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        ...
    ],
    "improved_rules":[
        "Rule ...: ...; Checking Method: ...",
        ...
    ],
    "new_rules":[
        "Rule ...: ...; Checking Method: ...",
        ...
    ],
    "final_rules":[
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        "Rule ...: ...; Checking Method: ...",
        ...
    ]
}

where
verified_rules: list rules that satisfy all the provided transitions.
conflicting_rules: list rules that contradict any of the transitions. Modify these rules if they can be modified correctly and put them in 'improved_rules'.
improved_rules: show modified 'conflicting_rules'.
new_rules: list new rules discovered. New rules must be different from the rules in 'verified_rules' and 'improved_rules' and satisfy all the transitions. otherwise, simply leave this section blank.
final_rules: combine all the rules from 'verified_rules', 'improved_rules', and 'new_rules'.


Instructions:
- Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, **no single quotes**, etc.
- Please use you knowledge in Alfworld, do inductive reasoning. You need to dig up as many rules as possible that satisfy all transitions.
- Extract and utilize only the features that influence the outcome of the action.
- Please generate general and universal rules; the rules should not reference any specific item or tool! You need to generalize across various items or tools.
- Generate only the rules under what conditions the action will fail.
- While generating a rule, you also need to state how to check if a transition satisfies this rule. Please be specific as to which and how 'state features' in 'inital state' need to be checked
        """)
        
        header = textwrap.dedent(f"""
My information is as follows: 
transitions：
        """)

        footer = textwrap.dedent(f"""
given rules：
        """)

        user_prompt = header + transitions_json_str + "\n" + footer + existing_rules_json_str

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
        
        AR_llm_prompt_imp_file_name = os.path.join(input_dir, "AR_llm_prompt_improve.txt")
        with open(AR_llm_prompt_imp_file_name, "w", encoding="utf-8") as f:
            f.write(log_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0,
            )

            generate_response = response.choices[0].message.content
            action_rules_data = json.loads(generate_response) 
            return action_rules_data

        except openai.APITimeoutError:
            print("[ActionRules] API request timed out.")
          
        except openai.APIConnectionError as e:
            print(f"[ActionRules] API connection error: {e}")

        except openai.RateLimitError:
            print("[ActionRules] OpenAI API rate limit exceeded. Waiting 5 seconds...")
            time.sleep(5)
            return self.generate_ActionRulesImprove(transitions_data)

        except openai.APIStatusError as e:
            print(f"[ActionRules] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[ActionRules] JSON decode error from LLM response")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    

    def save(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
  