import os
import openai
import json
import time
import copy
import re
import inspect
import textwrap
from typing import List, Callable, Tuple, Dict, Optional
import ast

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class STAGE3:

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client 

        if self.client is None:
            print("[CodeRule] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def generate_coderule(self, ar_data, input_dir):
       
        if self.client is None:
            print("[CodeRule] Error: OpenAI client is not available. Cannot generate action.")
        
        if isinstance(ar_data, str):
            ar_data = json.loads(ar_data)
    
        ar_str = json.dumps(ar_data.get("final_rules", []), indent=4, ensure_ascii=False)

        # プロンプト ===================================================================
        system_prompt = textwrap.dedent("""
Using your knowledge of Alfworld and Python programming, please implement the following rule as a Python function. The function should evaluate the current state and an action to return a Boolean value based on specified conditions.

The function should be defined as follows:

```python
def Rule_<rule_number>_<action_type_without_spaces>(state, action):
    # Your code here
    return success
where
success: a bool, whether the action is executed successfully, give 'True' or 'False'. If the action type is not the action type in the rule, count as success (e.g., success = True).


Here is several examples of the input format:

state:
{
    "reachable_locations": [
	    "bed 1",
		"desk 1",
        "drawer 1",
        "drawer 2",
        "drawer 3",
        "garbagecan 1",
        "laundryhamper 1",
        "shelf 1",
        "shelf 2",
        "sidetable 1"
    ],
    "items_in_locations": {
        "desk 1": {
	        "items": [
	          "bowl 2",
            "bowl 1",
            "cellphone 1",
            "keychain 1",
            "pencil 2",
            "remotecontrol 1"
          ],
          "status": null
        },
        "drawer 1": {
           "items": [],
           "status": "closed"
        },
        "drawer 2": {
           "items": [],
           "status": "closed"
        },
        "shelf 1": {
           "items": [
	           "pencil 1"
           ],
           "status": null
        }
    },
    "item_in_hand": {
	    "item_name": null,
	    "status": null
    },
    "current_position": {
	    "location_name": "drawer 2",
      "status": "closed"
    }
}

action:
* goto: {"action_name": "goto", "args": {"recep": "..."}}
* take: {"action_name": "take", "args": {"obj": "...", "recep": "..."}}
* put: {"action_name": "put", "args": {"obj": "...", "recep": "..."}}
* open: {"action_name": "open", "args": {"recep": "..."}}
* close: {"action_name": "close", "args": {"recep": "..."}}
* clean: {"action_name": "clean", "args": {"obj": "...", "recep": "..."}}
* heat: {"action_name": "heat", "args": {"obj": "...", "recep": "..."}}
* cool: {"action_name": "cool", "args": {"obj": "...", "recep": "..."}}
* use: {"action_name": "use", "args": {"tool": "..."}}

The function should return a Boolean (True or False) based on an internal rule which you must implement.

Ensure that the function handles the input and outputs the expected result based on Alfworld mechanics and the provided state, action and scene_graph.

Important:
Output all rule functions together as valid Python code, without markdown formatting or extra text.
Output only the raw Python code without any Markdown code blocks (no ```python or ```), no explanations or extra text.

You should only respond in the format as described below, and do not give example usage or anything else:
RESPONSE FORMAT:
def Rule_<rule_number>_<action_type_without_spaces>(state, action):
    # Your code here
        """)
        
        header = textwrap.dedent("""
My information is as follows: 
A list of Action Rules, each with:
- A rule number
- An action type (e.g., "go to", "open", "put")
- A condition description
- A checking method description
given rules：
        """)


        user_prompt = header + ar_str

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

        CR_file_name = os.path.join(input_dir, "CodeRule_llm_prompt.txt")
        with open(CR_file_name, "w", encoding="utf-8") as f:
            f.write(log_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0,
            )

            generate_response = response.choices[0].message.content
            return generate_response

        except openai.APITimeoutError:
            print("[CodeRule] API request timed out.")
          
        except openai.APIConnectionError as e:
            print(f"[CodeRule] API connection error: {e}")

        except openai.RateLimitError:
            print("[CodeRule] OpenAI API rate limit exceeded. Waiting 5 seconds...")
            time.sleep(5)

        except openai.APIStatusError as e:
            print(f"[CodeRule] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[CodeRule] JSON decode error from LLM response")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    # ===========================================================================

    def verify_code_rule_boolean(self, code_rule: str, model: str = "gpt-3.5-turbo") -> bool:

        if self.client is None:
            print("[CodeRule] Error: OpenAI client is not available. Cannot generate action.")
        
        try:
            ast.parse(code_rule)
            compile(code_rule, "<string>", "exec")
        except Exception as e:
            print(f"[Syntax/Compile Error Detected] {e}")
            return False

        # プロンプト ===================================================================
        system_prompt = textwrap.dedent("""
You are a strict Python code validator.
Check whether the following string is a valid, executable Python code.

If the code can be executed without syntax or import errors, output exactly "True".
If the code cannot be executed due to any errors, output exactly "False".

Do not explain or add any text other than "True" or "False".
        """)
        
        header = "--- CODE START ---\n"

        user_prompt = header + str(code_rule)

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

        parent_dir = "./Code_Check"
        os.makedirs(parent_dir, exist_ok=True) 
        file_path = os.path.join(parent_dir, "check_prompt.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(log_text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000,
            temperature=0,
        )

        result = response.choices[0].message.content.strip()
        print(f"=== LLM Verification Result: {result} ===")

        return result.strip().lower() == "true"

    # ===========================================================================


    def save(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
    
    def append(self, path, data):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(data)


    
