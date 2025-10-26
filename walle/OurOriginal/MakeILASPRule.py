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
import re

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class ILASP_LLM:

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
You are an AI that converts natural language action rules into valid and safe Prolog rules for execution via PySWIP.

Only use the following facts and predicates:
- current_position(Location)
- location_status(Location, Status)
- item_in_hand(Item, Status)
- reachable_location(Location)
- items_in_location(Item, Location)
- empty(Location)

Allowed action predicates:
- action(goto(Location))
- action(take(Item, Location))
- action(put(Item, Receptacle))
- action(open(Receptacle))
- action(close(Receptacle))
- action(clean(Item, Location))
- action(heat(Item, Location))
- action(cool(Item, Location))
- action(use(Tool))

Rules must follow:

1. Each rule defines action_failed(Action).
2. Use only the listed facts and predicates.
    If a rule cannot be expressed safely with these, skip it completely.
3. All variables used in the body must be explicitly bound before use.
    Never use unbound variables in comparisons or negations.
4. If a predicate (e.g., location_status/2) might be missing for some objects,
    always include a safety guard using nonvar/1, var/1, or location_status(Receptacle, _) before conditions like Status \== open.
5. Avoid any rule structure that could cause instantiation errors.
    For example, wrap comparisons like this:
        location_status(Receptacle, Status), nonvar(Status), Status \== open.
6. Use \+ only for negation, never not.
7. Use \== only for comparing bound atoms.
8. Output only valid Prolog syntax, no comments, no text, no explanations.
9. The generated rules must be executable in SWI-Prolog (via PySWIP) without warnings or errors.
                                        
Example conversion:

Natural language:
"For action 'goto', if the destination location is the same as the current location, the action will fail."
Prolog:
action_failed(goto(Destination)) :-
    current_position(Destination).

Another example (safe variable binding):
"For action 'take', if the receptacle where the object is located is not open, the action will fail."
Prolog:                         
action_failed(take(Item, Receptacle)) :-
    items_in_location(Item, Receptacle),
    location_status(Receptacle, Status),
    nonvar(Status),
    Status \== open.                                                                                               
        """)
        
        header = textwrap.dedent("""
Now, convert the following list of natural language action rules to Prolog:
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

        CR_file_name = os.path.join(input_dir, "ILASP_llm_prompt.txt")
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


    def save(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
    
    # ルールを正規化して比較用に統一
    @staticmethod
    def normalize_rule(rule: str) -> str:
        # 行ごとの空白を削除して単一スペースに連結
        return ' '.join(line.strip() for line in rule.splitlines() if line.strip())

    def save_connect(self, path, data):
        # action_failedごとに分割してルールを抽出
        new_rules = [r.strip() for r in re.split(r'(?=action_failed\()', data) if r.strip()]

        # 各ルールを1行に変換（改行を削除、連続空白は1つに）
        def flatten_rule(rule: str) -> str:
            # 改行削除 + 前後空白削除
            rule = ' '.join(line.strip() for line in rule.splitlines() if line.strip())
            # 連続スペースを1つに
            rule = re.sub(r'\s+', ' ', rule)
            return rule

        new_rules = [flatten_rule(r) for r in new_rules]

        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing_rules = [r.strip() for r in re.split(r'(?=action_failed\()', f.read()) if r.strip()]
            existing_rules = [flatten_rule(r) for r in existing_rules]
        except FileNotFoundError:
            existing_rules = []

        # 重複チェック
        existing_set = set(existing_rules)
        rules_to_add = [r for r in new_rules if r not in existing_set]

        if rules_to_add:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(rules_to_add) + '\n')



if __name__ == '__main__':
    ilasp = ILASP_LLM(model="gpt-3.5-turbo")

    with open("./check1/test.json", "r", encoding="utf-8") as f:
        action_rules = json.load(f)
    
    code_rule = ilasp.generate_coderule(action_rules, "./check1")
    ilasp.save("./PrologRule/generated_rules.pl", code_rule)

    print("生成したLASルール:")
    print(code_rule)
