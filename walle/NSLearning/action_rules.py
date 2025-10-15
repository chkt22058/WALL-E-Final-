import json
from openai import OpenAI

def AR_generate_llm_prompt_with_transitions(transitions_data: list, existing_rules: list = None) -> str:
    """
    LLM入力用のプロンプトと、変換された遷移データを結合した文字列を生成する関数

    Args:
        transitions_data (list): llm_input_traj.jsonから読み込まれた変換済みの遷移データ。
        existing_rules (list): 既存のルールの配列。提供されない場合は空のリストを使用。

    Returns:
        str: LLMに投げるための結合されたプロンプトテキスト。
    """
    if existing_rules is None:
        existing_rules = []

    # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
    transitions_json_str = json.dumps(transitions_data, indent=4, ensure_ascii=False)
    # 既存のルールをJSON文字列に変換
    existing_rules_json_str = json.dumps(existing_rules, indent=4, ensure_ascii=False)

    prompt_template = f"""
    You are responsible for mining new rules from the given transitions, ensuring that
    these rules differ from the ones already provided as much as possible. 
    However, if no genuinely new rules can be found, it is acceptable to retain the existing rules
    without forcing new ones.

    Focus on generating general and universal rules that are not tied to any specific item or tool.
    Your goal is to generalize across different objects, creating flexible rules that can be applied
    broadly to diverse contexts and situations.

    I will give you an array of transitions:
    {transitions_json_str}
    and an array of existing rules:
    {existing_rules_json_str}

    You should only respond in the format as described below:
    RESPONSE FORMAT:
    {{
        "rules":[
            "Rule ...: For action ...,...; Checking Method: ...",
            "Rule ...: For action ...,...; Checking Method: ...",
            ...
        ]
    }}
    
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
    satisfies this rule. Please be specific as to which and how ’features’ need to
    be checked
    - If a rule overlaps or duplicates an existing rule, include it again in 'rules' so the final output fully represents the existing knowledge along with any new findings.
    - If no new rules can be found beyond the existing ones, simply return the existing rules unchanged.
    """
    return prompt_template


def AR_generate_improve_prompt(transitions_data: list, existing_rules: list = None) -> str:
    """
    既存のアクションルールを改善するプロンプトを生成する関数

    Args:
        transitions_data (list): llm_input_traj.jsonから読み込まれた変換済みの遷移データ。
        existing_rules (list): 既存のルールの配列。提供されない場合は空のリストを使用。

    Returns:
        str: LLMに投げるための結合されたプロンプトテキスト。
    """
    if existing_rules is None:
        existing_rules = []

    # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
    transitions_json_str = json.dumps(transitions_data, indent=4, ensure_ascii=False)
    # 既存のルールをJSON文字列に変換
    existing_rules_json_str = json.dumps(existing_rules, indent=4, ensure_ascii=False)

    prompt_template = f"""
    You are responsible for improving the existing rules by verifying that they hold true for all transitions. 
    This involves identifying any conflicting rules, diagnosing potential issues, and making necessary modifications. 
    Ensure that the refined rules are consistent and correctly align with the transitions provided, avoiding any contradictions or overlaps.

    I will give you an array of transitions:
    {transitions_json_str}
    and an array of existing rules:
    {existing_rules_json_str}

    You should only respond in the format as described below:
    RESPONSE FORMAT:
    {{
        "final_rules":[
          "Rule ...: For action ...,...; Checking Method: ...",
          "Rule ...: For action ...,...; Checking Method: ...",
        ...
        ]
    }}

    where
    final_rules:
    The list of improved action rules after analysis.
    These rules have been refined to resolve any duplicates or contradictions found in the existing rules: "new_rules".
    They are consistent, generalized, and correctly aligned with all provided transitions.

    Instructions:
    - Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, **no single quotes**, etc.
    - Please use you knowledge in <ENV>, do inductive reasoning. You need to dig up as many rules as possible that satisfy all transitions.
    - Extract and utilize only the features that influence the outcome of the action.
    - Please generate general and universal rules; the rules should not reference any specific item or tool! You need to generalize across various items or tools.
    - Generate only the rules under what conditions the action will fail.
    - While generating a rule, you also need to state how to check if a transition satisfies this rule. Please be specific as to which and how 'features' need to be checked

    """
    return prompt_template


