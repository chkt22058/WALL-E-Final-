import json
from openai import OpenAI

def transform_kg_prompt(raw_trajectory_data: dict) -> list:
    """
    生のタクト軌跡データ (traj_real.json形式) を
    Knowledge Graph抽出プロンプトの要求する形式に変換します。

    Args:
        raw_trajectory_data (dict): traj_real.jsonから読み込まれたデータ。

    Returns:
        list: 変換された遷移のリスト。各要素は {'inital_state': '...', 'action': '...', 'action_result': '...'} 形式。
    """
    transform_transitions = []

    for task_name, steps in raw_trajectory_data.items():
        for step_data in steps:
            # === initial_state の文字列変換 ===
            state = step_data.get('state', {})
            current_pos = state.get('current_position', {}).get('location_name', 'unknown_location')
            item_in_hand = state.get('item_in_hand', {}).get('item_name', 'null')
            
            state_description_parts = []
            state_description_parts.append(f"Agent at {current_pos}")

            if item_in_hand and item_in_hand != "null":
                state_description_parts.append(f"holding {item_in_hand}")
            else:
                state_description_parts.append("hand empty")

            items_in_locations = state.get('items_in_locations', {})
            location_items_descriptions = []
            for loc, items in items_in_locations.items():
                if items: # アイテムが空でない場所のみ記述
                    location_items_descriptions.append(f"{loc} contains {', '.join(items)}")
            
            if location_items_descriptions:
                state_description_parts.append(". " + ". ".join(location_items_descriptions))
            else:
                state_description_parts.append(".") # アイテムがない場合もピリオドで締める

            initial_state_str = ", ".join(state_description_parts)

            # === action の文字列変換 ===
            action = step_data.get('action', {})
            action_type = action.get('action_type', 'unknown_action')
            action_object = action.get('object')
            action_in_location = action.get('in_location')

            action_str_parts = []
            action_str_parts.append(action_type)
            if action_object:
                action_str_parts.append(action_object)
            if action_in_location:
                action_str_parts.append(f"in {action_in_location}")
            
            action_str = " ".join(action_str_parts).strip()
            action_str = action_str.capitalize() # 先頭を大文字にする

            # === action_result の文字列変換 ===
            action_result_bool = step_data.get('action_result', False)
            action_result_str = "True" if action_result_bool else "False"

            transform_transitions.append({
                'inital_state': initial_state_str,
                'action': action_str,
                'action_result': action_result_str
            })
    
    return transform_transitions


def KG_generate_llm_prompt_with_transitions(transitions_data: list) -> str:
    """
    LLM入力用のプロンプトと、変換された遷移データを結合した文字列を生成する関数。

    Args:
        transitions_data (list): llm_input_traj.jsonから読み込まれた変換済みの遷移データ。
        existing_rules (list): 既存のルールの配列。提供されない場合は空のリストを使用。

    Returns:
        str: LLMに投げるための結合されたプロンプトテキスト。
    """

    # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
    transitions_json_str = json.dumps(transitions_data, indent=4, ensure_ascii=False)

    prompt_template = f"""
    You are a helpful assistant with inductive reasoning. Given the history trajectory,
    including action and observation, you need to reflect on the action execution results
    and identify and extract prerequisite or feasibility constraints, that is, discover
    when an action or item creation requires the presence of certain materials, resources, or other items.

    We define the Knowledge Graph as:
    {{{{
        "V": "the set of entities (e.g., items, materials, location-specific objects, or abstract concepts)",
        "E": "the set of directed edges, each capturing a relationship or prerequisite among entities"
    }}}}

    An edge takes the form:
    (u, v, label),
    where u and v are entities in V, and label indicates how u relates to v
    (for example, 'requires', 'consumes', 'collects', etc.).
    
    I will give you an array of transitions:
    {transitions_json_str}

    You should ONLY respond in the following format:
    [
    {{'u':'entity_u', 'v':'entity_v', 'label':{{'relation':'...', 'quantity':'...'}}}},
    {{'u':'entity_u', 'v':'entity_v', 'label':{{'relation':'...', 'quantity':'...'}}}},
    ...
    ]
    example:
    [
    {{'u':'wooden_sword', 'v':'table', 'label':{{'relation':'requires', 'quantity':None}}}},
    {{'u':'table', 'v':'wood', 'label':{{'relation':'consumes', 'quantity':'2'}}}}
    ]
    Instructions:
    - Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, **no single quotes**, etc.
    """
    return prompt_template


if __name__ == "__main__":

    ### STEP1: 入力プロンプトの生成---------------------------------------------------
    with open("../buffer_fact/traj_real.json", "r", encoding="utf-8") as f:
        raw_trajectory_data = json.load(f)

    # データを変換
    transform_data = transform_kg_prompt(raw_trajectory_data)
    output_filename = "check/transform_KG.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(transform_data, f, indent=4, ensure_ascii=False)

    # プロンプトの作成
    llm_prompt = generate_llm_prompt_with_transitions(transform_data)
    with open("input/KG_llm_prompt.txt", "w", encoding="utf-8") as f:
        f.write(llm_prompt)
    
    ### STEP2: LLMに入力---------------------------------------------------
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant with inductive reasoning."},
            {"role": "user", "content": llm_prompt},
        ],
        response_format={"type": "json_object"}
    )
    generate_response = response.choices[0].message.content
    action_rules_data = json.loads(generate_response) 
    with open("output/knowledge_graph.json", 'w', encoding='utf-8') as f:
        json.dump(action_rules_data, f, ensure_ascii=False, indent=4)
    """