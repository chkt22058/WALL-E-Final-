import re
import json
from typing import Dict, Any
import openai

client = openai.OpenAI()

def parse_initial_observation(text: str) -> Dict[str, Any]:
    """
    自然文の観測情報から reachable_locations を抽出し、タスク文を除外して JSON 状態を構築。
    """

    # 「you see」から「.」までの最初の完全な観測文を抽出
    match = re.search(r"you see (.+?)(?:\.|$)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("観測情報が 'you see ...' 形式で見つかりませんでした。")

    object_list_text = match.group(1)

    # 正規表現で "a XXX" のようなオブジェクト表現を抽出
    objects = re.findall(r"a ([\w\s\d]+?)(?=,| and|\.|$)", object_list_text)

    # トリム + ソート（番号付きの要素を自然順に並べる）
    def natural_key(s):
        parts = s.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (s, 0)

    locations = sorted(set(obj.strip() for obj in objects), key=natural_key)

    # 状態フォーマットを構築
    obs_json = {
        "state": {
            "reachable_locations": locations,
            "items_in_locations": {},
            "item_in_hand": {
                "item_name": None,
                "status": None
            },
            "current_position": {
                "location_name": "middle_of_room",
                "status": None
            }
        }
    }
    return obs_json


def get_updated_state_from_observation(prev_obs: Dict, next_obs_text: str) -> Dict:
    """
    Based on the previous state (S_{t-1}) and current observation text (S_t),
    generates the next state S_t using an LLM.
    """

    prev_obs_json = json.dumps(prev_obs, indent=4)

    # Construct the prompt
    prompt = f"""
    You are an intelligent state updater for an AI agent operating in a simulated environment (like ALFWorld).
    Your task is to accurately update the provided 'previous_state' JSON object based on the 'new_observation' text.

    **Strict Rules for State Update Logic:**

    1.  **Field Completeness:**
        -   The returned JSON MUST always contain the 'state' object and ALL its sub-fields: 'reachable_locations', 'items_in_locations', 'item_in_hand', and 'current_position'.
        -   Do NOT omit any field, even if its value has not changed.

    2.  **Preservation of Existing Values:**
        -   If a field or its content is not explicitly mentioned or clearly changed in the 'new_observation', its value should be carried over directly from the 'previous_state'.

    3.  **'current_position' Update:**
        -   If the observation explicitly states "You arrive at [location name].", update 'current_position.location_name' to that [location name].
        -   If the observation explicitly states that the arrived location has a specific status (e.g., "The [location name] is closed.", "The [location name] is open."), update 'current_position.status' to that status (e.g., "closed", "open").
        -   If no specific status for the location is mentioned in the current observation, 'current_position.status' should be set to "null".
        -   If no movement is indicated in the observation, 'current_position' should be carried over from the 'previous_state'.

    4.  **'item_in_hand' Update:**
        -   If the observation states "You take [item name] from [location name]." or "You are now holding a [item name].", update 'item_in_hand.item_name' to that [item name].
        -   If the observation states "You put [item name] in/on [location name]." or "You put down [item name].", or implies an empty hand, set 'item_in_hand.item_name' to "null" and its 'status' to "null".
        -   If an item has a status (e.g., "dirty", "clean", "heated", "cooled"), update 'item_in_hand.status' accordingly (e.g., "a dirty potato" means status: "dirty").

    5.  **'items_in_locations' Update (MOST CRITICAL):**
        -   Each location in 'items_in_locations' must be a JSON object with two keys: "items" (a list of item names) and "status" (status of the location itself, e.g., "open", "closed", or "null").
        -   **Update for Current Location's Visible Items:**
            -   If the observation explicitly lists items on a surface, replace the list of items for that specific location with the newly observed items.
        -   **Item Movement due to Actions:**
            -   If a "take [item] from [location]" action occurred, remove that item from the list in that location.
            -   If a "put [item] in/on [location]" action occurred, add that item to the list in that location.
            -   For actions that change item status (e.g., "clean [item] with [tool]"), update the status of the item within the location or in hand.
        -   **Location Status:**
            -   If the observation mentions the status of a location (e.g., "cabinet 1 is closed"), update 'status' for that location in 'items_in_locations'.
            -   For locations not mentioned, preserve the previous 'status'.
        -   **New Locations:**
            -   If a location appears for the first time, initialize it in 'items_in_locations' with an empty "items" list and "status": "null".
            -   If a location appears in the current observation (current_position) but does not exist in 'items_in_locations', initialize it.
        -   **Preservation Rule:**
            -   Always retain items and status for locations not explicitly updated by the current observation.

    6.  **'reachable_locations' Preservation:**
        -   The 'reachable_locations' list should be carried over completely from the 'previous_state', unless the observation explicitly indicates a change.

    ---

    Here is the previous state (JSON):
    {prev_obs_json}

    Here is the new observation (Text):
    "{next_obs_text.strip()}"

    Following all the strict rules above, return the **entire updated state** in **strict valid JSON** format.
    Each location in 'items_in_locations' must be an object with "items" and "status".
    Do NOT include any explanations, comments, or additional text outside the JSON object.

    Expected JSON Schema:

    {{
        "state": {{
            "reachable_locations": [...],
            "items_in_locations": {{
                "location name": {{
                    "items": ["item 1", "item 2", "item_with_status 1"],
                    "status": "..." or None
                }}
            }},
            "item_in_hand": {{
                "item_name": "..." or None,
                "status": "..." or None
            }},
            "current_position": {{
                "location_name": "..." or None,
                "status": "..." or None
            }}
        }}
    }}
    """

    # OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4o", # Consider upgrading to "gpt-4o" if issues persist
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that accurately updates environment state JSON based on new observations and strict rules."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0 # Keep temperature at 0 for deterministic and accurate updates
    )

    # Get response
    content = response.choices[0].message.content

    # JSON parsing and error handling
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(" JSON parse error:", e)
        print(" Raw content from GPT:\n", content)
        raise


# ============================================================================================================
#obs_text = """
#You are in the middle of a room. Looking quickly around you, you see a cabinet 20, a cabinet 19, a cabinet 18,
#a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10,
#a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2,
#a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2,
#a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1,
#a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

#Your task is to: put a clean potato in countertop.
#"""

#obs_state = parse_initial_observation(obs_text)

# JSON整形出力
#print(json.dumps(obs_state, indent=4))
#with open("./check.json", "w", encoding="utf-8") as f:
#    json.dump(obs_state, f, indent=4, ensure_ascii=False)

#with open("./check.json", 'r', encoding='utf-8') as f:
#    obs_state = json.load(f)

# ============================================================================================================

# 新しい観測（例：S₁）
#obs1_text = """
#You arrive at countertop 3. On the countertop 3, you see a apple 3, a bowl 2, a butterknife 2,
#a butterknife 1, a houseplant 1, a knife 3, a potato 2, a spatula 3, and a statue 1.
#"""

# 状態更新
#obs1_state = get_updated_state_from_observation(obs_state, obs1_text)

# 結果確認
#print(json.dumps(obs1_state, indent=4))
#with open("./check2.json", "w", encoding="utf-8") as f:
#    json.dump(obs1_state, f, indent=4, ensure_ascii=False)

# ============================================================================================================