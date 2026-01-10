import re
import json
from typing import Dict, Any
import openai
import textwrap

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
    prompt = textwrap.dedent(f"""
    You are an intelligent state updater for an AI agent operating in a simulated environment (like ALFWorld).
    Your task is to accurately update the provided 'previous_state' JSON object based on the 'new_observation' text.

    **Strict Rules for State Update Logic:**

    1.  **Field Completeness:**
        -   The returned JSON MUST always contain the 'state' object and ALL its sub-fields: 'reachable_locations', 'items_in_locations', 'item_in_hand', and 'current_position'.
        -   Do NOT omit any field.

    2.  **Preservation of Existing Values:**
        -   If a field or its content is not explicitly mentioned or clearly changed in the 'new_observation', its value should be carried over directly from the 'previous_state'.

    3.  **'current_position' Update:**
        -   If the observation explicitly states "You arrive at [location name].", update 'current_position.location_name' to that [location name].
        -   If the observation explicitly states that the arrived location has a specific status (e.g., "The [location name] is closed."), update 'current_position.status'.
        -   If no specific status is mentioned, 'current_position.status' should be set to null.
        -   **Note:** 'current_position' does NOT store adjacent information anymore.

    4.  **'item_in_hand' Update:**
        -   If "You take [item]..." or "holding [item]": update 'item_in_hand.item_name'.
        -   If "You put..." or empty hand implied: set 'item_in_hand.item_name' to null.
        -   Update 'status' (e.g., "dirty", "heated") if mentioned.

    5.  **'items_in_locations' Update (CRITICAL for Items & Topology):**
        -   Each location object MUST have three keys:
            1. "items" (list of strings)
            2. "status" (string or null)
            3. "adjacent" (list of strings or null)

        -   **Standard Item/Status Updates:**
            -   Update "items" list if observation shows items at a location.
            -   Update "status" (open/closed) if mentioned.
            -   Reflect "take"/"put" actions in the "items" lists.

        -   **Topology ('adjacent') Update Logic (The "Look" Rule):**
            -   **Trigger:** If the observation is from a 'look' action (e.g., "**You are facing the [location A], and [location B]...**").
            -   **Target:** Identify the entry in 'items_in_locations' that matches the `current_position.location_name`.
            -   **Action:**
                1. Extract all locations mentioned in the "You are facing..." text.
                2. Exclude the `current_position.location_name` itself from this list.
                3. The remaining locations are the "adjacent" locations.
                4. **Update the 'adjacent' field** for the CURRENT location entry only:
                   - If neighbors found: Set "adjacent": ["loc 1", "loc 2"].
                   - If NO neighbors found: Set "adjacent": ["No_other_locations"].
            -   **IMPORTANT:** Do NOT create new keys in `items_in_locations` for the neighbors found in step 3. Only store their names as strings in the `adjacent` list of the current location.

        -   **Initialization (Creation of New Location Entries):**
            -   **Condition:** You must ONLY create a new key (entry) in `items_in_locations` IF the observation says "You arrive at [location name]" (indicating the agent is physically there).
            -   **Logic:** If the agent arrives at a new location not yet in the JSON, initialize it with:
                `{{"items": [], "status": null, "adjacent": null}}`
            -   **STRICT PROHIBITION:** Do NOT create a new key in `items_in_locations` just because a location name appears in the "adjacent" list or "facing" text. If the agent has not visited it yet, it should NOT exist as a key in `items_in_locations`.

    6.  **'reachable_locations' Preservation:**
        -   Carry over from 'previous_state' unless explicitly changed.

    7.  **NULL Value Handling:**
        -   Use JSON `null` (not string "null") for unknown/empty values.

    **Non-Specific Observation Rule:**
        - If 'new_observation' is "Nothing happens.", return 'state' IDENTICAL to 'previous_state'.

    ---

    Here is the previous state (JSON):
    {prev_obs_json}

    Here is the new observation (Text):
    "{next_obs_text.strip()}"

    Following all strict rules, return the **entire updated state** in **strict valid JSON**.
    """)

    # OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that accurately updates environment state JSON based on new observations and strict rules. You MUST use JSON null values (not the string 'null') for empty/none values."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
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