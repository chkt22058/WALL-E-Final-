import re

def normalize_name(name: str) -> str:
    """
    Prologで扱いやすいように、スペースをアンダースコアに変換し、
    すべて小文字に揃える。
    Noneが来た場合は '_' を返す。
    """
    if name is None:
        return "_"
    return re.sub(r"\s+", "_", str(name).strip().lower())

def state_action_to_facts(sample: dict) -> str:
    facts = []
    state = sample["state"]
    action = sample["action"]

    # --- Action ---
    action_name = normalize_name(action.get("action_name"))
    args = action.get("args", {})

    obj = normalize_name(args.get("obj"))
    recep = normalize_name(args.get("recep"))
    tool = normalize_name(args.get("tool"))
    

    # アクションを一つの述語で表現
    if action_name == "goto":
        facts.append(f"action(goto({recep})).")
    elif action_name == "take":
        facts.append(f"action(take({obj}, {recep})).")
    elif action_name == "put":
        facts.append(f"action(put({obj}, {recep})).")
    elif action_name == "open":
        facts.append(f"action(open({recep})).")
    elif action_name == "close":
        facts.append(f"action(close({recep})).")
    elif action_name == "clean":
        facts.append(f"action(clean({obj}, {recep})).")
    elif action_name == "heat":
        facts.append(f"action(heat({obj}, {recep})).")
    elif action_name == "cool":
        facts.append(f"action(cool({obj}, {recep})).")
    elif action_name == "use":
        facts.append(f"action(use({tool})).")
    

    # --- State: current position ---
    if "current_position" in state:
        loc_info = state["current_position"]
        loc = normalize_name(loc_info["location_name"])
        facts.append(f"current_position({loc}).")
        status = loc_info.get("status")
        if status:
            facts.append(f"location_status({loc}, {normalize_name(status)}).")
        else:
            facts.append(f"location_status({loc}, null).")

    # --- State: item in hand ---
    if "item_in_hand" in state:
        item_info = state["item_in_hand"]
        item = normalize_name(item_info.get("item_name"))
        status = item_info.get("status")
        if item != "_":  # None の場合は Fact を作らない
            if status:
                facts.append(f"item_in_hand({item}, {normalize_name(status)}).")
            else:
                facts.append(f"item_in_hand({item}, null).")

        # 何も持っていない場合
        else:
            facts.append(f"item_in_hand(null, null).")


    # --- State: reachable locations ---
    if "reachable_locations" in state:
        for loc in state["reachable_locations"]:
            loc = normalize_name(loc)
            facts.append(f"reachable_location({loc}).")

    # --- State: items in locations ---
    items_in_location_generated = False
    empty_generated = False  # emptyフラグも追加

    if "items_in_locations" in state:
        items_in_locations = state["items_in_locations"]
    
        # items_in_locationsが空でない場合
        if items_in_locations:
            for location, info in items_in_locations.items():
                loc = normalize_name(location)
                items = info.get("items", [])

                # アイテムがある場合
                if items:
                    for item in items:
                        item = normalize_name(item)
                        facts.append(f"items_in_location({item}, {loc}).")
                        items_in_location_generated = True
                else:
                    # 空の棚を明示
                    facts.append(f"empty({loc}).")
                    empty_generated = True
            
                # 場所の status があれば追加
                loc_status = info.get("status")
                if loc_status:
                    facts.append(f"location_status({loc}, {normalize_name(loc_status)}).")

    # ダミーを追加
    if not items_in_location_generated:
        facts.append(f"items_in_location(null, null).")

    if not empty_generated:
        facts.append(f"empty(null).")

    return "\n".join(facts)
    # return facts

def save(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


if __name__ == '__main__':
    # サンプル入力
    sample = {
        "state": {
        "reachable_locations": [
            "bathtubbasin 1",
            "countertop 1",
            "drawer 1",
            "drawer 2",
            "drawer 3",
            "drawer 4",
            "drawer 5",
            "drawer 6",
            "drawer 7",
            "drawer 8",
            "garbagecan 1",
            "handtowelholder 1",
            "sinkbasin 1",
            "toilet 1",
            "toiletpaperhanger 1",
            "towelholder 1",
            "towelholder 2"
        ],
        "items_in_locations": {},
        "item_in_hand": {
            "item_name": None,
            "status": None
        },
        "current_position": {
            "location_name": "middle_of_room",
            "status": None
        }
    },
    "action": {"action_name": "goto", "args": {"recep": "drawer 1"}}
    }

    facts = state_action_to_facts(sample)
    print(facts)

    save("./check1/facts.pl", facts)

