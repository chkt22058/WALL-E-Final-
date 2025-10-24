from pyswip import Prolog
import re
from state_action_to_facts import normalize_name, state_action_to_facts, save

# --- Prolog でルール追加＆判定 ---
def check_action_failed_take_pyswip(sample):
    prolog = Prolog()
    facts_str = state_action_to_facts(sample)

    for f in facts_str.splitlines():
        f = f.strip()
        if f.endswith('.'):
            f = f[:-1]
        prolog.assertz(f)

    # --- ルール追加 ---
    prolog.assertz(
        "action_failed(put(Item, Receptacle)) :- current_position(Receptacle), \+ items_in_location(Item, Receptacle), \+ item_in_hand(Item, in_hand)"
    )

    Item = normalize_name(sample["action"]["args"]["obj"])
    Location = normalize_name(sample["action"]["args"]["recep"])

    query = f"action_failed(take({Item}, {Location}))"
    result = list(prolog.query(query))

    return len(result) > 0

# --- サンプル JSON ---
sample = {
            "state": {
                "reachable_locations": [
                    "bathtubbasin 1",
                    "cabinet 1",
                    "cabinet 2",
                    "cabinet 3",
                    "cabinet 4",
                    "cabinet 5",
                    "countertop 1",
                    "garbagecan 1",
                    "handtowelholder 1",
                    "handtowelholder 2",
                    "sinkbasin 1",
                    "toilet 1",
                    "toiletpaperhanger 1",
                    "towelholder 1"
                ],
                "items_in_locations": {
                    "cabinet 1": {
                        "items": [],
                        "status": "closed"
                    },
                    "cabinet 2": {
                        "items": [],
                        "status": "closed"
                    },
                    "countertop 1": {
                        "items": [
                            "candle 1",
                            "soapbottle 1",
                            "spraybottle 3"
                        ],
                        "status": None
                    }
                },
                "item_in_hand": {
                    "item_name": "cloth 1",
                    "status": None
                },
                "current_position": {
                    "location_name": "countertop 1",
                    "status": None
                }
            },
            "action": {
                "action_name": "put",
                "args": {
                    "obj": "cloth 1",
                    "recep": "sinkbasin 1"
                }
            },
            "action_result": {
                "feedback": "Nothing happens.",
                "success": False,
                "suggestion": ""
            }
        }

# --- 判定 ---
facts = state_action_to_facts(sample)
print(facts)
failed = check_action_failed_take_pyswip(sample)
print(f"Action failed? {failed}")