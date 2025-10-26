import json
from pyswip import Prolog
import tempfile
from walle.OurOriginal.JSONtoFacts import state_action_to_facts, normalize_name  

with open("./check2/test_obs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prolog 初期化
prolog = Prolog()
prolog.consult("./check3/all_generated_rules.pl")

action_success_list = []

for task_name, transitions in data.items():
    print(f"Processing task: {task_name}")
    
    for step_id, step in transitions.items():
        # ステップの JSON を Prolog ファクトに変換
        facts_str = state_action_to_facts(step)
    
        for line in facts_str.splitlines():
            prolog.assertz(line[:-1])  # 末尾の . は削除

        # Action の Prolog 表現を作成
        action_name = step["action"]["action_name"]
        recep = step["action"]["args"].get("recep")
        obj = step["action"]["args"].get("obj")
        tool = step["action"]["args"].get("tool")

        action_name = normalize_name(action_name)
        recep = normalize_name(recep)
        obj = normalize_name(obj)
        tool = normalize_name(tool)

        if action_name == "goto":
            query = f"action_failed(goto({recep}))."
        elif action_name == "open":
            query = f"action_failed(open({recep}))."
        elif action_name == "take":
            query = f"action_failed(take({obj},{recep}))."
        elif action_name == "put":
            query = f"action_failed(put({obj},{recep}))."
        elif action_name == "close":
            query = f"action_failed(close({recep}))."
        else:
            query = None  # 他のアクションは無視する場合

        if query:
            result = list(prolog.query(query))
            print("query:", query, "=> result:", result)  # ← デバッグ用
            if result:
                # action_failed が True になったステップの success を集める
                action_success_list.append(step["action_result"]["success"])
        
        # ステップ処理後に削除
        prolog.retractall("action(_)")
        prolog.retractall("current_position(_)")
        prolog.retractall("location_status(_,_)")
        prolog.retractall("reachable_location(_)")
        prolog.retractall("items_in_location(_,_)")
        prolog.retractall("empty(_)")

# True / False のカウント
true_count = sum(1 for x in action_success_list if x)
false_count = sum(1 for x in action_success_list if not x)
prob_success = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0
prob_failure = false_count / (true_count + false_count) if (true_count + false_count) > 0 else 0

print("成功確率:", prob_success)
print("失敗確率:", prob_failure)