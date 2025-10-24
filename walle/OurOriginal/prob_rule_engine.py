import json
import re
from pyswip import Prolog
from .state_action_to_facts import state_action_to_facts
import os

# --- 確率計算メイン関数 ---
def compute_rule_probabilities(json_path, prolog_rule_path, output, text_index=None):
    """
    Prologルールと実際の遷移データ(JSON)をもとに、
    各 action_failed/1 ルールの「正解率（確率）」を求めて保存する
    """
    # --- JSONロード ---
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # --- アクション一覧とカウンタ初期化 ---
    actions = ["goto", "take", "put", "open", "close", "clean", "heat", "cool", "use"]
    action_totals = {a: 0 for a in actions}

    # --- Prologインスタンス生成 ---
    prolog = Prolog()
    prolog.consult(prolog_rule_path)

    # --- Prologルール抽出 ---
    with open(prolog_rule_path, "r", encoding="utf-8") as f:
        prolog_text = f.read()

    # 条件付きルールも含めてすべて抽出
    rule_pattern = re.compile(r"(action_failed\(.*?\)\s*:-\s*.*?\.)", re.DOTALL)
    rules = rule_pattern.findall(prolog_text)

    # 単純なヘッドだけのルール（:-がないもの）も追加
    simple_rules = re.findall(r"action_failed\(.*?\)\.", prolog_text)
    for r in simple_rules:
        if r not in rules:
            rules.append(r)

    # 重複除去
    rules = list(set(rules))

    # --- 各ルールのTrueカウント初期化 ---
    rule_counters = {r: 0 for r in rules}

    # --- 各サンプルごとに評価 ---
    for command, entries in all_data.items():
        for idx, sample in entries.items():
            state = sample["state"]
            action = sample["action"]
            action_name = action["action_name"].lower()

            if action_name not in actions:
                continue

            # アクション回数カウント
            action_totals[action_name] += 1

            # state/actionをPrologにassert
            facts = state_action_to_facts(sample)
            for fact in facts:
                prolog.assertz(fact.rstrip("."))

            # 各ルールを評価
            for rule in rules:
                query = rule.split(":-")[0].strip().rstrip(".") + "."
                try:
                    if list(prolog.query(query)):
                        rule_counters[rule] += 1
                except Exception:
                    # クエリが不正な場合はスキップ
                    pass

            # --- 状態をリセット ---
            prolog.retractall("reachable_location(_)")      # 到達可能場所
            prolog.retractall("items_in_location(_,_)")     # アイテム位置
            prolog.retractall("current_position(_)")        # 現在位置
            prolog.retractall("item_in_hand(_,_)")          # 手持ちアイテム
            prolog.retractall("location_status(_,_)")       # 状態
            prolog.retractall("action(_)")                  # 実行アクション

    # --- 確率計算 ---
    rule_probs = {}
    for rule, count in rule_counters.items():
        match = re.search(r"action_failed\((\w+)", rule)
        if not match:
            continue
        action_name = match.group(1)
        total = action_totals.get(action_name, 0)
        rule_probs[rule] = count / total if total > 0 else 0.0

    # --- 結果を保存 ---
    if text_index is None:
        text_index = json_path.split("/")[-1].split(".")[0]

    output_path = os.path.join(output, f"action_fail_prob_{text_index}.pl")
    with open(output_path, "w", encoding="utf-8") as f:
        for rule, prob in rule_probs.items():
            f.write(f"{prob:.2f} :: {rule}\n")

    print(f"\n✅ 確率付きルールを保存しました → {output_path}")

    # --- 結果出力（確認用） ---
    print("\nAction totals:")
    print(action_totals)

    print("\nRule true counts:")
    print(rule_counters)

    print("\nRule probabilities:")
    for rule, prob in rule_probs.items():
        print(f"{prob:.2f} :: {rule}")

    return rule_probs, action_totals, rule_counters


# --- メイン実行例 ---
if __name__ == "__main__":
    rule_probs, totals, counts = compute_rule_probabilities(
        json_path="./check3/all_D_inc.json",
        prolog_rule_path="./check3/all_generated_rules.pl",
        text_index="all_D_inc"
    )
