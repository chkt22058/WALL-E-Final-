import json
import os
from typing import List, Callable, Tuple, Dict

def greedy_rule_selection(D_inc: Dict, R_code: List[Callable], l: int, output_dir: str):
    """
    Greedy Algorithm for Maximum Coverage Problem
    """

    print("===貪欲アルゴリズム===")

    # 1. D_inc をタスクごとに展開してリストにする
    transitions_list = []
    for task_name, task_data in D_inc.items():
        for idx, trans in task_data.items():
            transitions_list.append(trans)

    # 遷移が空なら早期リターン
    if not transitions_list:
        print("⚠️ D_inc に遷移が存在しません。R_star = [] を返します。")
        return []

    # 2. a_ij 行列の作成
    a_matrix = []
    for rule in R_code:
        row = []
        for trans in transitions_list:
            state = trans["real_transitions"]["state"]
            action = trans["real_transitions"]["action"]

            SG_file_name = os.path.join(output_dir, "scene_graph.json")
            with open(SG_file_name, "r", encoding="utf-8") as f:
                scene_graph = json.load(f)

            _, rule_success, _ = rule(state, action, scene_graph)

            real_success = trans["real_transitions"]["action_result"].get("success", False)
            pred_success = trans["predicted_transitions"]["action_result"].get("success", False)

            covers = (pred_success != real_success) and (rule_success == real_success)
            row.append(1 if covers else 0)
        a_matrix.append(row)

    # 3. 貪欲選択
    R_star = []
    D_cov = set()

    while len(D_cov) < len(transitions_list):
        gains = []
        for i, row in enumerate(a_matrix):
            covered_by_rule = {j for j, val in enumerate(row) if val == 1}
            gain = len(covered_by_rule - D_cov)
            gains.append(gain)

        i_star = max(range(len(gains)), key=lambda i: gains[i])
        if gains[i_star] == 0:
            break

        R_star.append(R_code[i_star])
        covered_by_rule = {j for j, val in enumerate(a_matrix[i_star]) if val == 1}
        D_cov |= covered_by_rule

        if len(R_star) == l:
            break

    return R_star

