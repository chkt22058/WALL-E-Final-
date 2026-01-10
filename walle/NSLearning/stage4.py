import json
import os
from typing import List, Callable, Tuple, Dict, Union

def greedy_rule_selection(D_inc: Dict, D_cor: Dict, R_code: List[Callable], l: int, out_dir: str):
    """
    Greedy Algorithm for Maximum Coverage Problem (WALL-E 2.0 Implementation)
    
    論文の仕様に基づく2段階プロセス:
    1. Validity Check (Appendix E.3):
       成功事例(D_cor)に対して「失敗」と誤予測するルールを「無効(Invalid)」として除外する。
       
    2. Maximum Set Coverage (Section 3.1.4, Algorithm 1):
       失敗事例(D_inc)を最も多く正しく予測(カバー)できるルールを貪欲法で選定する。
    """

    print("\n=== Stage 4: Code Rule Pruning (WALL-E 2.0 Logic) ===")
    
    # ---------------------------------------------------------
    # データの前処理: 辞書形式をフラットなリストに変換
    # ---------------------------------------------------------
    
    # 1. 成功事例 (D_cor) の展開
    success_transitions = []
    if D_cor:
        if isinstance(D_cor, dict):
            for task_data in D_cor.values():
                if task_data:
                    success_transitions.extend(task_data.values())
        elif isinstance(D_cor, list):
            success_transitions = D_cor
            
    # 2. 失敗事例 (D_inc) の展開
    inc_transitions_list = []
    for task_data in D_inc.values():
        if task_data:
            inc_transitions_list.extend(task_data.values())

    if not inc_transitions_list:
        print("⚠️ D_inc (失敗事例) が存在しません。ルール選定をスキップします。")
        return []

    initial_rule_count = len(R_code)
    valid_R_code = list(R_code) # 候補リストのコピー

    # ---------------------------------------------------------
    # Step 1: Validity Check (Appendix E.3)
    # 「実際には成功しているのに、失敗すると予測するルール」を排除
    # ---------------------------------------------------------
    if success_transitions:
        print(f"Checking validity against {len(success_transitions)} successful transitions...")
        temp_valid_rules = []
        
        for rule in valid_R_code:
            is_invalid = False
            
            for trans in success_transitions:
                # データの構造の揺れに対応 (action_resultの位置)
                real_success = False
                if "action_result" in trans:
                    real_success = trans["action_result"].get("success", False)
                elif "real_transitions" in trans:
                    real_success = trans["real_transitions"].get("action_result", {}).get("success", False)
                elif "step_data" in trans: # ラップされている場合
                     real_success = trans["step_data"].get("action_result", {}).get("success", False)
                     trans = trans["step_data"] # 中身を取り出す

                # 万が一、失敗データが混入していたらスキップ (成功データとの矛盾のみを検証するため)
                if not real_success:
                    continue

                # 必要なデータを取得
                state = trans.get("state", {})
                action = trans.get("action", {})
                # new_nslearning.py で注入されたシーングラフを使用
                sg = trans.get("scene_graph", {"nodes": [], "edges": []})

                try:
                    # ルールを実行 [cite: 245]
                    # return: feedback, success_flag, suggestion
                    _, rule_success_flag, _ = rule(state, action, sg)

                    # 【論文の核心ロジック】
                    # "predicting failure when the transition actually succeeds" -> Invalid 
                    # ルールが「False (失敗)」と判定したのに、実際は「True (成功)」だった場合
                    if rule_success_flag is False:
                        print(f"  [削除] {rule.__name__}: 成功事例を『失敗』と誤判定しました (False Positive).")
                        is_invalid = True
                        break # 1つでも矛盾があれば即アウト
                
                except Exception as e:
                    print(f"  [警告] {rule.__name__} 実行エラー: {e}")
                    # 安全のためエラーが出るルールは除外
                    is_invalid = True
                    break

            if not is_invalid:
                temp_valid_rules.append(rule)
        
        valid_R_code = temp_valid_rules
        print(f"Validity Check完了: {initial_rule_count} -> {len(valid_R_code)} ルールが通過")

    if not valid_R_code:
        print("⚠️ 有効なルールが残りませんでした。空リストを返します。")
        return []

    # ---------------------------------------------------------
    # Step 2: Maximum Coverage (Section 3.1.4)
    # 失敗事例 (D_inc) をカバーするルールを貪欲法で選定
    # ---------------------------------------------------------
    print(f"Solving Maximum Coverage for {len(inc_transitions_list)} failed transitions...")

    # a_ij 行列の作成 (行:ルール, 列:失敗遷移) [cite: 209]
    a_matrix = []
    
    for rule in valid_R_code:
        row = []
        for trans in inc_transitions_list:
            # データ構造の正規化
            if "step_data" in trans: trans = trans["step_data"]

            state = trans.get("state", {})
            action = trans.get("action", {})
            current_sg = trans.get("scene_graph", {"nodes": [], "edges": []})

            try:
                _, rule_success, _ = rule(state, action, current_sg)
            except:
                rule_success = True # エラーならカバーできていないとみなす

            # 実環境の結果
            if "action_result" in trans:
                real_success = trans["action_result"].get("success", False)
            else:
                real_success = trans.get("real_transitions", {}).get("action_result", {}).get("success", False)
            
            # カバーの定義:
            # 「実環境で失敗」かつ「ルールも失敗と予測」した場合にカバーとみなす 
            # (正しい失敗予測 = 1, それ以外 = 0)
            pred_success = rule_success
            covers = (not real_success) and (not pred_success)
            
            row.append(1 if covers else 0)
        a_matrix.append(row)

    # 貪欲選択ロジック (Algorithm 1) 
    R_star = []     # 選択されたルールセット
    D_cov = set()   # カバーされた遷移のインデックス集合

    # ルール数上限 l または 全カバーするまでループ
    while len(R_star) < l and len(D_cov) < len(inc_transitions_list):
        gains = []
        for i, row in enumerate(a_matrix):
            # すでに選択済みのルールは除外 (gain = -1)
            if valid_R_code[i] in R_star:
                gains.append(-1)
                continue

            # 新たにカバーできる遷移の数を計算 (Marginal Gain) [cite: 223]
            covered_indices = {j for j, val in enumerate(row) if val == 1}
            gain = len(covered_indices - D_cov)
            gains.append(gain)

        # 最大ゲインを持つルールを選択 [cite: 224]
        i_star = max(range(len(gains)), key=lambda i: gains[i])
        
        # ゲインが0なら、これ以上役に立つルールはないので終了 [cite: 225]
        if gains[i_star] <= 0:
            break 

        selected_rule = valid_R_code[i_star]
        R_star.append(selected_rule)
        
        # カバー集合を更新 [cite: 231]
        new_covered = {j for j, val in enumerate(a_matrix[i_star]) if val == 1}
        D_cov |= new_covered

    # 結果の出力と保存
    coverage_percentage = (len(D_cov) / len(inc_transitions_list)) * 100 if inc_transitions_list else 0
    text = f"最終選択ルール数: {len(R_star)}, カバー率: {coverage_percentage:.1f}% ({len(D_cov)}/{len(inc_transitions_list)})"
    print(text)

    text_dir = os.path.join(out_dir, "CoverRate")
    os.makedirs(text_dir, exist_ok=True)
    with open(os.path.join(text_dir, "cover_rate.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    
    return R_star