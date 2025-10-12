import json
from openai import OpenAI
import matplotlib.pyplot as plt
import os
import inspect
import re
import importlib.util
import sys
import ast


from .stage1 import *
from .new_action_rules import *
from .new_knowledge_graph import *
from .new_scene_graph import *
from .stage3 import *
from .stage4 import *

from networkx.readwrite import json_graph


def New_NSLearning(real_trajectory, predicted_trajectory, outdir, task_name):
    coderule_dir = os.path.join(outdir, "CodeRule")
    os.makedirs(coderule_dir, exist_ok=True)

    input_dir = os.path.join(coderule_dir, "input")
    output_dir = os.path.join(coderule_dir, "output")
    check_dir = os.path.join(coderule_dir, "check")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    # Stage1
    # ====================================================================================================
    # 既存D_incorrectのパス
    all_D_inc_path = "./CodeRule/all_D_inc.json"
    os.makedirs(os.path.dirname(all_D_inc_path), exist_ok=True)

    D_cor, D_inc = implement_stage1(real_trajectory, predicted_trajectory, task_name)

    Dcor_file_name = os.path.join(check_dir, "D_cor.json")
    with open(Dcor_file_name, "w", encoding="utf-8") as f:
        json.dump(D_cor, f, indent=4, ensure_ascii=False)

    Dinc_file_name = os.path.join(check_dir, "D_inc.json")
    with open(Dinc_file_name, "w", encoding="utf-8") as f:
        json.dump(D_inc, f, indent=4, ensure_ascii=False)
    

    if os.path.exists(all_D_inc_path):
        with open(all_D_inc_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {} 

    for task_name, task_data in D_inc.items():
        existing[task_name] = task_data
        print(f"✓ '{task_name}' を追加しました。")

    # 保存
    with open(all_D_inc_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

    print(f"✓ all_D_inc.json に '{task_name}' のデータを追加しました")


    # ====================================================================================================


    # Stage2
    # ====================================================================================================

    # Action rules =======================================================================================
    k = 3  # スライドウィンドウ幅
    max_step = max(int(key.split("_")[1]) for key in real_trajectory.keys() if key.startswith("state_"))
    
    AR = ActionRules(model="gpt-3.5-turbo")
    AR_file_name = os.path.join(output_dir, "action_rules.json")
    AR_imp_file_name = os.path.join(output_dir, "action_rules_improve.json")

    for end_step in range(max_step, k - 2, -1):
        # 各スライドウィンドウに対して TopK データを作成
        TopK_real_trajectory = get_history_window_dict(real_trajectory, end_step - k + 1, end_step)
        print(f"\n=== Window: state_{end_step - k + 1}〜state_{end_step} ===")
        # print(json.dumps(TopK_real_trajectory, ensure_ascii=False, indent=4))

        ar = AR.generate_ActionRules(TopK_real_trajectory, input_dir, output_dir)
        AR.save(AR_file_name, ar)

        improve_ar = AR.generate_ActionRulesImprove(TopK_real_trajectory, ar, input_dir)
        AR.save(AR_imp_file_name, improve_ar)
   
    # Knowledge graph =======================================================================================

    #KG = KnowledgeGraph(model="gpt-3.5-turbo")

    #kg = KG.generate_KnowledgeGraph(TopK_real_trajectory)
    #KG.save("./CodeRule/output/knowledge_graph.json", kg)


    # stage3: コードルールの作成
    # ====================================================================================================
    is_available_rule = False

    stage3 = STAGE3(model="gpt-4.1")

    with open(AR_imp_file_name, "r", encoding="utf-8") as f:
        action_rules = json.load(f)
    
    count = 0
    while not is_available_rule:
        print("コードルールリプランカウント:", count)
        code_rule = stage3.generate_coderule(action_rules, input_dir)
        is_available_rule = stage3.verify_code_rule_boolean(code_rule)
        count += 1
    
    CR_file_name = os.path.join(output_dir, "code_rules.py")
    stage3.save(CR_file_name, code_rule)
    
    print("===LLMが生成したコードルール===")
    print(code_rule)


    # stage4(修正版)
    # ====================================================================================================

    # 新規生成されたルールをロード
    print("\n===新規ルールをロード===")
    R_code_new = load_rules_from_file(CR_file_name)
    print(f"新規ルール数: {len(R_code_new)}")
    for rule in R_code_new:
        print(f"  - {rule.__name__} (source saved: {hasattr(rule, '__source_code__')})")
    
    # --------------------------------------------------------------------------------------

    # 既存ルールのパス
    all_rules_path = "./CodeRule/all_code_rules.py"
    os.makedirs(os.path.dirname(all_rules_path), exist_ok=True)

    # 既存ルールを読み込み
    existing_rules = []
    if os.path.exists(all_rules_path):
        existing_rules = load_rules_from_file(all_rules_path)
        print(f"✓ 既存ルール数: {len(existing_rules)}")
    else:
        print("✓ 既存ルールなし（新規作成）")
    
    # --------------------------------------------------------------------------------------
    
    # 今回生成されたルールと既存ルールをマージ
    merged_rules = merge_rules(existing_rules, R_code_new)
    print(f"✓ マージ後のルール数: {len(merged_rules)}")

    # --------------------------------------------------------------------------------------

    # 統合済み D_incをロード
    if os.path.exists(all_D_inc_path):
        with open(all_D_inc_path, "r", encoding="utf-8") as f:
            merged_D_inc = json.load(f)
    else:
        merged_D_inc = {}  # 無ければ空辞書

    # 統合済み D_inc を使ってルール選定
    R_star = greedy_rule_selection(merged_D_inc, merged_rules, 5, output_dir)

    print("\n===選ばれたルール===")
    for rule in R_star:
        has_source = hasattr(rule, '__source_code__')
        print(f"  - {rule.__name__} (source: {'✓' if has_source else '✗'})")
    
    save_pruned_rules(R_star, all_rules_path)

    return R_star

    # ===================================================================================================

    

# 実軌跡の上位K件の辞書作成
def get_recent_history_dict(data: dict, k: int) -> dict:
    """
    辞書形式の履歴データから、直近のK件のステップを抽出します。
    """
    # 履歴の総ステップ数を計算
    max_step = -1
    for key in data.keys():
        if key.startswith("state_"):
            try:
                step_num = int(key.split("_")[1])
                if step_num > max_step:
                    max_step = step_num
            except (ValueError, IndexError):
                continue

    # 抽出するステップの開始インデックスを計算
    start_step = max(0, max_step - k + 1)

    recent_data = {}

    # 直近K件の履歴を抽出
    for i in range(start_step, max_step + 1):
        state_key = f"state_{i}"
        action_key = f"action_{i}"
        action_result_key = f"action_result_{i}"

        # キーが存在する場合のみ辞書に追加
        if state_key in data:
            recent_data[state_key] = data[state_key]
        if action_key in data:
            recent_data[action_key] = data[action_key]
        if action_result_key in data:
            recent_data[action_result_key] = data[action_result_key]

    return recent_data


def load_rules_from_file(filepath: str):
    """
    ファイルからルールをロードし、元のソースコードも関数に保存
    """
    with open(filepath, "r", encoding="utf-8") as f:
        file_content = f.read()
    
    spec = importlib.util.spec_from_file_location("code_rules", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["code_rules"] = module
    spec.loader.exec_module(module)

    rules = []
    tree = ast.parse(file_content)
    lines = file_content.split('\n')
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("Rule_"):
            func = getattr(module, node.name, None)
            if func is None:
                continue
            
            # 関数のソースコードを抽出
            start = node.lineno - 1
            end = len(lines)
            
            for next_node in tree.body:
                if (isinstance(next_node, (ast.FunctionDef, ast.ClassDef)) and 
                    next_node.lineno > node.lineno):
                    end = next_node.lineno - 1
                    break
            
            while end > start and lines[end - 1].strip() == '':
                end -= 1
            
            # ✅ ソースコードを関数の属性として保存
            func.__source_code__ = '\n'.join(lines[start:end])
            
            rules.append(func)
    
    print(f"✓ Loaded {len(rules)} rules from {filepath}")
    return rules


def save_pruned_rules(R_star, filepath):
    """
    選択されたルールを保存（関数に保存されたソースコードを使用）
    
    Args:
        R_star: 選択された関数オブジェクトのリスト
        filepath: 保存先パス (code_rules_pruned.py)
    """
    saved = []
    failed = []
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Pruned code rules\n\n")
        
        for rule in R_star:
            if hasattr(rule, '__source_code__'):
                f.write(rule.__source_code__ + "\n\n")
                saved.append(rule.__name__)
                print(f"✓ Saved {rule.__name__}")
            else:
                failed.append(rule.__name__)
                print(f"✗ No source code for {rule.__name__}")
    
    print(f"\n{'='*50}")
    print(f"Pruned rules saved to {filepath}")
    print(f"✓ Successfully saved: {len(saved)}/{len(R_star)} rules")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")
    print(f"{'='*50}")


def merge_rules(past_rules, new_rules):
    """
    過去のルールと新規ルールを統合し、重複を除去する
    関数名が同じ場合は新規ルールを優先する
    """
    # 過去ルールの関数名セットを作成
    past_rule_names = {rule.__name__ for rule in past_rules}
    
    # 新規ルールの関数名セットを作成
    new_rule_names = {rule.__name__ for rule in new_rules}
    
    # 重複していない過去ルールを抽出
    unique_past_rules = [rule for rule in past_rules if rule.__name__ not in new_rule_names]
    
    # 過去ルール(重複除外) + 新規ルールを結合
    combined_rules = unique_past_rules + new_rules
    
    print(f"\n過去ルール: {len(past_rules)}件")
    print(f"新規ルール: {len(new_rules)}件")
    print(f"重複除外後: {len(combined_rules)}件")
    
    return combined_rules


def get_history_window_dict(data: dict, start_step: int, end_step: int) -> dict:
    """指定された範囲のstate/action/action_resultを抽出"""
    result = {}
    for i in range(start_step, end_step + 1):
        state_key = f"state_{i}"
        action_key = f"action_{i}"
        action_result_key = f"action_result_{i}"
        if state_key in data:
            result[state_key] = data[state_key]
        if action_key in data:
            result[action_key] = data[action_key]
        if action_result_key in data:
            result[action_result_key] = data[action_result_key]
    return result
