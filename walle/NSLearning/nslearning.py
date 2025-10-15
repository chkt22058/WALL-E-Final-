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


def NSLearning(real_trajectory, predicted_trajectory, outdir):
    coderule_dir = os.path.join(outdir, "CodeRule")
    os.makedirs(coderule_dir, exist_ok=True)

    input_dir = os.path.join(coderule_dir, "input")
    output_dir = os.path.join(coderule_dir, "output")
    check_dir = os.path.join(coderule_dir, "check")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    # RealTrajectory t-k:k
    # ====================================================================================================
    k = 5
    TopK_real_trajectory = get_recent_history_dict(real_trajectory, k)
    print("TopK確認")
    print(json.dumps(TopK_real_trajectory, ensure_ascii=False, indent=4))
    # ====================================================================================================

    # Stage1
    # ====================================================================================================
    D_cor, D_inc = implement_stage1(real_trajectory, predicted_trajectory)

    Dcor_file_name = os.path.join(check_dir, "D_cor.json")
    with open(Dcor_file_name, "w", encoding="utf-8") as f:
        json.dump(D_cor, f, indent=4, ensure_ascii=False)

    Dinc_file_name = os.path.join(check_dir, "D_inc.json")
    with open(Dinc_file_name, "w", encoding="utf-8") as f:
        json.dump(D_inc, f, indent=4, ensure_ascii=False)

    #with open("./CodeRule/check/D_cor.json", "w", encoding="utf-8") as f:
    #    json.dump(D_cor, f, indent=4, ensure_ascii=False)

    #with open("./CodeRule/check/D_inc.json", "w", encoding="utf-8") as f:
    #    json.dump(D_inc, f, indent=4, ensure_ascii=False)
    # ====================================================================================================


    # Stage2
    # ====================================================================================================

    # Action rules =======================================================================================

    AR_file_name = os.path.join(output_dir, "action_rules.json")
    AR_imp_file_name = os.path.join(output_dir, "action_rules_improve.json")
    AR = ActionRules(model="gpt-3.5-turbo")

    ar = AR.generate_ActionRules(TopK_real_trajectory, input_dir, output_dir)
    AR.save(AR_file_name, ar)
    
    improve_ar = AR.generate_ActionRulesImprove(TopK_real_trajectory, ar, input_dir)
    AR.save(AR_imp_file_name, improve_ar)
   
    # Knowledge graph =======================================================================================

    #KG = KnowledgeGraph(model="gpt-3.5-turbo")

    #kg = KG.generate_KnowledgeGraph(TopK_real_trajectory)
    #KG.save("./CodeRule/output/knowledge_graph.json", kg)

    # Scene Graph =======================================================================================

    sg = SceneGraph()
    sg.update_last_state(real_trajectory)
    SG_file_name = os.path.join(output_dir, "scene_graph.json")

    print("=== Scene Graph ===")
    sg.visualize()
    sg.save(SG_file_name)

    # ====================================================================================================
    """
    # ルールを学習しない(実験するとき)

     # 既存ルールのパス
    all_rules_path = "./CodeRule/all_code_rules.py"

    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(all_rules_path), exist_ok=True)

    existing_rules = load_rules_from_file(all_rules_path)

    return existing_rules
    """
    # ====================================================================================================

    # stage3: コードルールの作成
    # ====================================================================================================

    stage3 = STAGE3(model="gpt-4.1")

    with open(AR_imp_file_name, "r", encoding="utf-8") as f:
        action_rules = json.load(f)
    
    code_rule = stage3.generate_coderule(action_rules, input_dir)
    print("生成されたコードルールの型:")
    print(type(code_rule))
    print("生成されたコードルール:")
    print(code_rule)
    
    CR_file_name = os.path.join(output_dir, "code_rules.py")

    stage3.save(CR_file_name, code_rule)
    
    print("===LLMが生成したコードルール===")
    print(code_rule)


    # Stage4
    # ====================================================================================================

    # 新規生成されたルールをロード
    print("\n===新規ルールをロード===")
    R_code_new = load_rules_from_file(CR_file_name)
    print(f"新規ルール数: {len(R_code_new)}")
    for rule in R_code_new:
        print(f"  - {rule.__name__} (source saved: {hasattr(rule, '__source_code__')})")
    
    # 過去に剪定されたルールをロード
    R_code_past = []
    pruned_path = os.path.join(output_dir, "code_rules_pruned.py")
    if os.path.exists(pruned_path):
        print("\n===過去の剪定済みルールをロード===")
        R_code_past = load_rules_from_file(pruned_path)
        print(f"過去のルール数: {len(R_code_past)}")
        for rule in R_code_past:
            print(f"  - {rule.__name__} (source saved: {hasattr(rule, '__source_code__')})")
    
    # 過去のルールと新規ルールを統合(重複除去)
    R_code_combined = merge_rules(R_code_past, R_code_new)
    print(f"\n===統合後のルール数: {len(R_code_combined)}===")
    
    # 統合されたルールセットから剪定
    R_star = greedy_rule_selection(D_inc, R_code_combined, 5, output_dir)

    print("\n===選ばれたルール===")
    for rule in R_star:
        has_source = hasattr(rule, '__source_code__')
        print(f"  - {rule.__name__} (source: {'✓' if has_source else '✗'})")

    # ✅ 剪定後のルールを保存(引数は2つだけ!)
    save_pruned_rules(R_star, pruned_path)

    # return R_star

    # ======================================================================

    # 既存ルールのパス
    all_rules_path = "./CodeRule/all_code_rules.py"

    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(all_rules_path), exist_ok=True)

    # 既存ルールを読み込み
    existing_rules = []
    if os.path.exists(all_rules_path):
        existing_rules = load_rules_from_file(all_rules_path)
        print(f"✓ 既存ルール数: {len(existing_rules)}")
    else:
        print("✓ 既存ルールなし（新規作成）")

    # 今回剪定されたルール(R_star)と既存ルールをマージ
    merged_rules = merge_rules(existing_rules, R_star)
    print(f"✓ マージ後のルール数: {len(merged_rules)}")

    # マージした結果を保存
    save_pruned_rules(merged_rules, all_rules_path)

    return merged_rules  

    # ====================================================================================================

    

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