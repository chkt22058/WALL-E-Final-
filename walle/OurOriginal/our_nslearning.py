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
from .ilasp_llm import *

from networkx.readwrite import json_graph

from .prob_rule_engine import compute_rule_probabilities


def Our_NSLearning(real_trajectory, predicted_trajectory, outdir, task_name, t_index):
    coderule_dir = os.path.join(outdir, "OurRule")
    os.makedirs(coderule_dir, exist_ok=True)

    input_dir = os.path.join(coderule_dir, "input")
    output_dir = os.path.join(coderule_dir, "output")
    check_dir = os.path.join(coderule_dir, "check")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    # RealTrajectory t-k:k
    # ====================================================================================================
    k = 3
    TopK_real_trajectory = get_recent_history_dict(real_trajectory, k)
    # ====================================================================================================

    # Stage1 D_incとD_corの作成
    # ====================================================================================================
    # 既存D_incorrectのパス
    all_D_inc_path = "./OurRule/all_D_inc.json"
    os.makedirs(os.path.dirname(all_D_inc_path), exist_ok=True)

    D_cor, D_inc = implement_stage1(real_trajectory, predicted_trajectory, task_name)

    # ✅ 両方の辞書をマージ
    D_all = {**D_cor, **D_inc}

    Dcor_file_name = os.path.join(check_dir, "D_cor.json")
    with open(Dcor_file_name, "w", encoding="utf-8") as f:
        json.dump(D_cor, f, indent=4, ensure_ascii=False)

    Dinc_file_name = os.path.join(check_dir, "D_inc.json")
    with open(Dinc_file_name, "w", encoding="utf-8") as f:
        json.dump(D_inc, f, indent=4, ensure_ascii=False)
    
    Dall_file_name = os.path.join(check_dir, "D_all.json")
    with open(Dall_file_name, "w", encoding="utf-8") as f:
        json.dump(D_all, f, indent=4, ensure_ascii=False)


    # ====================================================================================================


    # Actionルールの作成
    # ====================================================================================================
    All_AR_dir = "./OurRule"
    os.makedirs(All_AR_dir, exist_ok=True)
    All_AR_file_name = os.path.join(All_AR_dir, "all_action_rules.json")

    # Action rules =======================================================================================
    AR_file_name = os.path.join(output_dir, "action_rules.json")
    AR_imp_file_name = os.path.join(output_dir, f"action_rules_improve_{t_index}.json")
    AR = ActionRules(model="gpt-3.5-turbo")

    ar = AR.generate_ActionRules(TopK_real_trajectory, input_dir, All_AR_dir)
    AR.save(AR_file_name, ar)
    
    improve_ar = AR.generate_ActionRulesImprove(TopK_real_trajectory, ar, input_dir)
    AR.save(AR_imp_file_name, improve_ar)
    AR.save(All_AR_file_name, improve_ar)


    # Prologルールの作成
    # ====================================================================================================
    ilasp = ILASP_LLM(model="gpt-4.1")

    with open(AR_imp_file_name, "r", encoding="utf-8") as f:
        action_rules = json.load(f)
    
    code_rule = ilasp.generate_coderule(action_rules, input_dir)
    ilasp_filename = os.path.join(output_dir, f"generated_rules_{t_index}.pl")
    all_ilasp_filename = os.path.join(output_dir, f"all_generated_rules.pl")
    ilasp.save(ilasp_filename, code_rule)
    ilasp.save_connect(all_ilasp_filename, code_rule)

    print("生成したLASルール:")
    print(code_rule)


    # Prologルールに確率の割り振り
    # ====================================================================================================
    rule_probs, totals, counts = compute_rule_probabilities(
        json_path=Dall_file_name,
        prolog_rule_path=all_ilasp_filename,
        text_index=t_index,
        output=output_dir
    )





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



def save_rule_probabilities(rule_probs: dict, text_index: int, output_dir="probabilistic_rules"):
    """
    rule_probs: { rule_text: probability }
    text_index: 各テキストやデータセットのインデックス（例: 1, 2, 3...）
    output_dir: 保存先ディレクトリ名
    """
    # 出力先ディレクトリを作成（存在しなければ）
    os.makedirs(output_dir, exist_ok=True)

    # 保存ファイルパス
    filename = f"action_fail_prob_{text_index}.pl"
    filepath = os.path.join(output_dir, filename)

    # 確率付きルールを書き込む
    with open(filepath, "w", encoding="utf-8") as f:
        for rule, prob in rule_probs.items():
            f.write(f"{prob:.2f} :: {rule}\n")

    print(f"✅ Saved probabilistic rules to {filepath}")