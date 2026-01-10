# stage1.py

import json
from typing import List, Dict, Any, Tuple
import os
import copy # deepcopy を使用

def implement_stage1(
    traj_real: Dict, 
    traj_pred: Dict, 
    all_inc_list_existing: List[Dict[str, Any]], 
    all_cor_list_existing: List[Dict[str, Any]],
    scene_graph,
    task_name: str = "unknown_task"
) -> Tuple[Dict, List[Dict], List[Dict]]: 

    D_cor = {}
    D_inc = {} 
    D_cor_all_updated = copy.deepcopy(all_cor_list_existing)
    D_inc_all_updated = copy.deepcopy(all_inc_list_existing)
    
    # --- マップ作成 ---
    
    # 失敗事例マップ
    existing_steps_map_inc = {}
    for index, entry in enumerate(D_inc_all_updated):
        key = (entry.get('task_id'), entry.get('step_id'))
        existing_steps_map_inc[key] = index 
    
    # 【修正1】成功事例マップ (参照先を D_cor_all_updated に修正)
    existing_steps_map_cor = {}
    for index, entry in enumerate(D_cor_all_updated): # ← ここを修正しました
        key = (entry.get('task_id'), entry.get('step_id'))
        existing_steps_map_cor[key] = index 

    # グローバルIDのベース
    global_id_base_inc = len(D_inc_all_updated) 
    global_id_base_cor = len(D_cor_all_updated) # 分けたほうが安全です

    real_transitions = extract_transitions(traj_real)
    predicted_transitions = extract_transitions(traj_pred)
    min_length = min(len(real_transitions), len(predicted_transitions))
    
    current_succ_count = 0
    current_fail_count = 0

    for i in range(min_length):
        real_transition = real_transitions[i]
        predicted_transition = predicted_transitions[i]

        real_success = real_transition.get("action_result", {}).get("success", False)
        predicted_success = predicted_transition.get("action_result", {}).get("success", False)
        
        current_step_id_tuple = (task_name, i) 

        # ---------------------------------------------------------------------
        # 成功/失敗の一致によって分類
        # ---------------------------------------------------------------------
        if real_success == predicted_success:
            D_cor[str(i)] = real_transition
            new_entry_data = copy.deepcopy(real_transition) 
            
            # 【重複チェック: 成功事例】
            if current_step_id_tuple in existing_steps_map_cor:
                index_to_update = existing_steps_map_cor[current_step_id_tuple]
                D_cor_all_updated[index_to_update]['step_data'] = new_entry_data
            else:
                new_entry = {
                    "global_id": global_id_base_cor + current_succ_count, 
                    "task_id": task_name,
                    "step_id": i, 
                    "step_data": new_entry_data 
                }
                D_cor_all_updated.append(new_entry) 
                current_succ_count += 1
                
        else:
            D_inc[str(i)] = real_transition
            new_entry_data = copy.deepcopy(real_transition) 
            
            # 【重複チェック: 失敗事例】
            if current_step_id_tuple in existing_steps_map_inc:
                index_to_update = existing_steps_map_inc[current_step_id_tuple]
                D_inc_all_updated[index_to_update]['step_data'] = new_entry_data
            else:
                new_entry = {
                    "global_id": global_id_base_inc + current_fail_count, 
                    "task_id": task_name,
                    "step_id": i, 
                    "step_data": new_entry_data 
                }
                D_inc_all_updated.append(new_entry) 
                current_fail_count += 1

    # 【修正2 & 3】ループ終了後に一括でSceneGraphを注入 (効率化 & エラー回避)
    # これにより D_cor_sg, D_inc_sg が確実に生成されます
    D_cor_sg = inject_scene_graph(D_cor_all_updated, scene_graph)
    D_inc_sg = inject_scene_graph(D_inc_all_updated, scene_graph)

    return D_cor, D_inc, D_cor_sg, D_inc_sg


def extract_transitions(trajectory_dict: Dict) -> List[Dict]:
    transitions = []
    index = 0
    while f"state_{index}" in trajectory_dict:
        transition = {
            "state": trajectory_dict.get(f"state_{index}", {}),
            "action": trajectory_dict.get(f"action_{index}", {}),
            "action_result": trajectory_dict.get(f"action_result_{index}", {})
        }
        transitions.append(transition)
        index += 1
    return transitions


def inject_scene_graph(inc_list: List[Dict[str, Any]], scene_graph: Dict):
    """
    D_inc/D_cor リストの各エントリの 'step_data' に scene_graph を注入する。
    """
    print(f"Injecting SceneGraph data to {len(inc_list)} items...")
    for entry in inc_list:
        step_id_str = str(entry["step_id"])
        step_data = entry["step_data"]
        sg_key = f"scene_graph_{step_id_str}"
        
        # 既に注入済みで、かつ内容に変更がない場合はスキップするなどしても良いが、
        # ここでは常に最新のSGで上書きする
        if sg_key in scene_graph:
            step_data["scene_graph"] = scene_graph[sg_key]
        else:
            # フォールバック (Stage 4 で必要)
            # 既存データにSGがない場合も空リストを入れておく
            if "scene_graph" not in step_data:
                step_data["scene_graph"] = {"nodes": [], "edges": []} 
    return inc_list