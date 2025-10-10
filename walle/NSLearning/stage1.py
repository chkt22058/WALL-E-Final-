import json

def implement_stage1(traj_real, traj_pred, task_name="unknown_task"):
    D_cor = {task_name: {}}  # ← タスク名を最上位キーに
    D_inc = {task_name: {}}  # 同様に

    real_transitions = extract_transitions(traj_real)
    predicted_transitions = extract_transitions(traj_pred)

    min_length = min(len(real_transitions), len(predicted_transitions))

    for i in range(min_length):
        real_transition = real_transitions[i]
        predicted_transition = predicted_transitions[i]

        real_success = real_transition.get("action_result", {}).get("success", False)
        predicted_success = predicted_transition.get("action_result", {}).get("success", False)

        # 成功/失敗の一致によって分類
        if real_success == predicted_success:
            D_cor[task_name][str(i)] = {
                "real_transitions": real_transition
            }
        else:
            D_inc[task_name][str(i)] = {
                "real_transitions": real_transition
            }

    return D_cor, D_inc


def extract_transitions(trajectory_dict):
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



if __name__ == '__main__':
    with open("../buffer_fact/traj_real.json", "r", encoding="utf-8") as f:
        traj_real = json.load(f)
    with open("../buffer_fact/traj_pred.json", "r", encoding="utf-8") as f:
        traj_pred = json.load(f)
    
    D_cor, D_inc = implement_stage1(traj_real, traj_pred)

     # 件数を表示
    print(f"正しく予測された遷移数: {len(D_cor)}")
    print(f"誤って予測された遷移数: {len(D_inc)}")

    with open("./check/D_cor.json", "w", encoding="utf-8") as f:
        json.dump(D_cor, f, indent=4, ensure_ascii=False)

    with open("./check/D_inc.json", "w", encoding="utf-8") as f:
        json.dump(D_inc, f, indent=4, ensure_ascii=False)
    