import json
import networkx as nx
import matplotlib.pyplot as plt

def extract_scene_graph_from_state(state: dict, current_scene_graph: nx.DiGraph):
    """
    与えられたstate情報からサブグラフを抽出し、既存のScene Graphに統合します。
    """
    new_nodes = set()
    new_edges = [] # (u, v, label)

    # 1. reachable_locationsからノードを追加
    for loc in state["reachable_locations"]:
        new_nodes.add(loc)

    # 2. items_in_locationsからノードと"located_in"エッジを追加
    for location, items in state["items_in_locations"].items():
        new_nodes.add(location)
        for item in items:
            new_nodes.add(item)
            new_edges.append((item, location, "located_in"))
            new_edges.append((location, item, "contains")) # 逆方向のエッジ

    # 3. item_in_handからノードと"has"エッジを追加
    if state["item_in_hand"]["item_name"] != "null":
        item_in_hand = state["item_in_hand"]["item_name"]
        item_status = state["item_in_hand"]["status"]
        new_nodes.add(item_in_hand)
        new_nodes.add("Agent") # エージェント自身もノードとして追加
        new_edges.append(("Agent", item_in_hand, "has"))
        if item_status != "normal": # normal以外の状態もエッジで表現
            new_edges.append((item_in_hand, f"is_{item_status}", "has_status"))
            new_nodes.add(f"is_{item_status}") # status自体もノードとして扱う場合 (例: "is_clean")

    # 4. current_positionからノードと"at"エッジを追加
    current_loc = state["current_position"]["location_name"]
    current_loc_status = state["current_position"]["status"]
    new_nodes.add(current_loc)
    new_nodes.add("Agent") # エージェント自身もノードとして追加
    new_edges.append(("Agent", current_loc, "at"))

    # Scene Graphの更新
    current_scene_graph.add_nodes_from(new_nodes)

    for u, v, label in new_edges:
        current_scene_graph.add_edge(u, v, label=label)

    return current_scene_graph

def build_scene_graph_from_trajectory(trajectory_data: dict):
    """
    軌跡データ全体からScene Graphを構築し、各ステップのSGを記録します。
    """
    global_scene_graph = nx.DiGraph() # 全体のScene Graph
    scene_graphs_per_step = [] # 各ステップのScene Graphのリスト

    # 軌跡データは、トップレベルのキー（タスク名）の下のリスト
    task_name = list(trajectory_data.keys())[0]
    steps = trajectory_data[task_name]

    print(f"Building Scene Graph for task: '{task_name}'")

    for i, step in enumerate(steps):
        print(f"\n--- Processing Step {i} ---") 

        current_state = step["state"]
        action_details = step["action"] # アクションの詳細を取得
        
        # 現在のステップの情報を統合してグローバルScene Graphを更新
        global_scene_graph = extract_scene_graph_from_state(current_state, global_scene_graph)
        
        # 各ステップのScene Graphのコピーを保存 (論文の G_scene_t を表現)
        scene_graphs_per_step.append(global_scene_graph.copy())

        # ログ出力
        print(f"  Agent's Current Position: {current_state['current_position']['location_name']}")
        print(f"  Action Executed:")
        print(f"    action_type: {action_details.get('action_type', 'N/A')}")
        # その他のアクションパラメータを表示
        for key, value in action_details.items():
            if key != 'action_type':
                print(f"    {key}: {value}")
        print(f"  Action Result: {step['action_result']}")
        
        print(f"  Current Accumulated SG Nodes ({len(global_scene_graph.nodes)}): {list(global_scene_graph.nodes)}")
        print(f"  Current Accumulated SG Edges ({len(global_scene_graph.edges)}):")
        for u, v, data in global_scene_graph.edges(data=True):
            print(f"    - {u} --({data['label']})--> {v}")
    
    print("\n--- Scene Graph Building Complete ---")
    print(f"Final Scene Graph has {len(global_scene_graph.nodes)} nodes and {len(global_scene_graph.edges)} edges.")
    return global_scene_graph, scene_graphs_per_step


def convert_real_trajectory_to_task_format_no_task_name(real_trajectory):
    steps = []
    i = 0
    while True:
        state_key = f"state_{i}"
        action_key = f"action_{i}"
        action_result_key = f"action_result_{i}"

        if state_key not in real_trajectory:
            break

        step = {
            "state": real_trajectory[state_key],
            "action": real_trajectory.get(action_key, {}),
            "action_result": real_trajectory.get(action_result_key, {})
        }
        steps.append(step)
        i += 1

    return {"": steps}


if __name__ == "__main__":
    json_file_path = '../buffer_fact/traj_real.json' 
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        exit()
    
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Check file format.")
        exit()
        
    # Scene Graphを構築
    final_sg, all_sgs = build_scene_graph_from_trajectory(traj_data)

    # 最終的なScene Graphのノードとエッジを出力
    print("\n--- Final Scene Graph Details ---")
    print("Nodes:", list(final_sg.nodes()))
    print("Edges:")
    for u, v, data in final_sg.edges(data=True):
        print(f"- {u} --({data['label']})--> {v}")
    
    # --- JSON形式で保存 ---
    output_json_file = 'output/scene_graph.json'
    
    # JSON形式で保存するためのデータ構造を準備
    sg_data_for_json = {
        "V": list(final_sg.nodes()),
        "E": []
    }
    # エッジ情報を適切な形式で追加
    for u, v, data in final_sg.edges(data=True):
        sg_data_for_json["E"].append({"u": u, "v": v, "label": data['label']})

    # ファイルに書き込み
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(sg_data_for_json, f, ensure_ascii=False, indent=4)
        
    print(f"\nFinal Scene Graph (JSON) saved to {output_json_file}")