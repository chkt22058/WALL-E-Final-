# 実験方法
"""
# 全タスクを昇順実行（A1→A2→...→F5）
python test.py ./results --all

# 単一タスク
python test.py ./results --task A1

# 複数タスク指定
python test.py ./results --tasks A1 B2 C3

# グループ内からランダム
python test.py ./results --group Examine_in_Light --random_per_group 3

# 全グループから1つずつ
python test.py ./results --random_all_groups

# 全グループからランダムに N 個（例：10個）
python test.py ./results --random_all_groups_n 10

"""

import os
import random
import openai
import json
import yaml
import time
import argparse
import sys

# コマンドライン引数保存の準備 ============================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('outdir', nargs='?', default='./default', help='Output directory (default: ./default)')

parser.add_argument('--group', type=str, help='カテゴリ名（例: pick_and_place_simple）')
parser.add_argument('--random_per_group', type=int, help='カテゴリ内からランダムに選ぶ数')
parser.add_argument('--random_all_groups', action='store_true', help='全カテゴリから1つずつランダム選択')
parser.add_argument('--random_all_groups_n', type=int, help='全グループからランダムにN個選択')  
parser.add_argument('--task', type=str, help='単一タスクID（例: F3）')
parser.add_argument('--tasks', nargs='+', help='複数タスクID（例: F1 F2 F3）')
parser.add_argument('--all', action='store_true', help='全タスクを昇順で実行')

args = parser.parse_args()

outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)

# === tasks.yaml 読み込み ===
with open('train.yaml', 'r') as f:
    tasks_config = yaml.safe_load(f)['tasks']
# =======================================================================================================================

from alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from alfworld.alfworld.agents.environment.alfred_thor_env import AlfredThorEnv 

from utils.state_parser import *
from utils.trajectory_parser import *
from utils.make_action_command import *

from copy import deepcopy

# Our rule process
from walle.OurOriginal.our_nslearning import *

from walle.OurOriginal.agent_multi_action import *
from walle.OurOriginal.world_best_select import *

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 


# 環境変数 OPENAI_API_KEY が設定されているか確認
if client is None:
    print("Skipping LLMAgent and LLMWorldModel tests due to OpenAI client initialization failure.")
elif not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set your OpenAI API key before running the script.")
else:
    print("OpenAI API Key is set. Running MPC with LLMAgent and LLMWorldModel...")

    # エージェントとワールドモデルのインスタンス化
    agent = LLMAgent_MultiAction(model="gpt-3.5-turbo")
    world_model = LLMWorldModel_BestSelect(model="gpt-3.5-turbo")

    # === 実行タスク選択ロジック ===
    selected_tasks = []

    if args.group and args.random_per_group:
        # 指定グループ内からランダム選択
        group_tasks = [tid for tid, t in tasks_config.items() if t['group'] == args.group]
        selected_tasks = random.sample(group_tasks, min(args.random_per_group, len(group_tasks)))
        print(f"グループ「{args.group}」からランダムに {len(selected_tasks)} タスク選択: {selected_tasks}")

    elif args.random_all_groups:
        # 全カテゴリから1つずつランダム選択
        groups = sorted(set(t['group'] for t in tasks_config.values()))
        for g in groups:
            g_tasks = [tid for tid, t in tasks_config.items() if t['group'] == g]
            selected_tasks.append(random.choice(g_tasks))
        print(f"全 {len(groups)} グループから1つずつ選択: {selected_tasks}")
    
    elif args.random_all_groups_n:
        # ★新機能：全グループからランダムにN個選択
        all_tasks = list(tasks_config.keys())
        selected_tasks = random.sample(all_tasks, min(args.random_all_groups_n, len(all_tasks)))
        print(f"全グループの中からランダムに {len(selected_tasks)} タスク選択: {selected_tasks}")

    elif args.task:
        selected_tasks = [args.task]

    elif args.tasks:
        selected_tasks = args.tasks
    
    elif args.all:
        # 全タスクを昇順で実行
        selected_tasks = sorted(tasks_config.keys())
        print(f"全 {len(selected_tasks)} タスクを昇順で実行: {selected_tasks}")

    else:
        # 対話選択モード
        print("利用可能なグループ一覧:")
        groups = sorted(set(t['group'] for t in tasks_config.values()))
        for g in groups:
            print(" -", g)
        args.group = input("実行したいグループ名を入力してください: ").strip()
        args.random_per_group = int(input("ランダムに選ぶタスク数を入力してください: ").strip())
        group_tasks = [tid for tid, t in tasks_config.items() if t['group'] == args.group]
        selected_tasks = random.sample(group_tasks, min(args.random_per_group, len(group_tasks)))

    # === 実行タスクの確認 ===
    print(f"\n=== 実行対象タスク一覧 ===")
    for task_id in selected_tasks:
        print(f"{task_id}: {tasks_config[task_id]['name']} ({tasks_config[task_id]['group']})")
    print("================================\n")

    # === 各タスクを順番に実行 ===
    for i, task_id in enumerate(selected_tasks, 1):
        task_info = tasks_config[task_id]
        task_name = task_info['name']
        SPECIFIC_GAME_PATH = task_info['path']

        # ✅ タスクごとの出力ディレクトリを作成
        task_outdir = os.path.join(outdir, f"{task_id}_{task_info['group']}")
        os.makedirs(task_outdir, exist_ok=True)

        print(f"\n=== [{i}/{len(selected_tasks)}] 実行中: {task_id} ===")
        print(f"出力ディレクトリ: {task_outdir}")
        print(f"タスク: {task_name}")

        with open("./alfworld/configs/base_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # テキストワールド(TW)環境の設定
        env = AlfredTWEnv(config, train_eval="train")
    
        env.game_files = [SPECIFIC_GAME_PATH]
        env.num_games = 1 
        print(f"\n特定のゲームファイルを設定しました: {env.game_files[0]}")
        env = env.init_env(batch_size=1)
        env.json_file_list = [SPECIFIC_GAME_PATH] #Thorの時は必要.
    
        # 1. 実環境データの読み込み．観測値Stの取得．==============================================================================
        obs, info = env.reset()
        obs_text = obs[0]
        print("最初の観測情報:\n", obs_text)
        # print("行動後の現在の環境で実行可能な行動:\n", info["admissible_commands"])


        # 初期観測値(テキスト)からJSON形式に変換
        #######################################
        obs_state = parse_initial_observation(obs_text)
        print(json.dumps(obs_state, indent=4))
        #######################################

        # ======================================================================================================================

        t_index = 0
        Rcode_t = []
        real_trajectory = {}
        predicted_trajectory = {}
        done_flag = False

        transition_dir = os.path.join(task_outdir, "transition_log")
        os.makedirs(transition_dir, exist_ok=True)

        agent_prompt_dir = os.path.join(task_outdir, "agent_prompts_log")
        os.makedirs(agent_prompt_dir, exist_ok=True)
        wm_prompt_dir = os.path.join(task_outdir, "wm_prompts_log")
        os.makedirs(wm_prompt_dir, exist_ok=True)
        
            
        while not done_flag and t_index < 50:
            # ===============================================================================================
            current_planned_action, agent_prompt = agent.generate_action(obs_state, task_name)
            agent_prompt_file_name = os.path.join(agent_prompt_dir, f"agent_prompt_{t_index}.txt")
            with open(agent_prompt_file_name, "w", encoding="utf-8") as f:
                f.write(agent_prompt)
            print("[Agent]: 計画された行動:")
            print(f"{current_planned_action}")

            best_action, wm_prompt = world_model.predict_transition_outcome(obs_state, current_planned_action)
            wm_prompt_file_name = os.path.join(wm_prompt_dir, f"wm_prompt_{t_index}.txt")
            with open(wm_prompt_file_name, "w", encoding="utf-8") as f:
                f.write(wm_prompt)
            print("[World Model]: 選択された行動:")
            print(f"{best_action}")
        
            # utilsフォルダのmake_action_commandを使って、アクションコマンド作成.
            action_command = make_action_command(best_action["selected_action"])
            print("アクションコマンドの確認\n")
            print(action_command)

            # アクションコマンドを入力して実環境から情報を取得．
            obs, reward, done, info = env.step([action_command])
            done_flag = done[0]
            print(f"\n--- Real State for Step {t_index} ---")
            print(f"Obs{t_index+1}: {obs[0]}")
            print(f"タスクが完了したかどうか: {done_flag}")

            # 行動後観測値(Ot+1)からJSON形式に変換
            obs_next_text = obs[0]
            obs_next_state = get_updated_state_from_observation(obs_state, obs_next_text)
            print(f"{action_command} 実行後の観測値:")
            print(json.dumps(obs_next_state, indent=4, ensure_ascii=False))
            # ===============================================================================================

            # 実軌跡の保存
            real_trajectory[f"state_{t_index}"] = deepcopy(obs_state["state"])
            real_trajectory[f"action_{t_index}"] = deepcopy(best_action["selected_action"])
            real_trajectory[f"action_result_{t_index}"] = generate_action_result_from_obs(obs_next_text)
        
            with open(os.path.join(transition_dir, f"real_trajectory_{t_index}.json"), "w", encoding="utf-8") as f:
                json.dump(real_trajectory, f, indent=4, ensure_ascii=False)
        
            # 予測軌跡の保存
            predicted_trajectory[f"state_{t_index}"] = deepcopy(obs_state["state"])
            predicted_trajectory[f"action_{t_index}"] = deepcopy(best_action["selected_action"])
            predicted_trajectory[f"action_result_{t_index}"] = {"feedback": "", "success": True, "suggestion": ""}

            with open(os.path.join(transition_dir, f"predicted_trajectory_{t_index}.json"), "w", encoding="utf-8") as f:
                json.dump(predicted_trajectory, f, indent=4, ensure_ascii=False)

            # Ourルールの箇所 ================================================================================

            Our_NSLearning(real_trajectory, predicted_trajectory, task_outdir, task_name, t_index)

            # ================================================================================
        
            obs_state = obs_next_state        
            t_index += 1
    
    print("\n✅ 全タスクの実行が完了しました！")