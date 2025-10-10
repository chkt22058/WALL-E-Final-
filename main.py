import os
import openai
import json
import yaml
import time
import argparse
import sys

# コマンドライン引数保存の準備 ============================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('outdir', nargs='?', default='./default', help='Output directory (default: ./default)')
args = parser.parse_args()

outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)
# =======================================================================================================================

from alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from alfworld.alfworld.agents.environment.alfred_thor_env import AlfredThorEnv 

from utils.state_parser import *
from utils.trajectory_parser import *
from utils.make_action_command import *

from walle.MPC.MPC import *
from copy import deepcopy

from walle.NSLearning.nslearning import *
from walle.NSLearning.new_nslearning import *

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
    agent = LLMAgent(model="gpt-4.1")
    world_model = LLMWorldModel(model="gpt-4.1")

    # AlfWorld環境のTW設定を定義 ====================================================================
    ### Task: put a clean potato in countertop.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_clean_then_place_in_recep-Potato-None-CounterTop-5/trial_T20190909_123141_081448/game.tw-pddl'
    
    ### Task: heat some plate and put it in countertop.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep-Plate-None-CounterTop-28/trial_T20190907_180239_626804/game.tw-pddl'
    
    ### Task: find two cloth and put them in sinkbasin.
    SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_two_obj_and_place-Cloth-None-SinkBasin-405/trial_T20190909_074547_875583/game.tw-pddl'

    ### Task: clean some cloth and put it in toilet.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_clean_then_place_in_recep-Cloth-None-Toilet-426/trial_T20190910_045202_007687/game.tw-pddl'
    
    ### Task: put some book on armchair. 〇
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_and_place_simple-Book-None-ArmChair-209/trial_T20190909_091451_936328/game.tw-pddl'
    
    ### Task: put some knife on sidetable. 〇
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_and_place_simple-Knife-None-SideTable-3/trial_T20190906_214318_430974/game.tw-pddl'

    ### Task: put a remotecontrol in armchair. 〇
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/train/pick_and_place_simple-RemoteControl-None-ArmChair-230/trial_T20190909_020906_064567/game.tw-pddl'

    ### Task: put a clean kettle in cabinet.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-Kettle-None-Cabinet-2/trial_T20190909_042935_977031/game.tw-pddl'    
    # ===================================================================================================

    # ALFWorld: TW環境基本6つのタスク ====================================================================
    ### Task: look at book under the desklamp. ✖
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/look_at_obj_in_light-Book-None-DeskLamp-302/trial_T20190909_085137_911990/game.tw-pddl'

    ### Task: put some alarmclock on desk. ✖
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072317_014092/game.tw-pddl'

    ### Task: clean some bowl and put it in shelf. 〇
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-Bowl-None-Shelf-7/trial_T20190908_152949_169018/game.tw-pddl'

    ### Task: put a cool bowl in cabinet. ✖
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_cool_then_place_in_recep-Bowl-None-Cabinet-18/trial_T20190908_144624_086654/game.tw-pddl'

    ### Task: put a hot apple in fridge. ✖
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_heat_then_place_in_recep-Apple-None-Fridge-6/trial_T20190908_153841_522662/game.tw-pddl'

    ### Task: find two book and put them in desk. ✖
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_two_obj_and_place-Book-None-Desk-302/trial_T20190906_181314_259738/game.tw-pddl'
    # ===================================================================================================

    # ALFWolrd: N = 20タスク =============================================================================

    ### TaskA1: examine the cd with the desklamp. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/look_at_obj_in_light-CD-None-DeskLamp-307/trial_T20190906_200435_424001/game.tw-pddl'
    ### TaskA2: look at creditcard under the desklamp. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/look_at_obj_in_light-CreditCard-None-DeskLamp-314/trial_T20190906_201548_667159/game.tw-pddl'
    ### TaskA3: look at tissuebox under the desklamp. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/look_at_obj_in_light-TissueBox-None-DeskLamp-216/trial_T20190908_033138_836240/game.tw-pddl'

    ### TaskB1: put some creditcard on armchair. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_and_place_simple-CreditCard-None-ArmChair-202/trial_T20190909_011606_013059/game.tw-pddl'
    ### TaskB2: put some soapbottle on cart. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_and_place_simple-SoapBottle-None-Cart-430/trial_T20190909_100609_714213/game.tw-pddl'
    ### TaskB3: put some tomato on microwave. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_and_place_simple-Tomato-None-Microwave-13/trial_T20190908_125127_168939/game.tw-pddl'
    ### TaskB4: put some toiletpaper on toiletpaperhanger.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-402/trial_T20190908_030828_744767/game.tw-pddl'

    ### TaskC1: put a clean dishsponge in shelf. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-DishSponge-None-Shelf-20/trial_T20190907_222456_204496/game.tw-pddl'
    ### TaskC2: put a clean egg in garbagecan. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-Egg-None-GarbageCan-11/trial_T20190906_181033_636334/game.tw-pddl'
    ### TaskC3: clean some lettuce and put it in countertop. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-15/trial_T20190907_070041_442493/game.tw-pddl'
    ### TaskC4: put a clean pan in fridge.
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_clean_then_place_in_recep-Pan-None-Fridge-1/trial_T20190908_105549_705446/game.tw-pddl'

    ### TaskD1: cool some cup and put it in cabinet. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_cool_then_place_in_recep-Cup-None-Cabinet-26/trial_T20190909_085908_816209/game.tw-pddl'
    ### TaskD2: put a cool lettuce in countertop. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_cool_then_place_in_recep-Lettuce-None-CounterTop-24/trial_T20190908_015109_682752/game.tw-pddl'
    ### TaskD3: put a cool plate in diningtable. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_cool_then_place_in_recep-Plate-None-DiningTable-23/trial_T20190910_025008_561989/game.tw-pddl'

    ### TaskE1: put a hot mug in coffeemachine. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-2/trial_T20190907_070838_688262/game.tw-pddl'
    ### TaskE2: put a hot potato in countertop. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_heat_then_place_in_recep-Potato-None-CounterTop-25/trial_T20190908_003752_653811/game.tw-pddl'
    ### TaskE3: put a hot tomato in countertop. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_heat_then_place_in_recep-Tomato-None-CounterTop-20/trial_T20190909_041153_700490/game.tw-pddl'

    ### TaskF1: put two candle in countertop. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_two_obj_and_place-Candle-None-CounterTop-406/trial_T20190908_045958_916084/game.tw-pddl'
    ### TaskF2: find two keychain and put them in dresser. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_two_obj_and_place-KeyChain-None-Dresser-318/trial_T20190907_181205_590674/game.tw-pddl'
    ### TaskF3: find two remotecontrol and put them in armchair. 
    # SPECIFIC_GAME_PATH = '/root/.cache/alfworld/json_2.1.1/valid_train/pick_two_obj_and_place-RemoteControl-None-ArmChair-208/trial_T20190908_000903_032547/game.tw-pddl'

    # ===================================================================================================
    # プロンプトに含めるタスク名
    task_name = "find two cloth and put them in sinkbasin."
    # ===================================================================================================

    
    with open("./alfworld/configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # テキストワールド(TW)環境の設定
    env = AlfredTWEnv(config, train_eval="train")

    # Vision(Thor)環境の設定
    # env = AlfredThorEnv(config, train_eval="train")
    
    env.game_files = [SPECIFIC_GAME_PATH]
    env.num_games = 1 
    print(f"\n特定のゲームファイルを設定しました: {env.game_files[0]}")
    env = env.init_env(batch_size=1)
    env.json_file_list = [SPECIFIC_GAME_PATH] #Thorの時は必要.

    # MPCの結果を保存するディレクトリ
    mpc_results_dir = "walle/MPC/mpc_results_log"
    if not os.path.exists(mpc_results_dir):
        os.makedirs(mpc_results_dir)
        print(f"Created directory: {mpc_results_dir}")
    
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
    real_trajectory = {}
    predicted_trajectory = {}
    done_flag = False

    transition_dir = os.path.join(outdir, "transition_log")
    os.makedirs(transition_dir, exist_ok=True)

    # 既存ルールのパス
    all_rules_path = "./CodeRule/all_code_rules.py"
    os.makedirs(os.path.dirname(all_rules_path), exist_ok=True)

    # 既存ルールを読み込み
    Rcode_t = []
    if os.path.exists(all_rules_path):
        Rcode_t = load_rules_from_file(all_rules_path)
            
    # 実環境で遷移を集める工程 & 実験工程
    while not done_flag and t_index < 50:
        print(f"\n--- Running MPC for Step {t_index} ---")
        # ===============================================================================================
        # MPCを実行し、計画されたアクションと予測された次の状態(Ot+1)を取得.
        current_planned_action = MPC(obs_state, Rcode_t, agent, world_model, t_index, outdir, 10, task_name)
        print(f"計画された行動:{current_planned_action}")
        
        # utilsフォルダのmake_action_commandを使って、アクションコマンド作成.
        action_command = make_action_command(current_planned_action)
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
        real_trajectory[f"action_{t_index}"] = deepcopy(current_planned_action)
        real_trajectory[f"action_result_{t_index}"] = generate_action_result_from_obs(obs_next_text)
        
        with open(os.path.join(transition_dir, f"real_trajectory_{t_index}.json"), "w", encoding="utf-8") as f:
            json.dump(real_trajectory, f, indent=4, ensure_ascii=False)

        # 予測軌跡の保存
        predicted_trajectory[f"state_{t_index}"] = deepcopy(obs_state["state"])
        predicted_trajectory[f"action_{t_index}"] = deepcopy(current_planned_action)
        predicted_trajectory[f"action_result_{t_index}"] = {"feedback": "", "success": True, "suggestion": ""}

        with open(os.path.join(transition_dir, f"predicted_trajectory_{t_index}.json"), "w", encoding="utf-8") as f:
            json.dump(predicted_trajectory, f, indent=4, ensure_ascii=False)

        obs_state = obs_next_state        
        t_index += 1


    # 得られた遷移で、コードルール作成!!
    code_rule = New_NSLearning(real_trajectory, predicted_trajectory, outdir, task_name)
    print("剪定されたコードルールの確認:")
    print(code_rule)