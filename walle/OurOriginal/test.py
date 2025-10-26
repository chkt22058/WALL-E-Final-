import json
import os

from walle.OurOriginal.JSONtoFacts import normalize_name, state_action_to_facts, save

INPUT_PATH = "./check2/test_obs.json"
OUTPUT_DIR = "./check2/facts_out"

def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for task_name, steps in data.items():
        all_facts = []

        for step_id, sample in steps.items():
            facts = state_action_to_facts(sample)
            all_facts.append(f"% Step {step_id}\n" + facts + "\n")

        # タスク名をファイル名として保存（スペース等は置換）
        filename = normalize_name(task_name) + ".pl"
        save(os.path.join(OUTPUT_DIR, filename), "\n".join(all_facts))

    print("✅ すべてのPrologファイルを保存しました:", OUTPUT_DIR)

if __name__ == "__main__":
    process_all()
