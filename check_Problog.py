from problog.program import PrologFile, PrologString
from problog import get_evaluatable

# ファイル読み込み
with open("./probabilistic_rules.pl", "r") as f:
    rules_text = f.read()

with open("./1027/E3_Heat_and_Place/OurRule/output/Fact/fact_0.pl", "r") as f:
    facts_text = f.read()

# 両方を結合して PrologString にする
model = PrologString(rules_text + "\n" + facts_text)

# 確率推論を実行
result = get_evaluatable().create_from(model).evaluate()

# 結果を表示
for k, v in result.items():
    print(f"{k} = {v:.4f}")
