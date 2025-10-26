import json
from pyswip import Prolog
from pathlib import Path
import re
import tempfile
import os

class PrologRuleProbabilityCalculator:
    def __init__(self, json_file, fact_folder, rules_file):
        """
        Args:
            json_file: D_all.jsonのパス
            fact_folder: fact_*.plファイルが入っているフォルダのパス
            rules_file: all_prolog_rules.plのパス
        """
        self.json_file = json_file
        self.fact_folder = fact_folder
        self.rules_file = rules_file
        
        # fact_*.plファイルを自動的に検索
        self.fact_files = self._load_fact_files()
    
    def _load_fact_files(self):
        """fact_folderから全てのfact_*.plファイルを読み込む"""
        fact_folder_path = Path(self.fact_folder)
        if not fact_folder_path.exists():
            raise FileNotFoundError(f"フォルダが見つかりません: {self.fact_folder}")
        
        # fact_*.plファイルを検索してソート
        fact_files = sorted(fact_folder_path.glob("fact_*.pl"), 
                           key=lambda x: int(x.stem.split('_')[1]))
        
        fact_files_dict = {}
        for f in fact_files:
            # fact_0.pl -> 0 を抽出
            step_id = f.stem.split('_')[1]
            fact_files_dict[step_id] = str(f)
        
        print(f"{len(fact_files_dict)}個のfactファイルを検出しました")
        return fact_files_dict
        
    def load_json_data(self):
        """JSONファイルを読み込む"""
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def load_rules(self):
        """ルールファイルを読み込んでパースする"""
        with open(self.rules_file, 'r') as f:
            content = f.read()
        
        # action_failed(...)のルールを抽出
        rules = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('%'):
                rules.append(line)
        return rules
    
    def parse_rule_head(self, rule):
        """ルールのヘッド部分（action_failed(...)）を抽出"""
        match = re.match(r'(.+?):-', rule)
        if match:
            return match.group(1).strip()
        return None
    
    def check_rule_applies(self, fact_file, rule_body):
        """
        ルールの条件部分が満たされるかチェック
        各チェックごとに新しいPrologインスタンスを使用
        Args:
            fact_file: factファイルのパス
            rule_body: ルールの条件部分（:- 以降）
        Returns:
            bool: 条件が満たされる場合True
        """
        try:
            # 新しいPrologインスタンスを作成（重要！）
            prolog = Prolog()
            
            # factファイルを読み込む
            prolog.consult(fact_file)
            
            # クエリを実行
            query = f"({rule_body})"
            result = list(prolog.query(query))
            
            # Prologインスタンスを明示的にクリーンアップ
            del prolog
            
            return len(result) > 0
        except Exception as e:
            # エラーメッセージを簡略化（existence_errorは述語が存在しない場合）
            if "existence_error" in str(e):
                # 述語が存在しない場合は条件が満たされないと判断
                return False
            print(f"    クエリエラー: {str(e)[:100]}")
            return False
    
    def extract_action_from_rule(self, rule):
        """
        ルールからアクション名を抽出
        例: action_failed(goto(Location)) -> 'goto'
        """
        match = re.search(r'action_failed\((\w+)\(', rule)
        if match:
            return match.group(1)
        return None
    
    def calculate_probabilities(self):
        """各ルールの確率を計算"""
        # JSONデータを読み込む
        data = self.load_json_data()
        
        # ルールを読み込む
        rules = self.load_rules()
        
        # 各ルールの統計を収集
        rule_stats = {}
        
        for rule in rules:
            rule_stats[rule] = {
                'true_count': 0,
                'false_count': 0,
                'probability': 0.0,
                'examples': []  # デバッグ用
            }
        
        # 各タスクの各ステップを処理
        for task_name, steps in data.items():
            print(f"\n処理中: {task_name}")
            
            for step_id, step_data in steps.items():
                print(f"  ステップ {step_id}", end="")
                
                # 対応するfactファイルを取得
                fact_file = self.fact_files.get(step_id)
                
                if not fact_file or not Path(fact_file).exists():
                    print(f" - 警告: fact_{step_id}.pl が見つかりません")
                    continue
                
                # action_resultのsuccessを取得
                action_result = step_data.get('action_result', {})
                success = action_result.get('success', True)
                
                # 実行されたアクションを取得
                action_data = step_data.get('action', {})
                action_name = action_data.get('action_name', '')
                
                print(f" - アクション: {action_name}, Success: {success}")
                
                # 各ルールをチェック
                for rule in rules:
                    # このルールが対象とするアクションと一致するか確認
                    rule_action = self.extract_action_from_rule(rule)
                    if rule_action and rule_action != action_name:
                        continue
                    
                    # ルールの条件部分を抽出
                    if ':-' in rule:
                        rule_body = rule.split(':-', 1)[1].strip().rstrip('.')
                        
                        # ルールの条件が満たされるかチェック
                        applies = self.check_rule_applies(fact_file, rule_body)
                        
                        if applies:
                            print(f"    ✓ ルールが適用: {rule[:60]}...")
                            if success:
                                rule_stats[rule]['true_count'] += 1
                            else:
                                rule_stats[rule]['false_count'] += 1
                            
                            # デバッグ用に例を保存
                            rule_stats[rule]['examples'].append({
                                'task': task_name,
                                'step': step_id,
                                'action': action_name,
                                'success': success
                            })
        
        # 確率を計算
        print("\n" + "="*80)
        print("確率計算結果:")
        print("="*80)
        
        for rule, stats in rule_stats.items():
            true_count = stats['true_count']
            false_count = stats['false_count']
            total = true_count + false_count
            
            if total > 0:
                probability = false_count / total
                stats['probability'] = probability
                print(f"\nルール: {rule}")
                print(f"  Success=True: {true_count}")
                print(f"  Success=False: {false_count}")
                print(f"  確率 p = {probability:.4f}")
                
                # 例を表示（最初の3つまで）
                if stats['examples']:
                    print(f"  適用例:")
                    for ex in stats['examples'][:3]:
                        print(f"    - {ex['task']}, step {ex['step']}: {ex['action']} -> {ex['success']}")
            else:
                print(f"\nルール: {rule}")
                print(f"  適用例なし（条件が一度も満たされませんでした）")
        
        return rule_stats
    
    def generate_probabilistic_rules(self, rule_stats, output_file):
        """確率付きルールファイルを生成"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("% Probabilistic Prolog Rules\n")
            f.write("% Format: p :: rule\n")
            f.write("% p = False / (True + False)\n\n")
            
            for rule, stats in rule_stats.items():
                prob = stats['probability']
                true_count = stats['true_count']
                false_count = stats['false_count']
                total = true_count + false_count
                
                if total > 0:
                    f.write(f"% Statistics: Success=True: {true_count}, Success=False: {false_count}, Total: {total}\n")
                    f.write(f"{prob:.4f} :: {rule}\n\n")
                else:
                    f.write(f"% No data for this rule (condition never satisfied)\n")
                    f.write(f"% {rule}\n\n")
        
        print(f"\n確率付きルールを {output_file} に保存しました")


# 使用例
if __name__ == "__main__":
    calculator = PrologRuleProbabilityCalculator(
        json_file="D_all.json",
        fact_folder="Fact",  # Factフォルダを指定
        rules_file="all_prolog_rules.pl"
    )
    
    # 確率を計算
    rule_stats = calculator.calculate_probabilities()
    
    # 確率付きルールを生成
    calculator.generate_probabilistic_rules(rule_stats, "probabilistic_rules.pl")