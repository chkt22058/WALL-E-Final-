import json
from pyswip import Prolog
from pathlib import Path
import re

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
    
    def extract_action_info(self, fact_file):
        """
        factファイルからaction述語の情報を抽出
        Returns:
            dict: {'action_name': str, 'args': [arg1, arg2, ...]} or None
        """
        try:
            prolog = Prolog()
            prolog.consult(fact_file)
            
            # action述語を取得
            result = list(prolog.query("action(Action)"))
            del prolog
            
            if result:
                action_term = str(result[0]['Action'])
                # 例: "take(egg_3, fridge_1)" -> action_name='take', args=['egg_3', 'fridge_1']
                match = re.match(r'(\w+)\((.+)\)', action_term)
                if match:
                    action_name = match.group(1)
                    args_str = match.group(2)
                    # 引数をカンマで分割（ネストした括弧も考慮）
                    args = [arg.strip() for arg in args_str.split(',')]
                    return {
                        'action_name': action_name,
                        'args': args
                    }
            return None
        except Exception as e:
            print(f"    action抽出エラー: {e}")
            return None
    
    def extract_variables_from_rule_head(self, head):
        """
        ルールのヘッドから変数名とその位置を抽出
        例: action_failed(take(Item, Location)) -> ['Item', 'Location']
        例: action_failed(goto(Target)) -> ['Target']
        """
        # action_failed(action_name(vars...)) のパターンをマッチ
        match = re.search(r'action_failed\(\w+\(([^)]+)\)\)', head)
        if match:
            vars_str = match.group(1)
            # カンマで分割して変数名を抽出
            variables = [var.strip() for var in vars_str.split(',')]
            return variables
        return []
    
    def bind_variables_in_body(self, rule_body, variables, values):
        """
        ルールの条件部分の変数を実際の値でバインド
        Args:
            rule_body: ルールの条件部分
            variables: 変数名のリスト ['Item', 'Location']
            values: 実際の値のリスト ['egg_3', 'fridge_1']
        Returns:
            str: 変数が置換された条件文
        """
        bound_body = rule_body
        
        # 各変数を対応する値で置換
        for var, val in zip(variables, values):
            # 変数全体を置換（部分一致を避けるため\bを使用）
            bound_body = re.sub(r'\b' + var + r'\b', val, bound_body)
        
        return bound_body
    
    def check_rule_applies_with_bindings(self, fact_file, rule, action_info):
        """
        ルールの条件部分が満たされるかチェック（複数変数のバインディングを考慮）
        Args:
            fact_file: factファイルのパス
            rule: 完全なルール文字列
            action_info: extractで取得したアクション情報
        Returns:
            bool: 条件が満たされる場合True
        """
        try:
            # ルールからヘッドと条件を分離
            if ':-' not in rule:
                return False
            
            head, body = rule.split(':-', 1)
            head = head.strip()
            body = body.strip().rstrip('.')
            
            # ヘッドから変数名を抽出
            variables = self.extract_variables_from_rule_head(head)
            
            if not variables:
                # 変数がない場合は通常の方法でチェック
                return self.check_rule_applies_simple(fact_file, body)
            
            # アクション情報から値を取得
            if not action_info or 'args' not in action_info:
                return False
            
            values = action_info['args']
            
            # 変数の数と値の数が一致するか確認
            if len(variables) != len(values):
                print(f"    警告: 変数数({len(variables)})と引数数({len(values)})が一致しません")
                return False
            
            # 新しいPrologインスタンスを作成
            prolog = Prolog()
            prolog.consult(fact_file)
            
            # 変数を実際の値でバインド
            bound_body = self.bind_variables_in_body(body, variables, values)
            
            # デバッグ出力
            # print(f"    元の条件: {body}")
            # print(f"    バインド後: {bound_body}")
            
            # クエリを実行
            query = f"({bound_body})"
            result = list(prolog.query(query))
            
            del prolog
            return len(result) > 0
                
        except Exception as e:
            if "existence_error" in str(e):
                return False
            print(f"    クエリエラー: {str(e)[:150]}")
            return False
    
    def check_rule_applies_simple(self, fact_file, rule_body):
        """
        変数バインディングが不要な単純なルールチェック
        Args:
            fact_file: factファイルのパス
            rule_body: ルールの条件部分（:- 以降）
        Returns:
            bool: 条件が満たされる場合True
        """
        try:
            prolog = Prolog()
            prolog.consult(fact_file)
            
            query = f"({rule_body})"
            result = list(prolog.query(query))
            
            del prolog
            return len(result) > 0
        except Exception as e:
            if "existence_error" in str(e):
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
                
                # factファイルからアクション情報を抽出
                action_info = self.extract_action_info(fact_file)
                
                print(f" - アクション: {action_name}", end="")
                if action_info and action_info.get('args'):
                    print(f"({', '.join(action_info['args'])})", end="")
                print(f", Success: {success}")
                
                # 各ルールをチェック
                for rule in rules:
                    # このルールが対象とするアクションと一致するか確認
                    rule_action = self.extract_action_from_rule(rule)
                    if rule_action and rule_action != action_name:
                        continue
                    
                    # ルールの条件が満たされるかチェック（変数バインディング考慮）
                    applies = self.check_rule_applies_with_bindings(fact_file, rule, action_info)
                    
                    if applies:
                        print(f"    ✓ ルールが適用: {rule[:70]}...")
                        if success:
                            rule_stats[rule]['true_count'] += 1
                        else:
                            rule_stats[rule]['false_count'] += 1
                        
                        # デバッグ用に例を保存
                        example_info = {
                            'task': task_name,
                            'step': step_id,
                            'action': action_name,
                            'success': success
                        }
                        if action_info and action_info.get('args'):
                            example_info['args'] = action_info['args']
                        
                        rule_stats[rule]['examples'].append(example_info)
        
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
                        args_info = f"({', '.join(ex['args'])})" if 'args' in ex else ""
                        print(f"    - {ex['task']}, step {ex['step']}: {ex['action']}{args_info} -> {ex['success']}")
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
        json_file="./1027/E3_Heat_and_Place/OurRule/trajectory/D_all.json",
        fact_folder="./1027/E3_Heat_and_Place/OurRule/output/Fact",  # Factフォルダを指定
        rules_file="./1027/E3_Heat_and_Place/OurRule/output/Prolog/all_prolog_rules.pl"
    )
    
    # 確率を計算
    rule_stats = calculator.calculate_probabilities()
    
    # 確率付きルールを生成
    calculator.generate_probabilistic_rules(rule_stats, "probabilistic_rules.pl")