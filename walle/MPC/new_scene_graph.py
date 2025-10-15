import networkx as nx
import json

class SceneGraph:
    """
    単一ステップ(state_t)ごとの観測を統合して、
    シーングラフを自動的に拡張・更新するバージョン。
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def update(self, state):
        """
        1つの state_t をもとに、既存グラフを更新。
        初回呼び出し時は新規グラフを生成。
        """

        # --- ロケーションとその中のアイテムを登録 ---
        for loc, content in state["items_in_locations"].items():
            self.graph.add_node(loc, type="location")

            for item in content["items"]:
                self.graph.add_node(item, type="item")

                # すでに同じ contains 関係がなければ追加
                if not any(
                    d.get("relation") == "contains"
                    for _, _, d in self.graph.edges(loc, data=True)
                    if _ == item
                ):
                    self.graph.add_edge(loc, item, relation="contains")

        # --- 現在位置（agent の at 関係）を更新 ---
        current_loc = state.get("current_position", {}).get("location_name")
        if current_loc:
            self.graph.add_node("agent", type="agent")
            self.graph.add_node(current_loc, type="location")

            # 既存の "at" エッジを削除して新しい位置を反映
            for u, v, d in list(self.graph.edges("agent", data=True)):
                if d.get("relation") == "at":
                    self.graph.remove_edge(u, v)
            self.graph.add_edge("agent", current_loc, relation="at")

        # --- 手に持っているアイテム（holding）を更新 ---
        hand_item = state.get("item_in_hand", {}).get("item_name")

        # まず既存の holding を全削除（どんな場合でも）
        for u, v, d in list(self.graph.edges("agent", data=True)):
            if d.get("relation") == "holding":
                self.graph.remove_edge(u, v)

        # その上で、新しいアイテムを持っていれば holding を追加
        if hand_item:
            self.graph.add_node(hand_item, type="item")
            self.graph.add_edge("agent", hand_item, relation="holding")

            # holding 中のアイテムを含む contains エッジを削除
            for u, v, d in list(self.graph.edges(data=True)):
                if v == hand_item and d.get("relation") == "contains":
                    self.graph.remove_edge(u, v)


    def visualize(self):
        """グラフの内容を表示"""
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} -[{data['relation']}]→ {v}")

    def save(self, path):
        """JSON 形式で保存"""
        data = {
            "nodes": [
                {"id": n, "type": self.graph.nodes[n].get("type")}
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, "relation": d.get("relation")}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def to_dict(self):
        """
        現在のシーングラフを辞書形式 (Rule_26_takeが期待する形式) で返します。
        """
        data = {
            "nodes": [
                {"id": n, "type": self.graph.nodes[n].get("type")}
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, "relation": d.get("relation")}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        return data # 辞書を返す
