import networkx as nx
import json

class SceneGraph:
    """
    WALL-E 2.0 に準拠した最小限の Scene Graph 実装。
    位置（location）と、その中に含まれるアイテム（contains）のみを表現。
    時間情報（timestep）は保持しない。
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def update(self, observation):
        """
        各ステップの観測から location-item 関係のみを更新。
        すでに存在するノード・エッジは再追加しない。
        """
        for loc, content in observation["items_in_locations"].items():
            self.graph.add_node(loc, type="location")
            for item in content["items"]:
                self.graph.add_node(item, type="item")
                # 重複エッジを避ける
                if not any(
                    d.get("relation") == "contains"
                    for _, _, d in self.graph.edges(loc, data=True)
                    if _ == item
                ):
                    self.graph.add_edge(loc, item, relation="contains")

    def update_all_states(self, trajectory):
        """
        trajectory (state_0, state_1, ...) 全体から Scene Graph を構築。
        """
        for key in sorted(trajectory.keys()):
            if key.startswith("state_"):
                self.update(trajectory[key])

    def visualize(self):
        """
        グラフを簡易表示
        """
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} -[{data['relation']}]→ {v}")

    def save(self, path):
        """
        JSON 形式で保存
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
