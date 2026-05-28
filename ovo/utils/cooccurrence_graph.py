from collections import defaultdict
from typing import List, Dict
import json

class CooccurrenceGraph:
    """
    Persistent co-occurrence graph for 3D instances.
    """
    def __init__(self):
        self.graph: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    def increment(self, i: int, j: int, kf_id: int) -> None:
        if kf_id not in self.graph[i][j]:
            self.graph[i][j].append(kf_id)
        if kf_id not in self.graph[j][i]:
            self.graph[j][i].append(kf_id)

    def weight(self, i: int, j: int) -> int:
        return len(self.graph[i][j])

    def get_kfs(self, i: int, j: int) -> List[int]:
        return sorted(self.graph[i][j])

    def get_neighbors(self, i: int) -> Dict[int, int]:
        return {neighbor: len(kfs) for neighbor, kfs in self.graph[i].items()}

    def merge(self, target: int, source: int) -> None:
        if source not in self.graph:
            return
        for neighbor, kfs in list(self.graph[source].items()):
            if neighbor == target:
                continue
            combined = set(self.graph[target][neighbor] + kfs)
            self.graph[target][neighbor] = list(combined)
            self.graph[neighbor][target] = list(combined)
            self.graph[neighbor].pop(source, None)
        del self.graph[source]

    def remove(self, node_id: int) -> None:
        if node_id not in self.graph:
            return
        for neighbor in list(self.graph[node_id].keys()):
            self.graph[neighbor].pop(node_id, None)
        del self.graph[node_id]

    def export_json(self, path: str) -> None:
        export_data = {"nodes": list(self.graph.keys()), "edges": []}
        for i, neighbors in self.graph.items():
            for j, kfs in neighbors.items():
                if i < j:
                    export_data["edges"].append({
                        "source": i, "target": j,
                        "weight": len(kfs), "keyframes": sorted(kfs)
                    })
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
