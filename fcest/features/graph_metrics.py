from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    pass

__all__ = ["GraphMetricsExtractor"]


class GraphMetricsExtractor:

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    def extract(self) -> dict[str, float | int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "diameter": nx.diameter(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_shortest_path_length": nx.average_shortest_path_length(self.graph),
        }
