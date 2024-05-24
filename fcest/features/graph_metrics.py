import networkx as nx


class GraphMetricsExtractor:

    def __init__(self, graph):
        self.graph = graph

    def extract(self):
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "diameter": nx.diameter(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_shortest_path_length": nx.average_shortest_path_length(self.graph),
        }
