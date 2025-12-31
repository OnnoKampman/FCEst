import logging
import unittest

import networkx as nx

from fcest.features.graph_metrics import GraphMetricsExtractor

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


class TestGraphMetricsExtractor(unittest.TestCase):
    """
    Test functions in ./features/graph_metrics.py.
    """

    def test_graph_metrics_extractor_initialization(self):
        """
        Test GraphMetricsExtractor initialization.
        """
        graph_metrics_extractor = GraphMetricsExtractor(
            graph=nx.Graph(),
        )


if __name__ == "__main__":
    unittest.main()
