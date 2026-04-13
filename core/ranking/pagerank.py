"""PageRank — iterative graph-based authority scoring.

Implements the original PageRank algorithm with damping factor:
  PR(p) = (1-d)/N + d × Σ PR(q)/L(q)  for all q linking to p

where:
  d = damping factor (typically 0.85)
  N = total number of pages
  L(q) = number of outbound links from page q

Also supports personalized PageRank by biasing the teleport vector.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PageRank:
    """Compute PageRank scores for a web graph."""

    def __init__(self, damping: float = 0.85, max_iterations: int = 100,
                 convergence_threshold: float = 1e-6):
        self.damping = damping
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._scores: Dict[int, float] = {}

    def compute(self, link_graph: Dict[int, List[int]]) -> Dict[int, float]:
        """Compute PageRank from adjacency list: {source_doc_id: [target_doc_ids]}."""
        all_nodes: Set[int] = set()
        for src, targets in link_graph.items():
            all_nodes.add(src)
            all_nodes.update(targets)

        if not all_nodes:
            return {}

        node_list = sorted(all_nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)

        scores = np.ones(N, dtype=np.float64) / N

        outlinks = np.zeros(N, dtype=np.float64)
        for src, targets in link_graph.items():
            valid_targets = [t for t in targets if t in node_to_idx]
            outlinks[node_to_idx[src]] = len(valid_targets)

        inlinks: Dict[int, List[int]] = defaultdict(list)
        for src, targets in link_graph.items():
            for t in targets:
                if t in node_to_idx:
                    inlinks[node_to_idx[t]].append(node_to_idx[src])

        for iteration in range(self.max_iterations):
            new_scores = np.full(N, (1 - self.damping) / N, dtype=np.float64)

            dangling_sum = 0.0
            for i in range(N):
                if outlinks[i] == 0:
                    dangling_sum += scores[i]

            new_scores += self.damping * dangling_sum / N

            for i in range(N):
                for j in inlinks.get(i, []):
                    if outlinks[j] > 0:
                        new_scores[i] += self.damping * scores[j] / outlinks[j]

            diff = np.abs(new_scores - scores).sum()
            scores = new_scores

            if diff < self.convergence_threshold:
                logger.info(f"PageRank converged after {iteration + 1} iterations (diff={diff:.2e})")
                break

        self._scores = {node_list[i]: float(scores[i]) for i in range(N)}
        return self._scores

    def compute_personalized(
        self,
        link_graph: Dict[int, List[int]],
        preference_docs: Set[int],
        bias_weight: float = 0.3,
    ) -> Dict[int, float]:
        """Personalized PageRank: bias teleport towards preferred documents."""
        all_nodes: Set[int] = set()
        for src, targets in link_graph.items():
            all_nodes.add(src)
            all_nodes.update(targets)

        if not all_nodes:
            return {}

        node_list = sorted(all_nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        N = len(node_list)

        teleport = np.ones(N, dtype=np.float64) / N
        if preference_docs:
            pref_indices = [node_to_idx[d] for d in preference_docs if d in node_to_idx]
            if pref_indices:
                uniform_part = (1 - bias_weight) / N
                bias_part = bias_weight / len(pref_indices)
                teleport = np.full(N, uniform_part, dtype=np.float64)
                for idx in pref_indices:
                    teleport[idx] += bias_part

        scores = teleport.copy()

        outlinks = np.zeros(N, dtype=np.float64)
        for src, targets in link_graph.items():
            valid_targets = [t for t in targets if t in node_to_idx]
            outlinks[node_to_idx[src]] = len(valid_targets)

        inlinks: Dict[int, List[int]] = defaultdict(list)
        for src, targets in link_graph.items():
            for t in targets:
                if t in node_to_idx:
                    inlinks[node_to_idx[t]].append(node_to_idx[src])

        for iteration in range(self.max_iterations):
            new_scores = (1 - self.damping) * teleport

            dangling_sum = sum(scores[i] for i in range(N) if outlinks[i] == 0)
            new_scores += self.damping * dangling_sum * teleport

            for i in range(N):
                for j in inlinks.get(i, []):
                    if outlinks[j] > 0:
                        new_scores[i] += self.damping * scores[j] / outlinks[j]

            diff = np.abs(new_scores - scores).sum()
            scores = new_scores

            if diff < self.convergence_threshold:
                break

        self._scores = {node_list[i]: float(scores[i]) for i in range(N)}
        return self._scores

    def get_score(self, doc_id: int) -> float:
        return self._scores.get(doc_id, 0.0)

    @property
    def scores(self) -> Dict[int, float]:
        return dict(self._scores)

    def top_k(self, k: int = 10) -> List[Tuple[int, float]]:
        return sorted(self._scores.items(), key=lambda x: x[1], reverse=True)[:k]
