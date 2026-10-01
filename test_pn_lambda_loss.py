import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "pointwise_reranker"))
from modeling import compute_pn_lambda_loss


def _manual_loss(scores, temperature):
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    discount = lambda rank: 1.0 / math.log2(rank + 1.0)
    weights = [
        abs(discount(ranks[0]) - discount(ranks[index]))
        for index in range(1, len(scores))
    ]
    pair_losses = [
        torch.nn.functional.softplus(torch.tensor(-(scores[0] - scores[index]) / temperature)).item()
        for index in range(1, len(scores))
    ]
    return sum(w * loss for w, loss in zip(weights, pair_losses)) / sum(weights)


def test_matches_binary_ndcg_swap_formula():
    scores = torch.tensor([[0.2, 0.8, -0.1, 0.5]], requires_grad=True)
    loss, mean_delta = compute_pn_lambda_loss(scores, temperature=0.25)
    expected = _manual_loss(scores.detach()[0].tolist(), 0.25)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    assert 0.0 < mean_delta.item() < 1.0
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 0] < 0


def test_per_query_normalization_prevents_pair_count_scaling():
    two_passages = torch.tensor([[0.0, 1.0]])
    duplicated_negatives = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    loss_two, _ = compute_pn_lambda_loss(two_passages, temperature=0.5)
    loss_many, _ = compute_pn_lambda_loss(duplicated_negatives, temperature=0.5)
    assert torch.allclose(loss_two, loss_many, atol=1e-6)


def test_rank_weights_are_detached():
    scores = torch.tensor([[0.1, 0.9, 0.4]], requires_grad=True)
    loss, mean_delta = compute_pn_lambda_loss(scores, temperature=0.25)
    assert not mean_delta.requires_grad
    loss.backward()
    assert torch.isfinite(scores.grad).all()
