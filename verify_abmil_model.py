"""GPU-side smoke test for the original ABMIL architecture and helper methods."""
import torch
import torch.nn.functional as F
from abmil import abmil

def main():
    torch.manual_seed(17)
    model = abmil(input_size=512)
    features = torch.randn(257, 512)
    target = torch.ones((1, 1), dtype=torch.float32)
    logits, attention = model(features)
    assert logits.shape == (1, 1), logits.shape
    assert attention.shape == (1, 257), attention.shape
    assert torch.isfinite(logits).all() and torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=1), torch.ones(1), atol=1e-6, rtol=1e-6)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    error, predictions = model.calculate_classification_error(features, target)
    objective, helper_attention = model.calculate_objective(features, target)
    assert 0.0 <= error <= 1.0
    assert predictions.shape == (1, 1) and objective.shape == (1, 1)
    assert torch.isfinite(objective).all()
    assert torch.allclose(attention, helper_attention)
    print("ABMIL MODEL SMOKE TEST PASSED")
    print("logits:", tuple(logits.shape))
    print("attention:", tuple(attention.shape))
    print("attention sum:", float(attention.sum()))

if __name__ == "__main__":
    main()
