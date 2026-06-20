import torch
from eurosat_cnn.model import LandCoverCNN, count_parameters


def test_forward_shape():
    m = LandCoverCNN(num_classes=10)
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 10)


def test_predict_returns_labels():
    m = LandCoverCNN(num_classes=10)
    pred = m.predict(torch.randn(2, 3, 64, 64))
    assert pred.shape == (2,)
    assert pred.dtype == torch.long


def test_param_count_reasonable():
    m = LandCoverCNN()
    n = count_parameters(m)
    assert 50_000 < n < 5_000_000


def test_overfits_tiny_batch():
    """One batch, many steps: loss must drop -> the net actually learns."""
    torch.manual_seed(0)
    m = LandCoverCNN(num_classes=3, width=8)
    x = torch.randn(6, 3, 64, 64)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    first = None
    m.train()
    for _ in range(40):
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.5
