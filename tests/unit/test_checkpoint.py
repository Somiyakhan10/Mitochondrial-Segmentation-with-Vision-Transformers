from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mitomorph.exceptions import MitoMorphError
from mitomorph.segmentation.checkpoint import load_checkpoint, resume_training, save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, epoch=3, extra={"note": "test"})

    new_model = _TinyModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
    state = load_checkpoint(path, new_model, new_optimizer)

    assert state["epoch"] == 3
    assert state["extra"] == {"note": "test"}
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_resume_training_returns_next_epoch(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, epoch=5)

    assert resume_training(path, model, optimizer) == 6


def test_load_checkpoint_missing_file_raises(tmp_path):
    model = _TinyModel()
    with pytest.raises(MitoMorphError):
        load_checkpoint(tmp_path / "missing.pt", model)
