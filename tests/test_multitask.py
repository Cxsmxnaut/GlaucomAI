import pytest
import torch
from torch.nn import functional as F

from glaucoma.multitask import ARCHITECTURE, TASKS, TASK_CLASSES, SeparateTaskModel, load_separate_tasks
from glaucoma.pipeline import PREPROCESSING, ModelError, build_model, load_model
from multitask_train import routed_loss


def test_prefix_cache_preserves_native_baseline_forward_exactly():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = build_model().eval()
    model = SeparateTaskModel(baseline).eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        expected = baseline(images)
        prefix = model.prefix(images)
        cached = model.task_logits(model.embedding(prefix), "eyepacs")
        direct = model(images, "eyepacs")
    assert prefix.shape == (2, 192, 7, 7)
    assert torch.allclose(expected, cached, atol=1e-6)
    assert torch.allclose(direct, cached, atol=1e-6)


def test_routed_loss_updates_suffix_and_matching_heads_without_touching_prefix():
    torch.manual_seed(42)
    model = SeparateTaskModel().train()
    frozen = model.prefix_sha256()
    before = model.features[8][0].weight.detach().clone()
    prefix = torch.randn(4, 192, 7, 7)
    labels = torch.tensor([0, 1, 1, 0])
    tasks = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss = routed_loss(model, prefix, labels, tasks)
    loss.backward()
    assert all(p.grad is None for p in model.features[:7].parameters())
    assert model.heads["eyepacs"].weight.grad is not None
    assert model.heads["acrima"].weight.grad is not None
    assert model.heads["papila"].weight.grad is None
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    optimizer.step()
    assert frozen == model.prefix_sha256()
    assert not torch.equal(before, model.features[8][0].weight)


def test_task_routing_equals_separate_classification_losses_in_eval():
    model = SeparateTaskModel().eval()
    prefix = torch.randn(4, 192, 7, 7)
    labels = torch.tensor([0, 1, 1, 0])
    tasks = torch.tensor([0, 0, 2, 2])
    embedding = model.embedding(prefix)
    expected = (F.cross_entropy(model.task_logits(embedding[:2], "eyepacs"), labels[:2], reduction="sum")
                + F.cross_entropy(model.task_logits(embedding[2:], "papila"), labels[2:], reduction="sum")) / 4
    assert torch.allclose(routed_loss(model, prefix, labels, tasks), expected)
    with pytest.raises(ValueError, match="explicit research task"):
        model.task_logits(embedding, "generic_glaucoma")


def test_separate_task_checkpoint_schema_and_original_dashboard_are_distinct(tmp_path):
    model = SeparateTaskModel().eval()
    path = tmp_path / "synthetic_multitask.pth"
    payload = {"architecture": ARCHITECTURE, "tasks": list(TASKS), "task_classes": TASK_CLASSES,
               "trained_tasks": ["eyepacs", "acrima"], "thresholds": {"eyepacs": .4, "acrima": .5},
               "preprocessing": PREPROCESSING, "frozen_prefix_sha256": model.prefix_sha256(),
               "state_dict": model.state_dict()}
    torch.save(payload, path)
    loaded, metadata = load_separate_tasks(path)
    assert loaded.prefix_sha256() == model.prefix_sha256()
    assert metadata["task_classes"] == TASK_CLASSES
    with pytest.raises(ModelError):
        load_model(path)
    payload["task_classes"] = {"eyepacs": ["healthy", "glaucoma"]}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Task labels"):
        load_separate_tasks(path)
