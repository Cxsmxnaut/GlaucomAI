"""Research-only task-specific heads; never interchangeable clinical outputs."""

from hashlib import sha256

import torch
from torch import nn

from glaucoma.pipeline import PREPROCESSING, build_model

TASKS = ("eyepacs", "acrima", "papila")
ARCHITECTURE = "efficientnet_b0_separate_tasks_v1"
TASK_CLASSES = {"eyepacs": ["non_referable", "referable"],
                "acrima": ["normal", "glaucoma"], "papila": ["healthy", "glaucoma"]}


class SeparateTaskModel(nn.Module):
    def __init__(self, baseline=None):
        super().__init__()
        baseline = build_model() if baseline is None else baseline
        self.features = baseline.features
        self.avgpool = baseline.avgpool
        self.dropout = baseline.classifier[0]
        self.heads = nn.ModuleDict({"eyepacs": baseline.classifier[1],
                                   "acrima": nn.Linear(1280, 2), "papila": nn.Linear(1280, 2)})
        for parameter in self.features[:7].parameters():
            parameter.requires_grad = False
        for parameter in self.features[7:].parameters():
            parameter.requires_grad = True
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.features[:7].eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def prefix(self, images):
        return self.features[:7](images)

    def embedding(self, prefix):
        return self.avgpool(self.features[7:](prefix)).flatten(1)

    def task_logits(self, embedding, task):
        if task not in TASKS:
            raise ValueError("Select an explicit research task, not a generic clinical output.")
        return self.heads[task](self.dropout(embedding))

    def forward(self, images, task):
        return self.task_logits(self.embedding(self.prefix(images)), task)

    def prefix_sha256(self):
        digest = sha256()
        for name, tensor in self.features[:7].state_dict().items():
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def load_separate_tasks(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE or payload.get("tasks") != list(TASKS):
        raise ValueError("This is not a compatible separate-task research checkpoint.")
    if payload.get("task_classes") != TASK_CLASSES or payload.get("preprocessing") != PREPROCESSING:
        raise ValueError("Task labels or preprocessing differ from the separate-task protocol.")
    trained = payload.get("trained_tasks", [])
    thresholds = payload.get("thresholds", {})
    if not trained or trained != [task for task in TASKS if task in trained] or set(thresholds) != set(trained):
        raise ValueError("Trained tasks and thresholds must match explicitly.")
    if any(not isinstance(value, (int, float)) or not 0 < value < 1 for value in thresholds.values()):
        raise ValueError("Task thresholds must be finite and between zero and one.")
    model = SeparateTaskModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint values.")
    if model.prefix_sha256() != payload["frozen_prefix_sha256"]:
        raise ValueError("Frozen-prefix provenance mismatch.")
    return model, {key: value for key, value in payload.items() if key != "state_dict"}
