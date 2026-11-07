from PIL import Image
import torch
from torch.utils.data import Dataset

from glaucoma.pipeline import build_model, preprocessing
from train import extract_features


def test_cached_features_match_full_model_and_allow_head_training():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(pretrained=False).eval()
    tensor = preprocessing()(Image.new("RGB", (64, 64), (40, 70, 80)))

    class Images(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return tensor, index

    features = extract_features(model, Images(), torch.device("cpu"), 2)
    with torch.inference_mode():
        direct = model(tensor.unsqueeze(0))
        cached = model.classifier(features[0][0].unsqueeze(0))
    assert torch.allclose(direct, cached, atol=1e-6)
    model.classifier.train()
    loss = torch.nn.functional.cross_entropy(model.classifier(features.tensors[0]), features.tensors[1])
    loss.backward()
    assert model.classifier[1].weight.grad is not None
    assert torch.isfinite(model.classifier[1].weight.grad).all()
