import ast
import json
from pathlib import Path
import runpy

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_valid_clean_and_matches_source():
    builders = runpy.run_path(str(ROOT / "scripts/build_notebook.py"))
    for filename, builder in [("GlaucomAI_Colab.ipynb", "build"), ("GlaucomAI_Demo.ipynb", "build_demo")]:
        path = ROOT / "notebooks" / filename
        nbformat.validate(nbformat.read(path, as_version=4))
        notebook = json.loads(path.read_text())
        assert notebook == builders[builder]()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
                ast.parse("".join(cell["source"]))


def test_ui_shows_real_scores_and_label_limitations():
    from PIL import Image
    from glaucoma.notebook_ui import result_html
    html = result_html(Image.new("RGB", (224, 224)), {
        "referable": 0.81, "non_referable": 0.19, "flagged": True,
        "threshold": 0.4, "elapsed_ms": 93,
    })
    assert "81.0" in html
    assert "19.0%" in html
    assert "40.0%" in html
    assert "not a diagnosis" in html
    assert "does not mean healthy" in html


def test_widget_upload_analysis_invalid_replacement_and_clear(monkeypatch):
    from datetime import datetime, timezone
    from io import BytesIO

    from PIL import Image
    import torch

    from glaucoma import notebook_ui
    from glaucoma.pipeline import LoadedModel

    class FixedOutput(torch.nn.Module):
        def forward(self, image):
            return torch.tensor([[0.0, 2.0]])

    monkeypatch.setattr(notebook_ui, "load_model", lambda _: LoadedModel(FixedOutput(), 0.4, {}))
    rendered = []
    monkeypatch.setattr(notebook_ui, "display", lambda value: rendered.append(value))
    dashboard = notebook_ui.display_assistant("test-only.pth")
    assert dashboard["analyze"].disabled
    buffer = BytesIO()
    Image.new("RGB", (50, 50)).save(buffer, format="PNG")

    def upload(content):
        dashboard["uploader"].value = ({"name": "retina.png", "type": "image/png",
                                      "size": len(content), "content": memoryview(content),
                                      "last_modified": datetime.now(timezone.utc)},)

    upload(buffer.getvalue())
    assert not dashboard["analyze"].disabled
    dashboard["analyze"].click()
    assert any("88.1" in getattr(value, "data", "") for value in rendered)
    upload(b"invalid replacement")
    assert dashboard["analyze"].disabled
    dashboard["reset"].click()
    assert dashboard["uploader"].value == ()
    assert dashboard["analyze"].disabled
