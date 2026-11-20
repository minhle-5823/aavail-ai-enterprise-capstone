from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from scripts.generate_demo import make_demo_csv
from aavail.modeling import train_from_csv
from aavail.api import app


@pytest.fixture(scope="session")
def trained_workspace(tmp_path_factory):
    """Everything, including model and charts, lives outside production directories."""
    root = tmp_path_factory.mktemp("aavail_test_isolated_")
    data = make_demo_csv(root/"test_only_synthetic.csv",n=480,seed=911)
    model_dir = root/"models"
    report = train_from_csv(data, model_dir, data_mode="demo")
    return {"root": root, "data": data, "models": model_dir, "report": report}


@pytest.fixture
def api_client(trained_workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(app.state, "model_dir", trained_workspace["models"], raising=False)
    monkeypatch.setattr(app.state, "log_dir", tmp_path/"private_test_logs", raising=False)
    with TestClient(app) as client:
        yield client
