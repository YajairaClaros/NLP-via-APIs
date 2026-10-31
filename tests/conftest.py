import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.tiny_models import build_bert, build_roberta  # noqa: E402


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    return build_bert(tmp_path_factory.mktemp("tiny-bert"))


@pytest.fixture(scope="session")
def tiny_roberta_dir(tmp_path_factory):
    return build_roberta(tmp_path_factory.mktemp("tiny-roberta"))


@pytest.fixture(scope="session", params=["bert", "roberta"])
def tiny_model_dir(request, tiny_bert_dir, tiny_roberta_dir):
    return tiny_bert_dir if request.param == "bert" else tiny_roberta_dir
