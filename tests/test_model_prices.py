import pytest

from batchata.providers.anthropic.parse_results import _calculate_cost as anthropic_cost
from batchata.providers.gemini.parse_results import _calculate_cost as gemini_cost
from batchata.providers.openai.parse_results import _calculate_cost as openai_cost


@pytest.mark.parametrize("model, list_price", [
    ("gpt-5.2", 15.75),
    ("gpt-5.2-pro", 189.0),
    ("o3-latest", 10.0),
])
def test_openai_batch_cost(model, list_price):
    assert openai_cost(1_000_000, 1_000_000, model, 0.5) == pytest.approx(list_price / 2)


def test_anthropic_batch_cost():
    assert anthropic_cost(1_000_000, 1_000_000, "claude-sonnet-4-5-20250929") == pytest.approx(9.0)


def test_gemini_batch_cost():
    assert gemini_cost("gemini-3-flash-preview", 1_000_000, 1_000_000, 0.5) == pytest.approx(1.75)
