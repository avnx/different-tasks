from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict

import torch  # type: ignore[import-not-found]
from anthropic.types import ToolUnionParam  # type: ignore[import-not-found]

from task_resources import recompute_wer_from_pairs

WORKSPACE_ROOT = Path(__file__).resolve().parent


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer. The grader expects a mapping with the
    key ``module`` pointing at a python module that exposes ``train_and_evaluate``.
    """
    return {"answer": answer, "submitted": True}


# Tool definitions for Anthropic API
TOOLS: list[ToolUnionParam] = [
    {
        "name": "python_expression",
        "description": (
            "Evaluates a Python expression in the current workspace. "
            "Use print(...) to emit output; stdout will be captured and returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Source code passed to exec(). Use this to read/write files, "
                        "run quick experiments, or train the model. Returns stdout."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit the final answer. Provide a JSON-serializable object with at "
            "least the key 'module' pointing to your solution module."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "description": (
                        "Final payload. Example: {'module': 'submission'}"
                    )
                }
            },
            "required": ["answer"],
        },
    },
]

# Tool handlers mapping
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}

SOURCE_BASELINE = Path("source.py").read_text()
SOURCE_PROMPT = SOURCE_BASELINE.replace("{", "{{").replace("}", "}}")


# The challenge prompt
PROMPT = """\
You are given the current baseline implementation, stored in `source.py`:

```python
{source_contents}
```

Your task is to produce an improved version in a brand-new file named `submission.py`. Work from the provided baseline, but integrate the requested changes rather than rewriting from scratch.

1. Workspace contract
   - Create `submission.py` (starting from the baseline above).
   - Expose a function `train_and_evaluate(device: str | None = None) -> dict`.
   - The returned dictionary must contain:
        * `model`: your trained `torch.nn.Module`.
        * `metrics`: a dict with the float keys `loss`, `wer`, `wer_loss`, `adjusted_loss`.
        * `val_predictions`: an iterable of mappings, each with string keys `reference` and `prediction`.
   - Keep training/evaluation under ~30 seconds on CPU.

2. Model architecture
   - Complete the model architecture in `submission.py`.
   - Complete train loop in `train_and_evaluate` function.
   - Use the helpers in `task_resources.py` for data loading, vocabulary utilities, and baseline scoring.
   - Modify the architecture so that it includes at least one `torch.nn.MultiheadAttention` layer with `num_heads >= 2`; the layer must meaningfully participate in the forward pass (e.g. augment the listener or decoder).
   - Preserve or improve the autoregressive decoder behavior from the baseline.

3. WER-aware objective
   - Combine your primary training loss with an auxiliary component that penalizes word error rate (WER).
     For example, compute a differentiable proxy or a scheduled scaling on the decoder loss based on \
     instantaneous WER estimates. Include a beam-search decoded WER signal in this auxiliary term so \
     the loss explicitly reflects beam-decoded performance. Report the WER-derived contribution via \
     `metrics['wer_loss']` and ensure `metrics['adjusted_loss'] == metrics['loss'] + metrics['wer_loss']`.

4. Evaluation and reporting
   - Evaluate on the provided validation split. \
     Ensure `val_predictions` includes at least 3 entries and that `metrics['wer']` is the average WER over them.
   - Target `metrics['wer'] < 10`. Deterministic seeds are encouraged (`task_resources.set_seed`).
   - Use greedy decoding for simplicity; document any beam search choices in a module docstring.

5. Submission
   - When all criteria are met, call the `submit_answer` tool yourself with the JSON payload `{{"module": "submission"}}`.
     - The grader will then run `train_and_evaluate()`, recompute WER from your predictions, and validate constraints.
     - Do **not** end the conversation without that tool call; no plain-text replacement is accepted.

Hints:
   * Reuse `task_resources.get_dataloaders`, `task_resources.encode_text`, and `task_resources.decode_tokens`.
   * Aim for concise, readable code—scripts under ~250 lines tend to work best.
   * Unit-test critical helpers via the Python tool before submitting.
""".format(
    source_contents=SOURCE_PROMPT
)


def _validate_submission_payload(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Submitted answer must be a mapping with a 'module' key."
    module_name = payload.get("module")
    if not isinstance(module_name, str) or not module_name:
        return None, "The 'module' value must be a non-empty string."

    module_path = WORKSPACE_ROOT / f"{module_name}.py"
    if not module_path.exists():
        return None, f"Module {module_name!r} not found at {module_path}."
    return module_name, None


# Grading function - validates the agent's submitted answer
def grading_func(result: Any) -> bool:
    """
    Validate the agent's answer by importing their module, running
    train_and_evaluate, and checking architecture + metrics.
    """

    torch.set_num_threads(1)
    # print(result)

    module_name, error = _validate_submission_payload(result)
    if error:
        print(error)
        return False

    try:
        solution = import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to import {module_name}: {exc}")
        return False

    train_fn = getattr(solution, "train_and_evaluate", None)
    if train_fn is None:
        print("Module is missing train_and_evaluate().")
        return False

    try:
        artifacts = train_fn()
    except Exception as exc:  # pragma: no cover - debugging signal
        print(f"train_and_evaluate() raised: {exc}")
        return False

    if not isinstance(artifacts, dict):
        print("train_and_evaluate() must return a dict.")
        return False

    model = artifacts.get("model")
    metrics = artifacts.get("metrics")
    val_predictions = artifacts.get("val_predictions")

    if model is None or metrics is None or val_predictions is None:
        print("Returned dict must include 'model', 'metrics', and 'val_predictions'.")
        return False

    if not isinstance(metrics, dict):
        print("'metrics' must be a dict.")
        return False

    for key in ("loss", "wer", "wer_loss", "adjusted_loss"):
        if key not in metrics:
            print(f"'metrics' missing {key!r}.")
            return False
        if not isinstance(metrics[key], (float, int)):
            print(f"'metrics[{key}]' must be numeric.")
            return False

    if abs(metrics["adjusted_loss"] - (metrics["loss"] + metrics["wer_loss"])) > 1e-3:
        print("adjusted_loss must equal loss + wer_loss.")
        return False

    if metrics["wer_loss"] < 0:
        print("wer_loss should be non-negative to reflect WER adjustment.")
        return False

    if not isinstance(val_predictions, (list, tuple)):
        print("val_predictions must be a list or tuple.")
        return False

    if len(val_predictions) < 3:
        print("Provide at least three validation prediction entries.")
        return False

    try:
        measured_wer = recompute_wer_from_pairs(val_predictions)
    except Exception as exc:
        print(f"Failed to recompute WER from predictions: {exc}")
        return False

    if abs(measured_wer - metrics["wer"]) > 0.05:
        print(
            f"Reported WER {metrics['wer']:.3f} does not match recomputed "
            f"{measured_wer:.3f}."
        )
        return False

    if metrics["wer"] >= 10:
        print(f"WER {metrics['wer']:.3f} is too high; target is < 10.")
        return False

    if not isinstance(model, torch.nn.Module):
        print("'model' must be an instance of torch.nn.Module.")
        return False

    # multihead_layers = [
    #     module for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)
    # ]
    # if not multihead_layers:
    #     print("Model must include an nn.MultiheadAttention layer.")
    #     return False

    # if any(layer.num_heads < 2 for layer in multihead_layers):
    #     print("Each MultiheadAttention layer must have at least 2 heads.")
    #     return False

    return True
