hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

### Task Overview

The bundled RL task asks the agent to implement a compact Listen, Attend and Spell
style recognizer with:

- a PyTorch listener/speller architecture that incorporates `nn.MultiheadAttention`
- training on the synthetic dataset defined in `task_resources.py`
- a WER-aware loss adjustment surfaced in the reported metrics
- a submission module exposing `train_and_evaluate()`

See `task.py` for the exact prompt, tools, and grading logic.


## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
