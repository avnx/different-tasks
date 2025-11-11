import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
from anthropic.types import MessageParam, ToolUnionParam  # type: ignore[import-not-found]

from task import PROMPT, TOOL_HANDLERS, TOOLS, grading_func
# print(grading_func({"module": "submission"}))

MAX_TOKENS = 63900


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        try:
            response = await client.messages.create(
                model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
            )
        except ValueError as exc:
            if "Streaming is required" not in str(exc):
                raise
            async with client.messages.stream(
                model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
            ) as stream:
                async for _ in stream:
                    pass
                response = await stream.get_final_message()

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
                stripped = content.text.strip()

                parsed_payload = None

                # Try direct JSON parsing
                try:
                    parsed_payload = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

                # Fallback: search for the first JSON object in the string
                if parsed_payload is None:
                    start = stripped.find("{")
                    end = stripped.rfind("}")
                    if 0 <= start < end:
                        candidate = stripped[start : end + 1]
                        try:
                            parsed_payload = json.loads(candidate)
                        except json.JSONDecodeError:
                            parsed_payload = None

                if isinstance(parsed_payload, dict):
                    result = tool_handlers["submit_answer"](parsed_payload)
                    submitted_answer = result["answer"]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(content, "id", "manual_json"),
                            "content": json.dumps(result),
                        }
                    )
                    has_tool_use = True
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )
        messages.append({"role": "assistant", "content": response.content})

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response; attempting automatic submission.")
            if Path("submission.py").exists():
                auto_payload = {"module": "submission"}
                result = tool_handlers["submit_answer"](auto_payload)
                if verbose:
                    print(
                        "\nAuto-submitted payload {'module': 'submission'} "
                        "after agent completion."
                    )
                return result["answer"]
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    if Path("submission.py").exists():
        auto_payload = {"module": "submission"}
        result = tool_handlers["submit_answer"](auto_payload)
        if verbose:
            print(
                "\nAuto-submitted payload {'module': 'submission'} "
                "after agent completion."
            )
        return result["answer"]
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=8,
        verbose=verbose,
    )

    success = grading_func(result)

    if success:
        print(f"✓ Run {run_id}: SUCCESS")
    else:
        print(f"✗ Run {run_id}: FAILURE - Payload: {result}")

    return run_id, success, result


async def main(concurrent: bool = True):
    # Run the test 10 times and track success rate
    num_runs = 10


    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=PROMPT,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=True))
