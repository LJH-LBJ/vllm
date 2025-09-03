# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.interfaces import supports_eagle3


@pytest.mark.parametrize(
    "model_path",
    [("nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized")])
def test_llama(vllm_runner, example_prompts, model_path, monkeypatch):
    # Set environment variable for V1 engine serialization
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        eagle3_supported = vllm_model.apply_model(supports_eagle3)
        assert eagle3_supported

        vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                  max_tokens=20)
        print(vllm_outputs)
        assert vllm_outputs


@pytest.mark.parametrize(
    "model_path",
    [("nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized")])
def test_qwen(vllm_runner, example_prompts, model_path, monkeypatch):
    # Set environment variable for V1 engine serialization
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        eagle3_supported = vllm_model.apply_model(supports_eagle3)
        assert eagle3_supported

        vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                  max_tokens=20)
        print(vllm_outputs)
        assert vllm_outputs


@pytest.mark.parametrize(
    "model_path",
    [("Qwen/Qwen2-VL-7B-Instruct")])  # Use a publicly available Qwen2-VL model
def test_qwen2_vl_eagle3_support(vllm_runner, example_prompts, model_path, monkeypatch):
    """Test that Qwen2-VL models support Eagle3 interface."""
    # Set environment variable for V1 engine serialization
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        # Test that the model supports Eagle3 interface
        eagle3_supported = vllm_model.apply_model(supports_eagle3)
        assert eagle3_supported, "Qwen2-VL model should support Eagle3 interface"

        # Basic functionality test - ensure model can generate text
        # Using simple text prompts for the multimodal model
        simple_prompts = ["Hello, how are you?", "What is the capital of France?"]
        vllm_outputs = vllm_model.generate_greedy(simple_prompts,
                                                  max_tokens=10)
        print("Qwen2-VL Eagle3 test outputs:", vllm_outputs)
        assert vllm_outputs
        assert len(vllm_outputs) == len(simple_prompts)
