"""
LLM backend factory.
Abstracts over Ollama, llama.cpp, and HuggingFace Transformers
so the rest of the codebase stays backend-agnostic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("gromacs_agent.backends")


def load_model(llm_cfg: dict):
    """
    Instantiate and return the appropriate SmolAgent model wrapper
    based on config.yaml llm.backend value.

    Supported backends:
        'ollama'       → LiteLLMModel (Ollama server)
        'llamacpp'     → LiteLLMModel (llama.cpp server)
        'transformers' → TransformersModel (local HF model)

    Args:
        llm_cfg: The 'llm' section of config.yaml.

    Returns:
        A SmolAgent-compatible model object.
    """
    backend = llm_cfg["backend"].lower()

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------
    if backend == "ollama":
        from smolagents import LiteLLMModel
        model = LiteLLMModel(
            model_id=llm_cfg["model_id"],
            api_base=llm_cfg.get("api_base", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )
        logger.info("Loaded Ollama model: %s", llm_cfg["model_id"])
        return model

    # ------------------------------------------------------------------
    # llama.cpp server backend
    # ------------------------------------------------------------------
    if backend == "llamacpp":
        from smolagents import LiteLLMModel
        model = LiteLLMModel(
            model_id=llm_cfg["model_id"],
            api_base=llm_cfg.get("api_base", "http://localhost:8080"),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )
        logger.info("Loaded llama.cpp model: %s", llm_cfg["model_id"])
        return model

    # ------------------------------------------------------------------
    # HuggingFace Transformers backend (fully local, no server)
    # ------------------------------------------------------------------
    if backend == "transformers":
        from smolagents import TransformersModel
        import torch

        device = llm_cfg.get("device", "cpu")
        load_in_4bit = llm_cfg.get("load_in_4bit", False)

        model_kwargs: dict = {}
        if load_in_4bit:
            # Requires bitsandbytes
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = TransformersModel(
            model_id=llm_cfg["model_id"],
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            temperature=llm_cfg.get("temperature", 0.1),
            max_new_tokens=llm_cfg.get("max_tokens", 4096),
            model_kwargs=model_kwargs,
        )
        logger.info(
            "Loaded Transformers model: %s (device=%s, 4bit=%s)",
            llm_cfg["model_id"], device, load_in_4bit
        )
        return model

    raise ValueError(
        f"Unknown LLM backend: '{backend}'. "
        "Supported: 'ollama', 'llamacpp', 'transformers'."
    )