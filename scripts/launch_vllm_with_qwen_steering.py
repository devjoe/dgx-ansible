#!/usr/bin/env python3
"""Launch vLLM after registering the experiment Qwen steering hook."""

from __future__ import annotations

import qwen_dir_steering_vllm_plugin  # noqa: F401
from vllm.entrypoints.cli.main import main


if __name__ == "__main__":
    main()
