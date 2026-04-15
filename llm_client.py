"""AWS Bedrock client for Claude models (converse API)."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
from botocore.exceptions import ClientError

import config as cfg

log = logging.getLogger(__name__)


class BedrockClient:

    def __init__(self, region: str = cfg.AWS_REGION):
        from botocore.config import Config
        boto_config = Config(
            read_timeout=300,
            connect_timeout=10,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=cfg.MAX_WORKERS + 5,
        )
        self.client = boto3.client("bedrock-runtime", region_name=region, config=boto_config)
        self.call_count = 0

    def invoke(
        self,
        model_key: str,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        model_id = cfg.BEDROCK_MODELS[model_key]
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": tools}

        last_err = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                resp = self.client.converse(**kwargs)
                self.call_count += 1
                return resp
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in ("ThrottlingException", "TooManyRequestsException",
                            "ServiceUnavailableException", "ModelTimeoutException"):
                    wait = cfg.RETRY_BACKOFF_BASE ** attempt
                    log.warning("Bedrock %s (attempt %d/%d), retrying in %.1fs",
                                code, attempt, cfg.MAX_RETRIES, wait)
                    time.sleep(wait)
                    last_err = exc
                else:
                    raise
        raise last_err  # type: ignore[misc]

    def invoke_tool(
        self,
        model_key: str,
        messages: list[dict],
        tool_spec: dict,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        tools = [{"toolSpec": tool_spec}]
        resp = self.invoke(
            model_key, messages,
            system=system, tools=tools,
            max_tokens=max_tokens, temperature=temperature,
        )
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                return block["toolUse"]["input"]
        # Fallback: try to extract JSON from text response
        for block in resp["output"]["message"]["content"]:
            if "text" in block:
                text = block["text"]
                # Try direct parse
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Try to find JSON object in text
                import re
                match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except (json.JSONDecodeError, ValueError):
                        pass
                # Last resort: return text as answer field for eval tools
                return {"answer": text.strip(), "confidence": 0.5, "reasoning": "parsed from text"}
        raise ValueError("No output in model response")

    def invoke_text(
        self,
        model_key: str,
        messages: list[dict],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        resp = self.invoke(
            model_key, messages,
            system=system, max_tokens=max_tokens, temperature=temperature,
        )
        parts = []
        for block in resp["output"]["message"]["content"]:
            if "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    @staticmethod
    def user_msg(text: str) -> list[dict]:
        return [{"role": "user", "content": [{"text": text}]}]
