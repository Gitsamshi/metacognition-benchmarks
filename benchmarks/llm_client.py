"""Bedrock LLM client using boto3 Converse API."""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

import boto3

from .config import BedrockConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around AWS Bedrock Converse API."""

    def __init__(self, config: BedrockConfig):
        self.config = config
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=config.region_name,
        )
        self._call_count = 0
        self._token_usage = {"input": 0, "output": 0}

    def prompt(
        self,
        user_message: str,
        *,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
        response_format: Optional[type] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            user_message: The user's message.
            system: Optional system prompt.
            history: Optional conversation history as list of
                     {"role": "user"|"assistant", "content": str}.
            response_format: If provided, append JSON schema instruction to prompt.
            model_id: Override model ID for this call.
            temperature: Override temperature for this call.
            max_tokens: Override max_tokens for this call.

        Returns:
            The assistant's text response.
        """
        messages = []
        if history:
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        if response_format is not None:
            user_message = self._append_json_instruction(user_message, response_format)

        messages.append({
            "role": "user",
            "content": [{"text": user_message}],
        })

        inference_config: dict[str, Any] = {
            "maxTokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        kwargs: dict[str, Any] = {
            "modelId": model_id or self.config.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system:
            kwargs["system"] = [{"text": system}]

        try:
            response = self.client.converse(**kwargs)
        except Exception as e:
            logger.error("Bedrock API error: %s", e)
            raise

        self._call_count += 1
        usage = response.get("usage", {})
        self._token_usage["input"] += usage.get("inputTokens", 0)
        self._token_usage["output"] += usage.get("outputTokens", 0)

        text = response["output"]["message"]["content"][0]["text"]
        return text

    def prompt_json(
        self,
        user_message: str,
        *,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Send a prompt and parse the response as JSON."""
        raw = self.prompt(
            user_message,
            system=system,
            history=history,
            model_id=model_id,
            temperature=temperature,
        )
        return self._extract_json(raw)

    @staticmethod
    def _append_json_instruction(message: str, dataclass_type: type) -> str:
        """Append JSON formatting instructions based on dataclass fields."""
        import dataclasses
        if dataclasses.is_dataclass(dataclass_type):
            fields = dataclasses.fields(dataclass_type)
            schema = {f.name: str(f.type) for f in fields}
            message += f"\n\nRespond with ONLY valid JSON matching this schema: {json.dumps(schema, ensure_ascii=False)}"
        return message

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            started = False
            for line in lines:
                if line.strip().startswith("```") and not started:
                    started = True
                    continue
                if line.strip() == "```" and started:
                    break
                if started:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    @property
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "token_usage": dict(self._token_usage),
        }


class ConversationSession:
    """Manages multi-turn conversation state."""

    def __init__(self, llm: LLMClient, system: Optional[str] = None):
        self.llm = llm
        self.system = system
        self.history: list[dict] = []

    def send(self, message: str, **kwargs) -> str:
        """Send a message and get a response, maintaining history."""
        response = self.llm.prompt(
            message,
            system=self.system,
            history=self.history,
            **kwargs,
        )
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        return response

    def send_json(self, message: str, **kwargs) -> dict:
        """Send a message and parse JSON response, maintaining history."""
        raw = self.send(message, **kwargs)
        return self.llm._extract_json(raw)

    def reset(self):
        """Clear conversation history."""
        self.history = []
