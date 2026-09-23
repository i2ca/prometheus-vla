"""Responses API adapter for the Direct visuomotor contract."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are the visuomotor policy for a Unitree G1 in MuJoCo.
Use only the RGB images and measured proprioception provided in this call.
Choose one short Cartesian end-effector action at a time, then inspect the next
observation. Never infer or request privileged simulator truth. Keep the hand
above the table during transit, avoid the torso and idle arm, and use small
corrections near objects. The reason field must be one concise sentence stating
visible evidence and the immediate purpose; do not include hidden reasoning.
Call actuate to move or finish_episode only when the task is complete or cannot
continue safely."""


ACTUATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "actuate",
    "description": "Execute one bounded Cartesian command for the active palm.",
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"},
            "steps": {"type": "integer", "minimum": 1, "maximum": 5},
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "quaternion_wxyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "gripper": {
                "type": "string",
                "enum": ["keep", "open", "closed"],
            },
        },
        "required": ["reason", "steps", "position", "quaternion_wxyz", "gripper"],
    },
}

FINISH_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "finish_episode",
    "description": "Stop because the task is complete or continuing is unsafe/impossible.",
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"},
            "outcome": {"type": "string", "enum": ["complete", "blocked", "unsafe"]},
        },
        "required": ["reason", "outcome"],
    },
}


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def observation_content(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Build model input without leaking absolute paths or privileged state."""
    public = {k: v for k, v in observation.items() if k not in {"images", "next_call"}}
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": "Current measured observation:\n" + json.dumps(public, ensure_ascii=False),
    }]
    for image in observation["images"]:
        content.append({"type": "input_text", "text": f"Camera: {image['camera']}"})
        content.append({"type": "input_image", "image_url": _data_url(Path(image["path"])), "detail": "high"})
    return content


@dataclass(frozen=True)
class PolicyDecision:
    response_id: str
    tool: str
    arguments: dict[str, Any]
    usage: dict[str, Any] | None


class AstraPolicy:
    def __init__(self, client: Any, model: str = "gpt-6-astra", reasoning_effort: str = "xhigh"):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    def decide(self, observation: dict[str, Any]) -> PolicyDecision:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            reasoning={"effort": self.reasoning_effort},
            input=[{"role": "user", "content": observation_content(observation)}],
            tools=[ACTUATE_TOOL, FINISH_TOOL],
            tool_choice="required",
            parallel_tool_calls=False,
        )
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if len(calls) != 1:
            raise RuntimeError(f"expected exactly one function call, got {len(calls)}")
        call = calls[0]
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return PolicyDecision(
            response_id=response.id,
            tool=call.name,
            arguments=json.loads(call.arguments),
            usage=usage,
        )

