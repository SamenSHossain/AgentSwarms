"""Message roles in agent-framework traces, for the chat adapter.

A framework trace mixes the agents' own turns with the task and with tool output.
Each message is one of:

* ``agent``: an OpenAI/Anthropic ``assistant``, a LangChain ``ai`` message, any
  AutoGen message or event (``TextMessage``, ``ToolCallRequestEvent``, ...); its
  author is the speaker the chat aliases find (``name``, ``source``, ...);
* ``task``: ``system``/``developer``, and ``user``/``human`` (AutoGen: source
  ``user``) unless it names another speaker; the author is ``task``;
* ``tool``: an OpenAI/LangChain ``tool`` or ``function`` message, an AutoGen
  ``ToolCallExecutionEvent``, an Anthropic user turn of ``tool_result`` blocks: the
  calling agent's private read, never a public post.

A tool result goes to the agent whose call id it answers (``tool_call_id``,
``call_id``, ``tool_use_id``), else to the one agent with a call still pending,
else to ``tool``.  A ``user`` turn that names its speaker keeps it: AutoGen 0.2
group chats and LangGraph supervisors pass agents' turns on as named user messages.
A message with no recognised ``role`` or ``type`` value is an ordinary chat post.
"""

from __future__ import annotations

ROLES = {"assistant": "agent", "ai": "agent", "model": "agent",
         "system": "task", "developer": "task", "user": "task", "human": "task",
         "tool": "tool", "function": "tool", "ipython": "tool"}
AUTOGEN_TOOL = ("ToolCallExecutionEvent", "FunctionExecutionResultMessage")
AUTOGEN_TASK = ("SystemMessage",)
TASK_SPEAKERS = ("user", "human", "system")   # an AutoGen source or a user turn's name that is not an agent


def unwrap(m: dict) -> dict:
    """LangChain's ``messages_to_dict`` form ``{"type": "ai", "data": {...}}`` as one flat message."""
    data = m.get("data")
    if isinstance(data, dict) and "content" not in m and str(m.get("type", "")).lower() in ROLES:
        return {**data, "type": m["type"]}
    return m


def _blocks(m: dict) -> list[dict]:
    c = m.get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def role(m: dict, speaker) -> str | None:
    """'agent', 'task' or 'tool', or None for a message that carries no framework role."""
    r, t = m.get("role"), m.get("type")
    task_speaker = str(speaker or "").lower() in TASK_SPEAKERS
    if isinstance(t, str) and not (isinstance(r, str) and r.lower() in ROLES) \
            and t[:1].isupper() and t.endswith(("Message", "Event")):   # AutoGen: every speaker is named, `user` too
        return "tool" if t in AUTOGEN_TOOL else "task" if t in AUTOGEN_TASK or task_speaker else "agent"
    r = r.lower() if isinstance(r, str) and r.lower() in ROLES else t.lower() if isinstance(t, str) else None
    if r not in ROLES:
        return None
    if r in ("user", "human"):
        if any(b.get("type") == "tool_result" for b in _blocks(m)):   # Anthropic: tool results come back as a user turn
            return "tool"
        return "agent" if speaker and not task_speaker else "task"
    return ROLES[r]


def call_ids(m: dict) -> list:
    """The tool calls an agent message makes; an OpenAI ``function_call`` has no id (None)."""
    calls = m.get("tool_calls")
    ids = [c.get("id") for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
    if isinstance(m.get("function_call"), dict):
        ids.append(None)
    blocks = _blocks(m)
    if m.get("type") == "ToolCallRequestEvent":
        ids += [b.get("id") for b in blocks]
    return ids + [b.get("id") for b in blocks if b.get("type") == "tool_use"]


def result_ids(m: dict) -> list:
    """The call ids a tool message answers."""
    ids = [m.get("tool_call_id"), m.get("call_id")]
    ids += [b.get("call_id") or b.get("tool_use_id") for b in _blocks(m)]
    return [i for i in ids if i is not None]


class Calls:
    """Tool calls made and not yet answered, by agent.  An agent's next turn closes its earlier calls (the APIs
    answer every call before the agent speaks again), so a call that was never answered cannot linger."""

    def __init__(self):
        self.pending: dict[object, str] = {}

    def open(self, agent: str, ids: list) -> None:
        self.pending = {k: a for k, a in self.pending.items() if a != agent}
        for i in ids:
            self.pending[i if i is not None else object()] = agent

    def answer(self, ids: list) -> str | None:
        """The caller: by call id, else the only agent with a call pending; None when neither settles it."""
        hit = {self.pending.pop(i) for i in ids if i in self.pending}
        if len(hit) == 1:
            return hit.pop()
        if not hit and len(self.pending) == 1:
            return self.pending.popitem()[1]
        return None
