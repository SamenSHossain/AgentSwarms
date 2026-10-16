"""Message roles in agent-framework traces, for the chat adapter.

A framework trace mixes the agents' own turns with the task and with tool output.
Each message is one of:

* ``agent``: an OpenAI/Anthropic ``assistant``, a LangChain ``ai`` message, an
  AutoGen chat message (``TextMessage``, ``ToolCallSummaryMessage``,
  ``HandoffMessage``, ...: any ``*Message`` but the ones below); its author is the
  speaker the chat aliases find (``name``, ``source``, ...);
* ``inner``: any other AutoGen ``*Event`` (``ThoughtEvent``, a streaming chunk,
  ``ToolCallRequestEvent``, ``SelectSpeakerEvent``, ...): a step inside one agent,
  which AgentChat does not pass to the others, so no post (a call request still
  opens the calls it makes);
* ``task``: ``system``/``developer``, and ``user``/``human`` (AutoGen: source
  ``user``) unless it names another speaker; the author is ``task``;
* ``tool``: an OpenAI/LangChain ``tool`` or ``function`` message, an AutoGen
  ``ToolCallExecutionEvent`` or ``CodeExecutionEvent`` (autogen-core:
  ``FunctionExecutionResultMessage``), an Anthropic user turn of ``tool_result``
  blocks: the calling agent's private read, never a public post.

A tool result goes to the agent whose call id it answers (``tool_call_id``,
``call_id``, ``tool_use_id``), else to the agent an AutoGen event names (its
``source``: the agent whose tool ran), else to the one agent with calls still
pending (unless the result names an id those calls do not have), else to
``tool``.  A ``user`` turn that names its speaker keeps it: AutoGen 0.2 group
chats and LangGraph supervisors pass agents' turns on as named user messages.
A message with no recognised ``role`` or ``type`` value is an ordinary chat post.
"""

from __future__ import annotations

ROLES = {"assistant": "agent", "ai": "agent", "model": "agent",
         "system": "task", "developer": "task", "user": "task", "human": "task",
         "tool": "tool", "function": "tool", "ipython": "tool"}
AUTOGEN_TOOL = ("ToolCallExecutionEvent", "CodeExecutionEvent", "FunctionExecutionResultMessage")
AUTOGEN_TASK = ("SystemMessage",)
AUTOGEN_CALLS = ("ToolCallRequestEvent", "AssistantMessage")   # their content lists FunctionCalls: {id, name, arguments}
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


def autogen_type(m: dict) -> str | None:
    """An AutoGen message's ``type`` (a capitalised ``*Message`` or ``*Event``) when no OpenAI-style role overrides it."""
    r, t = m.get("role"), m.get("type")
    if isinstance(t, str) and not (isinstance(r, str) and r.lower() in ROLES) and t[:1].isupper() \
            and t.endswith(("Message", "Event")):
        return t
    return None


def role(m: dict, speaker) -> str | None:
    """'agent', 'inner', 'task' or 'tool', or None for a message that carries no framework role."""
    r, t = m.get("role"), m.get("type")
    task_speaker = str(speaker or "").lower() in TASK_SPEAKERS
    if a := autogen_type(m):   # every speaker is named, `user` too
        return ("tool" if a in AUTOGEN_TOOL else "inner" if a.endswith("Event") else
                "task" if a in AUTOGEN_TASK or task_speaker else "agent")
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
    if autogen_type(m) in AUTOGEN_CALLS:
        ids += [b.get("id") for b in blocks]
    return ids + [b.get("id") for b in blocks if b.get("type") == "tool_use"]


def result_ids(m: dict) -> list:
    """The call ids a tool message answers."""
    ids = [m.get("tool_call_id"), m.get("call_id")]
    ids += [b.get("call_id") or b.get("tool_use_id") for b in _blocks(m)]
    return [i for i in ids if i is not None]


def result_text(m: dict):
    """A ``CodeExecutionEvent`` keeps its output under ``result``, not ``content``."""
    r = m.get("result")
    return r.get("output") if isinstance(r, dict) else None


class Calls:
    """Tool calls made and not yet answered, by agent.  An agent's next turn closes its earlier calls (the APIs
    answer every call before the agent speaks again), so a call that was never answered cannot linger."""

    def __init__(self):
        self.pending: dict[object, str] = {}

    def open(self, agent: str, ids: list) -> None:
        self.pending = {k: a for k, a in self.pending.items() if a != agent}
        for i in ids:
            self.pending[i if i is not None else object()] = agent

    def answer(self, ids: list, named: str | None = None) -> str | None:
        """The caller: by call id, else ``named`` (an AutoGen event's source), else the one agent with calls pending;
        None when nothing settles it.  A result naming an id no pending call has rules out the calls that have ids."""
        hit = {self.pending.pop(i) for i in ids if i in self.pending}
        if hit:
            return hit.pop() if len(hit) == 1 else None
        if named:
            return named
        cand = [k for k in self.pending if not ids or type(k) is object]   # object(): a call made with no id
        if cand and len({self.pending[k] for k in cand}) == 1:
            return self.pending.pop(cand[0])
        return None
