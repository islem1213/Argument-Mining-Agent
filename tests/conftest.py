import json

import pytest

SAMPLE = (
    "Schools should start later in the morning. Teenagers need more sleep, so later starts improve learning. "
    "However, critics worry about transport costs. Buses would have to run on new schedules, "
    "but many districts already share buses across schools."
)


class FakeClient:
    """Scripted stand-in for an OpenAI-compatible server: returns queued replies per schema name."""

    def __init__(self, replies: dict[str, list]):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, messages, schema):
        self.calls.append((schema.__name__, messages))
        reply = self.replies[schema.__name__].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture
def sample():
    return SAMPLE
