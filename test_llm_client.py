import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai import RateLimitError

from state.topic_classification.client import LLMClient


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeRateLimitError(RateLimitError):
    def __init__(self, retry_after):
        self.response = SimpleNamespace(
            headers={"retry-after": str(retry_after)})


class FakeAPI:
    def __init__(self, clock, responses):
        self.clock = clock
        self.responses = list(responses)
        self.call_times = []

    def ask(self, prompt, **kwargs):
        self.call_times.append(self.clock.time())
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class LLMClientRateLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = FakeClock()

    def client(self, name, api):
        client = LLMClient("test-team-code")
        client.client = api
        client._rate_limit_path = Path(self.temp.name) / name
        client._min_request_interval = 21.0
        client._clock = self.clock.time
        client._sleep = self.clock.sleep
        return client

    def test_clients_for_same_team_share_request_spacing(self):
        api = FakeAPI(self.clock, ["draft", "review"])
        first = self.client("team.lock", api)
        second = self.client("team.lock", api)

        self.assertEqual(first.ask("draft", "system", 10), "draft")
        self.assertEqual(second.ask("review", "system", 10), "review")

        self.assertEqual(api.call_times, [1000.0, 1021.0])
        self.assertEqual(self.clock.sleeps, [21.0])

    def test_retry_after_extends_shared_request_spacing(self):
        api = FakeAPI(self.clock, [FakeRateLimitError(45), "recovered"])
        client = self.client("retry.lock", api)

        with patch("state.topic_classification.client.random.uniform",
                   return_value=0.0):
            self.assertEqual(client.ask("draft", "system", 10), "recovered")

        self.assertEqual(api.call_times, [1000.0, 1045.0])
        self.assertEqual(self.clock.sleeps, [45.0])


if __name__ == "__main__":
    unittest.main()
