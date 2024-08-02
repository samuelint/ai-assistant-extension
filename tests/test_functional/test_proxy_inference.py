from fastapi.testclient import TestClient
import pytest
from base_assistant_extension import ExtensionHost
from openai import OpenAI
from tests.test_functional.fixture_extension import FixtureExtension
from tests.test_functional.inference_server import InferenceServer


@pytest.fixture
def inference_server() -> ExtensionHost:
    server = InferenceServer()
    return TestClient(
        base_url="http://testserver_inference",
        app=server.app,
    )


@pytest.fixture
def host(inference_server: TestClient) -> ExtensionHost:
    return ExtensionHost(
        extension=FixtureExtension(),
        inference_url="http://testserver_inference/openai/v1",
        inference_http_client=inference_server,
    )


@pytest.fixture
def host_test_server(host: ExtensionHost):
    return TestClient(
        app=host.app,
    )


@pytest.fixture
def openai_client(host_test_server: TestClient):
    return OpenAI(
        base_url="http://testserver/openai/v1",
        http_client=host_test_server,
    )


class TestProxyInference:
    def test_chat_completion(self, openai_client: OpenAI):
        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Say hi",
                }
            ],
        )

        assert len(chat_completion.choices[0].message.content) > 0

        result = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Is this a joke? Answer by yes or no."
                    + chat_completion.choices[0].message.content,
                }
            ],
        )
        assert "yes" in result.choices[0].message.content.lower()
