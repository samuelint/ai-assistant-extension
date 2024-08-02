from fastapi.testclient import TestClient
import pytest
from base_assistant_extension.extension_host import ExtensionHost
from openai import OpenAI
from tests.test_functional.fixture_extension import FixtureExtension


@pytest.fixture
def host() -> ExtensionHost:
    return ExtensionHost(
        extension=FixtureExtension(),
        port=7680,
    )


@pytest.fixture
def host_test_server(host: ExtensionHost):
    return TestClient(app=host.app)


@pytest.fixture
def openai_client(host_test_server):
    return OpenAI(
        base_url="http://testserver/openai/v1",
        http_client=host_test_server,
    )


class TestMetadata:
    def test_name_metadata(self, host_test_server):
        result = host_test_server.get("/metadata").json()

        assert result["name"] == "joker"

    def test_description_metadata(self, host_test_server):
        result = host_test_server.get("/metadata").json()

        assert result["description"] == "Tell jokes."


class TestOpenAIInference:
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
