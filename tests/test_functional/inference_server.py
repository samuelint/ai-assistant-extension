from typing import Optional
from langchain_openai_api_bridge.fastapi.langchain_openai_api_bridge_fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from fastapi import FastAPI
from langchain_openai import ChatOpenAI


class InferenceServer:
    def __init__(
        self,
        port: Optional[int] = 7777,
    ) -> None:
        self.port = port
        self.app = FastAPI()

        def create_agent(dto: CreateAgentDto):
            return ChatOpenAI(
                temperature=dto.temperature or 0.7,
                model=dto.model,
                max_tokens=dto.max_tokens,
                api_key=dto.api_key,
            )

        bridge = LangchainOpenaiApiBridgeFastAPI(
            app=self.app,
            agent_factory_provider=create_agent,
        )

        bridge.bind_openai_chat_completion()
