import asyncio

from bluesec1_agent.llm import OpenAIStructuredStepClient
from bluesec1_agent.settings import Settings


async def main() -> None:
    s = Settings()
    async with OpenAIStructuredStepClient(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key.get_secret_value(),
        model=s.llm_default_model,
        timeout_seconds=s.llm_timeout_seconds,
        max_completion_tokens=s.llm_max_completion_tokens,
        max_retries=s.llm_max_step_retries,
        response_format_mode=s.llm_response_format,
    ) as llm:
        step = await llm.next_step(
            [
                {
                    "role": "user",
                    "content": "Alert on host id host-1. Call get_entity for entity_id host-1.",
                }
            ]
        )
    print(step.model_dump_json(indent=2))


asyncio.run(main())
