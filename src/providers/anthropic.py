"""
Anthropic Provider

Provider class for Anthropic Claude models with batch processing.
"""

import json
from textwrap import dedent
from typing import List, Type, Dict, Any, Optional, Union
from pydantic import BaseModel
from anthropic import Anthropic
from .base import BaseBatchProvider


class AnthropicBatchProvider(BaseBatchProvider):
    """Anthropic batch processing provider."""
    
    # Batch limitations from https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#batch-limitations
    MAX_REQUESTS = 100_000      # Max requests per batch
    MAX_TOTAL_SIZE_MB = 256     # Max total batch size in MB
    
    def __init__(self, rate_limits: Dict[str, int] = None):
        super().__init__(rate_limits)
        self.client = Anthropic()  # Automatically reads ANTHROPIC_API_KEY from env
    
    def get_default_rate_limits(self) -> Dict[str, int]:
        """Get default rate limits for Anthropic (basic tier)."""
        return {
            "batches_per_minute": 5,
            "requests_per_minute": 500
        }
    
    def validate_batch(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]]) -> None:
        if not messages:
            return
            
        if len(messages) > self.MAX_REQUESTS:
            raise ValueError(f"Too many requests: {len(messages)} > {self.MAX_REQUESTS}")
        
        total_size = sum(len(str(msg)) for msg in messages)
        max_size_bytes = self.MAX_TOTAL_SIZE_MB * 1024 * 1024
        if total_size > max_size_bytes:
            raise ValueError(f"Batch too large: ~{total_size/1024/1024:.1f}MB > {self.MAX_TOTAL_SIZE_MB}MB")
    
    def prepare_batch_requests(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]], **kwargs) -> List[dict]:
        if not messages:
            return []
        
        batch_requests = []
        for i, conversation in enumerate(messages):
            system_messages = [msg["content"] for msg in conversation if msg.get("role") == "system"]
            user_messages = [msg for msg in conversation if msg.get("role") != "system"]
            
            if response_model:
                schema = response_model.model_json_schema()
                system_message = dedent(f"""
                    As a genius expert, your task is to understand the content and provide
                    the parsed objects in json that match the following json_schema:

                    {json.dumps(schema, indent=2, ensure_ascii=False)}

                    Make sure to return an instance of the JSON, not the schema itself
                """).strip()
                
                combined_system = system_message
                if system_messages:
                    original_system = "\n\n".join(system_messages)
                    combined_system = f"{original_system}\n\n{system_message}"
            else:
                combined_system = "\n\n".join(system_messages) if system_messages else None
            
            request = {
                "custom_id": f"request_{i}",
                "params": {
                    "messages": user_messages,
                    **kwargs
                }
            }
            if combined_system:
                request["params"]["system"] = combined_system
                
            batch_requests.append(request)
        
        return batch_requests
    
    def create_batch(self, requests: List[dict]) -> str:
        batch_response = self.client.messages.batches.create(requests=requests)
        return batch_response.id
    
    def get_batch_status(self, batch_id: str) -> str:
        batch_status = self.client.messages.batches.retrieve(batch_id)
        return batch_status.processing_status
    
    def _is_batch_completed(self, status: str) -> bool:
        return status == "ended"
    
    def _is_batch_failed(self, status: str) -> bool:
        return status in ["canceled", "expired"]
    
    def get_results(self, batch_id: str) -> List[Any]:
        return self.client.messages.batches.results(batch_id)
    
    def parse_results(self, results: List[Any], response_model: Optional[Type[BaseModel]]) -> Union[List[BaseModel], List[str]]:
        if not results:
            return []
            
        parsed_results = []
        errors = []
        
        for result in results:
            try:
                if result.result.type != "succeeded":
                    errors.append(result.model_dump())
                    continue
                    
                message_content = result.result.message.content[0].text
                
                if not response_model:
                    parsed_results.append(message_content)
                    continue

                start_idx = message_content.find('{')
                end_idx = message_content.rfind('}') + 1
                
                if start_idx == -1 or end_idx <= start_idx:
                    raise ValueError(f"No JSON found in response: {message_content}")
                    
                json_str = message_content[start_idx:end_idx]
                json_data = json.loads(json_str)
                parsed_results.append(response_model(**json_data))                    
                    
            except Exception as e:
                errors.append({"error": str(e), "result": result.model_dump()})
            
        if errors:
            print(f"\n❌ Batch processing errors ({len(errors)} failed):")
            for i, error in enumerate(errors, 1):
                print(f"\nError {i}:")
                if "error" in error:
                    print(f"  Exception: {error['error']}")
                if "result" in error:
                    result_data = error["result"]
                    if isinstance(result_data, dict):
                        if "result" in result_data and isinstance(result_data["result"], dict):
                            res = result_data["result"]
                            if "type" in res:
                                print(f"  Result type: {res['type']}")
                            if "error" in res:
                                print(f"  API error: {res['error']}")
                        if "custom_id" in result_data:
                            print(f"  Custom ID: {result_data['custom_id']}")
                    else:
                        print(f"  Result data: {result_data}")
                print(f"  Full error: {error}")
            raise RuntimeError(f"Some batch requests failed: {len(errors)} errors")
        
        return parsed_results