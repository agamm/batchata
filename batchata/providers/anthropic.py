"""
Anthropic Provider

Provider class for Anthropic Claude models with batch processing.
"""

import json
from textwrap import dedent
from typing import List, Type, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel
from anthropic import Anthropic
from tokencost import calculate_cost_by_tokens
from .base import BaseBatchProvider
from ..citations import Citation
from ..types import BatchResult



class AnthropicBatchProvider(BaseBatchProvider):
    """Anthropic batch processing provider."""
    
    # Supported models for this provider  
    SUPPORTED_MODELS = {
        # Claude 4 models
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        # Claude 3.7 models
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest",
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        # Claude 3 models
        "claude-3-haiku-20240307",
        # Legacy models (deprecated)
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-haiku-20240307",
    }
    
    # Models that support file/document input (PDFs, images, etc.)
    FILE_CAPABLE_MODELS = {
        # Claude 4 models
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        # Claude 3.7 models
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest",
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        # Legacy models (deprecated)
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-haiku-20240307",
    }
    
    # Batch limitations from https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#batch-limitations
    MAX_REQUESTS = 100_000      # Max requests per batch
    MAX_TOTAL_SIZE_MB = 256     # Max total batch size in MB
    
    # Batch API offers 50% discount on all usage
    BATCH_DISCOUNT = 0.5
    
    def __init__(self, rate_limits: Optional[Dict[str, int]] = None):
        super().__init__(rate_limits)
        self.client = Anthropic()  # Automatically reads ANTHROPIC_API_KEY from env
    
    def get_default_rate_limits(self) -> Dict[str, int]:
        """Get default rate limits for Anthropic (basic tier)."""
        return {
            "batches_per_minute": 5,
            "requests_per_minute": 500
        }
    
    @classmethod
    def get_supported_models(cls) -> set:
        """Get set of model names supported by this provider."""
        return cls.SUPPORTED_MODELS
    
    def validate_model_capabilities(self, model: str, has_files: bool, has_messages: bool) -> None:
        """Validate that the model supports the input type."""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' is not supported by Anthropic provider")
        
        if has_files and model not in self.FILE_CAPABLE_MODELS:
            raise ValueError(
                f"Model '{model}' does not support file/document input. "
                f"Please use a file-capable model like: {', '.join(sorted(self.FILE_CAPABLE_MODELS))}"
            )
        
        if has_messages and has_files:
            raise ValueError("Cannot provide both messages and files")
        
        if not has_messages and not has_files:
            raise ValueError("Must provide either messages or files")
    
    def validate_batch(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]]) -> None:
        if not messages:
            return
            
        if len(messages) > self.MAX_REQUESTS:
            raise ValueError(f"Too many requests: {len(messages)} > {self.MAX_REQUESTS}")
        
        total_size = sum(len(str(msg)) for msg in messages)
        max_size_bytes = self.MAX_TOTAL_SIZE_MB * 1024 * 1024
        if total_size > max_size_bytes:
            raise ValueError(f"Batch too large: ~{total_size/1024/1024:.1f}MB > {self.MAX_TOTAL_SIZE_MB}MB")
        
        # Validate message format
        for i, conversation in enumerate(messages):
            if not isinstance(conversation, list):
                raise ValueError(f"Message {i} must be a list of message objects")
            
            for j, message in enumerate(conversation):
                if not isinstance(message, dict):
                    raise ValueError(f"Message {i}[{j}] must be a dictionary")
                
                if "role" not in message:
                    raise ValueError(f"Message {i}[{j}] missing required 'role' field")
                
                if "content" not in message:
                    raise ValueError(f"Message {i}[{j}] missing required 'content' field")
                
                valid_roles = {"user", "assistant", "system"}
                if message["role"] not in valid_roles:
                    raise ValueError(f"Message {i}[{j}] has invalid role '{message['role']}'. Must be one of: {valid_roles}")
        
    
    def prepare_batch_requests(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]], enable_citations: bool = False, **kwargs) -> List[dict]:
        if not messages:
            return []
        
        
        # Filter kwargs to only include valid API parameters
        valid_api_params = {
            'model', 'max_tokens', 'temperature', 'top_k', 'top_p', 
            'metadata', 'stop_sequences', 'stream'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_api_params}
        
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
            
            # Add citation instruction if citations are enabled
            if enable_citations:
                citation_instruction = "Include citations in your response referencing the specific parts of the documents you used."
                if combined_system:
                    combined_system = f"{combined_system}\n\n{citation_instruction}"
                else:
                    combined_system = citation_instruction
            
            request = {
                "custom_id": f"request_{i}",
                "params": {
                    "messages": user_messages,
                    **filtered_kwargs
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
        return list(self.client.messages.batches.results(batch_id))
    
    def _parse_content_blocks(self, content_blocks: List[Any]) -> Tuple[str, List[Citation]]:
        """Parse content blocks to extract text and citations.
        
        Args:
            content_blocks: List of content blocks from API response
            
        Returns:
            Tuple of (full_text, citations)
        """
        text_parts = []
        all_citations = []
        
        for block in content_blocks:
            if not hasattr(block, 'text'):
                continue
                
            text_parts.append(block.text)
            
            if not hasattr(block, 'citations') or not block.citations:
                continue
                
            for cit in block.citations:
                citation_data = {
                    'type': getattr(cit, 'type', ''),
                    'cited_text': getattr(cit, 'cited_text', ''),
                    'document_index': getattr(cit, 'document_index', 0),
                    'document_title': getattr(cit, 'document_title', None),
                }
                
                # Add location fields based on type
                cit_type = getattr(cit, 'type', '')
                if cit_type == 'char_location':
                    citation_data['start_char_index'] = getattr(cit, 'start_char_index', None)
                    citation_data['end_char_index'] = getattr(cit, 'end_char_index', None)
                elif cit_type == 'page_location':
                    citation_data['start_page_number'] = getattr(cit, 'start_page_number', None)
                    citation_data['end_page_number'] = getattr(cit, 'end_page_number', None)
                elif cit_type == 'content_block_location':
                    citation_data['start_block_index'] = getattr(cit, 'start_block_index', None)
                    citation_data['end_block_index'] = getattr(cit, 'end_block_index', None)
                
                all_citations.append(Citation(**citation_data))
        
        return "".join(text_parts), all_citations
    
    def _extract_json_from_text(self, text: str, response_model: Type[BaseModel]) -> BaseModel:
        """Extract JSON from text and parse into Pydantic model.
        
        Args:
            text: Text containing JSON
            response_model: Pydantic model class
            
        Returns:
            Parsed Pydantic model instance
            
        Raises:
            ValueError: If no valid JSON found
        """
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"No JSON found in response: {text}")
            
        json_str = text[start_idx:end_idx]
        json_data = json.loads(json_str)
        return response_model(**json_data)
    
    def _map_citations_to_fields(self, content_blocks: List[Any], response_model: Type[BaseModel]) -> Dict[str, List[Citation]]:
        """
        Map citations to field names using proper JSON parsing instead of fragile regex.
        
        Args:
            content_blocks: List of content blocks from API response
            response_model: Pydantic model for field names
            
        Returns:
            Dict mapping field names to their citations
        """
        try:
            # Get the complete text and extract JSON using same logic as _extract_json_from_text
            full_text = "".join(getattr(block, 'text', '') for block in content_blocks)
            
            # Use the same JSON extraction logic as _extract_json_from_text
            start_idx = full_text.find('{')
            end_idx = full_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                return {}
                
            json_str = full_text[start_idx:end_idx]
            parsed_json = json.loads(json_str)
            
            # Map citations based on JSON structure and content block positions
            return self._extract_field_citations(content_blocks, parsed_json, response_model)
            
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, fall back to empty mapping
            # This is safer than the previous regex approach
            return {}
    
    def _extract_field_citations(self, content_blocks: List[Any], parsed_json: dict, response_model: Type[BaseModel]) -> Dict[str, List[Citation]]:
        """
        Extract citations for each field using JSON structure analysis.
        
        Args:
            content_blocks: List of content blocks from API response
            parsed_json: Parsed JSON object
            response_model: Pydantic model for field validation
            
        Returns:
            Dict mapping field names to their citations
        """
        field_citations = {}
        
        # Get valid field names from the response model
        valid_fields = set(response_model.model_fields.keys()) if hasattr(response_model, 'model_fields') else set()
        
        # Build a map of content block index to citations
        block_citations = {}
        for i, block in enumerate(content_blocks):
            if hasattr(block, 'citations') and block.citations:
                block_citations[i] = [
                    Citation(
                        type=getattr(cit, 'type', ''),
                        cited_text=getattr(cit, 'cited_text', ''),
                        document_index=getattr(cit, 'document_index', 0),
                        document_title=getattr(cit, 'document_title', None),
                        start_page_number=getattr(cit, 'start_page_number', None),
                        end_page_number=getattr(cit, 'end_page_number', None),
                        start_char_index=getattr(cit, 'start_char_index', None),
                        end_char_index=getattr(cit, 'end_char_index', None),
                        start_content_block_index=getattr(cit, 'start_content_block_index', None),
                        end_content_block_index=getattr(cit, 'end_content_block_index', None)
                    )
                    for cit in block.citations
                ]
        
        # Use a more sophisticated approach to map citations to fields
        # based on the structure of the JSON and content blocks
        for field_name, field_value in parsed_json.items():
            if field_name in valid_fields:
                # Find the approximate position of this field's value in the full text
                field_value_str = json.dumps(field_value) if not isinstance(field_value, str) else field_value
                
                # Look for citations in content blocks that likely correspond to this field
                field_citations[field_name] = []
                
                # Simple heuristic: assign citations from blocks based on field order
                # This works for flat JSON structures with simple field mappings
                for block_idx, citations in block_citations.items():
                    block_text = getattr(content_blocks[block_idx], 'text', '')
                    
                    # If the block text appears to be related to this field value
                    if self._text_likely_matches_field(block_text, field_value_str, field_name):
                        field_citations[field_name].extend(citations)
        
        return field_citations
    
    def _text_likely_matches_field(self, block_text: str, field_value: str, field_name: str) -> bool:
        """
        Heuristic to determine if a content block's text likely corresponds to a JSON field.
        
        Args:
            block_text: Text from the content block
            field_value: String representation of the field value
            field_name: Name of the field
            
        Returns:
            True if the text likely matches the field
        """
        # Simple heuristics that work better than regex parsing:
        
        # 1. Check if the block text contains the field value
        if field_value.strip('"') in block_text:
            return True
            
        # 2. Check if the block text appears after the field name in JSON
        if f'"{field_name}"' in block_text:
            return True
            
        # 3. For string values, check for partial matches
        if isinstance(field_value, str) and len(field_value) > 3:
            # Remove quotes and check for substantial overlap
            clean_value = field_value.strip('"')
            if len(clean_value) > 3 and clean_value in block_text:
                return True
        
        return False
    
    def parse_results(self, results: List[Any], response_model: Optional[Type[BaseModel]], enable_citations: bool) -> List[BatchResult]:
        """
        Parse results into unified format:
        [{"result": data, "citations": citations_data}, ...]
        
        Where:
        - result: parsed response model instance or plain text
        - citations: field citations dict or citation list or None
        """
        if not results:
            return []
            
        parsed_results = []
        errors = []
        
        for result in results:
            try:
                if result.result.type != "succeeded":
                    errors.append(result.model_dump())
                    continue
                    
                content = result.result.message.content
                
                # Parse the main result content
                if response_model:
                    if isinstance(content, list):
                        full_text, _ = self._parse_content_blocks(content)
                        result_data = self._extract_json_from_text(full_text, response_model)
                    else:
                        message_content = content.text if hasattr(content, 'text') else str(content)
                        result_data = self._extract_json_from_text(message_content, response_model)
                else:
                    # Plain text result
                    if isinstance(content, list):
                        result_data, _ = self._parse_content_blocks(content)
                    else:
                        result_data = content.text if hasattr(content, 'text') else str(content)
                
                # Parse citations if enabled
                citations_data = None
                if enable_citations:
                    if isinstance(content, list):
                        if response_model:
                            # Field citations - keep Citation objects for now, serialize later
                            citations_data = self._map_citations_to_fields(content, response_model)
                        else:
                            # Citation list
                            _, citations_list = self._parse_content_blocks(content)
                            citations_data = citations_list  # Keep Citation objects
                    else:
                        if hasattr(content, 'citations') and content.citations:
                            if response_model:
                                citations_data = self._map_citations_to_fields([content], response_model)
                            else:
                                _, citations_list = self._parse_content_blocks([content])
                                citations_data = citations_list  # Keep Citation objects
                
                # Create unified result entry  
                # Keep the original object for backward compatibility
                entry = {
                    "result": result_data,  # Keep original Pydantic instance or text
                    "citations": citations_data
                }
                parsed_results.append(entry)
                    
            except Exception as e:
                errors.append({"error": str(e), "result": result.model_dump()})
        
        self._handle_errors(errors)
        return parsed_results
    
    
    def _handle_errors(self, errors: List[dict]) -> None:
        """Handle and report batch processing errors."""
        if not errors:
            return
            
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
    
    def _extract_usage_from_result(self, result: Any) -> Dict[str, Any]:
        """
        Extract usage data from API response result.
        
        Args:
            result: API response result object
            
        Returns:
            Dictionary with usage data (input_tokens, output_tokens, service_tier)
        """
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "service_tier": "batch"
        }
        
        try:
            if (hasattr(result, 'result') and 
                hasattr(result.result, 'message') and 
                hasattr(result.result.message, 'usage')):
                
                usage = result.result.message.usage
                usage_data["input_tokens"] = getattr(usage, 'input_tokens', 0)
                usage_data["output_tokens"] = getattr(usage, 'output_tokens', 0)
                usage_data["service_tier"] = getattr(usage, 'service_tier', 'standard')
        except Exception:
            # If we can't extract usage, return default values
            pass
            
        return usage_data
    
    def _calculate_token_cost(self, num_tokens: int, model: str, token_type: Literal['input', 'output', 'cached']) -> float:
        """
        Calculate cost for tokens using tokencost library.
        
        Args:
            num_tokens: Number of tokens
            model: Model name
            token_type: Type of token ('input', 'output', or 'cached')
            
        Returns:
            Cost in USD
        """
        if num_tokens == 0:
            return 0.0
            
        try:
            # tokencost expects model names in specific format
            normalized_model = model.lower()
            cost = calculate_cost_by_tokens(num_tokens, normalized_model, token_type)
            return float(cost)
        except Exception:
            # If tokencost fails, return 0 rather than breaking
            return 0.0
    
    def _apply_batch_discount(self, cost: float, service_tier: str) -> float:
        """
        Apply batch discount if service_tier is 'batch'.
        
        Args:
            cost: Original cost
            service_tier: Service tier ('batch' or 'standard')
            
        Returns:
            Discounted cost
        """
        if service_tier == "batch":
            return cost * self.BATCH_DISCOUNT
        return cost
    
    def _aggregate_batch_usage(self, results: List[Any]) -> Dict[str, Any]:
        """
        Aggregate usage data across all results in a batch.
        
        Args:
            results: List of API result objects
            
        Returns:
            Dictionary with aggregated usage data
        """
        total_input_tokens = 0
        total_output_tokens = 0
        service_tier = "batch"
        request_count = 0
        
        for result in results:
            usage_data = self._extract_usage_from_result(result)
            total_input_tokens += usage_data["input_tokens"]
            total_output_tokens += usage_data["output_tokens"]
            service_tier = usage_data["service_tier"]  # Should be same for all in batch
            request_count += 1
            
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "service_tier": service_tier,
            "request_count": request_count
        }
    
    def get_batch_usage_costs(self, batch_id: str, model: str) -> Dict[str, Any]:
        """
        Get usage costs for a completed batch.
        
        Args:
            batch_id: ID of the batch
            model: Model name used for the batch
            
        Returns:
            Dictionary with cost information
        """
        if batch_id == "empty_batch":
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "service_tier": "batch",
                "request_count": 0
            }
        
        try:
            # Get raw results to extract usage
            raw_results = self.get_results(batch_id)
            usage_data = self._aggregate_batch_usage(raw_results)
            
            # Calculate costs
            input_cost = self._calculate_token_cost(
                usage_data["total_input_tokens"], 
                model, 
                "input"
            )
            output_cost = self._calculate_token_cost(
                usage_data["total_output_tokens"], 
                model, 
                "output"
            )
            
            # Apply batch discount
            input_cost = self._apply_batch_discount(input_cost, usage_data["service_tier"])
            output_cost = self._apply_batch_discount(output_cost, usage_data["service_tier"])
            
            total_cost = input_cost + output_cost
            
            return {
                "total_input_tokens": usage_data["total_input_tokens"],
                "total_output_tokens": usage_data["total_output_tokens"],
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "service_tier": usage_data["service_tier"],
                "request_count": usage_data["request_count"]
            }
            
        except Exception:
            # If we can't get costs, return zeros
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "service_tier": "batch",
                "request_count": 0
            }