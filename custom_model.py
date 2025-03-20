from smolagents import Model, TransformersModel, Tool, ChatMessage, MessageRole
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import torch
import logging
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)

def get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


class CustomTransformersModel(Model):
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
            For example, `"Qwen/Qwen2.5-Coder-32B-Instruct"`.
        device_map (`str`, *optional*):
            The device_map to initialize your model with.
        torch_dtype (`str`, *optional*):
            The torch_dtype to initialize your model with.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_new_tokens` or `device`.
        **kwargs:
            Additional keyword arguments to pass to `model.generate()`, for instance `max_new_tokens` or `device`.
    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        quantization_config = None,
        **kwargs,
    ):

        self.model_id = model_id

        default_max_tokens = 5000
        max_new_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        if not max_new_tokens:
            kwargs["max_new_tokens"] = default_max_tokens
            logger.warning(
                f"`max_new_tokens` not provided, using this default value for `max_new_tokens`: {default_max_tokens}"
            )

        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self._is_vlm = True
            else:
                raise e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(flatten_messages_as_text=not self._is_vlm, **kwargs)

    def make_stopping_criteria(self, stop_sequences: List[str], tokenizer) -> "StoppingCriteriaList":
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: List[str], tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
        )

        if max_new_tokens:
            completion_kwargs["max_new_tokens"] = max_new_tokens

        if hasattr(self, "processor"):
            prompt_tensor = self.processor.apply_chat_template(
                messages,
                tools=[get_tool_json_schema(tool) for tool in tools_to_call_from] if tools_to_call_from else None,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True if tools_to_call_from else False,
            )
        else:
            prompt_tensor = self.tokenizer.apply_chat_template(
                messages,
                tools=[get_tool_json_schema(tool) for tool in tools_to_call_from] if tools_to_call_from else None,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True if tools_to_call_from else False,
            )

        prompt_tensor = prompt_tensor.to(self.model.device)
        count_prompt_tokens = prompt_tensor["input_ids"].shape[1]

        if stop_sequences:
            stopping_criteria = self.make_stopping_criteria(
                stop_sequences, tokenizer=self.processor if hasattr(self, "processor") else self.tokenizer
            )
        else:
            stopping_criteria = None

        out = self.model.generate(
            **prompt_tensor,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.last_input_token_count = count_prompt_tokens
        self.last_output_token_count = len(generated_tokens)

        if stop_sequences is not None:
            output_text = remove_stop_sequences(output_text, stop_sequences)

        chat_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
        )
        # if tools_to_call_from:
        #     chat_message.tool_calls = [
        #         get_tool_call_from_text(output_text, self.tool_name_key, self.tool_arguments_key)
        #     ]
        return chat_message
