"""Hugging Face 聊天封装器。"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
    from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from langchain_huggingface._version import __version__
from langchain_huggingface.data._profiles import _PROFILES
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


@dataclass
class TGI_RESPONSE:
    """TextGenInference API 的响应。"""

    choices: list[Any]
    usage: dict


@dataclass
class TGI_MESSAGE:
    """发送至 TextGenInference API 的消息。"""

    role: str
    content: str
    tool_calls: list[dict]


def _lc_tool_call_to_hf_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_hf_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """将 LangChain 消息转换为字典。

    Args:
        message: LangChain 消息。

    Returns:
        字典。

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
                        # 如果仅调用函数，content 为 None 而非空字符串
            if message_dict["content"] == "":
                message_dict["content"] = None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_hf_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_hf_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
                # 如果仅调用工具，content 为 None 而非空字符串
        if "tool_calls" in message_dict and message_dict["content"] == "":
            message_dict["content"] = None
        else:
            pass
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """将字典转换为 LangChain 消息。

    Args:
        _dict: 字典。

    Returns:
        LangChain 消息。

    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name", "")
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role or "")


def _is_huggingface_hub(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_hub import (
            HuggingFaceHub,  # type: ignore[import-not-found]
        )

        return isinstance(llm, HuggingFaceHub)
    except ImportError:
                # 如果没有 langchain 社区，就不是 HuggingFaceHub
        return False


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list[ToolCallChunk] = []
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            with contextlib.suppress(KeyError):
                tool_call_chunks.append(
                    create_tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index"),
                    )
                )
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if usage := chunk.get("usage"):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _is_huggingface_textgen_inference(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_text_gen_inference import (
            HuggingFaceTextGenInference,  # type: ignore[import-not-found]
        )

        return isinstance(llm, HuggingFaceTextGenInference)
    except ImportError:
                # 如果没有 langchain 社区，就不是 HuggingFaceTextGenInference
        return False


def _is_huggingface_endpoint(llm: Any) -> bool:
    return isinstance(llm, HuggingFaceEndpoint)


def _is_huggingface_pipeline(llm: Any) -> bool:
    return isinstance(llm, HuggingFacePipeline)


class ChatHuggingFace(BaseChatModel):
    r"""Hugging Face LLM 作为 ChatModels。

    适用于 `HuggingFaceTextGenInference`、`HuggingFaceEndpoint`、
    `HuggingFaceHub` 和 `HuggingFacePipeline` LLM。

    Upon instantiating this class, the model_id is resolved from the url
    provided to the LLM, and the appropriate tokenizer is loaded from
    the HuggingFace Hub.

    设置：
        安装 `langchain-huggingface` 并确保你的 Hugging Face token
        已保存。

        ```bash
        pip install langchain-huggingface
        ```

        ```python
        from huggingface_hub import login

        login()  # You will be prompted for your HF key, which will then be saved locally
        ```

    关键初始化参数 —— 完成参数：
        llm:
            用于的 LLM。

    关键初始化参数 —— 客户端参数：
        custom_get_token_ids:
            可选编码器，用于统计 token 数量。
        metadata:
            添加到运行追踪的元数据。
        tags:
            添加到运行追踪的标签。
        verbose:
            是否打印响应文本。

    有关支持的初始化参数及其描述的完整列表，请参见参数部分。

    实例化：
        ```python
        from langchain_huggingface import HuggingFaceEndpoint,
        ChatHuggingFace

        model = HuggingFaceEndpoint(
            repo_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        chat = ChatHuggingFace(llm=model, verbose=True)
        ```

    调用：
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user
            sentence to French."),
            ("human", "I love programming."),
        ]

        chat(...).invoke(messages)
        ```

        ```python
        AIMessage(content='Je ai une passion pour le programme.\n\nIn
        French, we use "ai" for masculine subjects and "a" for feminine
        subjects. Since "programming" is gender-neutral in English, we
        will go with the masculine "programme".\n\nConfirmation: "J\'aime
        le programme." is more commonly used. The sentence above is
        technically accurate, but less commonly used in spoken French as
        "ai" is used less frequently in everyday speech.',
        response_metadata={'token_usage': ChatCompletionOutputUsage
        (completion_tokens=100, prompt_tokens=55, total_tokens=155),
        'model': '', 'finish_reason': 'length'},
        id='run-874c24b7-0272-4c99-b259-5d6d7facbc56-0')
        ```

    流式：
        ```python
        for chunk in chat.stream(messages):
            print(chunk)
        ```

        ```python
        content='Je ai une passion pour le programme.\n\nIn French, we use
        "ai" for masculine subjects and "a" for feminine subjects.
        Since "programming" is gender-neutral in English,
        we will go with the masculine "programme".\n\nConfirmation:
        "J\'aime le programme." is more commonly used. The sentence
        above is technically accurate, but less commonly used in spoken
        French as "ai" is used less frequently in everyday speech.'
        response_metadata={'token_usage': ChatCompletionOutputUsage
        (completion_tokens=100, prompt_tokens=55, total_tokens=155),
        'model': '', 'finish_reason': 'length'}
        id='run-7d7b1967-9612-4f9a-911a-b2b5ca85046a-0'
        ```

    异步：
        ```python
        await chat.ainvoke(messages)
        ```

        ```python
        AIMessage(content='Je déaime le programming.\n\nLittérale : Je
        (j\'aime) déaime (le) programming.\n\nNote: "Programming" in
        French is "programmation". But here, I used "programming" instead
        of "programmation" because the user said "I love programming"
        instead of "I love programming (in French)", which would be
        "J\'aime la programmation". By translating the sentence
        literally, I preserved the original meaning of the user\'s
        sentence.', id='run-fd850318-e299-4735-b4c6-3496dc930b1d-0')
        ```

    工具调用：
        ```python
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state,
            e.g. San Francisco, CA")

        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state,
            e.g. San Francisco, CA")

        chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
        ai_msg = chat_with_tools.invoke("Which city is hotter today and
        which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "0",
            }
        ]
        ```

    响应元数据
        ```python
        ai_msg = chat.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "token_usage": ChatCompletionOutputUsage(
                completion_tokens=100, prompt_tokens=8, total_tokens=108
            ),
            "model": "",
            "finish_reason": "length",
        }
        ```
    """  # noqa: E501

    llm: Any
    """LLM，必须是 HuggingFaceTextGenInference、HuggingFaceEndpoint、
        HuggingFaceHub 或 HuggingFacePipeline 类型。"""
    tokenizer: Any = None
    """模型的分词器。仅用于 HuggingFacePipeline。"""
    model_id: str | None = None
    """模型 ID。仅用于 HuggingFaceEndpoint。"""
    temperature: float | None = None
    """生成时使用的采样温度。"""
    stop: str | list[str] | None = Field(default=None, alias="stop_sequences")
    """默认停止序列。"""
    presence_penalty: float | None = None
    """惩罚重复 token。"""
    frequency_penalty: float | None = None
    """根据频率惩罚重复 token。"""
    seed: int | None = None
    """生成的随机种子"""
    logprobs: bool | None = None
    """是否返回 logprobs。"""
    top_logprobs: int | None = None
    """Number of most likely tokens to return at each token position, each with
     an associated log probability. `logprobs` must be set to true
     if this parameter is used."""
    logit_bias: dict[int, int] | None = None
    """修改指定 token 出现的可能性。"""
    streaming: bool = False
    """是否流式输出结果。"""
    stream_usage: bool | None = None
    """是否在流式输出中包含使用情况元数据。若为
True，流式过程中将额外生成一条包含使用情况元数据的消息块。"""
    n: int | None = None
    """为每个提示生成的聊天完成数量。"""
    top_p: float | None = None
    """每一步考虑的 token 总概率质量。"""
    max_tokens: int | None = None
    """最大生成 token 数量。"""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """持有 `create` 调用中有效的任何未明确指定的模型参数。"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

                # 如果未显式设置，则继承自 LLM 的属性
        self._inherit_llm_properties()

        self._resolve_model_id()

    def _inherit_llm_properties(self) -> None:
        """如果未显式设置，则继承被包装的 LLM 实例的属性。"""
        if not hasattr(self, "llm") or self.llm is None:
            return

                # ChatHuggingFace 属性到 LLM 属性的映射
        property_mappings = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",  # 不同的命名约定
            "top_p": "top_p",
            "seed": "seed",
            "streaming": "streaming",
            "stop": "stop_sequences",
        }

                # 从 LLM 继承属性，此处未显式设置
        for chat_prop, llm_prop in property_mappings.items():
            if hasattr(self.llm, llm_prop):
                llm_value = getattr(self.llm, llm_prop)
                chat_value = getattr(self, chat_prop, None)
                if not chat_value and llm_value:
                    setattr(self, chat_prop, llm_value)

                # 处理 HuggingFaceEndpoint 的特殊情况
        if _is_huggingface_endpoint(self.llm):
                        # 继承额外的 HuggingFaceEndpoint 特定属性
            endpoint_mappings = {
                "frequency_penalty": "repetition_penalty",
            }

            for chat_prop, llm_prop in endpoint_mappings.items():
                if hasattr(self.llm, llm_prop):
                    llm_value = getattr(self.llm, llm_prop)
                    chat_value = getattr(self, chat_prop, None)
                    if chat_value is None and llm_value is not None:
                        setattr(self, chat_prop, llm_value)

                # 如果未显式设置，则继承 model_kwargs
        if (
            not self.model_kwargs
            and hasattr(self.llm, "model_kwargs")
            and isinstance(self.llm.model_kwargs, dict)
        ):
            self.model_kwargs = self.llm.model_kwargs.copy()

    @model_validator(mode="after")
    def _set_huggingface_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-huggingface", __version__)
        return self

    @model_validator(mode="after")
    def validate_llm(self) -> Self:
        if (
            not _is_huggingface_hub(self.llm)
            and not _is_huggingface_textgen_inference(self.llm)
            and not _is_huggingface_endpoint(self.llm)
            and not _is_huggingface_pipeline(self.llm)
        ):
            msg = (
                "Expected llm to be one of HuggingFaceTextGenInference, "
                "HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline "
                f"received {type(self.llm)}"
            )
            raise TypeError(msg)
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        if self.model_id:
            return _get_default_model_profile(self.model_id) or None
        return None

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str | None = None,
        backend: Literal["pipeline", "endpoint", "text-gen"] = "pipeline",
        **kwargs: Any,
    ) -> ChatHuggingFace:
        """从 model_id 构造一个 ChatHuggingFace 模型。

        Args:
            model_id: Hugging Face 模型的模型 ID。
            task: 要执行的任务（例如 "text-generation"）。
            backend: 要使用的后端。可选值为 "pipeline"、"endpoint"、"text-gen"。
            **kwargs: 传递给后端或 ChatHuggingFace 的其他参数。
        """
        llm: (
            Any  # HuggingFacePipeline, HuggingFaceEndpoint, HuggingFaceTextGenInference
        )
        if backend == "pipeline":
            from langchain_huggingface.llms.huggingface_pipeline import (
                HuggingFacePipeline,
            )

            task = task if task is not None else "text-generation"

                        # 将管道特定的 kwargs 与 ChatHuggingFace kwargs 分离
                        # 应传递给 HuggingFacePipeline.from_model_id 的参数
            pipeline_specific_kwargs = {}

                        # 提取管道特定的参数
            pipeline_keys = [
                "backend",
                "device",
                "device_map",
                "model_kwargs",
                "pipeline_kwargs",
                "batch_size",
            ]
            for key in pipeline_keys:
                if key in kwargs:
                    pipeline_specific_kwargs[key] = kwargs.pop(key)

                        # 剩余的 kwargs（temperature、max_tokens 等）应传递给
                        # pipeline_kwargs 用于生成参数，这些参数由 ChatHuggingFace
                        # 将从 LLM 继承
            if "pipeline_kwargs" not in pipeline_specific_kwargs:
                pipeline_specific_kwargs["pipeline_kwargs"] = {}

                        # 将生成参数添加到 pipeline_kwargs
                        # 将 max_tokens 映射为 max_new_tokens 以用于 HuggingFace 管道
            generation_params = {}
            for k, v in list(kwargs.items()):
                if k == "max_tokens":
                    generation_params["max_new_tokens"] = v
                    kwargs.pop(k)
                elif k in (
                    "temperature",
                    "max_new_tokens",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "do_sample",
                ):
                    generation_params[k] = v
                    kwargs.pop(k)

            pipeline_specific_kwargs["pipeline_kwargs"].update(generation_params)

                        # 创建 HuggingFacePipeline
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_id, task=task, **pipeline_specific_kwargs
            )
        elif backend == "endpoint":
            from langchain_huggingface.llms.huggingface_endpoint import (
                HuggingFaceEndpoint,
            )

            llm = HuggingFaceEndpoint(repo_id=model_id, task=task, **kwargs)
        elif backend == "text-gen":
            from langchain_community.llms.huggingface_text_gen_inference import (  # type: ignore[import-not-found]
                HuggingFaceTextGenInference,
            )

            llm = HuggingFaceTextGenInference(inference_server_url=model_id, **kwargs)
        else:
            msg = f"Unknown backend: {backend}"
            raise ValueError(msg)

        return cls(llm=llm, **kwargs)

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_id,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming

        if _is_huggingface_textgen_inference(self.llm):
            message_dicts, params = self._create_message_dicts(messages, stop)
            answer = self.llm.client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        if _is_huggingface_endpoint(self.llm):
            if should_stream:
                stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {
                "stop": stop,
                **params,
                **({"stream": stream} if stream is not None else {}),
                **kwargs,
            }
            answer = self.llm.client.chat_completion(messages=message_dicts, **params)
            return self._create_chat_result(answer)
        llm_input = self._to_chat_prompt(messages)

        if should_stream:
            stream_iter = self.llm._stream(
                llm_input, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        if _is_huggingface_textgen_inference(self.llm):
            message_dicts, params = self._create_message_dicts(messages, stop)
            answer = await self.llm.async_client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        if _is_huggingface_endpoint(self.llm):
            should_stream = stream if stream is not None else self.streaming
            if should_stream:
                stream_iter = self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return await agenerate_from_stream(stream_iter)
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {
                **params,
                **({"stream": stream} if stream is not None else {}),
                **kwargs,
            }

            answer = await self.llm.async_client.chat_completion(
                messages=message_dicts, **params
            )
            return self._create_chat_result(answer)
        if _is_huggingface_pipeline(self.llm):
            msg = "async generation is not supported with HuggingFacePipeline"
            raise NotImplementedError(msg)
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _should_stream_usage(
        self, *, stream_usage: bool | None = None, **kwargs: Any
    ) -> bool | None:
        """确定是否在流式输出中包含使用情况元数据。

        For backwards compatibility, we check for `stream_options` passed
        explicitly to kwargs or in the model_kwargs and override self.stream_usage.
        """
        stream_usage_sources = [  # 优先级顺序
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if _is_huggingface_endpoint(self.llm):
            stream_usage = self._should_stream_usage(
                stream_usage=stream_usage, **kwargs
            )
            if stream_usage:
                kwargs["stream_options"] = {"include_usage": stream_usage}
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {**params, **kwargs, "stream": True}

            default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
            for chunk in self.llm.client.chat_completion(
                messages=message_dicts, **params
            ):
                if len(chunk["choices"]) == 0:
                    if usage := chunk.get("usage"):
                        usage_msg = AIMessageChunk(
                            content="",
                            additional_kwargs={},
                            response_metadata={},
                            usage_metadata={
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            },
                        )
                        yield ChatGenerationChunk(message=usage_msg)
                    continue

                choice = chunk["choices"][0]
                message_chunk = _convert_chunk_to_message_chunk(
                    chunk, default_chunk_class
                )
                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_id
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
                default_chunk_class = message_chunk.__class__
                generation_chunk = ChatGenerationChunk(
                    message=message_chunk, generation_info=generation_info or None
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                    )
                yield generation_chunk
        else:
            llm_input = self._to_chat_prompt(messages)
            stream_iter = self.llm._stream(
                llm_input, stop=stop, run_manager=run_manager, **kwargs
            )
            for chunk in stream_iter:  # chunk 是一个 GenerationChunk
                chat_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk.text),
                    generation_info=chunk.generation_info,
                )
                yield chat_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        stream_usage = self._should_stream_usage(stream_usage=stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk

        async for chunk in await self.llm.async_client.chat_completion(
            messages=message_dicts, **params
        ):
            if len(chunk["choices"]) == 0:
                if usage := chunk.get("usage"):
                    usage_msg = AIMessageChunk(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        usage_metadata={
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )
                    yield ChatGenerationChunk(message=usage_msg)
                continue

            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_id
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    def _to_chat_prompt(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """将消息列表转换为被包装的 LLM 所期望的提示词格式。"""
        if not messages:
            msg = "At least one HumanMessage must be provided!"
            raise ValueError(msg)

        if not isinstance(messages[-1], HumanMessage):
            msg = "Last message must be a HumanMessage!"
            raise ValueError(msg)

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """将 LangChain 消息转换为 ChatML 格式。"""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            msg = f"Unknown message type: {type(message)}"
            raise ValueError(msg)

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    def _resolve_model_id(self) -> None:
        """从 LLM 的 inference_server_url 解析 model_id。"""
        from huggingface_hub import list_inference_endpoints  # type: ignore[import]

        if _is_huggingface_hub(self.llm) or (
            hasattr(self.llm, "repo_id") and self.llm.repo_id
        ):
            self.model_id = self.llm.repo_id
            return
        if _is_huggingface_textgen_inference(self.llm):
            endpoint_url: str | None = self.llm.inference_server_url
        if _is_huggingface_pipeline(self.llm):
            from transformers import AutoTokenizer  # type: ignore[import]

            self.model_id = self.model_id or self.llm.model_id
            self.tokenizer = (
                AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer is None
                else self.tokenizer
            )
            return
        if _is_huggingface_endpoint(self.llm):
            self.model_id = self.llm.repo_id or self.llm.model
            return
        endpoint_url = self.llm.endpoint_url
        available_endpoints = list_inference_endpoints("*")
        for endpoint in available_endpoints:
            if endpoint.url == endpoint_url:
                self.model_id = endpoint.repository

        if not self.model_id:
            msg = (
                "Failed to resolve model_id:"
                f"Could not find model id for inference server: {endpoint_url}"
                "Make sure that your Hugging Face token has access to the endpoint."
            )
            raise ValueError(msg)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        假设模型与 OpenAI 工具调用 API 兼容。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                Supports any tool definition handled by [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                `'auto'` to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                `langchain.runnable.Runnable` constructor.
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                msg = (
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
                raise ValueError(msg)
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    msg = (
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """返回与给定模式匹配的输出格式的模型包装器。

        Args:
            schema: 输出模式。可以作为以下形式传入：

                - OpenAI 函数/工具模式，
                - JSON Schema，
                - `TypedDict` 类

                Pydantic 类目前受支持。

            method: 控制模型生成的方法，可选之一：

                - `'function_calling'`: 使用工具调用功能。
                - `'json_schema'`: 使用专用结构化输出功能。
                - `'json_mode'`: 使用 JSON 模式。

            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果在模型输出解析过程中发生错误，则会抛出异常。

                如果为 `True`，则同时返回原始模型响应（一个 `BaseMessage`）和
                解析后的模型响应。

                如果在输出解析过程中发生错误，该错误将被捕获并一同返回。

                最终输出始终是一个包含键 `'raw'`、`'parsed'` 和
                `'parsing_error'` 的 `dict`。

            kwargs:
                传递给底层 LLM 的
                `langchain_core.language_models.chat.BaseChatModel.bind`
                方法的额外参数，例如 `response_format` 或 `ls_structured_output_format`。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                如果 `include_raw` 为 `True`，则 `Runnable` 输出一个包含以下键的 `dict`：

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`
        """
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                msg = "Pydantic schema is not supported for function calling"
                raise NotImplementedError(msg)
            output_parser: JsonOutputKeyToolsParser | JsonOutputParser = (
                JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema},
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser = JsonOutputParser()  # type: ignore[arg-type]
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = JsonOutputParser()  # type: ignore[arg-type]
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 Hugging Face 推理提供者 API 的默认参数。"""
        params = {
            "model": self.model_id,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "stop": self.stop,
            **(self.model_kwargs if self.model_kwargs else {}),
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat-wrapper"
