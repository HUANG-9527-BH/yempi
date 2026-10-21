"""Ollama 聊天模型。

**输入流程（LangChain -> Ollama）**

`_convert_messages_to_ollama_messages()`：

- 将 LangChain 消息转换为 `ollama.Message` 格式
- 提取文本内容、图片（base64）和工具调用

`_chat_params()`：

- 将消息与模型参数（temperature、top_p 等）合并
- 若提供工具则附加工具
- 通过 `think` 参数配置推理/思考模式
- 设置输出格式（原始、JSON 或 JSON 模式）

**输出流程（Ollama -> LangChain）**

1. **Ollama 响应**

流式字典块包含：
- `message`：包含 `role`、`content`、`tool_calls`、`thinking` 的字典
- `done`：布尔值，表示是否完成
- `done_reason`：完成原因（`stop`、`length`、`load`）
- token 统计/时间元数据

2. **响应处理** (`_iterate_over_stream()`)

- 从 `message.content` 提取内容
- 将工具调用解析为 `ToolCall`
- 当 `reasoning=True` 时分离推理内容（存储在 `additional_kwargs` 中）
- 从 token 统计构建使用元数据

3. **LangChain 输出** (`ChatGenerationChunk` -> `AIMessage`)

- **流式**：生成 `ChatGenerationChunk`，其内容为 `AIMessageChunk`
- **非流式**：返回包含完整 `AIMessage` 的 `ChatResult`
- 工具调用附加至 `AIMessage.tool_calls`
- 推理内容存储在 `AIMessage.additional_kwargs['reasoning_content']`
"""

from __future__ import annotations

import ast
import json
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages import content as types
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from ollama import AsyncClient, Client, Message
from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, is_typeddict

from langchain_ollama._compat import _convert_from_v1_to_ollama
from langchain_ollama._utils import (
    merge_auth_headers,
    parse_url_with_auth,
    validate_model,
)
from langchain_ollama._version import __version__

log = logging.getLogger(__name__)


def _get_usage_metadata_from_generation_info(
    generation_info: Mapping[str, Any] | None,
) -> UsageMetadata | None:
    """从 Ollama 生成信息映射中获取使用元数据。"""
    if generation_info is None:
        return None
    input_tokens: int | None = generation_info.get("prompt_eval_count")
    output_tokens: int | None = generation_info.get("eval_count")
    if input_tokens is not None and output_tokens is not None:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return None


def _parse_json_string(
    json_string: str,
    *,
    raw_tool_call: dict[str, Any],
    skip: bool,
) -> Any:
    """尝试解析一个 JSON 字符串以进行工具调用。

    首先尝试使用标准的 `json.loads`。若失败，则回退到 `ast.literal_eval` 以安全解析
    Python 字面量，该方法对模型使用单引号或包含撇号的情况更具鲁棒性。

    Args:
        json_string: 要解析的 JSON 字符串。
        raw_tool_call: 在错误信息中包含的原始工具调用。
        skip: 是否忽略解析错误并仍返回值。

    Returns:
        解析后的 JSON 字符串或 Python 字面量。

    Raises:
        OutputParserException: 若字符串无效且 `skip=False` 时抛出。
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
                        # Use ast.literal_eval to safely parse Python-style dicts
                        # (e.g. with single quotes)
            return ast.literal_eval(json_string)
        except (SyntaxError, ValueError) as e:
                        # If both fail, and we're not skipping, raise an informative error.
            if skip:
                return json_string
            msg = (
                f"Function {raw_tool_call['function']['name']} arguments:\n\n"
                f"{raw_tool_call['function']['arguments']}"
                "\n\nare not valid JSON or a Python literal. "
                f"Received error: {e}"
            )
            raise OutputParserException(msg) from e
    except TypeError as e:
        if skip:
            return json_string
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{raw_tool_call['function']['arguments']}\n\nare not a string or a "
            f"dictionary. Received TypeError {e}"
        )
        raise OutputParserException(msg) from e


def _parse_arguments_from_tool_call(
    raw_tool_call: dict[str, Any],
) -> dict[str, Any] | None:
    """通过尝试解析任何浅层嵌套的字符串编码 JSON 来解析参数。

    临时修复 Ollama 中工具调用参数结构不一致的问题。
    若上游修复，应移除或更改此修复。

    参见 https://github.com/ollama/ollama/issues/6155
    """
    if "function" not in raw_tool_call:
        return None
    function_name = raw_tool_call["function"]["name"]
    arguments = raw_tool_call["function"]["arguments"]
    parsed_arguments: dict = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
                        # Filter out metadata fields like 'functionName' that echo function name
            if key == "functionName" and value == function_name:
                continue
            if isinstance(value, str):
                parsed_value = _parse_json_string(
                    value, skip=True, raw_tool_call=raw_tool_call
                )
                if isinstance(parsed_value, (dict, list)):
                    parsed_arguments[key] = parsed_value
                else:
                    parsed_arguments[key] = value
            else:
                parsed_arguments[key] = value
    else:
        parsed_arguments = _parse_json_string(
            arguments, skip=False, raw_tool_call=raw_tool_call
        )
    return parsed_arguments


def _get_tool_calls_from_response(
    response: Mapping[str, Any],
) -> list[ToolCall]:
    """从 Ollama 响应中获取工具调用。"""
    tool_calls = []
    if "message" in response and (
        raw_tool_calls := response["message"].get("tool_calls")
    ):
        tool_calls.extend(
            [
                tool_call(
                    id=str(uuid4()),
                    name=tc["function"]["name"],
                    args=_parse_arguments_from_tool_call(tc) or {},
                )
                for tc in raw_tool_calls
            ]
        )
    return tool_calls


def _lc_tool_call_to_openai_tool_call(tool_call_: ToolCall) -> dict:
    """将 LangChain 工具调用转换为 OpenAI 工具调用格式。"""
    return {
        "type": "function",
        "id": tool_call_["id"],
        "function": {
            "name": tool_call_["name"],
            "arguments": tool_call_["args"],
        },
    }


def _get_image_from_data_content_block(block: dict) -> str:
    """将标准数据内容块格式化为 Ollama 所期望的格式。"""
    if block["type"] == "image":
        if block.get("source_type") == "base64":
                        # v0 style
            return block["data"]
        if block.get("base64"):
                        # v1 content blocks
            return block["base64"]
        error_message = "Image data only supported through in-line base64 format."
        raise ValueError(error_message)

    error_message = f"Blocks of type {block['type']} not supported."
    raise ValueError(error_message)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class ChatOllama(BaseChatModel):
    r"""Ollama 聊天模型集成。

    ???+ note "设置"

        安装 `langchain-ollama` 并从 ollama 下载您想要使用的任何模型。

        ```bash
        ollama pull gpt-oss:20b
        pip install -U langchain-ollama
        ```

    关键初始化参数 —— 完成参数：
        model: str
            要使用的 Ollama 模型名称。
        reasoning: bool | None
            控制支持模型的推理/思考模式
            [支持的模型](https://ollama.com/search?c=thinking)

            - `True`: Enables reasoning mode. The model's reasoning process will be
                captured and returned separately in the `additional_kwargs` of the
                response message, under `reasoning_content`. The main response
                content will not include the reasoning tags.
            - `False`: Disables reasoning mode. The model will not perform any reasoning,
                and the response will not include any reasoning content.
            - `None` (Default): The model will use its default reasoning behavior. Note
                however, if the model's default behavior *is* to perform reasoning, think tags
                (`<think>` and `</think>`) will be present within the main response content
                unless you set `reasoning` to `True`.
        temperature: float
            Sampling temperature. Ranges from `0.0` to `1.0`.
        num_predict: int | None
            Max number of tokens to generate.

    有关支持的初始化参数及其描述的完整列表，请参见 params 部分。

    实例化：
        ```python
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model="gpt-oss:20b",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
            # other params ...
        )
        ```

    调用：
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        ```python
        AIMessage(content='J'adore le programmation. (Note: "programming" can also refer to the act of writing code, so if you meant that, I could translate it as "J'adore programmer". But since you didn\'t specify, I assumed you were talking about the activity itself, which is what "le programmation" usually refers to.)', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:37:50.182604Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 3576619666, 'load_duration': 788524916, 'prompt_eval_count': 32, 'prompt_eval_duration': 128125000, 'eval_count': 71, 'eval_duration': 2656556000}, id='run-ba48f958-6402-41a5-b461-5e250a4ebd36-0')
        ```

    流式：
        ```python
        for chunk in model.stream("Return the words Hello World!"):
            print(chunk.text, end="")
        ```

        ```python
        content='Hello' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
        content=' World' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
        content='!' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
        content='' response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:39:42.274449Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 411875125, 'load_duration': 1898166, 'prompt_eval_count': 14, 'prompt_eval_duration': 297320000, 'eval_count': 4, 'eval_duration': 111099000} id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'

        ```

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(
            content='Je adore le programmation.(Note: "programmation" is the formal way to say "programming" in French, but informally, people might use the phrase "le développement logiciel" or simply "le code")',
            response_metadata={
                "model": "llama3",
                "created_at": "2024-07-04T03:38:54.933154Z",
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "done": True,
                "total_duration": 1977300042,
                "load_duration": 1345709,
                "prompt_eval_duration": 159343000,
                "eval_count": 47,
                "eval_duration": 1815123000,
            },
            id="run-3c81a3ed-3e79-4dd3-a796-04064d804890",
        )
        ```

    异步：
        ```python
        await model.ainvoke("Hello how are you!")
        ```

        ```python
        AIMessage(
            content="Hi there! I'm just an AI, so I don't have feelings or emotions like humans do. But I'm functioning properly and ready to help with any questions or tasks you may have! How can I assist you today?",
            response_metadata={
                "model": "llama3",
                "created_at": "2024-07-04T03:52:08.165478Z",
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "done": True,
                "total_duration": 2138492875,
                "load_duration": 1364000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 297081000,
                "eval_count": 47,
                "eval_duration": 1838524000,
            },
            id="run-29c510ae-49a4-4cdd-8f23-b972bfab1c49-0",
        )
        ```

        ```python
        async for chunk in model.astream("Say hello world!"):
            print(chunk.content)
        ```

        ```python
        HEL
        LO
        WORLD
        !
        ```

        ```python
        messages = [("human", "Say hello world!"), ("human", "Say goodbye world!")]
        await model.abatch(messages)
        ```

        ```python
        [
            AIMessage(
                content="HELLO, WORLD!",
                response_metadata={
                    "model": "llama3",
                    "created_at": "2024-07-04T03:55:07.315396Z",
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": "stop",
                    "done": True,
                    "total_duration": 1696745458,
                    "load_duration": 1505000,
                    "prompt_eval_count": 8,
                    "prompt_eval_duration": 111627000,
                    "eval_count": 6,
                    "eval_duration": 185181000,
                },
                id="run-da6c7562-e25a-4a44-987a-2c83cd8c2686-0",
            ),
            AIMessage(
                content="It's been a blast chatting with you! Say goodbye to the world for me, and don't forget to come back and visit us again soon!",
                response_metadata={
                    "model": "llama3",
                    "created_at": "2024-07-04T03:55:07.018076Z",
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": "stop",
                    "done": True,
                    "total_duration": 1399391083,
                    "load_duration": 1187417,
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 230349000,
                    "eval_count": 31,
                    "eval_duration": 1166047000,
                },
                id="run-96cad530-6f3e-4cf9-86b4-e0f8abba4cdb-0",
            ),
        ]
        ```

    JSON 模式：
        ```python
        json_model = ChatOllama(format="json")
        json_model.invoke(
            "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. "
            "Respond using JSON only."
        ).content
        ```

        ```python
        '{"location": "Pune, India", "time_of_day": "morning"}'
        ```

    工具调用：
        ```python
        from langchain_ollama import ChatOllama
        from pydantic import BaseModel, Field


        class Multiply(BaseModel):
            a: int = Field(..., description="First integer")
            b: int = Field(..., description="Second integer")


        ans = await chat.invoke("What is 45*67")
        ans.tool_calls
        ```

        ```python
        [
            {
                "name": "Multiply",
                "args": {"a": 45, "b": 67},
                "id": "420c3f3b-df10-4188-945f-eb3abdb40622",
                "type": "tool_call",
            }
        ]
        ```

    Thinking / Reasoning:
        You can enable reasoning mode for models that support it by setting
        the `reasoning` parameter to `True` in either the constructor or
        the `invoke`/`stream` methods. This will enable the model to think
        through the problem and return the reasoning process separately in the
        `additional_kwargs` of the response message, under `reasoning_content`.

        如果 `reasoning` 设置为 `None`，模型将使用其默认的推理行为，且任何推理内容将
        *不会* 在 `reasoning_content` 键下被捕获，而是以思考标签（`<think>` 和
        `</think>`）的形式出现在主响应内容中。

        !!! note
            此功能仅对 [支持推理的模型](https://ollama.com/search?c=thinking) 可用。

        ```python
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model="deepseek-r1:8b",
            validate_model_on_init=True,
            reasoning=True,
        )

        model.invoke("how many r in the word strawberry?")

        # or, on an invocation basis:

        model.invoke("how many r in the word strawberry?", reasoning=True)
        # or model.stream("how many r in the word strawberry?", reasoning=True)

        # If not provided, the invocation will default to the ChatOllama reasoning
        # param provided (None by default).
        ```

        ```python
        AIMessage(content='The word "strawberry" contains **three \'r\' letters**. Here\'s a breakdown for clarity:\n\n- The spelling of "strawberry" has two parts ... be 3.\n\nTo be thorough, let\'s confirm with an online source or common knowledge.\n\nI can recall that "strawberry" has: s-t-r-a-w-b-e-r-r-y — yes, three r\'s.\n\nPerhaps it\'s misspelled by some, but standard is correct.\n\nSo I think the response should be 3.\n'}, response_metadata={'model': 'deepseek-r1:8b', 'created_at': '2025-07-08T19:33:55.891269Z', 'done': True, 'done_reason': 'stop', 'total_duration': 98232561292, 'load_duration': 28036792, 'prompt_eval_count': 10, 'prompt_eval_duration': 40171834, 'eval_count': 3615, 'eval_duration': 98163832416, 'model_name': 'deepseek-r1:8b'}, id='run--18f8269f-6a35-4a7c-826d-b89d52c753b3-0', usage_metadata={'input_tokens': 10, 'output_tokens': 3615, 'total_tokens': 3625})

        ```
    """  # noqa: E501, pylint: disable=line-too-long

    model: str
    """要使用的模型名称。"""

    reasoning: bool | str | None = None
    """控制 [支持的模型](https://ollama.com/search?c=thinking) 的推理/思考模式。

    - `True`: Enables reasoning mode. The model's reasoning process will be
        captured and returned separately in the `additional_kwargs` of the
        response message, under `reasoning_content`. The main response
        content will not include the reasoning tags.
    - `False`: Disables reasoning mode. The model will not perform any reasoning,
        and the response will not include any reasoning content.
    - `None` (Default): The model will use its default reasoning behavior. Note
        however, if the model's default behavior *is* to perform reasoning, think tags
        (`<think>` and `</think>`) will be present within the main response content
        unless you set `reasoning` to `True`.
    - `str`: e.g. `'low'`, `'medium'`, `'high'`. Enables reasoning with a custom
        intensity level. Currently, this is only supported `gpt-oss`. See the
        [Ollama docs](https://github.com/ollama/ollama-python/blob/da79e987f0ac0a4986bf396f043b36ef840370bc/ollama/_types.py#L210)
        for more information.
    """

    validate_model_on_init: bool = False
    """是否在初始化时验证模型是否存在于本地 Ollama 中。

    !!! version-added "Added in `langchain-ollama` 0.3.4"
    """

    mirostat: int | None = None
    """启用 Mirostat 采样以控制困惑度。

    (默认: `0`, `0` = 禁用, `1` = Mirostat, `2` = Mirostat 2.0)
    """

    mirostat_eta: float | None = None
    """影响算法对生成文本反馈的响应速度。

    A lower learning rate will result in slower adjustments, while a higher learning
    rate will make the algorithm more responsive.

    (默认: `0.1`)
    """

    mirostat_tau: float | None = None
    """控制输出连贯性与多样性的平衡。

    较低的值将导致更聚焦且连贯的文本。

    (默认: `5.0`)
    """

    num_ctx: int | None = None
    """设置用于生成下一个 token 的上下文窗口大小。

    (默认: `2048`)
    """

    num_gpu: int | None = None
    """要使用的 GPU 数量。

    在 macOS 上，默认为 `1` 以启用 metal 支持，`0` 以禁用。
    """

    num_thread: int | None = None
    """设置计算期间使用的线程数。

    By default, Ollama will detect this for optimal performance. It is recommended to
    set this value to the number of physical CPU cores your system has (as opposed to
    the logical number of cores).
    """

    num_predict: int | None = None
    """生成文本时预测的最大 token 数量。

    (默认: `128`, `-1` = 无限生成, `-2` = 填充上下文)
    """

    repeat_last_n: int | None = None
    """设置模型回溯以防止重复的距离。

    (默认: `64`, `0` = 禁用, `-1` = `num_ctx`)
    """

    repeat_penalty: float | None = None
    """设置对重复内容的惩罚强度。

    A higher value (e.g., `1.5`) will penalize repetitions more strongly, while a
    lower value (e.g., `0.9`) will be more lenient. (Default: `1.1`)
    """

    temperature: float | None = None
    """模型的温度。

    提高温度会使模型的回答更具创造性。

    (默认: `0.8`)
    """

    seed: int | None = None
    """设置生成时使用的随机数种子。

    Setting this to a specific number will make the model generate the same text for the
    same prompt.
    """

    logprobs: bool | None = None
    """是否返回 logprobs。

    !!! note

        流式输出时，每个中间块的 token 级别 logprobs 可通过
        `response_metadata["logprobs"]` 获取，并在使用 `invoke()`
        时累积到最终聚合响应中。
    """

    top_logprobs: int | None = None
    """Number of most likely tokens to return at each token position, each with
    an associated log probability. Must be a positive integer.

    若未设置 `logprobs=True`，则会自动启用 `logprobs`。
    """

    @field_validator("top_logprobs")
    @classmethod
    def _validate_top_logprobs(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            msg = "`top_logprobs` must be a positive integer."
            raise ValueError(msg)
        return v

    stop: list[str] | None = None
    """设置要使用的停止 token。"""

    tfs_z: float | None = None
    """尾部自由采样。

    用于降低输出中不太可能 token 的影响。

    A higher value (e.g., `2.0`) will reduce the impact more, while a value of `1.0`
    disables this setting.

    (默认: `1`)
    """

    top_k: int | None = None
    """降低生成无意义内容的概率。

    A higher value (e.g. `100`) will give more diverse answers, while a lower value
    (e.g. `10`) will be more conservative.

    (默认: `40`)
    """

    top_p: float | None = None
    """与 top-k 一起使用。

    A higher value (e.g., `0.95`) will lead to more diverse text, while a lower value
    (e.g., `0.5`) will generate more focused and conservative text.

    (默认: `0.9`)
    """

    format: Literal["", "json"] | JsonSchemaValue | None = None
    """指定输出格式（选项：`'json'`，JSON schema）。"""

    keep_alive: int | str | None = None
    """模型在内存中保持加载的时间长度。"""

    base_url: str | None = None
    """模型托管的基础 URL。

    若未指定，则默认使用 Ollama 客户端默认值。

    Supports `userinfo` auth in the format `http://username:password@localhost:11434`.
    Useful if your Ollama server is behind a proxy.

    !!! warning
        `userinfo` is not secure and should only be used for local testing or
        in secure environments. Avoid using it in production or over unsecured
        networks.

    !!! note 使用 `userinfo` 时，请确保 Ollama 服务器已配置为接受并验证这些凭据。

    !!! note
        `userinfo` 头信息会传递给同步和异步客户端。

    """

    client_kwargs: dict | None = {}
    """传递给 httpx 客户端的额外关键字参数。在此处传递 headers。

    这些参数会同时传递给同步和异步客户端。

    使用 `sync_client_kwargs` 和 `async_client_kwargs`
    来为同步和异步客户端分别传递不同的参数。
    """

    async_client_kwargs: dict | None = {}
    """在传递给 httpx 客户端之前，与 `client_kwargs` 合并的额外关键字参数。

    这些是仅适用于异步客户端的参数；共享参数请使用 `client_kwargs`。

    完整参数列表请参见 [httpx 文档](https://www.python-httpx.org/api/#asyncclient)。
    """

    sync_client_kwargs: dict | None = {}
    """在传递给 httpx 客户端之前，与 `client_kwargs` 合并的额外关键字参数。

    这些是仅适用于同步客户端的参数；共享参数请使用 `client_kwargs`。

    完整参数列表请参见 [httpx 文档](https://www.python-httpx.org/api/#client)。
    """

    _client: Client = PrivateAttr()
    """用于发起请求的客户端。"""

    _async_client: AsyncClient = PrivateAttr()
    """用于发起请求的异步客户端。"""

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """组装聊天补全请求的参数。

        Args:
            messages: 要发送给模型的 LangChain 消息列表。
            stop: 可选的停止 token 列表，用于本次调用。
            **kwargs: 要包含在请求中的额外关键字参数。

        Returns:
            传递给 Ollama 客户端的参数字典。
        """
        ollama_messages = self._convert_messages_to_ollama_messages(messages)

        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop

        options_dict = kwargs.pop("options", None)
        if options_dict is None:
                        # Only include parameters that are explicitly set (not None)
            options_dict = {
                k: v
                for k, v in {
                    "mirostat": self.mirostat,
                    "mirostat_eta": self.mirostat_eta,
                    "mirostat_tau": self.mirostat_tau,
                    "num_ctx": self.num_ctx,
                    "num_gpu": self.num_gpu,
                    "num_thread": self.num_thread,
                    "num_predict": self.num_predict,
                    "repeat_last_n": self.repeat_last_n,
                    "repeat_penalty": self.repeat_penalty,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "stop": self.stop if stop is None else stop,
                    "tfs_z": self.tfs_z,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }.items()
                if v is not None
            }

        format_param = self._resolve_format_param(
            kwargs.pop("format", self.format),
            kwargs.pop("response_format", None),
        )

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": format_param,
            "logprobs": kwargs.pop("logprobs", self.logprobs),
            "top_logprobs": kwargs.pop("top_logprobs", self.top_logprobs),
            "options": options_dict,
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

                # 过滤掉可能存在的 'strict' 参数，因为 Ollama 不支持该参数
                # 但可能由上游库（例如 LangChain ProviderStrategy）传递
        if "strict" in params:
            params.pop("strict")

        if tools := kwargs.get("tools"):
            params["tools"] = tools

        return params

    def _resolve_format_param(
        self,
        format_param: str | dict[str, Any] | None,
        response_format: Any | None,
    ) -> str | dict[str, Any] | None:
        """解析 format 参数。

        将 OpenAI 风格的 `response_format` 字典转换为 Ollama 所需的 `format` 参数。

        Args:
            format_param: 显式的 `format` 值（优先级更高）。
            response_format: OpenAI 风格的 `response_format` 字典。

        Returns:
            传递给 Ollama 客户端的解析后 format 值。
        """
        if format_param is not None:
            if response_format is not None:
                warnings.warn(
                    "Both 'format' and 'response_format' were provided. "
                    "'response_format' will be ignored in favor of 'format'.",
                    UserWarning,
                    stacklevel=2,
                )
            return format_param

        if response_format is None:
            return None

        return self._convert_response_format(response_format)

    def _convert_response_format(
        self,
        response_format: Any,
    ) -> str | dict[str, Any] | None:
        """将 OpenAI 风格的 `response_format` 转换为 Ollama 的 `format` 值。

        Args:
            response_format: 需要转换的 `response_format` 值。

        Returns:
            Ollama 兼容的 `format` 值，如果转换失败则返回 `None`。
        """
        if not isinstance(response_format, dict):
            warnings.warn(
                f"Ignored invalid 'response_format' type: {type(response_format)}. "
                "Expected a dictionary.",
                UserWarning,
                stacklevel=2,
            )
            return None

        fmt_type = response_format.get("type")
        if fmt_type == "json_object":
            return "json"
        if fmt_type == "json_schema":
            return self._extract_json_schema(response_format)

        warnings.warn(
            f"Ignored unrecognized 'response_format' type: {fmt_type}. "
            "Expected 'json_object' or 'json_schema'.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def _extract_json_schema(
        self,
        response_format: dict[str, Any],
    ) -> dict[str, Any] | None:
        """从 OpenAI 的 `json_schema` 包装中提取原始 JSON 模式。

        Args:
            response_format: 包含 `type: "json_schema"` 的字典。

        Returns:
            提取的原始 JSON 模式字典，如果提取失败则返回 `None`。
        """
        json_schema_block = response_format.get("json_schema")
        if not isinstance(json_schema_block, dict):
            warnings.warn(
                "response_format has type 'json_schema' but 'json_schema' "
                f"value is {type(json_schema_block)}, expected a dict "
                "containing a 'schema' key. "
                "The format parameter will not be set.",
                UserWarning,
                stacklevel=2,
            )
            return None
        schema = json_schema_block.get("schema")
        if schema is None:
            warnings.warn(
                "response_format has type 'json_schema' but no 'schema' "
                "key was found in 'json_schema'. "
                "The format parameter will not be set.",
                UserWarning,
                stacklevel=2,
            )
        return schema

    @model_validator(mode="after")
    def _set_ollama_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-ollama", __version__)
        return self

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """设置用于 ollama 的客户端。"""
        if self.top_logprobs is not None and self.logprobs is not True:
            if self.logprobs is False:
                msg = (
                    "`top_logprobs` is set but `logprobs` is explicitly `False`. "
                    "Either set `logprobs=True` to use `top_logprobs`, or remove "
                    "`top_logprobs`."
                )
                raise ValueError(msg)
                        # logprobs 为 None（未设置）——自动启用以提供便利
            self.logprobs = True
            warnings.warn(
                "`top_logprobs` is set but `logprobs` was not explicitly enabled. "
                "Setting `logprobs=True` automatically.",
                UserWarning,
                stacklevel=2,
            )

        client_kwargs = self.client_kwargs or {}

        cleaned_url, auth_headers = parse_url_with_auth(self.base_url)
        merge_auth_headers(client_kwargs, auth_headers)

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=cleaned_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=cleaned_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    def _convert_messages_to_ollama_messages(
        self, messages: list[BaseMessage]
    ) -> Sequence[Message]:
        """将 BaseMessage 列表转换为 Ollama 可消费的消息列表。

        Args:
            messages: 需要转换的 BaseMessage 列表。

        Returns:
            Ollama 格式的消息列表。
        """
        messages = list(messages)  # 浅拷贝以避免修改调用方的列表
        for idx, message in enumerate(messages):
                        # 处理以 v1 格式编写的消息内容
            if (
                isinstance(message, AIMessage)
                and message.response_metadata.get("output_version") == "v1"
            ):
                                # 将已知的 v1 内容解包为 Ollama 格式以用于请求
                                # 大多数类型保持不变地传递
                messages[idx] = message.model_copy(
                    update={
                        "content": _convert_from_v1_to_ollama(
                            cast("list[types.ContentBlock]", message.content),
                            message.response_metadata.get("model_provider"),
                        )
                    }
                )

        ollama_messages: list = []
        for message in messages:
            role: str
            tool_call_id: str | None = None
            tool_calls: list[dict[str, Any]] | None = None
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
                tool_calls = (
                    [
                        _lc_tool_call_to_openai_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    if message.tool_calls
                    else None
                )
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ChatMessage):
                role = message.role
            elif isinstance(message, ToolMessage):
                role = "tool"
                tool_call_id = message.tool_call_id
            else:
                msg = "Received unsupported message type for Ollama."
                raise TypeError(msg)

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:  # List
                for content_part in message.content:
                    if isinstance(content_part, str):
                        if content:
                            content += "\n"
                        content += content_part
                    elif content_part.get("type") == "text":
                        if content:
                            content += "\n"
                        content += content_part["text"]
                    elif content_part.get("type") == "tool_use":
                        continue
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = temp_image_url
                        elif (
                            isinstance(temp_image_url, dict)
                            and "url" in temp_image_url
                            and isinstance(temp_image_url["url"], str)
                        ):
                            image_url = temp_image_url["url"]
                        else:
                            msg = (
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )
                            raise ValueError(msg)

                        image_url_components = image_url.split(",")
                                                # 支持 data:image/jpeg;base64,<image> 格式
                                                # 以及 base64 字符串
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])
                    elif is_data_content_block(content_part):
                                                # 处理 v1 的 "image" 类型
                        image = _get_image_from_data_content_block(content_part)
                        images.append(image)
                    else:
                        msg = (
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )
                        raise ValueError(msg)
                        # 应在 Message 中包含 tool 和 tool_call_id 时转换为 ollama.Message
                        # 应在 Message 中包含 tool 和 tool_call_id 时转换为 ollama.Message
            msg_: dict = {
                "role": role,
                "content": content,
                "images": images,
            }
            if tool_calls:
                msg_["tool_calls"] = tool_calls
            if tool_call_id:
                msg_["tool_call_id"] = tool_call_id
            if isinstance(message, AIMessage):
                thinking = message.additional_kwargs.get("reasoning_content")
                if thinking is not None:
                    msg_["thinking"] = thinking
            ollama_messages.append(msg_)

        return ollama_messages

    async def _acreate_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Mapping[str, Any] | str]:
        if not self._async_client:
            msg = (
                "Ollama async client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            async for part in await self._async_client.chat(**chat_params):
                yield part
        else:
            yield await self._async_client.chat(**chat_params)

    def _create_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Mapping[str, Any] | str]:
        if not self._client:
            msg = (
                "Ollama sync client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            yield from self._client.chat(**chat_params)
        else:
            yield self._client.chat(**chat_params)

    def _chat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        verbose: bool = False,  # noqa: FBT002
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        verbose: bool = False,  # noqa: FBT002
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        return final_chunk

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """获取追踪的标准参数。"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = self._chat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(
                    "AIMessageChunk", final_chunk.message
                ).usage_metadata,
                tool_calls=cast("AIMessageChunk", final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _iterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                content = (
                    stream_resp["message"]["content"]
                    if "message" in stream_resp and "content" in stream_resp["message"]
                    else ""
                )

                                # 警告并跳过 done_reason 为 'load' 且内容为空的响应
                                # 这些表示模型已加载但未实际生成任何内容
                is_load_response_with_empty_content = (
                    stream_resp.get("done") is True
                    and stream_resp.get("done_reason") == "load"
                    and not content.strip()
                )

                if is_load_response_with_empty_content:
                    log.warning(
                        "Ollama returned empty response with done_reason='load'."
                        "This typically indicates the model was loaded but no content "
                        "was generated. Skipping this response."
                    )
                    continue

                if stream_resp.get("done") is True:
                    generation_info = dict(stream_resp)
                    if "model" in generation_info:
                        generation_info["model_name"] = generation_info["model"]
                    generation_info["model_provider"] = "ollama"
                    _ = generation_info.pop("message", None)
                else:
                    chunk_logprobs = stream_resp.get("logprobs")
                    generation_info = (
                        {"logprobs": chunk_logprobs}
                        if chunk_logprobs is not None
                        else None
                    )

                additional_kwargs = {}
                if (
                    reasoning
                    and "message" in stream_resp
                    and (thinking_content := stream_resp["message"].get("thinking"))
                ):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=generation_info,
                )

                yield chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _aiterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                content = (
                    stream_resp["message"]["content"]
                    if "message" in stream_resp and "content" in stream_resp["message"]
                    else ""
                )

                                # 警告并跳过 done_reason 为 'load' 且内容为空的响应
                                # 这些表示模型已加载但未实际生成任何内容
                is_load_response_with_empty_content = (
                    stream_resp.get("done") is True
                    and stream_resp.get("done_reason") == "load"
                    and not content.strip()
                )

                if is_load_response_with_empty_content:
                    log.warning(
                        "Ollama returned empty response with done_reason='load'. "
                        "This typically indicates the model was loaded but no content "
                        "was generated. Skipping this response."
                    )
                    continue

                if stream_resp.get("done") is True:
                    generation_info = dict(stream_resp)
                    if "model" in generation_info:
                        generation_info["model_name"] = generation_info["model"]
                    generation_info["model_provider"] = "ollama"
                    _ = generation_info.pop("message", None)
                else:
                    chunk_logprobs = stream_resp.get("logprobs")
                    generation_info = (
                        {"logprobs": chunk_logprobs}
                        if chunk_logprobs is not None
                        else None
                    )

                additional_kwargs = {}
                if (
                    reasoning
                    and "message" in stream_resp
                    and (thinking_content := stream_resp["message"].get("thinking"))
                ):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=generation_info,
                )

                yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = await self._achat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(
                    "AIMessageChunk", final_chunk.message
                ).usage_metadata,
                tool_calls=cast("AIMessageChunk", final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        """返回聊天模型的返回类型。"""
        return "chat-ollama"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,  # noqa: PYI051, ARG002
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        假设模型与 OpenAI 工具调用 API 兼容。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                Supports any tool definition handled by [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: If provided, which tool for model to call. **This parameter
                is currently ignored as it is not supported by Ollama.**
            kwargs: Any additional parameters are passed directly to
                `self.bind(**kwargs)`.
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        r"""返回与给定模式匹配格式输出的模型包装器。

        Args:
            schema: 输出模式。可作为以下形式传入：

                - OpenAI 函数/工具模式。
                - JSON Schema，
                - 一个 `TypedDict` 类，
                - 或一个 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                有关如何正确指定模式字段的类型和描述，请参阅
                `langchain_core.utils.function_calling.convert_to_openai_tool`。

            method: 控制模型生成的方法，可选之一是：

                - `'json_schema'`:
                    Uses Ollama's [structured output API](https://ollama.com/blog/structured-outputs)
                - `'function_calling'`:
                    Uses Ollama's tool-calling API
                - `'json_mode'`:
                    Specifies `format='json'`. Note that if using JSON mode then you
                    must include instructions for formatting the output into the
                    desired schema into the model call.

            include_raw:
                若为 `False`，则仅返回解析后的结构化输出。

                若在模型输出解析过程中发生错误，将抛出异常。

                若为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                若在输出解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            kwargs: 不支持额外的关键字参数。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                若 `include_raw` 为 `True`，则 `Runnable` 输出一个包含以下键的 `dict`:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        !!! warning "行为变更：`langchain-ollama` 0.2.2 版本起"

            增加对通过 `format` 参数使用结构化输出 API 的支持。

        !!! warning "行为变更：`langchain-ollama` 0.3.0 版本起"

            将默认 `method` 更新为 `'json_schema'`。

        ??? note "示例：`schema=Pydantic` 类，`method='json_schema'`，`include_raw=False`"

            ```python
            from typing import Optional

            from langchain_ollama import ChatOllama
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=...,
                    description="A justification for the answer.",
                )


            model = ChatOllama(model="llama3.1", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

            # -> AnswerWithJustification(
            #     answer='They weigh the same',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            # )
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='json_schema'`，`include_raw=True`"

            ```python
            from langchain_ollama import ChatOllama
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatOllama(model="llama3.1", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification,
                include_raw=True,
            )

            structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
            # -> {
            #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
            #     'parsing_error': None
            # }
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='function_calling'`，`include_raw=False`"

            ```python
            from typing import Optional

            from langchain_ollama import ChatOllama
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=...,
                    description="A justification for the answer.",
                )


            model = ChatOllama(model="llama3.1", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification,
                method="function_calling",
            )

            structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

            # -> AnswerWithJustification(
            #     answer='They weigh the same',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            # )
            ```

        ??? note "示例：`schema=TypedDict` 类，`method='function_calling'`，`include_raw=False`"

            ```python
            from typing_extensions import Annotated, TypedDict

            from langchain_ollama import ChatOllama


            class AnswerWithJustification(TypedDict):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: Annotated[str | None, None, "A justification for the answer."]


            model = ChatOllama(model="llama3.1", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
            # -> {
            #     'answer': 'They weigh the same',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            # }
            ```

        ??? note "示例：`schema=OpenAI` 函数模式，`method='function_calling'`，`include_raw=False`"

            ```python
            from langchain_ollama import ChatOllama

            oai_schema = {
                'name': 'AnswerWithJustification',
                'description': 'An answer to the user question along with justification for the answer.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'answer': {'type': 'string'},
                        'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                    },
                    'required': ['answer']
                }

                model = ChatOllama(model="llama3.1", temperature=0)
                structured_model = model.with_structured_output(oai_schema)

                structured_model.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='json_mode'`，`include_raw=True`"

            ```python
            from langchain_ollama import ChatOllama
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = ChatOllama(model="llama3.1", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            # -> {
            #     'raw': AIMessage(content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'),
            #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
            #     'parsing_error': None
            # }
            ```

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # ty: ignore[invalid-argument-type]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                format="json",
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # ty: ignore[invalid-argument-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            if is_pydantic_schema:
                schema = cast("TypeBaseModel", schema)
                if issubclass(schema, BaseModelV1):
                    response_format = schema.schema()
                else:
                    response_format = schema.model_json_schema()
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    response_format = convert_to_json_schema(schema)
                    if "required" not in response_format:
                        response_format["required"] = list(
                            response_format["properties"].keys()
                        )
                else:
                                        # 是 JSON schema
                    response_format = cast("dict", schema)
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": response_format,
                    },
                )
                output_parser = JsonOutputParser()
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', or 'json_mode'. Received: '{method}'"
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
