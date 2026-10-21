"""从 JSON 字符串或对象加载 LangChain 对象。

## 工作原理

每个 `Serializable` LangChain 对象都有一个唯一标识符（其 "类路径"），
该标识符是表示模块路径和类名的字符串列表。例如：

- `AIMessage` -> `["langchain_core", "messages", "ai", "AIMessage"]`
- `ChatPromptTemplate` -> `["langchain_core", "prompts", "chat", "ChatPromptTemplate"]`

When deserializing, the class path from the JSON `'id'` field is checked against an
allowlist. If the class is not in the allowlist, deserialization raises a `ValueError`.

## 威胁模型

A serialized LangChain payload crosses a trust boundary because the manifest
may contain serialized objects and configuration that affect runtime behavior.
For example, a payload can configure a chat model with a custom `base_url`,
custom headers, a different model name, or other constructor arguments. These
are supported features, but they also mean the payload contents should be
treated as executable configuration rather than plain text.

Concretely, deserialization instantiates Python objects, so any constructor
(`__init__`) or validator on an allowed class can run during `load()`. A
crafted payload that is allowed to reach an unintended class — or an intended
class with attacker-controlled kwargs — could cause network calls, file
operations, or environment-variable access while the object is being built.

!!! warning "不要对不受信任的输入使用"

    If the source is untrusted, avoid calling `load()` / `loads()` on it. If
    you must, restrict `allowed_objects` to types that do not execute logic
    during init — `allowed_objects='messages'` (or an explicit list of
    message classes) is the safe choice. Keep `secrets_from_env=False`.

`allowed_objects` 参数控制哪些类可以被反序列化：

- **Explicit list of classes** (recommended for untrusted input): only those
    specific classes are allowed.
- **`'messages'`**: chat-message classes only (e.g. `AIMessage`,
    `HumanMessage`). Safe for untrusted input.
- **`'core'` (current default)** — *unsafe with untrusted manifests.*
    Classes defined in the serialization mappings under `langchain_core`
    (messages, documents, prompts, etc.).
- **`'all'`** — *unsafe with untrusted manifests.* Every class in the
    serialization mappings, including partner chat models and LLMs and their
    constructor kwargs (endpoint URLs, headers, model names, etc.).

!!! note "允许类中的副作用"

    反序列化会调用允许类的
    `__init__`。如果这些类在初始化期间执行副作用（网络调用、文件操作等），这些副作用将会发生。允许列表防止实例化允许列表外的类，但不会对允许类本身进行沙箱隔离或限制其构造函数参数。

    For example, an untrusted manifest could deserialize a chat model whose
    `base_url` (or `endpoint_url`) points at an attacker-controlled host. Any
    request that model makes is then directed there — a Server-Side Request
    Forgery (SSRF) vector. This is *expected behavior*: deserialization
    faithfully reconstructs the configuration carried by the manifest, custom
    endpoints included, and LangChain does not special-case or strip such
    kwargs. The mitigation is to **only deserialize manifests you trust**,
    and for untrusted input to restrict `allowed_objects` to `'messages'`
    or an explicit list of classes that take no endpoint configuration.

导入路径在导入任何模块前也会根据受信任的命名空间进行验证。

### 最佳实践

- Use the most restrictive `allowed_objects` possible. For untrusted input,
    pass an explicit list of classes or `'messages'`. `'core'` and `'all'`
    are unsafe with untrusted manifests — only use them when the source
    serves the entire payload, including its configuration.
- Keep `secrets_from_env` set to `False` (the default). If you must use it,
    ensure the serialized data comes from a fully trusted source, as a crafted
    payload can read arbitrary environment variables.
- When using `secrets_map`, include only the specific secrets that the
    serialized object requires.

### 注入防护（基于转义）

During serialization, plain dicts that contain an `'lc'` key are escaped by wrapping
them: `{"__lc_escaped__": {...}}`. During deserialization, escaped dicts are unwrapped
and returned as plain dicts, NOT instantiated as LC objects.

这是一种白名单方法：只有由 `Serializable.to_json()`
显式生成的字典（这些字典不会被转义）才会被视为 LC 对象；其他所有内容都被视为用户数据。

Even if an attacker's payload includes `__lc_escaped__` wrappers, it will be unwrapped
to plain dicts and NOT instantiated as malicious objects.

## 示例

```python
from langchain_core.load import load
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Use default allowlist (classes from mappings) - recommended
obj = load(data)

# Allow only specific classes (most restrictive)
obj = load(
    data,
    allowed_objects=[
        ChatPromptTemplate,
        AIMessage,
        HumanMessage,
    ],
)
```
"""

import importlib
import json
import os
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from langchain_core._api import beta
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.load._validation import _is_escaped_dict, _unescape_value
from langchain_core.load.mapping import (
    _JS_SERIALIZABLE_MAPPING,
    _OG_SERIALIZABLE_MAPPING,
    OLD_CORE_NAMESPACES_MAPPING,
    SERIALIZABLE_MAPPING,
)
from langchain_core.load.serializable import Serializable

DEFAULT_NAMESPACES = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_anthropic",
    "langchain_groq",
    "langchain_google_genai",
    "langchain_aws",
    "langchain_openai",
    "langchain_google_vertexai",
    "langchain_mistralai",
    "langchain_fireworks",
    "langchain_xai",
    "langchain_sambanova",
    "langchain_perplexity",
]
# 仅允许通过 SERIALIZABLE_MAPPING 进行反序列化的命名空间。
# 通过路径加载不被允许。
DISALLOW_LOAD_FROM_PATH = [
    "langchain_community",
    "langchain",
]

ALL_SERIALIZABLE_MAPPINGS = {
    **SERIALIZABLE_MAPPING,
    **OLD_CORE_NAMESPACES_MAPPING,
    **_OG_SERIALIZABLE_MAPPING,
    **_JS_SERIALIZABLE_MAPPING,
}

# 现代消息类由 `allowed_objects='messages'` 接受。旧类型
# (BaseMessage / BaseMessageChunk, ChatMessage / ChatMessageChunk, FunctionMessage /
# FunctionMessageChunk) 被有意排除 — `BaseMessage` 是抽象类，
# 而聊天/函数变体已被 `ToolMessage` 和工具调用取代。
_MESSAGES_ALLOWED_CLASS_NAMES = frozenset(
    {
        "AIMessage",
        "AIMessageChunk",
        "HumanMessage",
        "HumanMessageChunk",
        "SystemMessage",
        "SystemMessageChunk",
        "ToolMessage",
        "ToolMessageChunk",
        "RemoveMessage",
    }
)

# 为默认允许的类路径缓存，基于映射计算
# 将模式（"all", "core", 或 "messages"）映射到缓存的路径集合
_default_class_paths_cache: dict[str, set[tuple[str, ...]]] = {}


def _get_default_allowed_class_paths(
    allowed_object_mode: Literal["all", "core", "messages"],
) -> set[tuple[str, ...]]:
    """从序列化映射中获取默认允许的类路径。

    This uses the mappings as the source of truth for what classes are allowed
    by default. Both the legacy paths (keys) and current paths (values) are included.

    Args:
        allowed_object_mode: 取值为 `'all'`、`'core'` 或 `'messages'`。

    Returns:
        默认允许的类路径元组集合。
    """
    if allowed_object_mode in _default_class_paths_cache:
        return _default_class_paths_cache[allowed_object_mode]

    allowed_paths: set[tuple[str, ...]] = set()
    for key, value in ALL_SERIALIZABLE_MAPPINGS.items():
        if allowed_object_mode == "core" and value[0] != "langchain_core":
            continue
        if allowed_object_mode == "messages" and (
            value[0] != "langchain_core"
            or value[-1] not in _MESSAGES_ALLOWED_CLASS_NAMES
        ):
            continue
        allowed_paths.add(key)
        allowed_paths.add(value)

    _default_class_paths_cache[allowed_object_mode] = allowed_paths
    return _default_class_paths_cache[allowed_object_mode]


def _block_jinja2_templates(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """在反序列化期间阻止 jinja2 模板以保障安全。

    Jinja2 templates can execute arbitrary code, so they are blocked by default when
    deserializing objects with `template_format='jinja2'`.

    Note:
        We intentionally do NOT check the `class_path` here to keep this simple and
        future-proof. If any new class is added that accepts `template_format='jinja2'`,
        it will be automatically blocked without needing to update this function.

    Args:
        class_path: 正在反序列化的类路径元组（未使用）。
        kwargs: 类构造函数的参数字典。

    Raises:
        ValueError: 如果 `template_format` 为 `'jinja2'`。
    """
    _ = class_path  # 未使用 - 详见文档说明。保留以满足签名要求。
    if kwargs.get("template_format") == "jinja2":
        msg = (
            "Jinja2 templates are not allowed during deserialization for security "
            "reasons. Use 'f-string' template format instead, or explicitly allow "
            "jinja2 by providing a custom init_validator."
        )
        raise ValueError(msg)


def default_init_validator(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """默认初始化验证器，阻止 jinja2 模板。

    这是 `load()` 和 `loads()` 在未提供自定义验证器时使用的默认验证器。

    Args:
        class_path: 正在反序列化的类路径元组。
        kwargs: 类构造函数的参数字典。

    Raises:
        ValueError: 如果 template_format 为 `'jinja2'`。
    """
    _block_jinja2_templates(class_path, kwargs)


AllowedObject = type[Serializable]
"""Type alias for classes that can be included in the `allowed_objects` parameter.

Must be a `Serializable` subclass (the class itself, not an instance).
"""

InitValidator = Callable[[tuple[str, ...], dict[str, Any]], None]
"""Type alias for a callable that validates kwargs during deserialization.

The callable receives:

- `class_path`: A tuple of strings identifying the class being instantiated
    (e.g., `('langchain', 'schema', 'messages', 'AIMessage')`).
- `kwargs`: The kwargs dict that will be passed to the constructor.

The validator should raise an exception if the object should not be deserialized.
"""


def _compute_allowed_class_paths(
    allowed_objects: Iterable[AllowedObject],
    import_mappings: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """从显式类列表中返回允许的类路径。

    A class path is a tuple of strings identifying a serializable class, derived from
    `Serializable.lc_id()`. For example: `('langchain_core', 'messages', 'AIMessage')`.

    Args:
        allowed_objects: 要允许的 `Serializable` 子类的可迭代对象。
        import_mappings: 旧类路径到当前类路径的映射。

    Returns:
        允许的类路径集合。

    Example:
        ```python
        # Allow a specific class
        _compute_allowed_class_paths([MyPrompt], {}) ->
            {("langchain_core", "prompts", "MyPrompt")}

        # Include legacy paths that map to the same class
        import_mappings = {("old", "Prompt"): ("langchain_core", "prompts", "MyPrompt")}
        _compute_allowed_class_paths([MyPrompt], import_mappings) ->
            {("langchain_core", "prompts", "MyPrompt"), ("old", "Prompt")}
        ```
    """
    allowed_objects_list = list(allowed_objects)

    allowed_class_paths: set[tuple[str, ...]] = set()
    for allowed_obj in allowed_objects_list:
        if not isinstance(allowed_obj, type) or not issubclass(
            allowed_obj, Serializable
        ):
            msg = "allowed_objects must contain Serializable subclasses."  # type: ignore[unreachable]
            raise TypeError(msg)

        class_path = tuple(allowed_obj.lc_id())
        allowed_class_paths.add(class_path)
                # 添加映射到同一类的旧路径。
        for mapping_key, mapping_value in import_mappings.items():
            if tuple(mapping_value) == class_path:
                allowed_class_paths.add(mapping_key)
    return allowed_class_paths


class Reviver:
    """JSON 对象的还原器。

    Used as the `object_hook` for `json.loads` to reconstruct LangChain objects from
    their serialized JSON representation.

    只有允许列表中的类才能被实例化。
    """

    def __init__(
        self,
        allowed_objects: Iterable[AllowedObject]
        | Literal["all", "core", "messages"]
        | None = None,
        secrets_map: dict[str, str] | None = None,
        valid_namespaces: list[str] | None = None,
        secrets_from_env: bool = False,  # noqa: FBT001,FBT002
        additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]]
        | None = None,
        *,
        ignore_unserializable_fields: bool = False,
        init_validator: InitValidator | None = default_init_validator,
    ) -> None:
        """初始化恢复器。

        See the module docstring for the threat model around `load()`/`loads()`:
        a serialized payload may carry constructor configuration that affects
        runtime behavior (custom `base_url`, headers, model name, etc.). Do not
        use `'core'` or `'all'` with untrusted manifests.

        Args:
            allowed_objects: Allowlist of classes that can be deserialized.
                - Explicit list of classes (recommended for untrusted input):
                    only those specific classes are allowed.
                - `'messages'`: chat-message classes only (e.g. `AIMessage`,
                    `HumanMessage`). Safe for untrusted input.
                - `'core'` (current default): unsafe with untrusted manifests.
                    Classes defined in the serialization mappings under
                    `langchain_core`.
                - `'all'`: unsafe with untrusted manifests. Every class in the
                    serialization mappings, including partner chat models and
                    LLMs and their constructor kwargs. See
                    `langchain_core.load.mapping` for the full list.
            secrets_map: A map of secrets to load.

                Only include the specific secrets the serialized object
                requires. If a secret is not found in the map, it will be loaded
                from the environment if `secrets_from_env` is `True`.
            valid_namespaces: Additional namespaces (modules) to allow during
                deserialization, beyond the default trusted namespaces.
            secrets_from_env: Whether to load secrets from the environment.

                精心构造的负载可能在其 `secret`
                字段中命名任意环境变量，因此在不可信数据上启用此功能可能导致敏感值泄露。除非序列化数据完全可信，否则请保持
                `False`（默认值）。 additional_import_mappings: 额外的命名空间映射字典。

                可用于覆盖默认映射或添加新映射。

                When `allowed_objects` is `None` (using defaults), paths from these
                mappings are also added to the allowed class paths.
            ignore_unserializable_fields: Whether to ignore unserializable fields.
            init_validator: Optional callable to validate kwargs before instantiation.

                If provided, this function is called with `(class_path, kwargs)` where
                `class_path` is the class path tuple and `kwargs` is the kwargs dict.
                The validator should raise an exception if the object should not be
                deserialized, otherwise return `None`.

                默认为 `default_init_validator`，会阻止 jinja2 模板。
        """
        if allowed_objects is None:
            warn_deprecated(
                since="1.3.3",
                message=(
                    "The default value of `allowed_objects` will change in a future "
                    "version. Pass an explicit value (e.g., "
                    "allowed_objects='messages' or allowed_objects='core') to suppress "
                    "this warning."
                ),
                pending=True,
            )
            allowed_objects = "core"

        self.secrets_from_env = secrets_from_env
        self.secrets_map = secrets_map or {}
                # 默认情况下，仅支持 langchain，但用户可传入额外命名空间
        self.valid_namespaces = (
            [*DEFAULT_NAMESPACES, *valid_namespaces]
            if valid_namespaces
            else DEFAULT_NAMESPACES
        )
        self.additional_import_mappings = additional_import_mappings or {}
        self.import_mappings = (
            {
                **ALL_SERIALIZABLE_MAPPINGS,
                **self.additional_import_mappings,
            }
            if self.additional_import_mappings
            else ALL_SERIALIZABLE_MAPPINGS
        )
                # 计算允许的类路径：
                # - "all" -> 使用映射中的默认路径 (+ additional_import_mappings)
                # - 显式列表 -> 从这些类中计算
        if allowed_objects in ("all", "core", "messages"):
            self.allowed_class_paths: set[tuple[str, ...]] | None = (
                _get_default_allowed_class_paths(allowed_objects).copy()
            )
                        # 将 additional_import_mappings 中的路径添加到默认值中
            if self.additional_import_mappings:
                for key, value in self.additional_import_mappings.items():
                    self.allowed_class_paths.add(key)
                    self.allowed_class_paths.add(value)
        else:
            self.allowed_class_paths = _compute_allowed_class_paths(
                cast("Iterable[AllowedObject]", allowed_objects), self.import_mappings
            )
        self.ignore_unserializable_fields = ignore_unserializable_fields
        self.init_validator = init_validator

    def __call__(self, value: dict[str, Any]) -> Any:
        """恢复值。

        Args:
            value: 要恢复的值。

        Returns:
            恢复后的值。

        Raises:
            ValueError: 如果命名空间无效。 ValueError: 如果尝试反序列化当前
            langchain-core 版本中无法反序列化的内容。 NotImplementedError:
            如果对象未实现且 `ignore_unserializable_fields` 为 False。
        """
        if (
            value.get("lc") == 1
            and value.get("type") == "secret"
            and value.get("id") is not None
        ):
            [key] = value["id"]
            if key in self.secrets_map:
                return self.secrets_map[key]
            if self.secrets_from_env and key in os.environ and os.environ[key]:
                return os.environ[key]
            return None

        if (
            value.get("lc") == 1
            and value.get("type") == "not_implemented"
            and value.get("id") is not None
        ):
            if self.ignore_unserializable_fields:
                return None
            msg = (
                "Trying to load an object that doesn't implement "
                f"serialization: {value}"
            )
            raise NotImplementedError(msg)

        if (
            value.get("lc") == 1
            and value.get("type") == "constructor"
            and value.get("id") is not None
        ):
            [*namespace, name] = value["id"]
            mapping_key = tuple(value["id"])

            if (
                self.allowed_class_paths is not None
                and mapping_key not in self.allowed_class_paths
            ):
                msg = (
                    f"Deserialization of {mapping_key!r} is not allowed. "
                    "The default (allowed_objects='core') only permits core "
                    "langchain-core classes. To allow trusted partner integrations, "
                    "use allowed_objects='all'. Alternatively, pass an explicit list "
                    "of allowed classes via allowed_objects=[...]. "
                    "See langchain_core.load.mapping for the full allowlist."
                )
                raise ValueError(msg)

            if (
                namespace[0] not in self.valid_namespaces
                                # 根命名空间 ["langchain"] 不是有效的标识符。
                or namespace == ["langchain"]
            ):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)
                        # 确定显式导入路径
            if mapping_key in self.import_mappings:
                import_path = self.import_mappings[mapping_key]
                                # 拆分为模块和名称
                import_dir, name = import_path[:-1], import_path[-1]
            elif namespace[0] in DISALLOW_LOAD_FROM_PATH:
                msg = (
                    "Trying to deserialize something that cannot "
                    "be deserialized in current version of langchain-core: "
                    f"{mapping_key}."
                )
                raise ValueError(msg)
            else:
                                # 否则，将命名空间视为路径。
                import_dir = namespace

                        # 在导入前验证导入路径是否在受信任命名空间内
            if import_dir[0] not in self.valid_namespaces:
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

                        # 我们不需要对 kwargs 进行递归
                        # 因为 json.loads 会为我们处理。
            kwargs = value.get("kwargs", {})

                        # 在导入前运行 init_validator（例如 jinja2 阻断）
                        # 以便在安全违规时快速失败。
            if self.init_validator is not None:
                self.init_validator(mapping_key, kwargs)

            mod = importlib.import_module(".".join(import_dir))

            cls = getattr(mod, name)

                        # 该类必须是 Serializable 的子类。
            if not issubclass(cls, Serializable):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            return cls(**kwargs)

        return value


@beta()
def loads(
    text: str,
    *,
    allowed_objects: Iterable[AllowedObject]
    | Literal["all", "core", "messages"]
    | None = None,
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """从 JSON 字符串还原一个 LangChain 类。

    等价于 `load(json.loads(text))`。

    Only classes in the allowlist can be instantiated. The default allowlist
    includes core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "不要对不可信输入使用"

        A serialized payload may carry constructor kwargs that affect runtime
        behavior (custom `base_url`, headers, model name, etc.), so it should be
        treated as executable configuration rather than plain text. For example,
        deserializing a model whose `base_url` points at an attacker-controlled
        host can result in Server-Side Request Forgery (SSRF); this is expected
        behavior, since `loads()` faithfully reconstructs the configuration in
        the manifest. If the source is untrusted, avoid calling `loads()` on it;
        if you must, pass `allowed_objects='messages'` or an explicit list of
        message classes. See the module-level threat model for details.

    Args:
        text: 要加载的字符串。
        allowed_objects: 可反序列化的类的允许列表。

            - Explicit list of classes (recommended for untrusted input): only
                those specific classes are allowed.
            - `'messages'`: chat-message classes only. Safe for untrusted input.
            - `'core'` (current default): unsafe with untrusted manifests.
                Classes defined in the serialization mappings under
                `langchain_core`.
            - `'all'`: unsafe with untrusted manifests. Every class in the
                serialization mappings, including partner chat models and LLMs
                and their constructor kwargs. See `langchain_core.load.mapping`
                for the full list.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            Only include the specific secrets the serialized object requires. If
            a secret is not found in the map, it will be loaded from the
            environment if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.

            伪造的负载可以在其 `secret`
            字段中指定任意环境变量，因此在不受信任的数据上启用此功能可能会导致敏感值泄露。除非序列化数据完全可信，否则请保持
            `False`（默认值）。 additional_import_mappings: 额外的命名空间映射字典。

            可用于覆盖默认映射或添加新映射。

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            默认为 `default_init_validator`，会阻止 jinja2 模板。

    Returns:
        已恢复的 LangChain 对象。

    Raises:
        ValueError: 如果对象的类路径不在 `allowed_objects` 允许列表中。
    """
    if allowed_objects is None:
        warn_deprecated(
            since="1.3.3",
            message=(
                "The default value of `allowed_objects` will change in a future "
                "version. Pass an explicit list of allowed classes (or "
                "'messages' for untrusted input that contains only chat "
                "messages) to suppress this warning."
            ),
            pending=True,
        )
        allowed_objects = "core"

        # 解析 JSON 并委托给 load() 以正确处理转义
    raw_obj = json.loads(text)
    return load(
        raw_obj,
        allowed_objects=allowed_objects,
        secrets_map=secrets_map,
        valid_namespaces=valid_namespaces,
        secrets_from_env=secrets_from_env,
        additional_import_mappings=additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )


@beta()
def load(
    obj: Any,
    *,
    allowed_objects: Iterable[AllowedObject]
    | Literal["all", "core", "messages"]
    | None = None,
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """从 JSON 对象恢复一个 LangChain 类。

    Use this if you already have a parsed JSON object, eg. from `json.load` or
    `orjson.loads`.

    Only classes in the allowlist can be instantiated. The default allowlist
    includes core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "不要对不可信输入使用"

        A serialized payload may carry constructor kwargs that affect runtime
        behavior (custom `base_url`, headers, model name, etc.), so it should be
        treated as executable configuration rather than plain text. For example,
        deserializing a model whose `base_url` points at an attacker-controlled
        host can result in Server-Side Request Forgery (SSRF); this is expected
        behavior, since `load()` faithfully reconstructs the configuration in
        the manifest. If the source is untrusted, avoid calling `load()` on it;
        if you must, pass `allowed_objects='messages'` or an explicit list of
        message classes. See the module-level threat model for details.

    Args:
        obj: 要加载的对象。
        allowed_objects: 可以反序列化的类的允许列表。

            - Explicit list of classes (recommended for untrusted input): only
                those specific classes are allowed.
            - `'messages'`: chat-message classes only. Safe for untrusted input.
            - `'core'` (current default): unsafe with untrusted manifests.
                Classes defined in the serialization mappings under
                `langchain_core`.
            - `'all'`: unsafe with untrusted manifests. Every class in the
                serialization mappings, including partner chat models and LLMs
                and their constructor kwargs. See `langchain_core.load.mapping`
                for the full list.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            仅包含序列化对象所需的特定秘密。

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.

            精心构造的负载可能在其 `secret`
            字段中命名任意环境变量，因此在不受信任的数据上启用此功能可能导致
            敏感值泄露。除非序列化数据完全可信，否则请保持此值为 `False`（默认值）。
            additional_import_mappings: 额外的命名空间映射字典。

            可用于覆盖默认映射或添加新映射。

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            默认为 `default_init_validator`，会阻止 jinja2 模板。

    Returns:
        已恢复的 LangChain 对象。

    Raises:
        ValueError: 如果对象的类路径不在 `allowed_objects` 允许列表中。

    Example:
        ```python
        from langchain_core.load import load, dumpd
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Hello")
        data = dumpd(msg)

        # Deserialize using default allowlist
        loaded = load(data)

        # Or with explicit allowlist
        loaded = load(data, allowed_objects=[AIMessage])

        # Or extend defaults with additional mappings
        loaded = load(
            data,
            additional_import_mappings={
                ("my_pkg", "MyClass"): ("my_pkg", "module", "MyClass"),
            },
        )
        ```
    """
    if allowed_objects is None:
        warn_deprecated(
            since="1.3.3",
            message=(
                "The default value of `allowed_objects` will change in a future "
                "version. Pass an explicit list of allowed classes (or "
                "'messages' for untrusted input that contains only chat "
                "messages) to suppress this warning."
            ),
            pending=True,
        )
        allowed_objects = "core"

    reviver = Reviver(
        allowed_objects,
        secrets_map,
        valid_namespaces,
        secrets_from_env,
        additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )

    def _load(obj: Any) -> Any:
        if isinstance(obj, dict):
                        # 首先检查是否为转义字典（在递归前）。
                        # 转义字典是用户数据，不应作为 LC 对象处理。
            if _is_escaped_dict(obj):
                return _unescape_value(obj)

                        # 未转义 - 递归进入子项，然后应用 reviver
            loaded_obj = {k: _load(v) for k, v in obj.items()}
            return reviver(loaded_obj)
        if isinstance(obj, list):
            return [_load(o) for o in obj]
        return obj

    return _load(obj)
