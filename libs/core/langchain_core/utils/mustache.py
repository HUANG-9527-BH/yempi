"""源自 https://github.com/noahmorrison/chevron。

MIT 许可证。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

if TYPE_CHECKING:
    from typing import TypeAlias

logger = logging.getLogger(__name__)


Scopes: TypeAlias = list[Literal[False, 0] | Mapping[str, Any]]


# Globals
_CURRENT_LINE = 1
_LAST_TAG_LINE = None


class ChevronError(SyntaxError):
    """Chevron 错误的自定义异常。"""


#
# Helper functions
#


def grab_literal(template: str, l_del: str) -> tuple[str, str]:
    """从模板中解析一个字面量。

    Args:
        template: 要解析的模板。
        l_del: 左分隔符。

    Returns:
        字面量和模板。
    """
    global _CURRENT_LINE

    try:
                # 查找下一个标签并将模板移动到该位置
        literal, template = template.split(l_del, 1)
        _CURRENT_LINE += literal.count("\n")

        # 模板中已无更多标签？
    except ValueError:
                # 则模板剩余部分为字面量
        return (template, "")

    return (literal, template)


def l_sa_check(
    template: str,  # noqa: ARG001
    literal: str,
    is_standalone: bool,  # noqa: FBT001
) -> bool:
    """进行初步检查，判断标签是否可能是独立的。

    Args:
        template: 模板。（未使用。）
        literal: 字面量。
        is_standalone: 标签是否为独立标签。

    Returns:
        标签是否可能是独立的。
    """
        # 如果存在换行符，或上一个标签是独立的
    if literal.find("\n") != -1 or is_standalone:
        padding = literal.rsplit("\n", maxsplit=1)[-1]

                # 如果自上次换行以来的所有字符都是空格
                # 则下一个标签可能是独立的
                # 否则不可能是独立的
        return padding.isspace() or not padding
    return False


def r_sa_check(
    template: str,
    tag_type: str,
    is_standalone: bool,  # noqa: FBT001
) -> bool:
    """进行最终检查，判断标签是否可能是独立的。

    Args:
        template: 模板。
        tag_type: 标签类型。
        is_standalone: 标签是否为独立标签。

    Returns:
        标签是否可能是独立的。
    """
        # 检查右侧我们是否可能是独立的
    if is_standalone and tag_type not in {"variable", "no escape"}:
        on_newline = template.split("\n", 1)

                # 如果我们在右侧的字符都是空格，则我们是独立的
        return on_newline[0].isspace() or not on_newline[0]

        # 如果我们是标签，则不能是独立的
    return False


def parse_tag(template: str, l_del: str, r_del: str) -> tuple[tuple[str, str], str]:
    """从模板中解析一个标签。

    Args:
        template: 模板。
        l_del: 左分隔符。
        r_del: 右分隔符。

    Returns:
        标签和模板。

    Raises:
        ChevronError: 如果标签未闭合。
        ChevronError: 如果设置分隔符的标签未闭合。
    """
    tag_types = {
        "!": "comment",
        "#": "section",
        "^": "inverted section",
        "/": "end",
        ">": "partial",
        "=": "set delimiter?",
        "{": "no escape?",
        "&": "no escape",
    }

        # 获取标签
    try:
        tag, template = template.split(r_del, 1)
    except ValueError as e:
        msg = f"unclosed tag at line {_CURRENT_LINE}"
        raise ChevronError(msg) from e

        # 检查空标签
    if not tag.strip():
        msg = f"empty tag at line {_CURRENT_LINE}"
        raise ChevronError(msg)

        # 查找第一个字符的类型含义
    tag_type = tag_types.get(tag[0], "variable")

        # 如果类型不是变量
    if tag_type != "variable":
                # 则第一个字符不需要
        tag = tag[1:]

        # 如果我们可能是设置分隔符的标签
    if tag_type == "set delimiter?":
                # 再次确认我们确实是
        if tag.endswith("="):
            tag_type = "set delimiter"
                        # 移除等号
            tag = tag[:-1]

                # 否则我们应该报错
        else:
            msg = f"unclosed set delimiter tag\nat line {_CURRENT_LINE}"
            raise ChevronError(msg)

    elif (
                # 如果我们可能是不转义 HTML 的标签
        tag_type == "no escape?"
                # 并且我们有第三个大括号
                # （且使用大括号作为分隔符）
        and l_del == "{{"
        and r_del == "}}"
        and template.startswith("}")
    ):
                # 则我们是不转义 HTML 的标签
        template = template[1:]
        tag_type = "no escape"

        # 去掉键的空白字符并返回
    return ((tag_type, tag.strip()), template)


#
# 主标记化函数
#


def tokenize(
    template: str, def_ldel: str = "{{", def_rdel: str = "}}"
) -> Iterator[tuple[str, str]]:
    """对 Mustache 模板进行分词。

    以生成器方式对 Mustache 模板进行分词，支持文件类对象。它也接受包含模板的字符串。

    Args:
        template: 文件类对象，或 Mustache 模板的字符串
        def_ldel: 默认左分隔符
            （默认为 `'{{'`，符合规范的 Mustache）
        def_rdel: 默认右分隔符
            （默认为 `'}}'`，符合规范的 Mustache）

    Yields:
        以元组 `(tag_type, tag_key)` 形式返回 Mustache 标签，其中 `tag_type`
        为以下之一：

            * literal
            * section
            * inverted section
            * end
            * partial
            * no escape

            ...and `tag_key` is either the key or in the case of a literal tag, the
            literal itself.

    Raises:
        ChevronError: 如果模板中存在语法错误。
    """
    global _CURRENT_LINE, _LAST_TAG_LINE
    _CURRENT_LINE = 1
    _LAST_TAG_LINE = None

    is_standalone = True
    open_sections = []
    l_del = def_ldel
    r_del = def_rdel

    while template:
        literal, template = grab_literal(template, l_del)

                # 如果模板已完成
        if not template:
                        # 则生成字面量并退出
            yield ("literal", literal)
            break

                # 首先检查我们是否可能是独立的
        is_standalone = l_sa_check(template, literal, is_standalone)

                # 解析标签
        tag, template = parse_tag(template, l_del, r_del)
        tag_type, tag_key = tag

                # 特殊标签逻辑

                # 如果我们是设置分隔符的标签
        if tag_type == "set delimiter":
                        # 则获取并设置分隔符
            dels = tag_key.strip().split(" ")
            l_del, r_del = dels[0], dels[-1]

                # 如果我们是区块标签
        elif tag_type in {"section", "inverted section"}:
                        # 则打开新区块
            open_sections.append(tag_key)
            _LAST_TAG_LINE = _CURRENT_LINE

                # 如果我们是结束标签
        elif tag_type == "end":
                        # 则检查最后一个打开的区块
                        # 是否与我们相同
            try:
                last_section = open_sections.pop()
            except IndexError as e:
                msg = (
                    f'Trying to close tag "{tag_key}"\n'
                    "Looks like it was not opened.\n"
                    f"line {_CURRENT_LINE + 1}"
                )
                raise ChevronError(msg) from e
            if tag_key != last_section:
                                # 否则我们需要报错
                msg = (
                    f'Trying to close tag "{tag_key}"\n'
                    f'last open tag is "{last_section}"\n'
                    f"line {_CURRENT_LINE + 1}"
                )
                raise ChevronError(msg)

                # 进行第二次检查以判断我们是否是独立的
        is_standalone = r_sa_check(template, tag_type, is_standalone)

                # 如果是
        if is_standalone:
                        # 则移除换行符之前的内容
            template = template.split("\n", 1)[-1]

                        # 部分需要保留其左侧的空格
            if tag_type != "partial":
                                # 但其他标签不需要
                literal = literal.rstrip(" ")

                # 开始生成
                # 忽略空的字面量
        if literal:
            yield ("literal", literal)

                # 忽略注释和设置分隔符
        if tag_type not in {"comment", "set delimiter?"}:
            yield (tag_type, tag_key)

        # 当完成时如果有任何打开的区块
    if open_sections:
                # 则我们需要报错
        msg = (
            "Unexpected EOF\n"
            f'the tag "{open_sections[-1]}" was never closed\n'
            f"was opened at line {_LAST_TAG_LINE}"
        )
        raise ChevronError(msg)


#
# 辅助函数
#


def _html_escape(string: str) -> str:
    """返回 HTML 转义后的字符串，转义这些字符：`" & < >`。"""
    html_codes = {
        '"': "&quot;",
        "<": "&lt;",
        ">": "&gt;",
    }

        # & 必须首先处理
    string = string.replace("&", "&amp;")
    for char, code in html_codes.items():
        string = string.replace(char, code)
    return string


def _get_key(
    key: str,
    scopes: Scopes,
    *,
    warn: bool,
    keep: bool,
    def_ldel: str,
    def_rdel: str,
) -> Any:
    """使用点分路径从当前作用域中检索值。

    使用点表示法遍历嵌套的字典和列表。

    支持特殊键 `'.'` 以返回当前作用域。

    Args:
        key: 点分路径（例如 `'user.name'` 或 `'.'` 表示当前作用域）。
        scopes: 要搜索的作用域字典列表。
        warn: 当键未找到时是否记录警告。
        keep: 当键未找到时是否返回原始模板标签。
        def_ldel: 模板的左分隔符（当 keep 为 `True` 时使用）。
        def_rdel: 模板的右分隔符（当 keep 为 `True` 时使用）

    Returns:
        在键路径处找到的值。

            如果未找到，当 keep 为 `True` 时返回原始模板标签，
            否则返回空字符串。
    """
        # 如果键是点
    if key == ".":
                # 则直接返回当前作用域
        return scopes[0]

        # 遍历作用域
    for scope in scopes:
        try:
                        # 如果为假值则返回空字符串，有两个例外
                        # 0 应返回 0，False 应返回 False
            if scope in (0, False):
                return scope

            resolved_scope: Literal[False, 0] | Mapping[str, Any] | Sequence[Any] = (
                scope
            )
                        # 对于每个由点分隔的键
            for child in key.split("."):
                                # 如果为假值则返回空字符串，有两个例外
                                # 0 应返回 0，False 应返回 False
                if resolved_scope in (0, False):
                    return resolved_scope
                                # 进入作用域
                if isinstance(resolved_scope, Mapping):
                    try:
                        resolved_scope = resolved_scope[child]
                    except (KeyError, TypeError):
                                                # 键未找到 - 将由外层 try-except 捕获
                        msg = f"Key {child!r} not found in dict"
                        raise KeyError(msg) from None
                elif isinstance(resolved_scope, (list, tuple)):
                    try:
                        resolved_scope = resolved_scope[int(child)]
                    except (ValueError, IndexError, TypeError):
                                                # 无效索引 - 将由外层 try-except 捕获
                        msg = f"Invalid index {child!r} for list/tuple"
                        raise IndexError(msg) from None
                else:
                                        # 拒绝所有其他内容以确保安全
                                        # 这可防止遍历任意 Python 对象
                    msg = (
                        f"Cannot traverse into {type(resolved_scope).__name__}. "
                        "Mustache templates only support dict, list, and tuple. "
                        f"Got: {type(resolved_scope)}"
                    )
                    raise TypeError(msg)  # noqa: TRY301

            try:
                                # 这允许自定义假值数据类型
                                # https://github.com/noahmorrison/chevron/issues/35
                if resolved_scope._CHEVRON_return_scope_when_falsy:  # type: ignore[union-attr] # noqa: SLF001
                    return resolved_scope
            except AttributeError:
                if resolved_scope in (0, False):
                    return resolved_scope
                return resolved_scope or ""
        except (AttributeError, KeyError, IndexError, ValueError, TypeError):
                        # 在当前作用域中找不到该键
                        # TypeError: 试图遍历非字典/列表类型
                        # 将在下一次迭代中再次尝试
            pass

        # 在任何作用域中都找不到该键

    if warn:
        logger.warning("Could not find key '%s'", key)

    if keep:
        return f"{def_ldel} {key} {def_rdel}"

    return ""


def _get_partial(name: str, partials_dict: Mapping[str, str]) -> str:
    """加载一个部分模板。

    Returns:
        部分模板。
    """
    try:
                # 也许该部分模板在字典中
        return partials_dict[name]
    except KeyError:
        return ""


#
# 主渲染函数
#
g_token_cache: dict[str, list[tuple[str, str]]] = {}

EMPTY_DICT: MappingProxyType[str, str] = MappingProxyType({})


def render(
    template: str | list[tuple[str, str]] = "",
    data: Mapping[str, Any] = EMPTY_DICT,
    partials_dict: Mapping[str, str] = EMPTY_DICT,
    padding: str = "",
    def_ldel: str = "{{",
    def_rdel: str = "}}",
    scopes: Scopes | None = None,
    warn: bool = False,  # noqa: FBT001,FBT002
    keep: bool = False,  # noqa: FBT001,FBT002
) -> str:
    """渲染一个 Mustache 模板。

    使用数据作用域和内联部分能力渲染 Mustache 模板。

    Args:
        template: A file-like object or a string containing the template.
        data: A python dictionary with your data scope.
        partials_dict: A python dictionary which will be search for partials
            before the filesystem is.

            `{'include': 'foo'}` 等同于名为 include.mustache 的文件
            （默认为 `{}`）。
        padding: 用于填充部分模板，不应使用
            （但如果你真的想用也可以）。
        def_ldel: 默认左分隔符

            （默认为 `'{{'`，符合 Mustache 规范）。
        def_rdel: 默认右分隔符

            （默认为 `'}}'`，符合 Mustache 规范）。
        scopes: `get_key` 将遍历的作用域列表。
        warn: 当模板替换在 data 中找不到时记录警告。
        keep: 当替换在 data 中找不到时保留未替换的标签。

    Returns:
        一个包含渲染后模板的字符串。
    """
        # 如果模板是序列但不是字符串的派生类
    if isinstance(template, Sequence) and not isinstance(template, str):
                # 则无需对其进行分词
                # 但它必须是一个生成器
        tokens: Iterator[tuple[str, str]] = (token for token in template)
    elif template in g_token_cache:
        tokens = (token for token in g_token_cache[template])
    else:
                # 否则创建一个生成器
        tokens = tokenize(template, def_ldel, def_rdel)

    output = ""

    if scopes is None:
        scopes = [data]

        # 遍历所有标记
    for tag, key in tokens:
                # 设置当前作用域
        current_scope = scopes[0]

                # 如果当前是结束标签
        if tag == "end":
                        # 弹出最新作用域
            del scopes[0]

                # 如果当前作用域为假值且不是唯一作用域
        elif not current_scope and len(scopes) != 1:
            if tag in {"section", "inverted section"}:
                                # 将最近的作用域设为假值
                scopes.insert(0, False)

                # 如果当前是字面量标签
        elif tag == "literal":
                        # 为键添加填充并加入输出
            output += key.replace("\n", "\n" + padding)

                # 如果当前是变量标签
        elif tag == "variable":
                        # 将 HTML 转义后的键加入输出
            thing = _get_key(
                key, scopes, warn=warn, keep=keep, def_ldel=def_ldel, def_rdel=def_rdel
            )
            if thing is True and key == ".":
                                # 如果意外被强制转换为布尔值
                                # (反向标签会这样做)
                                # 则获取未强制转换的对象（栈中的下一个）
                thing = scopes[1]
            if not isinstance(thing, str):
                thing = str(thing)
            output += _html_escape(thing)

                # 如果当前是不进行 HTML 转义的标签
        elif tag == "no escape":
                        # 只需查找键并加入
            thing = _get_key(
                key, scopes, warn=warn, keep=keep, def_ldel=def_ldel, def_rdel=def_rdel
            )
            if not isinstance(thing, str):
                thing = str(thing)
            output += thing

                # 如果当前是区块标签
        elif tag == "section":
                        # 获取区块的作用域
            scope = _get_key(
                key, scopes, warn=warn, keep=keep, def_ldel=def_ldel, def_rdel=def_rdel
            )

                        # 如果作用域是可调用对象（如
                        # https://mustache.github.io/mustache.5.html）中所述
            if callable(scope):
                                # 从标签生成模板文本
                text = ""
                tags: list[tuple[str, str]] = []
                for token in tokens:
                    if token == ("end", key):
                        break

                    tags.append(token)
                    tag_type, tag_key = token
                    if tag_type == "literal":
                        text += tag_key
                    elif tag_type == "no escape":
                        text += f"{def_ldel}& {tag_key} {def_rdel}"
                    else:
                        text += "{}{} {}{}".format(
                            def_ldel,
                            {
                                "comment": "!",
                                "section": "#",
                                "inverted section": "^",
                                "end": "/",
                                "partial": ">",
                                "set delimiter": "=",
                                "no escape": "&",
                                "variable": "",
                            }[tag_type],
                            tag_key,
                            def_rdel,
                        )

                g_token_cache[text] = tags

                rend = scope(
                    text,
                    lambda template, data=None: render(
                        template,
                        data={},
                        partials_dict=partials_dict,
                        padding=padding,
                        def_ldel=def_ldel,
                        def_rdel=def_rdel,
                        scopes=(data and [data, *scopes]) or scopes,
                        warn=warn,
                        keep=keep,
                    ),
                )

                output += rend

                        # 如果作用域是序列、迭代器或生成器但不是
                        # 字符串的派生类
            elif isinstance(scope, (Sequence, Iterator)) and not isinstance(scope, str):
                                # 则需要进行一些循环处理

                                # 收集区块内的所有标签
                                # (不要被具有相同键的嵌套结束标签所误导)
                                # TODO: 这似乎仍有边缘情况，对吧？
                tags = []
                tags_with_same_key = 0
                for token in tokens:
                    if token == ("section", key):
                        tags_with_same_key += 1
                    if token == ("end", key):
                        tags_with_same_key -= 1
                        if tags_with_same_key < 0:
                            break
                    tags.append(token)

                                # 对作用域中的每一项
                for thing in scope:
                                        # 将其追加为最新作用域并渲染
                    new_scope = [thing, *scopes]
                    rend = render(
                        template=tags,
                        scopes=new_scope,
                        padding=padding,
                        partials_dict=partials_dict,
                        def_ldel=def_ldel,
                        def_rdel=def_rdel,
                        warn=warn,
                        keep=keep,
                    )

                    output += rend

            else:
                                # 否则我们只是一个作用域区块
                scopes.insert(0, scope)

                # 如果当前是反向区块
        elif tag == "inverted section":
                        # 将翻转后的作用域加入作用域列表
            scope = _get_key(
                key, scopes, warn=warn, keep=keep, def_ldel=def_ldel, def_rdel=def_rdel
            )
            scopes.insert(0, cast("Literal[False]", not scope))

                # 如果当前是部分模板
        elif tag == "partial":
                        # 加载部分模板
            partial = _get_partial(key, partials_dict)

                        # 确定要为部分模板添加的填充内容
            left = output.rpartition("\n")[2]
            part_padding = padding
            if left.isspace():
                part_padding += left

                        # 渲染部分模板
            part_out = render(
                template=partial,
                partials_dict=partials_dict,
                def_ldel=def_ldel,
                def_rdel=def_rdel,
                padding=part_padding,
                scopes=scopes,
                warn=warn,
                keep=keep,
            )

                        # 如果部分模板有缩进
            if left.isspace():
                                # 则移除末尾的空格
                part_out = part_out.rstrip(" \t")

                        # 将部分模板的输出加入输出
            output += part_out

    return output
