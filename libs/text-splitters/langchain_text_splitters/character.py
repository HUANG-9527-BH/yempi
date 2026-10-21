"""字符文本分割器。"""

from __future__ import annotations

import re
from typing import Any, Literal

from typing_extensions import override

from langchain_text_splitters.base import Language, TextSplitter


class CharacterTextSplitter(TextSplitter):
    """按字符进行文本分割。"""

    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """创建一个新的 TextSplitter。"""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    @override
    def split_text(self, text: str) -> list[str]:
        """在不重新插入前瞻分隔符的情况下分割为块。

        Args:
            text: 需要分割的文本。

        Returns:
            文本块的列表。
        """
                # 1. 确定分割模式：原始正则表达式或转义字面量
        sep_pattern = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )

                # 2. 初始分割（如果需要则保留分隔符）
        splits = _split_text_with_regex(
            text, sep_pattern, keep_separator=self._keep_separator
        )

                # 3. 检测零宽度前瞻，确保从不重新插入
        lookaround_prefixes = ("(?=", "(?<!", "(?<=", "(?!")
        is_lookaround = self._is_separator_regex and any(
            self._separator.startswith(p) for p in lookaround_prefixes
        )

                # 4. 决定合并分隔符：
                #    - 如果 keep_separator 或 lookaround -> 不重新插入
                #    - 否则 -> 重新插入字面量分隔符
        merge_sep = ""
        if not (self._keep_separator or is_lookaround):
            merge_sep = self._separator

                # 5. 合并相邻的分割结果并返回
        return self._merge_splits(splits, merge_sep)


def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: bool | Literal["start", "end"]
) -> list[str]:
        # 现在有了分隔符，开始分割文本
    if separator:
        if keep_separator:
                        # 模式中的括号会将分隔符保留在结果中。
            splits_ = re.split(f"({separator})", text)
            splits = (
                ([splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)])
                if keep_separator == "end"
                else ([splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)])
            )
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = (
                ([*splits, splits_[-1]])
                if keep_separator == "end"
                else ([splits_[0], *splits])
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class RecursiveCharacterTextSplitter(TextSplitter):
    """通过递归检查字符来分割文本。

    递归尝试使用不同的字符进行分割，以找到合适的分割方式。
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,  # noqa: FBT001,FBT002
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """创建一个新的 TextSplitter。"""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """分割输入文本并返回块。"""
        final_chunks = []
                # 获取适当的分隔符使用
        separator = separators[-1]
        new_separators = []
        for i, s_ in enumerate(separators):
            separator_ = s_ if self._is_separator_regex else re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, separator_, keep_separator=self._keep_separator
        )

                # 现在开始合并内容，递归分割较长的文本。
        good_splits = []
        separator_ = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator_)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator_)
            final_chunks.extend(merged_text)
        return final_chunks

    @override
    def split_text(self, text: str) -> list[str]:
        """根据预定义的分隔符将输入文本分割为较小的块。

        Args:
            text: 需要分割的输入文本。

        Returns:
            分割后获得的文本块列表。
        """
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
        cls, language: Language, **kwargs: Any
    ) -> RecursiveCharacterTextSplitter:
        """根据特定语言返回此类的实例。

        此方法使用语言特定的分隔符初始化文本分割器。

        Args:
            language: 用于配置文本分割器的语言。
            **kwargs: 用于自定义分割器的其他关键字参数。

        Returns:
            针对指定语言配置的文本分割器实例。
        """
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> list[str]:
        """获取与给定语言相关的分隔符列表。

        Args:
            language: 获取分隔符的语言。

        Returns:
            适用于指定语言的分隔符列表。

        Raises:
            ValueError: 如果该语言未实现或不受支持。
        """
        if language in {Language.C, Language.CPP}:
            return [
                                # 沿类定义分割
                "\nclass ",
                                # 沿函数定义分割
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.GO:
            return [
                                # 沿函数定义分割
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.JAVA:
            return [
                                # 沿类定义分割
                "\nclass ",
                                # 沿方法定义分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.KOTLIN:
            return [
                                # 沿类定义分割
                "\nclass ",
                                # 沿方法定义分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\nelse ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.JS:
            return [
                                # 沿函数定义分割
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                                # 沿类定义分割
                "\nclass ",
                                # 沿函数定义分割
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                                # 沿控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PHP:
            return [
                                # 沿函数定义分割
                "\nfunction ",
                                # 沿类定义分割
                "\nclass ",
                                # 沿控制流语句分割
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PROTO:
            return [
                                # 沿消息定义分割
                "\nmessage ",
                                # 沿服务定义分割
                "\nservice ",
                                # 沿枚举定义分割
                "\nenum ",
                                # 沿选项定义分割
                "\noption ",
                                # 沿导入语句分割
                "\nimport ",
                                # 沿语法声明分割
                "\nsyntax ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PYTHON:
            return [
                                # 首先尝试沿类定义分割
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                                # 现在按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.R:
            return [
                                # 沿函数定义分割
                "\nfunction ",
                                # 沿 S4 类和方法定义分割
                "\nsetClass\\(",
                "\nsetMethod\\(",
                "\nsetGeneric\\(",
                                # 沿控制流语句分割
                "\nif ",
                "\nelse ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                                # 按包加载分割
                "\nlibrary\\(",
                "\nrequire\\(",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RST:
            return [
                                # 按章节标题分割
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                                # 按指令标记分割
                "\n\n.. *\n\n",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RUBY:
            return [
                                # 按方法定义分割
                "\ndef ",
                "\nclass ",
                                # 按控制流语句分割
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.ELIXIR:
            return [
                                # 按函数和模块定义分割
                "\ndef ",
                "\ndefp ",
                "\ndefmodule ",
                "\ndefprotocol ",
                "\ndefmacro ",
                "\ndefmacrop ",
                                # 按控制流语句分割
                "\nif ",
                "\nunless ",
                "\ncase ",
                "\ncond ",
                "\nwith ",
                "\nfor ",
                "\ndo ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RUST:
            return [
                                # 按函数定义分割
                "\nfn ",
                "\nconst ",
                "\nlet ",
                                # 按控制流语句分割
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SCALA:
            return [
                                # 按类定义分割
                "\nclass ",
                "\nobject ",
                                # 按方法定义分割
                "\ndef ",
                "\nval ",
                "\nvar ",
                                # 按控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SWIFT:
            return [
                                # 按函数定义分割
                "\nfunc ",
                                # 按类定义分割
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                                # 按控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.MARKDOWN:
            return [
                                # 首先尝试按 Markdown 标题分割（从二级开始）
                "\n#{1,6} ",
                                # 注意，标题的替代语法（如下所示）此处未处理
                                # 二级标题
                # ---------------
                                # 代码块结束
                "```\n",
                                # 水平线
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                                # 注意，此分割器不处理由 *三个或更多* 的 ***, ---, 或 ___ 定义的水平线
                                # 但此情况未被处理
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.LATEX:
            return [
                                # 首先尝试按 Latex 章节分割
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                                # 现在按环境分割
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                                # 现在按数学环境分割
                "\n\\\\begin{align}",
                "$$",
                "$",
                                # 现在按常规行类型分割
                " ",
                "",
            ]
        if language == Language.HTML:
            return [
                                # 首先尝试按 HTML 标签分割
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        if language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\ndelegate ",
                "\nevent ",
                                # 按类定义分割
                "\nclass ",
                "\nabstract ",
                                # 按方法定义分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                                # 按控制流语句分割
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                                # 按异常分割
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SOL:
            return [
                                # 按编译器信息定义分割
                "\npragma ",
                "\nusing ",
                                # 按契约定义分割
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                                # 按方法定义分割
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                                # 按控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.COBOL:
            return [
                                # 按分部分割
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                                # 按 DATA DIVISION 内的章节分割
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                                # 按 PROCEDURE DIVISION 内的章节分割
                "\nINPUT-OUTPUT SECTION.",
                                # 按段落和常见语句分割
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                                # 按常规行类型分割
                "\n",
                " ",
                "",
            ]
        if language == Language.LUA:
            return [
                                # 按变量和表定义分割
                "\nlocal ",
                                # 按函数定义分割
                "\nfunction ",
                                # 按控制流语句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.HASKELL:
            return [
                                # 按函数定义分割
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                                # 按类型声明分割
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                                # 按模块声明分割
                "\nmodule ",
                                # 按导入语句分割
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                                # 按类型类声明分割
                "\nclass ",
                "\ninstance ",
                                # 按 case 表达式分割
                "\ncase ",
                                # 按函数定义中的 guards 分割
                "\n| ",
                                # 按记录字段声明分割
                "\n= {",
                "\n, ",
                                # 按常规行类型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.POWERSHELL:
            return [
                                # 按函数定义分割
                "\nfunction ",
                                # 按参数声明分割（转义括号）
                "\nparam ",
                                # 按控制流语句分割
                "\nif ",
                "\nforeach ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                                # 按类定义分割（适用于 PowerShell 5.0 及以上版本）
                "\nclass ",
                                # 按 try-catch-finally 块分割
                "\ntry ",
                "\ncatch ",
                "\nfinally ",
                                # 按正常行和空格分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.VISUALBASIC6:
            vis = r"(?:Public|Private|Friend|Global|Static)\s+"
            return [
                                # 按定义分割
                rf"\n(?!End\s){vis}?Sub\s+",
                rf"\n(?!End\s){vis}?Function\s+",
                rf"\n(?!End\s){vis}?Property\s+(?:Get|Let|Set)\s+",
                rf"\n(?!End\s){vis}?Type\s+",
                rf"\n(?!End\s){vis}?Enum\s+",
                                # 按控制流语句分割
                r"\n(?!End\s)If\s+",
                r"\nElseIf\s+",
                r"\nElse\s+",
                r"\nSelect\s+Case\s+",
                r"\nCase\s+",
                r"\nFor\s+",
                r"\nDo\s+",
                r"\nWhile\s+",
                r"\nWith\s+",
                                # 按常规行类型分割
                r"\n\n",
                r"\n",
                " ",
                "",
            ]

        if language in Language._value2member_map_:
            msg = f"Language {language} is not implemented yet!"
            raise ValueError(msg)
        msg = (
            f"Language {language} is not supported! Please choose from {list(Language)}"
        )
        raise ValueError(msg)
