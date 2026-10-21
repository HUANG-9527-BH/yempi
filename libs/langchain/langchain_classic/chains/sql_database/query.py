from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_classic.chains.sql_database.prompt import PROMPT, SQL_PROMPTS

if TYPE_CHECKING:
    from langchain_community.utilities.sql_database import SQLDatabase


def _strip(text: str) -> str:
    return text.strip()


class SQLInput(TypedDict):
    """SQL 链的输入。"""

    question: str


class SQLInputWithTables(TypedDict):
    """SQL 链的输入。"""

    question: str
    table_names_to_use: list[str]


def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: BasePromptTemplate | None = None,
    k: int = 5,
    *,
    get_col_comments: bool | None = None,
) -> Runnable[SQLInput | SQLInputWithTables | dict[str, Any], str]:
    r"""创建一个生成 SQL 查询的链。

    *安全提示*：此链为给定数据库生成 SQL 查询。

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        为降低泄露敏感数据的风险，请限制权限为只读，并将作用范围限定在所需表上。

        可选地，使用 SQLInputWithTables 输入类型来指定允许访问的表。

        控制谁可以向此链提交请求。

        详情参见 https://docs.langchain.com/oss/python/security-policy。

    Args:
        llm: The language model to use.
        db: The SQLDatabase to generate the query for.
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect.  See Prompt section below for more.
        k: The number of results per select statement to return.
        get_col_comments: Whether to retrieve column comments along with table info.

    Returns:
        一个接收问题并生成回答该问题的 SQL 查询的链。

    Example:
        ```python
        # pip install -U langchain langchain-community langchain-openai
        from langchain_openai import ChatOpenAI
        from langchain_classic.chains import create_sql_query_chain
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        model = ChatOpenAI(model="gpt-5.5", temperature=0)
        chain = create_sql_query_chain(model, db)
        response = chain.invoke({"question": "How many employees are there"})
        ```

    提示词: 若未提供提示词，则根据 SQLDatabase
    的方言选择默认提示词。若提供了自定义提示词，必须支持以下输入变量：

            * input: 用户问题加上后缀 "\\nSQLQuery: " 会传入此处。 * top_k: 每个 select
            语句返回的结果数量（即此函数的 `k` 参数）会传入此处。 * table_info:
            表定义和样本行数据会传入此处。若用户在调用链时指定了
            "table_names_to_use"，则仅包含这些表；否则包含所有表。 *
            dialect（可选）：若提示词中包含 dialect 输入变量，数据库方言将传入此处。

        以下是示例提示词：

        ```python
        from langchain_core.prompts import PromptTemplate

        template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:

        {table_info}.

        Question: {input}'''
        prompt = PromptTemplate.from_template(template)
        ```
    """  # noqa: E501
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables),
    ):
        msg = (
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
        raise ValueError(msg)
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    table_info_kwargs = {}
    if get_col_comments:
        if db.dialect not in ("postgresql", "mysql", "oracle"):
            msg = (
                f"get_col_comments=True is only supported for dialects "
                f"'postgresql', 'mysql', and 'oracle'. Received dialect: "
                f"{db.dialect}"
            )
            raise ValueError(msg)
        table_info_kwargs["get_col_comments"] = True

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use"),
            **table_info_kwargs,
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore[return-value]
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )
