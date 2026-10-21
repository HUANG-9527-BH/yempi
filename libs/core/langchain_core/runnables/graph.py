"""在 `Runnable` 对象中使用的图。"""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypedDict,
    overload,
)
from uuid import UUID, uuid4

from langchain_core.load.serializable import to_json_not_implemented
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    _IgnoreUnserializable,
    is_basemodel_subclass,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.runnables.base import Runnable as RunnableType


class Stringifiable(Protocol):
    """可转换为字符串的对象的协议。"""

    def __str__(self) -> str:
        """将对象转换为字符串。"""


class LabelsDict(TypedDict):
    """图中节点和边的标签字典。"""

    nodes: dict[str, str]
    """节点的标签。"""
    edges: dict[str, str]
    """边的标签。"""


def is_uuid(value: str) -> bool:
    """检查字符串是否为有效的 UUID。

    Args:
        value: 要检查的字符串。

    Returns:
        如果字符串是有效的 UUID，则返回 `True`，否则返回 `False`。
    """
    try:
        UUID(value)
    except ValueError:
        return False
    return True


class Edge(NamedTuple):
    """图中的一条边。"""

    source: str
    """源节点 ID。"""
    target: str
    """目标节点 ID。"""
    data: Stringifiable | None = None
    """与边相关联的可选数据。"""
    conditional: bool = False
    """该边是否为条件边。"""

    def copy(self, *, source: str | None = None, target: str | None = None) -> Edge:
        """返回带有可选新源和目标节点的边副本。

        Args:
            source: 新的源节点 ID。
            target: 新的目标节点 ID。

        Returns:
            带有新源和目标节点的边副本。
        """
        return Edge(
            source=source or self.source,
            target=target or self.target,
            data=self.data,
            conditional=self.conditional,
        )


class Node(NamedTuple):
    """图中的一个节点。"""

    id: str
    """节点的唯一标识符。"""
    name: str
    """节点的名称。"""
    data: TypeBaseModel | RunnableType[Any, Any] | None
    """节点的数据。"""
    metadata: dict[str, Any] | None
    """节点的可选元数据。"""

    def copy(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> Node:
        """返回带有可选新 ID 和名称的节点副本。

        Args:
            id: 新的节点 ID。
            name: 新的节点名称。

        Returns:
            带有新 ID 和名称的节点副本。
        """
        return Node(
            id=id or self.id,
            name=name or self.name,
            data=self.data,
            metadata=self.metadata,
        )


class Branch(NamedTuple):
    """图中的分支。"""

    condition: Callable[..., str]
    """返回条件字符串表示形式的可调用对象。"""
    ends: dict[str, str] | None
    """分支的可选结束节点 ID 字典。"""


class CurveStyle(Enum):
    """Mermaid 支持的不同曲线样式枚举。"""

    BASIS = "basis"
    BUMP_X = "bumpX"
    BUMP_Y = "bumpY"
    CARDINAL = "cardinal"
    CATMULL_ROM = "catmullRom"
    LINEAR = "linear"
    MONOTONE_X = "monotoneX"
    MONOTONE_Y = "monotoneY"
    NATURAL = "natural"
    STEP = "step"
    STEP_AFTER = "stepAfter"
    STEP_BEFORE = "stepBefore"


@dataclass
class NodeStyles:
    """不同节点类型对应的十六进制颜色代码模式。

    Args:
        default: 默认颜色代码。
        first: 第一个节点的颜色代码。
        last: 最后一个节点的颜色代码。
    """

    default: str = "fill:#f2f0ff,line-height:1.2"
    first: str = "fill-opacity:0"
    last: str = "fill:#bfb6fc"


class MermaidDrawMethod(Enum):
    """Mermaid 支持的不同绘制方法枚举。"""

    PYPPETEER = "pyppeteer"
    """使用 Pyppeteer 渲染图。"""
    API = "api"
    """使用 Mermaid.INK API 渲染图。"""


def node_data_str(
    id: str,
    data: TypeBaseModel | RunnableType[Any, Any] | None,
) -> str:
    """将节点的数据转换为字符串。

    Args:
        id: 节点 ID。
        data: 节点数据。

    Returns:
        数据的字符串表示形式。
    """
    if not is_uuid(id) or data is None:
        return id
    data_str = data.get_name() if isinstance(data, Runnable) else data.__name__
    return data_str if not data_str.startswith("Runnable") else data_str[8:]


def node_data_json(
    node: Node, *, with_schemas: bool = False
) -> dict[str, str | dict[str, Any]]:
    """将节点的数据转换为 JSON 可序列化格式。

    Args:
        node: The `Node` to convert.
        with_schemas: Whether to include the schema of the data if it is a Pydantic
            model.

    Returns:
        包含数据类型和数据本身的字典。
    """
    if node.data is None:
        json: dict[str, Any] = {}
    elif isinstance(node.data, RunnableSerializable):
        json = {
            "type": "runnable",
            "data": {
                "id": node.data.lc_id(),
                "name": node_data_str(node.id, node.data),
            },
        }
    elif isinstance(node.data, Runnable):
        json = {
            "type": "runnable",
            "data": {
                "id": to_json_not_implemented(node.data)["id"],
                "name": node_data_str(node.id, node.data),
            },
        }
    elif inspect.isclass(node.data) and is_basemodel_subclass(node.data):
        json = (
            {
                "type": "schema",
                "data": node.data.model_json_schema(
                    schema_generator=_IgnoreUnserializable
                ),
            }
            if with_schemas
            else {
                "type": "schema",
                "data": node_data_str(node.id, node.data),
            }
        )
    else:
        json = {
            "type": "unknown",
            "data": node_data_str(node.id, node.data),
        }
    if node.metadata is not None:
        json["metadata"] = node.metadata
    return json


@dataclass
class Graph:
    """节点和边的图。

    Args:
        nodes: 图中的节点字典。默认为空字典。
        edges: 图中的边列表。默认为空列表。
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def to_json(self, *, with_schemas: bool = False) -> dict[str, list[dict[str, Any]]]:
        """将图转换为 JSON 可序列化格式。

        Args:
            with_schemas: Whether to include the schemas of the nodes if they are
                Pydantic models.

        Returns:
            包含图中节点和边的字典。
        """
        stable_node_ids = {
            node.id: i if is_uuid(node.id) else node.id
            for i, node in enumerate(self.nodes.values())
        }
        edges: list[dict[str, Any]] = []
        for edge in self.edges:
            edge_dict = {
                "source": stable_node_ids[edge.source],
                "target": stable_node_ids[edge.target],
            }
            if edge.data is not None:
                edge_dict["data"] = edge.data  # type: ignore[assignment]
            if edge.conditional:
                edge_dict["conditional"] = True
            edges.append(edge_dict)

        return {
            "nodes": [
                {
                    "id": stable_node_ids[node.id],
                    **node_data_json(node, with_schemas=with_schemas),
                }
                for node in self.nodes.values()
            ],
            "edges": edges,
        }

    def __bool__(self) -> bool:
        """返回图是否包含任何节点。"""
        return bool(self.nodes)

    def next_id(self) -> str:
        """返回一个新的唯一节点标识符。

        可用于将节点添加到图中。
        """
        return uuid4().hex

    def add_node(
        self,
        data: TypeBaseModel | RunnableType[Any, Any] | None,
        id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        """向图中添加节点并返回它。

        Args:
            data: 节点的数据。
            id: 节点的标识符。
            metadata: 节点的可选元数据。

        Returns:
            已添加到图中的节点。

        Raises:
            ValueError: 如果已存在相同 id 的节点。
        """
        if id is not None and id in self.nodes:
            msg = f"Node with id {id} already exists"
            raise ValueError(msg)
        id_ = id or self.next_id()
        node = Node(id=id_, data=data, metadata=metadata, name=node_data_str(id_, data))
        self.nodes[node.id] = node
        return node

    def remove_node(self, node: Node) -> None:
        """从图中移除一个节点及其所有连接的边。

        Args:
            node: 要移除的节点。
        """
        self.nodes.pop(node.id)
        self.edges = [
            edge for edge in self.edges if node.id not in {edge.source, edge.target}
        ]

    def add_edge(
        self,
        source: Node,
        target: Node,
        data: Stringifiable | None = None,
        conditional: bool = False,  # noqa: FBT001,FBT002
    ) -> Edge:
        """向图中添加一条边并返回它。

        Args:
            source: 边的源节点。
            target: 边的目标节点。
            data: 与边关联的可选数据。
            conditional: 是否为条件边。

        Returns:
            已添加到图中的边。

        Raises:
            ValueError: 如果源节点或目标节点不在图中。
        """
        if source.id not in self.nodes:
            msg = f"Source node {source.id} not in graph"
            raise ValueError(msg)
        if target.id not in self.nodes:
            msg = f"Target node {target.id} not in graph"
            raise ValueError(msg)
        edge = Edge(
            source=source.id, target=target.id, data=data, conditional=conditional
        )
        self.edges.append(edge)
        return edge

    def extend(
        self, graph: Graph, *, prefix: str = ""
    ) -> tuple[Node | None, Node | None]:
        """将另一个图的所有节点和边添加到当前图中。

        注意此操作不会检查重复项，也不会连接两个图。

        Args:
            graph: 要添加的图。
            prefix: 添加到节点 id 上的前缀。

        Returns:
            子图的第一个和最后一个节点组成的元组。
        """
        if all(is_uuid(node.id) for node in graph.nodes.values()):
            prefix = ""

        def prefixed(id_: str) -> str:
            return f"{prefix}:{id_}" if prefix else id_

                # prefix each node
        self.nodes.update(
            {prefixed(k): v.copy(id=prefixed(k)) for k, v in graph.nodes.items()}
        )
                # prefix each edge's source and target
        self.edges.extend(
            [
                edge.copy(source=prefixed(edge.source), target=prefixed(edge.target))
                for edge in graph.edges
            ]
        )
                # return (prefixed) first and last nodes of the subgraph
        first, last = graph.first_node(), graph.last_node()
        return (
            first.copy(id=prefixed(first.id)) if first else None,
            last.copy(id=prefixed(last.id)) if last else None,
        )

    def reid(self) -> Graph:
        """返回一个新图，其中所有节点都重新识别。

        尽可能使用其唯一且可读的名称。
        """
        node_name_to_ids = defaultdict(list)
        for node in self.nodes.values():
            node_name_to_ids[node.name].append(node.id)

        unique_labels = {
            node_id: node_name if len(node_ids) == 1 else f"{node_name}_{i + 1}"
            for node_name, node_ids in node_name_to_ids.items()
            for i, node_id in enumerate(node_ids)
        }

        def _get_node_id(node_id: str) -> str:
            label = unique_labels[node_id]
            if is_uuid(node_id):
                return label
            return node_id

        return Graph(
            nodes={
                _get_node_id(id_): node.copy(id=_get_node_id(id_))
                for id_, node in self.nodes.items()
            },
            edges=[
                edge.copy(
                    source=_get_node_id(edge.source),
                    target=_get_node_id(edge.target),
                )
                for edge in self.edges
            ],
        )

    def first_node(self) -> Node | None:
        """查找唯一不作为任何边目标的节点。

        如果不存在这样的节点，或存在多个，则返回 `None`。
        绘制图时，该节点将是起点。

        Returns:
            The first node, or None if there is no such node or multiple
            candidates.
        """
        return _first_node(self)

    def last_node(self) -> Node | None:
        """查找唯一不作为任何边源的节点。

        如果不存在这样的节点，或存在多个，则返回 `None`。
        绘制图时，该节点将是终点。

        Returns:
            The last node, or None if there is no such node or multiple
            candidates.
        """
        return _last_node(self)

    def trim_first_node(self) -> None:
        """如果第一个节点存在且仅有一条出边，则将其移除。

        即：移除它不会导致图失去“第一个”节点。
        """
        first_node = self.first_node()
        if (
            first_node
            and _first_node(self, exclude=[first_node.id])
            and len({e for e in self.edges if e.source == first_node.id}) == 1
        ):
            self.remove_node(first_node)

    def trim_last_node(self) -> None:
        """如果最后一个节点存在且仅有一条入边，则将其移除。

        即：移除它不会导致图失去“最后一个”节点。
        """
        last_node = self.last_node()
        if (
            last_node
            and _last_node(self, exclude=[last_node.id])
            and len({e for e in self.edges if e.target == last_node.id}) == 1
        ):
            self.remove_node(last_node)

    def draw_ascii(self) -> str:
        """以 ASCII 艺术字符串形式绘制图。

        Returns:
            ASCII 艺术字符串。
        """
                # Import locally to prevent circular import
        from langchain_core.runnables.graph_ascii import draw_ascii  # noqa: PLC0415

        return draw_ascii(
            {node.id: node.name for node in self.nodes.values()},
            self.edges,
        )

    def print_ascii(self) -> None:
        """以 ASCII 艺术字符串形式打印图。"""
        print(self.draw_ascii())  # noqa: T201

    @overload
    def draw_png(
        self,
        output_file_path: str,
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> None: ...

    @overload
    def draw_png(
        self,
        output_file_path: None,
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> bytes: ...

    def draw_png(
        self,
        output_file_path: str | None = None,
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> bytes | None:
        """以 PNG 图像形式绘制图。

        Args:
            output_file_path: The path to save the image to. If `None`, the image
                is not saved.
            fontname: The name of the font to use.
            labels: Optional labels for nodes and edges in the graph. Defaults to
                `None`.

        Returns:
            如果 output_file_path 为 None，则返回 PNG 图像的字节，否则返回 None。
        """
                # Import locally to prevent circular import
        from langchain_core.runnables.graph_png import PngDrawer  # noqa: PLC0415

        default_node_labels = {node.id: node.name for node in self.nodes.values()}

        return PngDrawer(
            fontname,
            LabelsDict(
                nodes={
                    **default_node_labels,
                    **(labels["nodes"] if labels is not None else {}),
                },
                edges=labels["edges"] if labels is not None else {},
            ),
        ).draw(self, output_file_path)

    def draw_mermaid(
        self,
        *,
        with_styles: bool = True,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeStyles | None = None,
        wrap_label_n_words: int = 9,
        frontmatter_config: dict[str, Any] | None = None,
    ) -> str:
        """以 Mermaid 语法字符串形式绘制图。

        Args:
            with_styles: 是否包含样式在语法中。 curve_style: 边的样式。 node_colors:
            节点的颜色。 wrap_label_n_words: 节点标签换行的词数。 frontmatter_config:
            Mermaid 前置配置。 可用于自定义主题和样式。将转换为 YAML 并添加到 Mermaid
            图的开头。

                See more here: 了解更多：https://mermaid.js.org/config/configuration.html。

                示例配置：

                ```python
                {
                    "config": {
                        "theme": "neutral",
                        "look": "handDrawn",
                        "themeVariables": {"primaryColor": "#e2e2e2"},
                    }
                }
                ```
        Returns:
            Mermaid 语法字符串。
        """
                # Import locally to prevent circular import
        from langchain_core.runnables.graph_mermaid import draw_mermaid  # noqa: PLC0415

        graph = self.reid()
        first_node = graph.first_node()
        last_node = graph.last_node()

        return draw_mermaid(
            nodes=graph.nodes,
            edges=graph.edges,
            first_node=first_node.id if first_node else None,
            last_node=last_node.id if last_node else None,
            with_styles=with_styles,
            curve_style=curve_style,
            node_styles=node_colors,
            wrap_label_n_words=wrap_label_n_words,
            frontmatter_config=frontmatter_config,
        )

    def draw_mermaid_png(
        self,
        *,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeStyles | None = None,
        wrap_label_n_words: int = 9,
        output_file_path: str | None = None,
        draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
        background_color: str = "white",
        padding: int = 10,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        frontmatter_config: dict[str, Any] | None = None,
        base_url: str | None = None,
        proxies: dict[str, str] | None = None,
    ) -> bytes:
        """使用 Mermaid 绘制图形为 PNG 图像。

        Args:
            curve_style: 边的样式。 node_colors: 节点的颜色。 wrap_label_n_words:
            节点标签换行的词数。 output_file_path: 保存图像的路径。如果为
            `None`，则不保存图像。 draw_method: 用于绘制图形的方法。 background_color:
            背景色。 padding: 图形周围的填充。 max_retries:
            最大重试次数（`MermaidDrawMethod.API`）。 retry_delay:
            重试之间的延迟（`MermaidDrawMethod.API`）。 frontmatter_config: Mermaid
            前置元数据配置。 可用于自定义主题和样式。将转换为 YAML 并添加到 Mermaid
            图表开头。

                See more here: 了解更多：https://mermaid.js.org/config/configuration.html。

                示例配置：

                ```python
                {
                    "config": {
                        "theme": "neutral",
                        "look": "handDrawn",
                        "themeVariables": {"primaryColor": "#e2e2e2"},
                    }
                }
                ```
            base_url: 用于通过 API 渲染的 Mermaid 服务器基础 URL。
            proxies: 请求的 HTTP/HTTPS 代理（例如 `{"http": "http://127.0.0.1:7890"}`）。

        Returns:
            PNG 图像的字节数据。
        """
                # Import locally to prevent circular import
        from langchain_core.runnables.graph_mermaid import (  # noqa: PLC0415
            draw_mermaid_png,
        )

        mermaid_syntax = self.draw_mermaid(
            curve_style=curve_style,
            node_colors=node_colors,
            wrap_label_n_words=wrap_label_n_words,
            frontmatter_config=frontmatter_config,
        )
        return draw_mermaid_png(
            mermaid_syntax=mermaid_syntax,
            output_file_path=output_file_path,
            draw_method=draw_method,
            background_color=background_color,
            padding=padding,
            max_retries=max_retries,
            retry_delay=retry_delay,
            proxies=proxies,
            base_url=base_url,
        )


def _first_node(graph: Graph, exclude: Sequence[str] = ()) -> Node | None:
    """查找不作为任何边的目标的单个节点。

    排除在排除列表中的节点/源节点。

    如果不存在这样的节点，或存在多个，返回 `None`。

    在绘制图时，该节点将是起点。
    """
    targets = {edge.target for edge in graph.edges if edge.source not in exclude}
    found: list[Node] = [
        node
        for node in graph.nodes.values()
        if node.id not in exclude and node.id not in targets
    ]
    return found[0] if len(found) == 1 else None


def _last_node(graph: Graph, exclude: Sequence[str] = ()) -> Node | None:
    """查找不作为任何边的源的单个节点。

    排除在排除列表中的节点/目标节点。

    如果不存在这样的节点，或存在多个，返回 `None`。

    在绘制图时，该节点将是终点。
    """
    sources = {edge.source for edge in graph.edges if edge.target not in exclude}
    found: list[Node] = [
        node
        for node in graph.nodes.values()
        if node.id not in exclude and node.id not in sources
    ]
    return found[0] if len(found) == 1 else None
