"""辅助类，用于将状态图绘制到 PNG 文件中。"""

from itertools import groupby
from typing import Any, cast

from langchain_core.runnables.graph import Graph, LabelsDict

try:
    import pygraphviz as pgv  # type: ignore[import-not-found]

    _HAS_PYGRAPHVIZ = True
except ImportError:
    _HAS_PYGRAPHVIZ = False


class PngDrawer:
    """辅助类，用于将状态图绘制到 PNG 文件中。

    需要安装 `graphviz` 和 `pygraphviz`。

    Example:
        ```python
        drawer = PngDrawer()
        drawer.draw(state_graph, "graph.png")
        ```
    """

    def __init__(
        self, fontname: str | None = None, labels: LabelsDict | None = None
    ) -> None:
        """初始化 PNG 绘制器。

        Args:
            fontname: The font to use for the labels. Defaults to "arial".
            labels: A dictionary of label overrides. The dictionary
                should have the following format:
                {
                    "nodes": {
                        "node1": "CustomLabel1",
                        "node2": "CustomLabel2",
                        "__end__": "End Node"
                    },
                    "edges": {
                        "continue": "ContinueLabel",
                        "end": "EndLabel"
                    }
                }
                The keys are the original labels, and the values are the new labels.

        """
        self.fontname = fontname or "arial"
        self.labels = labels or LabelsDict(nodes={}, edges={})

    def get_node_label(self, label: str) -> str:
        """返回节点使用的标签。

        Args:
            label: 原始标签。

        Returns:
            新标签。
        """
        label = self.labels.get("nodes", {}).get(label, label)
        return f"<<B>{label}</B>>"

    def get_edge_label(self, label: str) -> str:
        """返回边使用的标签。

        Args:
            label: 原始标签。

        Returns:
            新标签。
        """
        label = self.labels.get("edges", {}).get(label, label)
        return f"<<U>{label}</U>>"

    def add_node(self, viz: Any, node: str) -> None:
        """向图中添加节点。

        Args:
            viz: Graphviz 对象。
            node: 要添加的节点。
        """
        viz.add_node(
            node,
            label=self.get_node_label(node),
            style="filled",
            fillcolor="yellow",
            fontsize=15,
            fontname=self.fontname,
        )

    def add_edge(
        self,
        viz: Any,
        source: str,
        target: str,
        label: str | None = None,
        conditional: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """向图中添加边。

        Args:
            viz: Graphviz 对象。
            source: 源节点。
            target: 目标节点。
            label: 边的标签。
            conditional: 是否为条件边。
        """
        viz.add_edge(
            source,
            target,
            label=self.get_edge_label(label) if label else "",
            fontsize=12,
            fontname=self.fontname,
            style="dotted" if conditional else "solid",
        )

    def draw(self, graph: Graph, output_path: str | None = None) -> bytes | None:
        """将给定的状态图绘制到 PNG 文件中。

        需要安装 `graphviz` 和 `pygraphviz`。

        Args:
            graph: 要绘制的图
            output_path: 保存 PNG 的路径。如果为 `None`，则返回 PNG 字节。

        Raises:
            ImportError: 如果未安装 `pygraphviz`。

        Returns:
            如果 `output_path` 为 None，则返回 PNG 字节，否则返回 None。
        """
        if not _HAS_PYGRAPHVIZ:
            msg = "Install pygraphviz to draw graphs: `pip install pygraphviz`."
            raise ImportError(msg)

                # 创建有向图
        viz = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0)

                # 添加节点、条件边和边到图中
        self.add_nodes(viz, graph)
        self.add_edges(viz, graph)
        self.add_subgraph(viz, [node.split(":") for node in graph.nodes])

                # 更新入口点和 END 样式
        self.update_styles(viz, graph)

                # 将图保存为 PNG
        try:
            return cast("bytes | None", viz.draw(output_path, format="png", prog="dot"))
        finally:
            viz.close()

    def add_nodes(self, viz: Any, graph: Graph) -> None:
        """向图中添加节点。

        Args:
            viz: Graphviz 对象。
            graph: 要绘制的图。
        """
        for node in graph.nodes:
            self.add_node(viz, node)

    def add_subgraph(
        self,
        viz: Any,
        nodes: list[list[str]],
        parent_prefix: list[str] | None = None,
    ) -> None:
        """向图中添加子图。

        Args:
            viz: Graphviz 对象。
            nodes: 要添加的节点。
            parent_prefix: 父子图的前缀。
        """
        for prefix, grouped in groupby(
            [node[:] for node in sorted(nodes)],
            key=lambda x: x.pop(0),
        ):
            current_prefix = (parent_prefix or []) + [prefix]
            grouped_nodes = list(grouped)
            if len(grouped_nodes) > 1:
                subgraph = viz.add_subgraph(
                    [":".join(current_prefix + node) for node in grouped_nodes],
                    name="cluster_" + ":".join(current_prefix),
                )
                self.add_subgraph(subgraph, grouped_nodes, current_prefix)

    def add_edges(self, viz: Any, graph: Graph) -> None:
        """向图中添加边。

        Args:
            viz: Graphviz 对象。
            graph: 要绘制的图。
        """
        for start, end, data, cond in graph.edges:
            self.add_edge(
                viz, start, end, str(data) if data is not None else None, cond
            )

    @staticmethod
    def update_styles(viz: Any, graph: Graph) -> None:
        """更新入口点和 END 节点的样式。

        Args:
            viz: Graphviz 对象。
            graph: 要绘制的图。
        """
        if first := graph.first_node():
            viz.get_node(first.id).attr.update(fillcolor="lightblue")
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(fillcolor="orange")
