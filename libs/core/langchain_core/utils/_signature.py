"""不评估注解的签名检查。"""

import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

if sys.version_info >= (3, 14):
    from annotationlib import Format


def _signature_parameters(obj: Callable[..., Any]) -> Mapping[str, inspect.Parameter]:
    """在不评估注解的情况下获取可调用对象的参数。

    On Python 3.14+, annotations are evaluated lazily, so `inspect.signature` raises
    `NameError` for annotations that reference names imported only under
    `TYPE_CHECKING`. Use this when only parameter names or kinds are needed.

    Args:
        obj: 要检查的可调用对象。

    Returns:
        可调用对象的参数，以名称为键。
    """
    if sys.version_info >= (3, 14):
        signature = inspect.signature(obj, annotation_format=Format.FORWARDREF)
    else:
        signature = inspect.signature(obj)
    return signature.parameters
