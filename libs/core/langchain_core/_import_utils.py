from importlib import import_module


def import_attr(
    attr_name: str,
    module_name: str | None,
    package: str | None,
) -> object:
    """从位于包中的模块导入属性。

    This utility function is used in custom `__getattr__` methods within `__init__.py`
    files to dynamically import attributes.

    Args:
        attr_name: 要导入的属性名称。
        module_name: 要从中导入的模块名称。

            如果为 `None`，则从包本身导入该属性。
        package: 包的名称，模块位于其中。

    Raises:
        ImportError: 如果找不到模块。
        AttributeError: 如果属性在模块或包中不存在。

    Returns:
        导入的属性。
    """
    if module_name == "__module__" or module_name is None:
        try:
            result = import_module(f".{attr_name}", package=package)
        except ModuleNotFoundError:
            msg = f"module '{package!r}' has no attribute {attr_name!r}"
            raise AttributeError(msg) from None
    else:
        try:
            module = import_module(f".{module_name}", package=package)
        except ModuleNotFoundError as err:
            msg = f"module '{package!r}.{module_name!r}' not found ({err})"
            raise ImportError(msg) from None
        result = getattr(module, attr_name)
    return result
