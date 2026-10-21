"""为 asyncio 功能提供的兼容层，这些功能在较旧的 Python 版本中可能缺失。"""

import sys

if sys.version_info[:2] < (3, 11):
    from async_timeout import timeout as asyncio_timeout
else:
    from asyncio import timeout as asyncio_timeout


__all__ = ["asyncio_timeout"]
