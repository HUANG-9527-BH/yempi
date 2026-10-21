"""此模块检查给定的 Python 文件是否可以无错误地导入。"""

import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            SourceFileLoader("x", file).load_module()
        except Exception:  # noqa: BLE001
            has_failure = True
            traceback.print_exc()

    sys.exit(1 if has_failure else 0)
