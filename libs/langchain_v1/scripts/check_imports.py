"""检查导入脚本。

快速验证一组 Python 文件能否被 Python
解释器加载，而不会引发任何错误。在运行更耗时的测试之前执行。适用于 Makefiles。

如果加载文件失败，该脚本会打印出有问题的文件名和详细的错误追踪信息。
"""

import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = "".join(
                random.choice(string.ascii_letters)  # noqa: S311
                for _ in range(20)
            )
            SourceFileLoader(module_name, file).load_module()
        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
