"""通过 stdio 运行其中一个共享服务器，由 argv 选择。

`MCPAdapter` 以此方式作为子进程启动；它不属于所演示的 API 的一部分。
"""

import sys

from _servers import run_calculator_stdio, run_weather_stdio

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "weather"
    (run_calculator_stdio if which == "calculator" else run_weather_stdio)()
