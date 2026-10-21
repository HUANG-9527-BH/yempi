"""SSRF 保护例外情况。"""


class SSRFBlockedError(Exception):
    """当请求被 SSRF 保护策略阻止时抛出。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"SSRF blocked: {reason}")
