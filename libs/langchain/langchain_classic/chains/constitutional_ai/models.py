"""用于宪法式 AI 链的模型。"""

from pydantic import BaseModel


class ConstitutionalPrinciple(BaseModel):
    """宪法原则类。"""

    critique_request: str
    revision_request: str
    name: str = "Constitutional Principle"
