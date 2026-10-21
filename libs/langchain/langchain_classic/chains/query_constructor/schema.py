from pydantic import BaseModel, ConfigDict


class AttributeInfo(BaseModel):
    """关于数据源属性的信息。"""

    name: str
    description: str
    type: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
