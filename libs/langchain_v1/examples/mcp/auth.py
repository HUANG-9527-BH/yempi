"""自定义认证，使 `langgraph dev` 知道某次运行属于哪个用户。

The handler maps the `x-user-id` header to a user identity. That identity is
what `graph_factory.make_graph` reads off `runtime.user` and mints an MCP token
for, so each run talks to the fleet as its own user. A real deployment
validates a credential against an identity provider and returns the user it
resolves to; here the header is trusted directly, which is enough to
demonstrate per-user routing. (`x-api-key` is reserved by the LangGraph SDK for
its own auth, so the demo uses a different header.)

通过 `langgraph.json` 配置接入。
"""

from __future__ import annotations

from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    """根据 `x-user-id` 解析调用方身份，对无此头部的请求拒绝处理。"""
    user_id = headers.get(b"x-user-id")
    if not user_id:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing x-user-id")
    identity = user_id.decode() if isinstance(user_id, bytes) else str(user_id)
    return {"identity": identity}
