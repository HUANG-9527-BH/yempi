"""自动生成的模型配置文件。

请勿手动编辑此文件。
此文件由 langchain-profiles CLI 工具生成。

其中包含来自 models.dev 项目的衍生数据。

Source: 来源: https://github.com/sst/models.dev
License: 许可证: MIT 许可证

如需更新这些数据，请参考此处的说明：

https://docs.langchain.com/oss/python/langchain/models#updating-or-overwriting-profile-data
"""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek-flash": {
        "name": "DeepSeek V4.1 Flash 1017",
        "release_date": "2026-10-17",
        "last_updated": "2026-10-17",
        "open_weights": True,
        "max_input_tokens": 1000000,
        "max_output_tokens": 393216,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "attachment": True,
        "temperature": True,
        "tool_call_streaming": True,
    },
    "deepseek-v4-flash": {
        "name": "DeepSeek V4 Flash",
        "status": "deprecated",
        "release_date": "2026-10-17",
        "last_updated": "2026-10-17",
        "open_weights": True,
        "max_input_tokens": 1000000,
        "max_output_tokens": 393216,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "attachment": True,
        "temperature": True,
        "tool_call_streaming": True,
    },
    "deepseek-v4-flash-vision-exp": {
        "name": "DeepSeek V4 Flash Vision Exp",
        "status": "deprecated",
        "release_date": "2026-10-17",
        "last_updated": "2026-10-17",
        "open_weights": True,
        "max_input_tokens": 1000000,
        "max_output_tokens": 393216,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "attachment": True,
        "temperature": True,
        "tool_call_streaming": True,
    },
    "deepseek-v4-pro": {
        "name": "DeepSeek V4 Pro",
        "release_date": "2026-08-12",
        "last_updated": "2026-08-22",
        "open_weights": True,
        "max_input_tokens": 1000000,
        "max_output_tokens": 393216,
        "text_inputs": True,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "attachment": False,
        "temperature": True,
        "tool_call_streaming": True,
    },
}
