"""自动生成的模型配置文件。

请勿手动编辑此文件。
此文件由 langchain-profiles CLI 工具生成。

其中包含来自 models.dev 项目的数据。

Source: 来源: https://github.com/sst/models.dev
License: 许可证: MIT 许可证

如需更新这些数据，请参考此处的说明：

https://docs.langchain.com/oss/python/langchain/models#updating-or-overwriting-profile-data
"""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "sonar": {
        "name": "Sonar",
        "release_date": "2024-01-01",
        "last_updated": "2025-09-01",
        "open_weights": False,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "text_inputs": True,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": False,
        "tool_calling": False,
        "attachment": False,
        "temperature": True,
    },
    "sonar-deep-research": {
        "name": "Perplexity Sonar Deep Research",
        "release_date": "2025-02-01",
        "last_updated": "2025-09-01",
        "open_weights": False,
        "max_input_tokens": 128000,
        "max_output_tokens": 32768,
        "text_inputs": True,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": False,
        "attachment": False,
        "temperature": False,
    },
    "sonar-pro": {
        "name": "Sonar Pro",
        "release_date": "2024-01-01",
        "last_updated": "2025-09-01",
        "open_weights": False,
        "max_input_tokens": 200000,
        "max_output_tokens": 8192,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": False,
        "tool_calling": False,
        "attachment": True,
        "temperature": True,
    },
    "sonar-reasoning-pro": {
        "name": "Sonar Reasoning Pro",
        "release_date": "2024-01-01",
        "last_updated": "2025-09-01",
        "open_weights": False,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": False,
        "attachment": True,
        "temperature": True,
    },
}
