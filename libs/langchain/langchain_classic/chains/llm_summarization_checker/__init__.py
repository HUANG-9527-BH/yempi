"""用于验证文本生成准确性的摘要检查链。

Chain that tries to verify the accuracy of text generation by splitting it into a
list of facts, then checking if those facts are true or not, and rewriting
the text to make it more truthful. It will repeat this loop until it hits `max_tries` or
gets to a "true" output.
"""
