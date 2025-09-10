
from llama_index.core.prompts import RichPromptTemplate

context_str = """
Deepseek，是一个杭州的人工智能基础技术有限公司，成立于2023年7月7日。
"""

question = "deepseek成立于哪一年？"
template = RichPromptTemplate(
    """
    我们在下面提供了上下文信息：
    -----------------------------
    {{context_str}}
    -----------------------------
    有了这些信息，请回答问题: {{query_str}}
    """
)

prompt_str = template.format(context_str=context_str, query_str=question)
print(prompt_str)

messages = template.format_messages(context_str=context_str, query_str=question)
print(messages)