from llama_index.core.prompts import RichPromptTemplate

qa_prompt_str = """\
上下文信息如下：
--------------------
{{ context_str }}
--------------------
根据给定的上下文信息而不是先前的知识，回答查询
请以 {{ tone_name }}的风格给出答案，
查询：{{ query_str }}
答案：
"""

prompt_templ = RichPromptTemplate(qa_prompt_str)

partial_prompt_tmpl = prompt_templ.partial_format(tone_name="莎士比亚")

fmt_prompt = partial_prompt_tmpl.format(
    context_str = "在这项工作中，我们开发了adadsadasdasdadada",
    query_str = "adasdadsadadsdad 是什么？"
)
print(fmt_prompt)
fmt_prompt = partial_prompt_tmpl.format_messages(
    context_str="在这项工作中，我们开发了adadsadasdasdadada",
    query_str = "adasdadsadadsdad 是什么？"
)
print(fmt_prompt)
