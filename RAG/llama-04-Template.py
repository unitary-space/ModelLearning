from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
    {% chat role = "system"%}
    给定一个列表，包含文本，请你尽你所能回答这个问题：
    {% endchat %}
    
    {% chat role = "user"%}
    {% for text in texts %}
    
    这是一些文本：{{text}}
    {% endfor %}
    {% endchat %}
    """
)

messages = template.format_messages(
    texts = ["阿珍喜欢阿强",
             "阿强和阿亮是夫妻关系",
             "阿亮不喜欢阿强，但是没有离婚",
             "阿强不喜欢阿珍"
    ]
)
print(messages)