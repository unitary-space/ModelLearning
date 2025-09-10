from dotenv import load_dotenv
from langchain.chains.prompt_selector import is_chat_model
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.dashscope import DashScope
import os

load_dotenv()
model = "qwen-plus-1125"
api_key = os.getenv("QWEN_KEY")
api_base_url = os.getenv("QWEN_URL")

Settings.llm = DashScope(model_name=model, api_key=api_key, api_base_url=api_base_url)

Settings.embed_model = HuggingFaceEmbedding(r'./models/BAAI/bge-large-zh-v1___5')
text_to_sql = """\
你是一个SQL专家，给定一个自然语言查询，您的工作是将其转化为SQL查询
下面是一些如何将自然语言转化为SQL语句的例子，你先参考这里的例子，如果没有再动用你的知识库：

<example>
{{ examples }}
</example>

现在轮到你了。
查询： {{ query_str }}
SQL:
"""

example_nodes = [
    TextNode(
        text="Query: llama2 有多少个参数？\n SQL: SELECT COUNT(*) FROM llama_2_params;"
    ),
    TextNode(
        text="Query: llama2 有多少层？\n SQL: SELECT COUNT(*) FROM llama_2_layers;"
    ),
    TextNode(
        text="Query: 今天吃了什么？\n 回答: 吃了豆腐脑。"
    )
]

index = VectorStoreIndex(nodes=example_nodes)
retriever = index.as_retriever()


def get_examples_fn(**kwargs):
    query = kwargs["query_str"]
    examples = retriever.retrieve(query)
    return "\n\n".join(node.text for node in examples)

#此处会使用检索器找寻其对应的样例填充到 examples 中
prompt_tmpl = RichPromptTemplate(
    text_to_sql,
    function_mappings={"examples": get_examples_fn}
)

prompt = prompt_tmpl.format(
    query_str = "llama2模型的参数是多少？"
)
print(prompt)

response = Settings.llm.complete(prompt)
print(response.text)

