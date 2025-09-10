from llama_index.core.prompts import RichPromptTemplate

qa_prompt_str = """
上下文信息如下：
--------------------
{{ context_str }}
--------------------
根据给定的上下文信息而不是先前的知识，回答查询
查询：{{ query_str }}
答案：
"""

def format_context_fn(**kwargs):
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join(f"- {c}" for c in context_list)
    return fmtted_context

prompt_templ = RichPromptTemplate(
    qa_prompt_str, function_mappings={"context_str": format_context_fn}
)
context_str = """\
Day 1: 项目启动与架构设计
今天团队正式启动了Llama 2的开发。Meta的目标很明确：在LLaMA 1的基础上提升模型性能，同时保持开源和可商用性。我们决定从以下方向突破：

扩展参数规模：增加70B版本，预训练数据从1万亿token提升至2万亿。

优化注意力机制：引入Group Query Attention（GQA）以减少推理时的KV Cache内存占用。

强化对话能力：计划通过监督微调（SFT）和人类反馈强化学习（RLHF）训练专用聊天模型Llama 2-Chat。

架构团队讨论了GQA的实现细节——它介于多头注意力（MHA）和多查询注意力（MQA）之间，通过分组共享键值对平衡计算效率和性能。

Day 15: 训练数据与分词器优化
数据组报告了预处理进展：

沿用LLaMA 1的SentencePiece BPE分词器（词汇量32k），但改进了对数字和UTF-8字符的处理。

清除了数据中的重复内容，并增加了代码和学术文本的比例，以提升模型推理能力。

训练稳定性是个挑战。我们采用了Pre-normalization（RMSNorm）和SwiGLU激活函数，初步实验显示损失曲线更平滑。

Day 30: 预训练与硬件瓶颈
预训练启动后，70B模型的显存占用惊人。基础设施团队不得不重新设计分布式训练策略：

使用模型并行+数据并行组合，将不同层分配到不同GPU。

遇到梯度同步延迟问题，通过优化通信协议缓解。

工程师Yinghai Lu提出用bfloat16混合精度训练，既节省内存又避免数值溢出。

Day 60: 微调与人类反馈
开始训练Llama 2-Chat！监督微调阶段使用了100万条人类标注的对话数据。
RLHF阶段更复杂：

通过拒绝采样和近端策略优化（PPO）迭代调整模型输出。

安全性团队（包括Yuchen Zhang）加入，设计奖励模型以减少有害内容。

发现模型有时过于“谨慎”，反而回答空洞。调整奖励权重后有所改善。
"""

fmt_prompt = prompt_templ.format(
    context_str=context_str, query_str = "llama 2 模型有多少个参数？"
)
print(fmt_prompt)