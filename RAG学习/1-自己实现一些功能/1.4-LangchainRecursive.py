from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == '__main__':
    """
    langchain 中包含了一个模块化的分割器类，只要引用实例化就能分割字符。__init__() 一般使用以下三个参数：
    (chunk_size, chunk_overlap, length_function)，表示的是分割的文本长度，和前文后面重叠的文本长度，输入内容的长度函数
    """
    text = ("自然语言处理（NLP），"
            "作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。"
            "在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。"
            "从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。"
            "随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。"
            "如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。"
            "NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        length_function=len
    )

    print(splitter)
    spl_text = splitter.split_text(text)
    print(spl_text)
