def overlap_split(text_input, subtext_length, stride):
    spl = []
    for i in range(0, len(text_input), stride):
        spl.append(text_input[i: i + subtext_length])
    return spl


if __name__ == '__main__':
    """
    当步长和给定长度不同（小于给定长度）时，会产生重叠分割。这种分割在处理中很常见，称为 overlapping window
    这种方法可以减少固定长度分割导致的句意不连贯问题。即将 abcdefg 分割为
    abc, bcd, cde, def, efg 这样的小文本
    """

    length = 50  # 给定长度为 50
    text = ("自然语言处理（NLP），"
            "作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。"
            "在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。"
            "从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。"
            "随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。"
            "如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。"
            "NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。")
    spl_text = overlap_split(text, length, 20)
    print(spl_text)