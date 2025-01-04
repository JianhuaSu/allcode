from openai import OpenAI
import time
# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-5Ve6gz2t7J1Edjq0p3eAfznkGAAr3aa6eEtDQ4NSCZdthWWF",
    base_url="https://api.moonshot.cn/v1",
)

# 初始对话历史，包含系统消息
HISTORY = [
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"
    },
    {
        "role": "system",
        "content": "系统会根据用户输入的意图标签来确定如何回应，并提供个性化的对话建议和响应。意图标签包括：'Complain'（抱怨）、'Praise'（赞扬）、'Apologise'（道歉）、'Thank'（感谢）、'Criticize'（批评）、'Agree'（同意）、'Taunt'（嘲讽）、'Flaunt'（炫耀）、'Joke'（开玩笑）、'Oppose'（反对）、'Comfort'（安慰）、'Care'（关心）、'Inform'（告知）、'Advise'（建议）、'Arrange'（安排）、'Introduce'（介绍）、'Leave'（离开）、'Prevent'（防止）、'Greet'（问候）、'Ask for help'（求助）。"
    },
    {
        "role": "system",
        "content": "根据用户提供的意图标签，系统将采取相应的对话策略。每个标签对应的响应策略如下：\n情感类（'Complain', 'Comfort', 'Apologise', 'Care'）：提供情感共鸣、安慰等；\n交流类（'Praise', 'Criticize', 'Agree', 'Taunt', 'Flaunt', 'Oppose'）：提供肯定、批评、理性讨论等；\n信息类（'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 'Prevent', 'Ask for help'）：提供信息、建议、安排等。"
    },
        {
        "role": "system",
        "content": "回答时不要有称呼，大概回答10到30个字左右"
    }
]

# 定义函数，直接根据用户提供的意图生成适当的响应和建议
def chat_with_intent(content, intent):
    time.sleep(1)
    # 第一次请求生成基础回答
    response = generate_response_by_intent(intent, content)
    HISTORY.append({"role": "system", "content": f'以下是回答风格：“{response}”，请用这种风格进行回答。'})
    
    # 将用户的内容和意图加入对话历史
    HISTORY.append({"role": "user", "content": content})

    # 向OpenAI API请求生成响应
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=HISTORY,
        temperature=0.3,
    )

    # 获取助手生成的第一次回复
    assistant_response = completion.choices[0].message.content

    # 将助手的第一次回复加入历史
    HISTORY.append({"role": "assistant", "content": assistant_response})

    # 第二次请求：根据第一次的回复生成建议
    suggestion_input = f"根据以下回答，给出一些建议：\n{assistant_response}"
    HISTORY.append({"role": "user", "content": suggestion_input})

    # 向OpenAI API请求生成建议
    completion_suggestion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=HISTORY,
        temperature=0.3,
    )

    # 获取助手生成的第二次建议
    additional_suggestion = completion_suggestion.choices[0].message.content

    # 返回第一次的回答和第二次的建议
    return assistant_response, additional_suggestion

# 根据意图分类生成不同类型的响应
def generate_response_by_intent(intent, content):
    if intent in ["Complain", "Comfort", "Apologise", "Care"]:
        return generate_emotional_response(content)
    elif intent in ["Praise", "Criticize", "Agree", "Taunt", "Flaunt", "Oppose"]:
        return generate_discussion_response(content)
    elif intent in ["Inform", "Advise", "Arrange", "Introduce", "Leave", "Prevent", "Ask for help"]:
        return generate_information_response(content)
    else:
        return "抱歉，我无法理解这个意图。"

# 情感类（安慰、抱怨等）响应
def generate_emotional_response(content):
    return f"我理解你的感受，抱怨是正常的。也许我们可以考虑一下如何改变现状，或者尝试一些解决办法。"

# 交流类（赞扬、批评等）响应
def generate_discussion_response(content):
    return f"感谢你的分享！你的意见非常有价值，我会考虑你的建议并进行调整。"

# 信息类（建议、安排等）响应
def generate_information_response(content):
    return f"根据你的需求，这里是相关的信息以及你可以考虑的方案。"


