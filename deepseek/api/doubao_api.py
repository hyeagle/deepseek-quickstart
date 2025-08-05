import os
from openai import OpenAI

prompt = """请帮我用 HTML 生成一个五子棋游戏，所有代码都保存在一个 HTML 中。"""

def main():
    api_key = load_env()
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {"role": "system", "content": "你是一个专业的 Web 开发助手，擅长用 HTML/CSS/JavaScript 编写游戏。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=False
    )

    # 提取生成的 HTML 内容
    if response.choices and len(response.choices) > 0:
        html_content = response.choices[0].message.content
        
        # 保存到文件
        with open("doubao.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("五子棋游戏已保存为 doubao.html")
    else:
        print("未收到有效响应")


def load_env():
    api_key = os.getenv("DOUBAO_API_KEY")
    if not api_key:
        raise ValueError("请设置 DOUBAO_API_KEY 环境变量")
    return api_key

if __name__ == "__main__":
    main()
    