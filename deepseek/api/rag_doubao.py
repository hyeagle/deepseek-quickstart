import os
from glob import glob
from pymilvus import model as milvus_model
from tqdm import tqdm
from pymilvus import MilvusClient
import json
from openai import OpenAI


# 这是一个预训练轻量级嵌入模型
embedding_model = milvus_model.DefaultEmbeddingFunction()
collection_name = "my_rag_collection"
milvus_client = None
question = "How is data stored in milvus?"
client = None
SYSTEM_PROMPT = None
USER_PROMPT = None


def split_text():
    lines = []
    # 根据通配符访问文件
    for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        # 为了方便，只使用 # 来分割文本
        lines += file_text.split("# ")
    return lines


def init_db_client():
    global milvus_client 
    milvus_client = MilvusClient(uri="./milvus_demo.db")
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=768,  # milvus 模型默认维度
        metric_type="IP",  # 内积距离
        consistency_level="Session",  # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)
    )


def insert_data(text_lines):
    data = []
    embedding_model = milvus_model.DefaultEmbeddingFunction()
    doc_embeddings = embedding_model.encode_documents(text_lines)
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": doc_embeddings[i], "text": line})
    milvus_client.insert(collection_name=collection_name, data=data)


# 测试模型
def test_model():
    test_embedding = embedding_model.encode_queries(["This is a test"])[0]
    # 打印嵌入向量的维度:768
    embedding_dim = len(test_embedding)
    print(embedding_dim)
    # 打印嵌入向量的前10个元素
    print(test_embedding[:10])


def search_rag():
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries(
            [question]
        ),  # 将问题转换为嵌入向量
        limit=3,  # 返回前3个结果
        search_params={"metric_type": "IP", "params": {}},  # 内积距离
        output_fields=["text"],  # 返回 text 字段
    )
    # for hit in search_res[0]:
        # print(f"text: {hit['entity']['text']}")
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # print(json.dumps(retrieved_lines_with_distances, indent=4))

    global context
    context = "\n".join(
        [
            f"距离: {line_with_distance[1]}\n{line_with_distance[0]}"
            for line_with_distance in retrieved_lines_with_distances
        ]
    )
    return context


def load_env():
    api_key = os.getenv("DOUBAO_API_KEY")
    if not api_key:
        raise ValueError("请设置 DOUBAO_API_KEY 环境变量")
    return api_key


def init_llm_client():
    global client
    api_key = load_env()
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


def init_prompt(context, question):
    global SYSTEM_PROMPT, USER_PROMPT
    SYSTEM_PROMPT = """
    Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
    """
    USER_PROMPT = f"""
    请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    <translated>
    </translated>
    """


def llm_deal():
    response = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    text_lines = split_text()
    init_db_client()
    insert_data(text_lines)
    context = search_rag()
    init_llm_client()
    init_prompt(context, question)
    llm_deal()