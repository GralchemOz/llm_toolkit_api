import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
# 读取JSON文件并提取content
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except:
        data = file_path
    contents = [element['content'][0] for element in data['elements'] if 'content' in element]
    return contents

# 比较文本的重复度
def is_highly_redundant(text1, text2, threshold=0.99):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return 1 - cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1]) > threshold

# 切分文本并去重
def split_and_deduplicate(text, segment_length):
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    unique_segments = []
    for segment in segments:
        # 检查是否有重复的文本
        redundant = [existing for existing in unique_segments if is_highly_redundant(segment, existing)]
        if not redundant:
            # 如果没有重复的，直接添加
            unique_segments.append(segment)
        else:
            # 如果有重复的，保留最长的文本
            longest_redundant = max(redundant, key=len)
            if len(segment) > len(longest_redundant):
                # 移除较短的重复文本，添加较长的文本
                unique_segments = [s for s in unique_segments if s != longest_redundant]
                unique_segments.append(segment)
    return unique_segments

# 处理文本并进行embedding
def process_texts_for_embedding(unique_contents, model, segment_length):
    embeddings = []
    for content in unique_contents:
        # 切分文本并去重
        split_contents = split_and_deduplicate(content, segment_length)
        for split_content in split_contents:
            #print(split_content)
            result_json = process_long_text(segment_length, split_content, model)
            embeddings.append(json.loads(result_json))
    return embeddings

def process_long_text(segment_length, input_text, model):
    # 将输入文本分割为指定长度的段落
    segments = [input_text[i:i+segment_length] for i in range(0, len(input_text), segment_length)]

    # 存储所有段落的embedding
    embeddings = []

    for segment in segments:
        
        # 获取模型输出
        with torch.no_grad():
            output = model.encode(segment)

        cls_embedding = output.tolist()
        embeddings.append(cls_embedding)

    # 将结果转换为JSON格式
    result_json = json.dumps({
        'segments': segments,
        'embeddings': embeddings
    }, ensure_ascii=False)

    return result_json

if __name__ == '__main__':
    model_emb = SentenceTransformer("").to("cuda")
    # 示例JSON文件路径
    json_file_path = 'output.json'
    # 读取JSON文件中的文本内容
    contents = read_json_file(json_file_path)
    # 过滤重复文本
    unique_contents = list(set(contents))  # 使用集合去重
    # 处理文本并获取embedding
    embeddings = process_texts_for_embedding(unique_contents, model_emb, segment_length=256)

    
    #结果返回为json
    result = {
        'unqique_contents': unique_contents,
        'embeddings': embeddings
    }
    #存储结果
    with open('embedding_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("Embedding result saved to embedding_result.json")