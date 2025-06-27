from fastapi import FastAPI, Body
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import requests
import torch
import httpx
import base64
import argparse
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import os
import time
import uuid
from pydantic import BaseModel,Field 
from typing import List, Optional

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
#model args
#parser.add_argument('--model_name', type=str, default='Florence-2-large-ft', help='Name of the model to use')
parser.add_argument('--model_path', type=str, default=None, help='Path to the florence-2 model')
parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether to trust remote code')
parser.add_argument('--dtype', type=str, default='float16', help='Data type to use for the model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for the model')
parser.add_argument('--embedding_model_path', type=str, default=None, help='Path to the embedding model')
parser.add_argument('--reranker_model_path',type=str, default=None, help='Path to the reranker model')
parser.add_argument('--fetch', type=bool, default=False, help='Whether to fetch a web page')
parser.add_argument('--guard_model_path', type=str, default=None, help='Whether to start the guard model')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose mode')
parser.add_argument('--html2markdown_model_path', type=str, default=None, help='Path to the html2markdown model')

args = parser.parse_args()

if args.dtype == 'float16':
    torch_dtype = torch.float16 
elif args.dtype == 'float32':
    torch_dtype = torch.float32
elif args.dtype == 'bfloat16':
    torch_dtype = torch.bfloat16
else:
    raise ValueError(f"Unsupported data type: {args.dtype}")
# 初始化FastAPI应用
app = FastAPI(    title="llm_toolkit_api",
    description="A simple API for extra functionality for large language models",
    version="1.1.0")
# 初始化模型和处理器
if args.model_path:
    try:
        processor = AutoProcessor.from_pretrained(args.model_path ,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype ,trust_remote_code=args.trust_remote_code).to(args.device)
    except ImportError:
        # A quick fix for the issue with flash_attn from https://huggingface.co/microsoft/phi-1_5/discussions/72
        def fixed_get_imports(filename):
            #if not str(filename).endswith("/modeling_florence2.py"):
            #    return get_imports(filename)
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports
    
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype ,trust_remote_code=args.trust_remote_code)
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
            model.to(args.device)    
    
    @app.post("/generate/")
    async def generate(body: dict = Body(...,example={
        "prompt": "<CAPTION>",
        "task_type": "<CAPTION>",
        "file_or_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    })):
        """
        Send a image into the florence-2 model and get the replay
        """
        prompt = body.get("prompt", None)
        task_type = body.get("task_type", None)
        file_or_url = body.get("file_or_url", None)
        # 读取图像
        try:
            #image = Image.open(file_or_url)
            byte_data = base64.b64decode(file_or_url)
    
            # 创建 BytesIO 对象
            image_file = io.BytesIO(byte_data)
    
            # 打开图像
            image = Image.open(image_file)
        except:
            url = file_or_url
            #image = Image.open(requests.get(url, stream=True).raw)
            # 发送HTTP GET请求以获取图片
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()  # 确保请求成功
    
                # 将响应内容（即图片数据）存储在变量中
                image_data = response.content
    
            # 使用Pillow打开图片
            image = Image.open(io.BytesIO(image_data))        
    
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # 处理输入
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device,torch_dtype)
    
    
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    
        # 将生成结果移回CPU
        generated_ids = generated_ids.to('cpu')
        #释放显存
        inputs["input_ids"], inputs["pixel_values"] = None, None
        torch.cuda.empty_cache()
        # 解码文本
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
        # 后处理
        #if task_type:
        parsed_answer = processor.post_process_generation(generated_text, task= task_type, image_size=(image.width, image.height))
        #else:
        #    parsed_answer = processor.post_process_generation(generated_text, task= "<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
    
        # 返回结果
        return parsed_answer

#support for embedding models
if args.embedding_model_path:
    from sentence_transformers import SentenceTransformer
    from asgiref.sync import sync_to_async
    #from asgiref.sync import sync_to_async
    try:
        model_emb = SentenceTransformer(args.embedding_model_path,trust_remote_code=args.trust_remote_code,model_kwargs={"torch_dtype":torch_dtype,"attn_implementation":"sdpa"}).to(args.device)
    except:
        print("Loading embedding model failed, trust_remote_code setting doesn't work, try to set it to False\n")
        model_emb = SentenceTransformer(args.embedding_model_path,model_kwargs={"torch_dtype":torch_dtype,"attn_implementation":"sdpa"}).to(args.device)
    #async def encode2list(encode):
    #    return encode

    @app.post("/embed/")
    async def embed_new(body: dict = Body(..., example={"input": ["Hello, world!", "你好，世界！"],
                                                        "model": "BAAI/bge-large-zh-v1.5",
                                                        "encoding_format": "float"})):
        """
        Use the sentence-transformers model to embed text (New Standardized Format)
        Now supports multiple inputs in a single request.
        """
        start_time = time.time()

        input_texts = body.get("input", None)
        model_name = body.get("model", "BAAI/bge-large-zh-v1.5")  # 提供默认值
        encoding_format = body.get("encoding_format", "float")  # 提供默认值

        if not isinstance(input_texts, list):
            input_texts = [input_texts]  # 如果input不是列表，转换为列表

        embeddings_list = []
        for i, input_text in enumerate(input_texts):
            embeddings = await sync_to_async(model_emb.encode)(input_text)
            embeddings_list.append({
                "object": "embedding",
                "embedding": embeddings.tolist(),
                "index": i
            })

        elapsed_time = time.time() - start_time
        if args.verbose:
            print(f"Embedding context: {input_texts}")
            print(f"Embedding time: {elapsed_time:.4f} seconds")

        # Format the response to match the OpenAI structure
        response_data = {
            "model": model_name,
            "data": embeddings_list,
            "usage": {
                "prompt_tokens": 10,  # 占位符
                "completion_tokens": 10,  # 占位符
                "total_tokens": 10  # 占位符
            }
        }
        return response_data
        
    @app.post("/embed/legacy/")
    async def embed(body: dict = Body(...,example={"text": "Hello, world!"})):
        """
        Use the sentence-transformers model to embed text
        """
        start_time = time.time()

        text = body.get("text", None)
        #inputs = processor(text=text, return_tensors="pt").to(args.device,torch_dtype)
        inputs = text
        with torch.no_grad():
            embeddings = model_emb.encode(inputs)
        #embeddings = await sync_to_async(model_emb.encode)(inputs)
        #embeddings = await encode2list(embeddings)        
        elapsed_time = time.time() - start_time
        if args.verbose:
            print(f"Embedding context: {inputs}")
            print(f"Embedding time: {elapsed_time:.4f} seconds")
        return embeddings.tolist()

#support for reranker model
if args.reranker_model_path:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs


    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.reranker_model_path,torch_dtype=torch.float16).eval().to(args.device)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    @app.post("/reranker/singletest/")
    async def reranker(body: dict = Body(...,example={"query": "What is the capital of France?", "doc": "The capital of France is Paris."})):
        """
        Use the reranker model to get the score of the document
        """
        start_time = time.time()
        query = body.get("query", None)
        doc = body.get("doc", None)
        inputs = process_inputs([format_instruction(task, query, doc)])
        scores = compute_logits(inputs)
        elasped_time = time.time() - start_time
        if args.verbose:
            print(f"Reranker query: {query}")
            print(f"Reranker doc: {doc}")
            print(f"Reranker time: {elasped_time:.4f} seconds")
        return scores[0]
    
    class RerankRequest(BaseModel):
        """
        comptaible with standred API
        """
        model: str = Field("Qwen-3-Reranker-0.6B", description="Not used")
        query: str = Field("What is the capital of France?", description="The query to search for")
        documents: List[str] = Field([
                        "The capital of France is Paris.",
                        "Paris is the capital of France.",
                        "The capital of France is Lyon.",
                        "Lyon is the capital of France.",
                        "The capital of France is Marseille.",
                    ], description="The documents to search for")
        top_n: int = Field(10, description="The number of documents to return")
        return_documents: bool = Field(True, description="Whether to return the documents")

    
    class RerankResult(BaseModel):
        """
        comptaible with standred API
        """
        index: int = Field(..., description="The original index of the document")
        relevance_score: float = Field(..., description="The relevance score of the document")
        document: Optional[str] = Field(None, description="The document")

    class RerankResponse(BaseModel):
        """
        comptaible with standred API
        """
        id: str = Field(default_factory=lambda: f"rerank-{uuid.uuid4()}", description="The id of the document")
        object: str = Field(default="rerank", description="The object of the response")
        crated: int = Field(default=int(time.time()), description="The time the document was created")
        model: str = Field(..., description="The model used to generate the response")
        results: List[RerankResult] = Field(..., description="The results of the rerank")

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank_endpoint(request: RerankRequest):
        """
        get request and return response
        """
        start_time = time.time()
        pairs = [format_instruction(task, query, doc) for query, doc in [[request.query, doc] for doc in request.documents]]
        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)
        elasped_time = time.time() - start_time
        results_with_scores = list(zip(range(len(request.documents)),scores,request.documents))
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        if request.top_n is not None and request.top_n > 0:
            top_results = results_with_scores[:request.top_n]
        else:
            top_results = results_with_scores

        #formalize results    
        response_results: List[RerankResult] = []
        for index, score, doc in top_results:
            result_item = RerankResult(
                index=index,
                relevance_score=float(score)
            )
            if request.return_documents:
                result_item.document = doc
            response_results.append(result_item)
        if args.verbose:
            print(f"Reranker: {request.model}")
            print(f"query: {request.query}")
            print(f"results: {top_results}")
            print(f"time: {time.time() - start_time}s")
        return RerankResponse(model = request.model,results=response_results)

#support for HTML2markdown model
if args.html2markdown_model_path:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.html2markdown_model_path)
    model_html2markdown = AutoModelForCausalLM.from_pretrained(args.html2markdown_model_path,torch_dtype=torch_dtype).to(args.device)
    @app.post("/html2markdown/")
    async def html2markdown(body: dict = Body(...,example={"html_or_url": "https://www.example.com"})):
        """
        Use the HTML2markdown model to convert HTML to markdown
        """
        start_time = time.time()
        #judge if the input is a url or html
        html_or_url = body.get("html_or_url", None)

        if html_or_url.startswith("http") or html_or_url.startswith("www"):
            async with httpx.AsyncClient() as client:
                response = await client.get(html_or_url,follow_redirects=True)
                html_content = response.content.decode("utf-8")
        else:
            html_content = html_or_url
        messages = [{"role": "user", "content": html_content}]
        input_text=tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(args.device)
        outputs = model_html2markdown.generate(inputs, max_new_tokens=128, temperature=0, do_sample=False, repetition_penalty=1.2)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs, outputs)]
        #release vram
        del inputs
        torch.cuda.empty_cache()
        if args.verbose:
            print(f"HTML2markdown context: {html_content}")
            print(f"HTML2markdown time: {time.time() - start_time:.4f} seconds")
        raw_text = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
        #strip assistant 
        raw_text = raw_text.split("assistant")[1].strip()
        return raw_text
        


#support for guard model
if args.guard_model_path:
    from transformers import pipeline
    classifier = pipeline("text-classification", model=args.guard_model_path,device=args.device)
    @app.post("/guard/")
    async def guard(body: dict = Body(...,example={"text": "Hello, world!"})):
        """
        Use the guard model to classify text
        """
        text = body.get("text", None)
        #inputs = processor(text=text, return_tensors="pt").to(args.device,torch_dtype)
        inputs = text
        with torch.no_grad():
            result = classifier(inputs)
        return result

#support for fetch and parse a web page
if args.fetch:
    if not args.embedding_model_path:
        raise Exception("embedding_model_path is required for fetch and parse")
    import httpx
    from bs4 import BeautifulSoup
    import json
    import asyncio
    from utils.web_parser import *
    from utils.embedding_content import *
    import time
    from selenium.webdriver.chrome.options import Options

    async def fetch_and_process(url: str):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式下运行
        chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速，某些系统/配置需要
        chrome_options.add_argument("--ignore-ssl-errors")  
        chrome_options.add_argument("--no-sandbox")  # 在某些环境中需要
        chrome_options.add_argument("--disable-dev-shm-usage")  # 在某些环境中需要
        #chrome_options.add_argument(f"--proxy-server=http://192.168.1.100:8080")  # 设置代理服务器
        driver = webdriver.Chrome(options=chrome_options)
        async with httpx.AsyncClient() as client:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            raw_html = driver.page_source
            soup = BeautifulSoup(raw_html, 'html.parser')
            window_rect = driver.get_window_rect()
            readableNodes = get_all_readable_nodes(driver, timeout_seconds=10)
            print(f"number of readable nodes: {len(readableNodes)}")
            clusters, noise = cluster_readable_nodes(readableNodes)
            critical_clusters = find_critical_clusters(window_rect, readableNodes, clusters)

            cluster_membership = {}
            for cluster in critical_clusters:
                for index in cluster:
                    cluster_membership[index] = True
            filtered_nodes = [node for i, node in enumerate(readableNodes) if i in cluster_membership]

            # 输出节点数量
            print(f"number of readable nodes: {len(filtered_nodes)}")
            elements = [serialize_node(node["node"]) for node in filtered_nodes]
            metadata = get_page_metadata(soup)
            data = {**metadata, "elements": elements}
            return data

    @app.post("/fetch/raw/")
    async def fetch_and_parse(body: dict = Body(..., example={"url": "https://example.com"})):
        """Fetch and parse a web page with raw format."""
        url = body.get("url", None)
        return await fetch_and_process(url)

    @app.post("/fetch/embed/")
    async def fetch_and_emb(body: dict = Body(..., example={"url": "https://example.com"})):
        """Fetch and parse a web page with split and embedding."""
        url = body.get("url", None)
        data = await fetch_and_process(url)
        # 读取JSON文件中的文本内容
        contents = read_json_file(data)
        # 过滤重复文本
        unique_contents = list(set(contents))  # 使用集合去重
        # 处理文本并获取embedding
        embeddings = process_texts_for_embedding(unique_contents, model_emb, segment_length=256)
        return embeddings

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
