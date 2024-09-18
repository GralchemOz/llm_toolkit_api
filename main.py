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
    version="0.3.1")
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
    #from asgiref.sync import sync_to_async
    try:
        model_emb = SentenceTransformer(args.embedding_model_path,trust_remote_code=args.trust_remote_code,model_kwargs={"torch_dtype":torch_dtype,"attn_implementation":"sdpa"}).to(args.device)
    except:
        print("Loading embedding model failed, trust_remote_code setting doesn't work, try to set it to False\n")
        model_emb = SentenceTransformer(args.embedding_model_path,model_kwargs={"torch_dtype":torch_dtype,"attn_implementation":"sdpa"}).to(args.device)
    #async def encode2list(encode):
    #    return encode
    @app.post("/embed/")
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