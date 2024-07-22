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

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
#model args
#parser.add_argument('--model_name', type=str, default='Florence-2-large-ft', help='Name of the model to use')
parser.add_argument('--model_path', type=str, default='microsoft/Florence-2-large-ft', help='Path to the model')
parser.add_argument('--trust_remote_code', type=bool, default=True, help='Whether to trust remote code')
parser.add_argument('--dtype', type=str, default='float16', help='Data type to use for the model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for the model')
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
    version="0.2.0")
# 初始化模型和处理器
try:
    processor = AutoProcessor.from_pretrained(args.model_path ,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype ,trust_remote_code=args.trust_remote_code).to(args.device)
except ImportError:
    # A quick fix for the issue with flash_attn from https://huggingface.co/microsoft/phi-1_5/discussions/72
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("/modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        model.to(device)    

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


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)