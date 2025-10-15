from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import asyncio
from PIL import Image
from io import BytesIO
from fastapi.responses import HTMLResponse
import logging

import inspireface as isf  # import SDK 主包

logging.basicConfig(level=logging.INFO)

load_dotenv()
app = FastAPI()

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8066"))

face_session = None

@app.on_event("startup")
async def startup_event():
    global face_session
    logging.info("Starting InspireFace RKNN server...")

    try:
        # 自动下载并加载 Gundam_RK3588 模型（如果本地已有，则直接加载）
        isf.reload("Gundam_RK3588")
        logging.info("Model Gundam_RK3588 reloaded successfully.")

        # 创建会话配置，这里启用人脸质量检测
        opt = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_RECOGNITION # 根据需要可用 | 运算组合多功能，例如 isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_RECOGNITION

        # 创建 InspireFace 会话，模式为一直检测（Always Detect）
        face_session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)
        logging.info("InspireFaceSession created successfully.")
    except Exception as e:
        logging.exception(f"Failed to initialize InspireFace session: {e}")
        sys.exit(1)

async def verify_header(api_key: str = Header(...)):
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """
    <!DOCTYPE html>
    <html><head><title>MT Photos InspireFace RKNN Server</title></head>
    <body>
    <h2>MT Photos 人脸识别服务 (InspireFace RKNN)</h2>
    <p>服务状态：运行中</p>
    <p>文档：https://mtmt.tech/docs/advanced/insightface_api/</p>
    </body></html>
    """
    return HTMLResponse(content=html_content)

@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-inspireface-rknn",
        "help": "https://mtmt.tech/docs/advanced/insightface_api",
        "detector_backend": "inspireface_rknn_gundam",
        "recognition_model": "Gundam_RK3588",
        "facial_min_score": 0.5,
        "facial_max_distance": 0.5,
    }

@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    global face_session
    if face_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = file.content_type
    image_bytes = await file.read()
    
    try:
        img = None
        if content_type == 'image/gif':
            with Image.open(BytesIO(image_bytes)) as pil_img:
                if pil_img.is_animated:
                    pil_img.seek(0)
                frame = pil_img.convert('RGB')
                img = np.array(frame)
        if img is None:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            return {'result': [], 'msg': f"Invalid or corrupted image: {file.filename}"}
        
        h, w, _ = img.shape
        if h > 10000 or w > 10000:
            return {'result': [], 'msg': 'Image size too large'}

        # 异步调用同步推理接口
        embedding_objs = await asyncio.get_running_loop().run_in_executor(None, lambda: _represent(img))
        
        return {
            "detector_backend": "inspireface_rknn_gundam",
            "recognition_model": "Gundam_RK3588",
            "result": embedding_objs
        }
    except Exception as e:
        logging.exception(f"Error processing image: {e}")
        return {'result': [], 'msg': str(e)}

def _represent(img):
    faces = face_session.face_detection(img)
    results = []
    for face in faces:
        feature = face_session.face_feature_extract(img, face)
        if feature is not None:
            res = {
                "embedding": feature.tolist(),
                "facial_area": {
                    "x": int(face.location[0]),
                    "y": int(face.location[1]),
                    "w": int(face.location[2] - face.location[0]),
                    "h": int(face.location[3] - face.location[1])
                },
                "face_confidence": float(face.detection_confidence)
            }
            results.append(res)
    return results

if __name__ == "__main__":
    uvicorn.run("server_rknn:app", host="0.0.0.0", port=http_port)
