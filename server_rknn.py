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
import threading
import queue
from contextlib import asynccontextmanager

import inspireface as isf  # import SDK 主包

logging.basicConfig(level=logging.INFO)

load_dotenv()

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8066"))
num_threads = int(os.getenv("NUM_THREADS", "3"))

# 线程安全的任务队列
task_queue = queue.Queue()
# 结果字典
results = {}

class Worker(threading.Thread):
    def __init__(self, core_id):
        super().__init__()
        self.core_id = core_id
        self.face_session = None
        self.daemon = True

    def run(self):
        self.init_model()
        self.process_tasks()

    def init_model(self):
        try:
            core_mask = 0
            if self.core_id is not None and self.core_id >= 0:
                core_mask = 1 << self.core_id
            if core_mask:
                isf.set_rknn_core_mask(core_mask)
            # 自动下载并加载 Gundam_RK3588 模型
            isf.reload("Gundam_RK3588")
            # 创建会话配置，启用人脸质量检测和识别
            opt = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_RECOGNITION
            # 创建 InspireFace 会话
            self.face_session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)
            logging.info(f"Worker {self.core_id} initialized successfully (core_mask=0x{core_mask:x}).")
        except Exception as e:
            logging.exception(f"Worker {self.core_id} failed to initialize: {e}")

    def process_tasks(self):
        while True:
            task_id, img_bytes, content_type = task_queue.get()
            try:
                img = self.preprocess_image(img_bytes, content_type)
                if isinstance(img, str):
                    results[task_id] = {'result': [], 'msg': img}
                else:
                    embedding_objs = self._represent(img)
                    results[task_id] = {
                        "detector_backend": "inspireface_rknn_gundam",
                        "recognition_model": "Gundam_RK3588",
                        "result": embedding_objs
                    }
            except Exception as e:
                logging.exception(f"Worker {self.core_id} error processing task {task_id}: {e}")
                results[task_id] = {'result': [], 'msg': str(e)}
            finally:
                task_queue.task_done()

    def preprocess_image(self, image_bytes, content_type):
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
            return "Invalid or corrupted image"
        
        h, w, _ = img.shape
        if h > 10000 or w > 10000:
            return 'Image size too large'
        return img

    def _represent(self, img):
        faces = self.face_session.face_detection(img)
        results = []
        for face in faces:
            feature = self.face_session.face_feature_extract(img, face)
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting InspireFace RKNN server with %d threads...", num_threads)
    for i in range(num_threads):
        worker = Worker(i)
        worker.start()
    yield

app = FastAPI(lifespan=lifespan)

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
    task_id = os.urandom(16).hex()
    image_bytes = await file.read()
    task_queue.put((task_id, image_bytes, file.content_type))
    
    # 使用循环和 asyncio.sleep 来异步等待结果
    while task_id not in results:
        await asyncio.sleep(0.01)
        
    return results.pop(task_id)

if __name__ == "__main__":
    uvicorn.run("server_rknn:app", host="0.0.0.0", port=http_port)
