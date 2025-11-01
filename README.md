# MT Photos人脸识别API

- 基于[deepinsight/insightface](https://github.com/deepinsight/insightface)实现的人脸识别API，利用 RKNN 进行加速。

## 更新日志

> * 2025/10/30
>   * 项目现已支持三线程并行处理，能够同时调用 RK3588 芯片的三个 NPU 核心，充分利用硬件资源
>     需要在MT-Photos中将人脸识别任务的并发数设置成3

## 镜像说明

您可以通过 Docker 来快速部署应用。

### 打包Docker镜像

```bash
# 在arm机器上打包
sudo docker build -t mt-photos-insightface-rknn:latest .
```

### 运行Docker容器

```yaml
services:
  mt-photos-insightface:
    image: a15355447898a/mt-photos-insightface-rknn:latest
    container_name: mt-photos-insightface
    hostname: mt-photos-insightface
    environment:
      - API_AUTH_KEY=1234567890
    ports:
       - 8066:8066
    restart: always
    devices:
      - /dev/dri:/dev/dri
    privileged: true
    volumes:
        - /proc/device-tree/compatible:/proc/device-tree/compatible
        - /usr/lib/librknnrt.so:/usr/lib/librknnrt.so
```

> - `API_AUTH_KEY` 为 MT Photos 连接时需要填写的 `api_key`。
> - 端口 `8060` 可根据需要自行修改。


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8066/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /represent

```bash
curl --location --request POST 'http://127.0.0.1:8066/represent' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- detector_backend : 人脸检测模型
- recognition_model : 人脸特征提取模型
- result : 识别到的结果

### 返回数据示例
```json
{
  "detector_backend": "insightface",
  "recognition_model": "buffalo_l",
  "result": [
    {
      "embedding": [ 0.5760641694068909,... 512位向量 ],
      "facial_area": {
        "x": 212,
        "y": 112,
        "w": 179,
        "h": 250,
        "left_eye": [ 271, 201 ],
        "right_eye": [ 354, 205 ]
      },
      "face_confidence": 1.0
    }
  ]
}
```
