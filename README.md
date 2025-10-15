# MT Photos人脸识别API

- 基于[deepinsight/insightface](https://github.com/deepinsight/insightface)实现的人脸识别API


## 安装方法

- 下载镜像

去releases下载

- 创建容器

```
services:
  mt-photos-insightface:
    image: mt-photos-insightface-rknn:latest
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



## 打包docker镜像

```bash
sudo docker build -t mt-photos-insightface-rknn:latest .
```


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
