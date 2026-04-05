# GLM-OCR API

> OCR Web API powered by [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) — a 0.9B multimodal model ranked **#1** on OmniDocBench V1.5.

---

## 👤 Thông tin sinh viên

| Thông tin | Chi tiết |
|---|---|
| Họ và tên | Đỗ Nguyên Tiến Phú |
| MSSV | 24120119 |
| Lớp | 24CTT3 |
| Môn học | Tư Duy Tính Toán |
| GVHD | Lê Đức Khoan |

---

## 🤖 Model

| | |
|---|---|
| **Tên model** | GLM-OCR |
| **Link HuggingFace** | https://huggingface.co/zai-org/GLM-OCR |
| **Tác giả** | Z.ai (zai-org) |
| **Kích thước** | 0.9B parameters / 2.65 GB |
| **License** | MIT |
| **Kiến trúc** | GLM-V encoder–decoder + CogViT visual encoder |

---

## 📋 Mô tả hệ chức năng thống

API cho phép trích xuất văn bản từ ảnh và tài liệu bằng mô hình GLM-OCR. Hỗ trợ 2 chế độ:

- **text**: Nhận dạng văn bản đơn giản từ ảnh
- **document**: Phân tích tài liệu đầy đủ (bảng biểu, công thức, layout)

Hỗ trợ 2 kiểu input:
- **Upload file**: jpg, png, pdf
- **URL**: Link ảnh từ internet

---

## 🗂 Cấu trúc project

```
glm-ocr-api/
├── main.py           # FastAPI application (routes, startup)
├── model.py          # Load & run GLM-OCR model
├── schemas.py        # Pydantic request/response schemas
├── test_api.py       # Test script dùng thư viện requests
├── requirements.txt  # Danh sách thư viện
└── README.md
```

---

## ⚙️ Cài đặt thư viện

### 1. Tạo virtual environment (khuyến khích)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2. Cài transformers từ source (bắt buộc cho GLM-OCR)
```bash
pip install git+https://github.com/huggingface/transformers.git
```

### 3. Cài các thư viện còn lại
```bash
pip install -r requirements.txt
```

> **Lưu ý GPU**: Cần NVIDIA GPU với ít nhất 6GB VRAM. Cài CUDA toolkit tương ứng với PyTorch version.

---

## 🚀 Chạy chương trình

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Lần đầu chạy sẽ tự động tải model (~2.65 GB) từ HuggingFace. Sau khi thấy log:
```
[INFO] Model loaded successfully!
```
thì API đã sẵn sàng.

Truy cập tài liệu tương tác tại: **http://localhost:8000/docs**

---

## 📡 Hướng dẫn gọi API

### `GET /` — Thông tin API
```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "name": "GLM-OCR API",
  "version": "1.0.0",
  "model": "zai-org/GLM-OCR",
  "endpoints": { ... }
}
```

---

### `GET /health` — Kiểm tra trạng thái
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0",
  "message": "Model is ready."
}
```

---

### `POST /predict/upload` — OCR từ file ảnh

```bash
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@test_image.png" \
  -F "mode=document"
```

**Python:**
```python
import requests

with open("test_image.png", "rb") as f:
    res = requests.post(
        "http://localhost:8000/predict/upload",
        files={"file": ("test_image.png", f, "image/png")},
        data={"mode": "document"}
    )
print(res.json())
```

**Response:**
```json
{
  "success": true,
  "mode": "document",
  "result": "<table class=\"table table-bordered\"><thead><tr><th rowspan=\"2\">Disability Category</th><th rowspan=\"2\">Participants</th><th rowspan=\"2\">Ballots Completed</th><th rowspan=\"2\">Ballots Incomplete/ Terminated</th><th colspan=\"2\">Results</th></tr><tr><th>Accuracy</th><th>Time to complete</th></tr></thead><tbody><tr><td>Blind</td><td>5</td><td>1</td><td>4</td><td>34.5%</td><td>n=1</td><td>1199 sec, n=1</td></tr><tr><td>Low Vision</td><td>5</td><td>2</td><td>3</td><td>98.3%</td><td>n=2<br>(97.7%, n=3)</td><td>1716 sec, n=3<br>(1934 sec, n=2)</td></tr><tr><td>Dexterity</td><td>5</td><td>4</td><td>1</td><td>98.3%</td><td>n=4</td><td>1672.1 sec, n=4</td></tr><tr><td>Mobility</td><td>3</td><td>3</td><td>0</td><td>95.4%</td><td>n=3</td><td>1416 sec, n=3</td></tr></tbody></table><|user|>",
  "message": "OCR completed on uploaded file: test_image.png"
}
```

---

### `POST /predict/url` — OCR từ URL ảnh

```bash
curl -X POST http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://funteacherfiles.com/wp-content/uploads/2022/10/Slide2-3.jpg", "mode": "text"}'
```

**Python:**
```python
import requests

res = requests.post(
    "http://localhost:8000/predict/url",
    json={
        "url": "https://funteacherfiles.com/wp-content/uploads/2022/10/Slide2-3.jpg",
        "mode": "text"
    }
)
print(res.json())
```

**Response:**
```json
{
  "success": true,
  "mode": "text",
  "result": "The Tent\n\nTed has a red tent.\nHe is in the tent.\nBen is in the tent,\ntoo. Ten hens ran\ninto the tent. But,\nthe tent fell on Ted,\nBen and the ten\nhens.\n\nfunteacherfiles.com<|user|>",
  "message": "OCR completed from URL."
}
```

---

### Xử lý lỗi

| HTTP Code | Ý nghĩa |
|---|---|
| `200` | Thành công |
| `422` | Dữ liệu đầu vào sai định dạng |
| `500` | Lỗi trong quá trình OCR |
| `503` | Model chưa sẵn sàng |

---

## 🧪 Chạy test

```bash
# Đảm bảo server đang chạy trước
python test_api.py
```

---

## 🎬 Video demo

[![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red)](https://youtu.be/pyelA4hDWAM)
