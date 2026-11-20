# Hướng dẫn nộp Capstone lên GitHub

**Đây là bài yêu cầu URL GitHub, không phải upload PDF/PNG.** Đề có 12 tiêu chí đối chiếu code. Repo cần ở chế độ **Public** trước khi nộp.

## 1. Chuẩn bị dữ liệu thật

Trong thư mục source, mở terminal:

```bash
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.train --data data/aavail-target.csv --output models
python run_tests.py
```

Nếu máy chưa có internet, cậu có thể chép file `aavail-target.csv` do khóa học cung cấp vào `data/`. Bộ nguồn ban đầu có mô hình DEMO dùng dữ liệu mô phỏng, **không nên dùng để báo cáo kết quả thị trường thật**. Để cập nhật hình từ dataset mới: copy 4 ảnh trong `models/` vào `docs/figures/` và chỉnh `docs/EDA.md` theo số liệu thật. Nếu tải dữ liệu từ nguồn công khai, ghi rõ nguồn trong README.

## 2. Kiểm tra tại máy

```bash
python run_tests.py
uvicorn aavail.api:app --host 127.0.0.1 --port 8080
```

Truy cập http://127.0.0.1:8080/docs, kiểm tra API `/predict?country=singapore`, `/predict?country=all`, `/monitor`, `/markets/summary`.

Nếu có Docker Desktop:

```bash
docker build -t aavail-capstone .
docker run --rm -p 8080:8080 aavail-capstone
```

Vì AI không có Docker Engine trong môi trường tạo file, **cần chạy thử ở máy cậu hoặc để GitHub Actions xác nhận**. Code không có lỗi unit test tại thời điểm đóng gói, nhưng không thay thế Docker smoke test.

## 3. Đưa lên GitHub

1. Đăng nhập GitHub, chọn **New repository**.
2. Đặt tên ví dụ `aavail-ai-enterprise-capstone`, để visibility **Public**, tạo repo.
3. Chọn **Add file → Upload files**. Giải nén source ZIP, upload **nội dung thư mục** (README.md, Dockerfile, aavail/, tests/, scripts/, docs/, .github/, ...), không chỉ upload file ZIP.
4. Chờ GitHub Actions CI hoàn tất. Nếu CI lỗi, mở log và sửa trước khi submit.
5. Lấy URL repo thực, dạng `https://github.com/<username>/aavail-ai-enterprise-capstone` và dán vào ô `URL` trên Coursera.

**Project Title gợi ý:** `AAVAIL Customer Churn Prediction: End-to-End AI Workflow`

## 4. Chú ý

- Không lấy URL từ repository của người khác, không dùng link GitHub ví dụ để nộp.
- Repo này là mẫu hoàn chỉnh về mặt source và unit test, **không thể đảm bảo được full rubric trước khi xác nhận dữ liệu thật + GitHub CI/Docker**.
- `README.md` có bảng ánh xạ 12 tiêu chí, người chấm dễ tìm evidence.
