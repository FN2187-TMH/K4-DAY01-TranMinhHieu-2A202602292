# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- Record này mô tả toàn ảnh như thế nào?
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
- Vì sao model score không phải ground truth?

* **Record hạng 1:**
  * `class_id`: `468`[cite: 2]
  * `class_name`: `"cab"`[cite: 2]
  * `rank`: `1`[cite: 2]
  * `score`: `0.510915` (tương đương $\approx 51.09\%$)[cite: 2]
  * `taxonomy_name`: `"ImageNet-1K"`[cite: 2]
* **Mô tả toàn ảnh:** Record này gán một nhãn phân loại tổng thể cho toàn bộ bức ảnh là chiếc xe taxi ("cab") với xác suất $51.09\%$[cite: 2], phản ánh chủ thể/ngữ cảnh chính của ảnh mà không xác định vị trí tọa độ cụ thể của vật thể.
* **Đơn vị định nghĩa class list:** Đơn vị/tác giả xây dựng bộ dữ liệu chuẩn **ImageNet-1K** (gồm 1,000 lớp đối tượng)[cite: 2].
* **Lý do cần giữ ID, tên lớp và tên taxonomy:**
  * **ID (`class_id`)**: Đảm bảo tính nhất quán và tối ưu hóa khi xử lý bằng lập trình/máy tính[cite: 2].
  * **Tên lớp (`class_name`)**: Giúp con người (annotator, reviewer) đọc hiểu ngữ nghĩa trực quan[cite: 2].
  * **Tên taxonomy (`taxonomy_name`)**: Định danh không gian nhãn chuẩn (`ImageNet-1K`), tránh xung đột ngữ nghĩa khi chuyển đổi hoặc so sánh với các bộ dữ liệu khác (như COCO hay Pascal VOC)[cite: 2].
* **Guideline khi ảnh có nhiều chủ thể:** Guideline cần quy định rõ nguyên tắc ưu tiên: gán nhãn theo chủ thể chính/chiếm diện tích lớn nhất ở trung tâm (dominant object), gán theo ngữ cảnh bao trùm toàn cảnh (scene-level), hoặc quy tắc xử lý khi các chủ thể có độ lớn tương đương.
* **Lý do model score không phải ground truth:** Model score chỉ là xác suất/độ tin cậy toán học (confidence score) được mô hình dự đoán dựa trên trọng số đã huấn luyện[cite: 2], có thể xuất hiện sai số hoặc nhiễu. Ground truth là thông tin nhãn chuẩn xác tuyệt đối được con người kiểm duyệt và gán thực tế.

---


## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- Diễn giải vị trí box bằng lời:
- So sánh số prediction ở hai threshold:
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
- Đề xuất một quy tắc box chặt:
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

* **Một record tiêu biểu (record 1 của sample `kitchen`):**
  * `class_name`: `"person"`[cite: 3]
  * `score`: `0.912625`[cite: 3]
  * `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`[cite: 3]
  * `bbox_width`: `113.58`[cite: 3]
  * `bbox_height`: `279.68`[cite: 3]
* **Diễn giải vị trí box bằng lời:** Đối tượng "person" được khoanh vùng bởi khung chữ nhật có tọa độ góc trên-bên trái là $(385.33, 69.24)$ px và góc dưới-bên phải là $(498.92, 348.92)$ px[cite: 3]. Khung bao này có kích thước chiều rộng $113.58$ px và chiều cao $279.68$ px trên ảnh[cite: 3].
* **So sánh số prediction ở hai threshold:** Ở ngưỡng thấp (ví dụ $0.35$)[cite: 3], số lượng bounding box trả về nhiều hơn, bao phủ được nhiều vật thể mờ/nhỏ nhưng lẫn nhiều nhiễu (dương tính giả). Ở ngưỡng cao (ví dụ $0.70$), số lượng box giảm đáng kể, chỉ giữ lại các đối tượng có độ tin cậy cao nhưng dễ bỏ sót vật thể bị che khuất (âm tính giả).
* **Biến động độ bao phủ và khối lượng reviewer:** Ngưỡng threshold thấp làm tăng độ bao phủ (Recall) nhưng làm tăng khối lượng công việc reviewer phải lọc bớt box sai. Ngưỡng cao làm giảm tải công việc của reviewer nhưng dễ gây thiếu sót dữ liệu gán nhãn.
* **Quy tắc box chặt (Tight box):** Bounding box phải ôm vừa sát viền cực trị ngoại tiếp của vật thể, không để thừa quá $2-5\%$ diện tích khoảng trống nền và không được xén lấn vào phần thân vật thể.
* **Quy định đối với object bị che khuất/cắt mép:** Guideline cần định rõ ngưỡng diện tích nhìn thấy tối thiểu để vẽ box (ví dụ: chỉ gán nhãn nếu thấy $>15\%$ vật thể), quy định vẽ box theo phần nhìn thấy (visible bbox) hay ước lượng phần bị che (full bbox), và quy trình escalation báo cáo Lead khi đối tượng bị cắt mép quá nặng.

---

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

* **Một record tiêu biểu (instance `traffic-001`):**
  * `instance_id`: `"traffic-001"`[cite: 5]
  * `class_name`: `"bus"`[cite: 5]
  * `score`: `0.925745`[cite: 5]
  * Số điểm (`polygon_point_count`): `120`[cite: 5]
  * Một phần `polygon_xy`: `[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], ...]`[cite: 5]
* **Chi tiết bổ sung của Polygon so với Bounding Box:** Polygon cung cấp chính xác đường biên ranh giới (contour/boundary) theo từng pixel của đối tượng, giúp loại bỏ toàn bộ phần điểm ảnh thuộc về nền (background) bị lẫn bên trong khung chữ nhật.
* **Vai trò của `instance_id`:** `instance_id` dùng để phân biệt và định danh duy nhất từng đối tượng riêng lẻ cụ thể trong cùng một bức ảnh[cite: 5]. Nó **không phải** là `class_id` (mã lớp chung) và **không phải** là `sample_id`/`coco_image_id` (mã nhận dạng bức ảnh)[cite: 5].
* **Quy tắc biên mask:** Đường biên polygon phải bám sát ranh giới thật của vật thể với độ chính xác ở cấp độ pixel (sai số lệch biên không quá $1-2$ px) và không được đè lấn sang mask của vật thể kế bên.
* **Quy định vùng mờ/tiếp xúc/che khuất:** Guideline cần quy định việc tách thành nhiều polygon độc lập (multi-polygon) khi vật thể bị che ở giữa, và cần cơ chế escalation để reviewer/lead xử lý khi hai vật thể dính liền có màu sắc/kết cấu quá tương đồng không thể phân định biên bằng mắt thường.

---

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| :--- | :--- | :--- | :--- | :--- |
| **Phân loại ảnh** | Nhãn ID/Text (Single/Multi-label Class ID, ví dụ: `468 - cab`)[cite: 2]. | Ảnh chứa nhiều chủ thể phức tạp dẫn đến mô hình chọn sai ngữ cảnh chính (ví dụ: ảnh bếp bị phân loại thành "gong")[cite: 2]. | Đọc quy tắc ưu tiên trong guideline, chọn $1$ nhãn đại diện chính xác nhất cho ngữ cảnh toàn ảnh[cite: 2]. | Kiểm tra nhãn gán có tuân thủ đúng thứ tự ưu tiên chủ thể trong guideline hay không. |
| **Phát hiện vật thể** | Tọa độ khung Bounding Box `[x_min, y_min, x_max, y_max]` kèm `class_id`[cite: 3]. | Box bao quá rộng thừa nhiều nền, gán nhầm lớp các xe gần nhau, hoặc bỏ sót đối tượng nhỏ[cite: 3]. | Vẽ khung bao ôm sát (tight) đường viền ngoại tiếp vật thể, gán đúng `class_id`[cite: 3]. | Kiểm tra độ ôm sát của khung (IoU), phát hiện box thừa/thiếu và kiểm tra việc gán đúng nhãn[cite: 3]. |
| **Instance segmentation** | Tập tọa độ đa giác `[(x1, y1), (x2, y2), ...]` kèm `instance_id` và `class_id`[cite: 5]. | Viền mask chập chạp, lem sang vùng nền hoặc không phân tách rõ ranh giới giữa các instance sát nhau[cite: 5]. | Chấm các điểm polygon bám chi tiết theo đường biên vật thể, gán đúng `instance_id` riêng[cite: 5]. | Kiểm tra độ chính xác cấp độ pixel của đường biên, đảm bảo không trùng lấn mask giữa các instance[cite: 5]. |

---

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

* **Một quy tắc bảo vệ dữ liệu:** Không chia sẻ, trích xuất hay phát tán dữ liệu dự án ra khỏi môi trường quy định; tuyệt đối không đính kèm thông tin định danh cá nhân (PII như Họ tên, MSSV, Email, SĐT) vào báo cáo hoặc các kho lưu trữ (repository) công khai.
* **Báo cáo sự cố dữ liệu:** Nếu phát hiện ảnh hoặc dữ liệu nằm ngoài phạm vi công việc, tôi sẽ lập tức dừng thao tác và báo cho **Project Lead / Trưởng nhóm dự án** hoặc **Giảng viên hướng dẫn**.

---

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.


* [x] `classification_predictions.json`[cite: 2]
* [x] `detection_predictions.json`[cite: 3]
* [x] `segmentation_predictions.json`[cite: 5]
* [x] `IMAGE_ATTRIBUTION.md`[cite: 4]
* [x] `visuals/classification_top5.png`
* [x] `visuals/detection_predictions.png`
* [x] `visuals/segmentation_prediction.png`
* [x] Ô validation cuối notebook báo `PASS`.
* [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
