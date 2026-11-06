# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603552
- **Họ và tên:** Nguyễn Thị Thùy Dung
- **SEED** (in ở ô đầu notebook):2307

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Mô hình | Precision | Recall | F1 test (775) | F1 SUB (40) | Train (s) | Inference (tin/s cả lô) | p95 (ms/tin) | USD/1.000 tin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 0.990 | 0.841 | 0.909 | 0.947 | 12.365 | 7,672.070 | **0.271** | 0.000 |
| 2. BoW + MLP | 0.981 | 0.912 | 0.945 | 0.974 | 12.106 | 13,823.438 | **2.326** | 0.000 |
| 3a. MiniLM + kNN | 0.907 | 0.947 | 0.926 | 0.974 | 78.133 | 22.161 | 71.466 | 0.000 |
| 3b. MiniLM + LogReg | 0.955 | 0.947 | 0.951 | **1.000** | 78.773 | 24.073 | **37.491** | 0.000 |
| 3c. MiniLM + MLP head | 0.991 | 0.938 | **0.964** | 0.974 | 79.134 | 28.453 | **41.326** | 0.000 |
| 4a. GPT-6 Luna zero-shot | 0.000* | 0.000* | 0.000* | 0.000* | 0.000 | 1.066 | 1,418.362 | 0.007† |
| 4b. GPT-6.1 Terra zero-shot | 0.000* | 0.000* | 0.000* | 0.000* | 0.000 | 1.127 | 1,132.124 | 0.073† |
| 4c. GPT-6 Luna few-shot | 0.000* | 0.000* | 0.000* | 0.000* | 0.000 | 1.256 | 1,150.242 | 0.017† |

* Không diễn giải F1 của LLM thành năng lực thực của model: các thử nghiệm 4a, 4b và 4c đều ghi nhận 40/40 nhãn invalid. Pipeline đã chuyển nhãn không hợp lệ thành ham, dẫn đến F1 lớp spam bằng 0. Hàng LLM chỉ đánh giá trên SUB (40 mẫu), không phải toàn bộ 775 test.
† Chi phí do hàm thực hành ước tính từ số từ/token, chưa phải usage được nhà cung cấp xác nhận. USD/1.000 tin bằng 0 đối với các model chạy cục bộ là không có phí API được hạch toán, chứ không phải tổng chi phí hạ tầng bằng 0.

Nhận xét từ dữ liệu đã đo: 
Trên toàn bộ tập test 775 tin, MiniLM + MLP head đạt F1 cao nhất (0.964), còn trên SUB 40 tin, MiniLM + LogReg đạt F1 = 1.000. Hai kết luận không mâu thuẫn vì chúng dùng tập đánh giá khác nhau; SUB nhỏ nên kết quả có biến động lớn.

Theo biểu đồ Pareto sử dụng 1 − F1 trên SUB và p95, ba mô hình nằm trên đường Pareto là BoW + LogReg, BoW + MLP, MiniLM + LogReg. Bốn mô hình ML/DL đáp ứng yêu cầu p95 < 50 ms/tin trong lần đo: BoW + LogReg (0.271 ms), BoW + MLP (2.326 ms), MiniLM + LogReg (37.491 ms), MiniLM + MLP head (41.326 ms). MiniLM + kNN (71.466 ms) và các LLM đều vượt ngưỡng.

## 3. Dự đoán và thực tế

Dự đoán 1 – MLP so với Logistic Regression (thí nghiệm 2.A). Em dự đoán MLP có thể cho F1 ngang hoặc thấp hơn LogReg vì dữ liệu nhỏ và BoW đơn giản. Thực tế MLP tốt hơn, F1 test 0.945 so với 0.909. Validation loss thấp nhất ở epoch 14, trong khi model được đánh giá ở cuối 30 epoch; vì vậy cần phân biệt thời điểm tốt nhất trên validation với trọng số dùng để tính F1.

Dự đoán 2 – Bỏ ReLU (2.B). Em dự đoán F1 giảm đáng kể về khoảng 0.90–0.93. Thực tế, F1 giảm rất nhẹ từ 0.945 xuống 0.941, cho thấy một phép phân loại tuyến tính đã đủ mạnh với biểu diễn BoW trong bài toán này.

Dự đoán 3 – Model lớn, nhỏ và few-shot (4.A, 4.B). Em dự đoán model lớn hơn sẽ có F1 tốt hơn khoảng 0.02–0.05 và few-shot có thể cải thiện thêm 0.01–0.03. Tuy nhiên, cả ba cấu hình đều có invalid = 40/40, nên không thể kiểm định giả thuyết F1. Kết quả chi phí được notebook ước tính: model lớn zero-shot 0.073 USD/1.000 tin, model nhỏ zero-shot 0.007 USD/1.000 tin (xấp xỉ 10 lần theo số đã làm tròn), few-shot model nhỏ 0.017 USD/1.000 tin. Bài học là phải kiểm tra response thô, giới hạn output token và parser trước khi đánh giá mô hình.

## 4. Phân tích 50 case khó

Bộ stress test ban đầu có 50 case; em đã thêm 5 case tự viết trong MY_CASES, và sau đó chạy lại ô đánh giá. Output cuối cùng trong notebook phản ánh 55 case, dù tiêu đề và tên cột của code vẫn viết “50 case”. Vì vậy các số mới nhất dưới đây khác những ảnh kết quả 50 case đã gửi trước đó.
Mô hình	F1 SUB thường	F1 stress test sau khi bổ sung 5 case	Accuracy stress test

| Mô hình | F1 SUB thường | F1 stress test sau khi bổ sung 5 case | Accuracy stress test |
|---|---:|---:|---:|
| BoW + LogReg | 0.947 | 0.308 | 0.509 |
| BoW + MLP | 0.974 | 0.409 | 0.527 |
| MiniLM + kNN | 0.974 | **0.721** | **0.691** |
| MiniLM + LogReg | 1.000 | 0.655 | 0.655 |
| MiniLM + MLP head | 0.974 | 0.655 | 0.655 |
| LLM zero-shot | 0.000* | 0.000* | 0.418* |
| LLM + prompt phòng thủ | 0.000* | 0.000* | 0.418* |

Trong các pipeline ML/DL cho kết quả hợp lệ, BoW + LogReg tụt F1 nhiều nhất, từ 0.947 xuống 0.308, giảm 0.639 điểm F1. MiniLM + kNN dẫn đầu stress test, nhưng accuracy 69.1% vẫn chưa đủ để kết luận phù hợp triển khai thực tế.

Hai case điển hình:
- Case #12, spam giọng tự nhiên: “hi there, sorry to bother you, my friend gave me your number, I'm new in town ... check my profile at meetlocal.club”. Nội dung mở đầu như một cuộc trò chuyện lịch sự nhưng hướng người nhận sang trang không mong muốn; không có tín hiệu spam từ khóa rõ rệt. Đây là một trong 5 case mà toàn bộ pipeline hiện tại xử lý sai.
- Case #14, spam giọng tự nhiên: “Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back”. Cách diễn đạt giống kể chuyện cá nhân nhưng chứa mồi chào việc làm thu nhập cao; mô hình dễ đánh giá nhầm là ham.
Theo bảng accuracy từng nhóm của lần chạy cuối, spam giọng tự nhiên là nhóm khó nhất với BoW (cả hai chỉ đúng 9%); MiniLM + kNN đúng 36% ở nhóm này. LLM và prompt phòng thủ cho kết quả giống nhau tại nhiều nhóm, nhưng vì vấn đề invalid nhãn, không thể kết luận prompt phòng thủ có hoặc không có tác dụng.

Bộ case khó được xây dựng có chủ đích; nên dùng làm stress test và phân tích lỗi, không dùng làm đại diện duy nhất cho phân phối spam thực tế.

## 5. Thử thách cá nhân

Đề theo SEED 2307 – Thử thách #4: Chạy knn_predict với k = 1, 5, 25 và giải thích vì sao k quá nhỏ hoặc quá lớn có thể bất lợi.

| k | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.9703 | 0.8879 | 0.9115 | 0.8996 |
| **5** | **0.9781** | **0.9068** | **0.9469** | **0.9264** |
| 25 | 0.9639 | 0.8400 | 0.9292 | 0.8824 |

Trong ba giá trị đã thử, k = 5 đạt F1 cao nhất (0.9264). k = 1 phụ thuộc mạnh vào một hàng xóm và dễ nhạy với nhiễu/nhãn sai; k = 25 dễ đưa vào nhiều điểm kém tương đồng và chịu ảnh hưởng của lớp ham chiếm đa số. Lựa chọn k chuẩn nên dựa trên validation, rồi đánh giá test sau cùng.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày, tải trung bình khoảng 116 SMS/giây, SLA p95 < 50 ms/tin.

Phương án đề xuất: Dùng BoW + MLP làm tầng phân loại nhanh mặc định. Kết quả notebook: F1 test 0.945, F1 SUB 0.974, p95 2.326 ms, tốc độ batch 13,823 tin/s trong điều kiện Colab. Điều này tạo dư địa độ trễ lớn, nhưng không đảm bảo throughput hoặc SLA khi triển khai production. Cần kiểm thử tải cao điểm, phần cứng, hàng đợi và các bước tiền xử lý.

Bổ sung tầng 2 MiniLM + LogReg để xử lý các tin không chắc chắn: F1 test 0.951, F1 SUB 1.000, p95 37.491 ms trong thí nghiệm, cao hơn F1 test BoW + MLP. Router đề xuất chuyển các tin có xác suất spam nằm trong [0.3, 0.7] sang tầng 2; các ngưỡng cần được hiệu chỉnh bằng validation và hiệu chuẩn xác suất. Độ trễ tổng hai tầng cần đo lại để đảm bảo end-to-end p95 < 50 ms. Các mẫu nghi vấn có thể được đưa vào kiểm tra bất đồng bộ nếu vượt SLA.

Đánh đổi: BoW + MLP triển khai rẻ/nhanh nhưng yếu trên tin spam giọng tự nhiên; MiniLM cung cấp thêm ngữ nghĩa nhưng tăng thời gian encode và tiêu thụ tài nguyên. Không nên đưa LLM API vào đường xử lý đồng bộ mặc định lúc pipeline hiện tại còn invalid 100% và p95 đều vượt 1.100 ms.

Ước tính chi phí LLM theo giả định của bài tính tay, không phải usage thực tế: 10 triệu SMS/ngày × (100 input token × 0.10 USD/1M + 1 output token × 0.50 USD/1M) = 105 USD/ngày, tương đương 3.150 USD/30 ngày. Chưa tính prompt dài hơn, token suy luận, sai số giá, hạ tầng và chi phí mạng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
|Phần 2 – ReLU|Vì sao bỏ ReLU thì hai lớp Linear tương đương mô hình tuyến tính?|Giải thích phép biến đổi tuyến tính và vai trò của activation function.|Chạy thí nghiệm: F1 giảm từ 0.945 xuống 0.941, sau đó đối chiếu với dự đoán.|
| Phần 3 – Embedding | Hướng dẫn xây dựng Cosine Similarity, top-k và kNN. | AI cung cấp công thức cosine, cách chuẩn hóa vector và bỏ phiếu đa số. | Kiểm thử TODO 3.1–3.3 PASS và đánh giá trên tập test. |
| | | | |

Điều học được: Thứ nhất, điểm F1 trên test cùng phân phối có thể quá lạc quan so với các tin nhắn cố tình ngụy trang; vì vậy cần stress test và phân tích theo nhóm lỗi. Thứ hai, model lớn hoặc kiến trúc phức tạp hơn không bảo đảm cải thiện kết quả; nhất là khi pipeline parse đầu ra bị lỗi, F1 có thể bằng 0 dù chưa đánh giá được năng lực thực tế của model. Thứ ba, lựa chọn giải pháp cần cân nhắc đồng thời F1, p95, chi phí, tài nguyên tính toán và khả năng chạy ở tải production.