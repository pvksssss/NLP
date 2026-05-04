# CLAUDE.md

## YÊU CẦU BẮT BUỘC: TRẢ LỜI BẰNG TIẾNG VIỆT

Mọi câu trả lời, giải thích, và hướng dẫn phải được viết bằng **tiếng Việt có dấu**.

---

# Hướng dẫn hành vi Agent cho dự án NLP

## 1. Nguyên tắc cốt lõi: Suy nghĩ trước khi code

**Không đưa ra giả định. Không che giấu sự bối rối. Nêu rõ trade-offs.**

### Trước khi triển khai:
- Nêu rõ các giả định của bạn. Nếu không chắc chắn, hãy hỏi.
- Nếu có nhiều cách diễn giải, trình bày tất cả - đừng chọn âm thầm.
- Nếu có cách tiếp cận đơn giản hơn, nói ra. Phản bác khi cần thiết.
- Nếu có điều gì chưa rõ, dừng lại. Nêu rõ điều gì gây bối rối. Hỏi.

---

## 2. Ưu tiên sự đơn giản

**Tối thiểu hóa code. Không có tính năng thừa.**

- Không thêm tính năng ngoài yêu cầu.
- Không tạo abstraction cho code chỉ dùng một lần.
- Không có "tính linh hoạt" hay "khả năng cấu hình" không được yêu cầu.
- Không có xử lý lỗi cho kịch bản bất khả thi.
- Nếu viết 200 dòng mà có thể là 50 dòng, hãy viết lại.

**Tự hỏi:** "Một senior engineer có nói đây là quá phức tạp không?" Nếu có, đơn giản hóa.

---

## 3. Thay đổi có chọn lọc

**Chỉ đụng vào những gì cần thiết. Dọn dẹp chỉ code của mình.**

### Khi chỉnh sửa code hiện có:
- Không "cải thiện" code xung quanh, comments, hay formatting.
- Không refactor những thứ không hỏng.
- Giữ style hiện có, kể cả khi bạn sẽ làm khác.
- Nếu nhận thấy dead code không liên quan, đề cập - không xóa.

### Khi thay đổi tạo orphan code:
- Xóa imports/variables/functions mà THAY ĐỔI CỦA BẠN làm unused.
- Không xóa dead code có sẵn trừ khi được yêu cầu.

**Tiêu chuẩn:** Mỗi dòng thay đổi phải có nguồn gốc trực tiếp từ yêu cầu của người dùng.

---

## 4. Thực thi theo mục tiêu

**Định nghĩa tiêu chí thành công. Lặp cho đến khi xác minh.**

### Biến task thành mục tiêu kiểm chứng được:
- "Thêm validation" → "Viết tests cho input không hợp lệ, rồi làm cho chúng pass"
- "Fix bug" → "Viết test reproduce bug, rồi làm cho nó pass"
- "Refactor X" → "Đảm bảo tests pass trước và sau"

### Với multi-step tasks, nêu kế hoạch:
```
1. [Bước] → xác minh: [kiểm tra]
2. [Bước] → xác minh: [kiểm tra]
3. [Bước] → xác minh: [kiểm tra]
```

**Tiêu chí thành công rõ ràng** cho phép bạn lặp độc lập. Tiêu chí yếu ("làm cho hoạt động") cần xác minh liên tục.

---

## 5. Cấu trúc Project

### Thư mục chính:
```
NLP/
├── src/              # Source code Python
│   ├── data/        # Data loading, preprocessing
│   ├── models/      # Model wrappers
│   ├── training/    # Training logic
│   ├── evaluation/  # Metrics, evaluation
│   └── inference/   # Inference, translation
├── configs/         # Configuration files
├── scripts/         # Executable scripts
├── outputs/         # Training outputs
├── notebook/        # Jupyter notebooks
├── .agent/          # Agent instructions
└── .cursor/         # Cursor rules
```

---

## 6. Tiêu chuẩn Code

### Python:
- Tuân thủ PEP 8
- Type hints cho functions
- Docstrings cho classes và functions quan trọng

### Notebooks:
- Clear markdown headers
- Giải thích ngắn gọn mỗi cell
- Output có ý nghĩa

### Commits:
- Message ngắn gọn, rõ ràng
- Mô tả "why" không phải "what"

---

## 7. Đánh dấu tiến độ

Khi được yêu cầu, sử dụng TodoWrite để theo dõi:
- Tiến độ hiện tại
- Task tiếp theo
- Deadline

---

**Các hướng dẫn này hoạt động tốt nếu:** ít thay đổi không cần thiết trong diffs, ít rewrite do quá phức tạp, và câu hỏi làm rõ đến trước thay vì sau mistakes.
