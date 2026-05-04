# BÁO CÁO TỔNG HỢP

## Dự án NLP: Neural Machine Translation - Xây dựng hệ thống dịch đa ngôn ngữ

---

## 1. Tổng quan dự án

### 1.1 Mục tiêu
Xây dựng hệ thống dịch máy đa ngôn ngữ (Multilingual Neural Machine Translation) sử dụng mô hình mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`), có khả năng dịch giữa nhiều cặp ngôn ngữ khác nhau trong cùng một mô hình.

### 1.2 Dataset
- **Nguồn dữ liệu**: OPUS-100 (Helsinki-NLP/opus-100)
- **Các cặp ngôn ngữ**:

| Cặp | Nguồn | Đích | Mô tả |
|-----|-------|------|-------|
| en-vi | Tiếng Anh | Tiếng Việt | Phổ biến |
| en-fr | Tiếng Anh | Tiếng Pháp | Phổ biến |
| de-en | Tiếng Đức | Tiếng Anh | Phổ biến |

### 1.3 Mô hình sử dụng

| Mô hình | Loại | Tham số | Ngôn ngữ |
|---------|------|---------|----------|
| facebook/mbart-large-50-many-to-many-mmt | mBART-50 | ~610M | 50 ngôn ngữ |

### 1.4 Thống kê Dataset

| Cặp ngôn ngữ | Train | Validation | Test |
|-------------|-------|------------|------|
| en-vi       | 50,000 | 2,000 | 2,000 |
| en-fr       | 50,000 | 2,000 | 2,000 |
| de-en       | 50,000 | 2,000 | 2,000 |
| **Tổng**    | **150,000** | **6,000** | **6,000** |

---

## 2. Kiến trúc kỹ thuật

### 2.1 Mô hình mBART-50

**Đặc điểm:**
- Kiến trúc: Encoder-Decoder (Transformer-based)
- Pre-training: Denoising autoencoder trên 50 ngôn ngữ
- Language Codes: `en_XX`, `vi_VN`, `fr_XX`, `de_DE`, ...

**Cơ chế điều khiển ngôn ngữ:**
```python
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "vi_VN"
# Mô hình tự động sinh văn bản tiếng Việt
```

### 2.2 Balanced Loss

**Vấn đề:** Dữ liệu giữa các cặp ngôn ngữ thường mất cân bằng.

**Giải pháp:** Áp dụng trọng số theo cặp ngôn ngữ:

```python
# Công thức:
L_balanced = Σ(w_pair × L_sample) / Σ(w_pair)

# Trọng số:
w_pair = √(N_total / N_pair)
```

**Ý nghĩa:** Cặp ít dữ liệu được gán trọng số cao hơn, giúp cân bằng đóng góp gradient.

### 2.3 Chiến lược Sampling

| Chiến lược | Mô tả | Tham số |
|------------|-------|---------|
| Greedy | Chọn token có xác suất cao nhất | `num_beams=1, do_sample=False` |
| Beam Search | Giữ top-k beams, chọn beam có xác suất cao nhất | `num_beams=4` |
| Top-k | Lấy mẫu từ k token có xác suất cao nhất | `top_k=50, do_sample=True` |
| Top-p (Nucleus) | Lấy mẫu từ nucleus tokens (tổng xác suất >= p) | `top_p=0.9` |

---

## 3. Pipeline huấn luyện

```
+-----------------------------------------------------------------+
|                    PIPELINE HUẤN LUYỆN                          |
+-----------------------------------------------------------------+
|                                                                 |
|  +---------+    +--------------+    +-------------+             |
|  |  Data   |-->|  Preprocess  |-->|   Tokenize  |             |
|  | OPUS-100|    |  (Filter)    |    |  (mBART-50) |             |
|  +---------+    +--------------+    +-------------+             |
|                                              |                  |
|                                              |                  |
|  +---------+    +--------------+    +-------------+             |
|  |  Model  |<--|  Train Loop  |<--|    Batch    |             |
|  | mBART-50|    |   (Epoch)    |    |  (8 samples)|             |
|  +---------+    +--------------+    +-------------+             |
|         |               |                                       |
|         |               v                                       |
|         |        +--------------+                               |
|         |        | Balanced Loss|                               |
|         |        | (Pair Weight)|                               |
|         |        +--------------+                               |
|         v                                                     |
|  +--------------+                                             |
|  |  Evaluation  |                                             |
|  |   (SacreBLEU)|                                             |
|  +--------------+                                             |
|                                                                 |
+-----------------------------------------------------------------+
```

### 3.1 Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Learning rate | 3e-5 |
| Batch size | 8 |
| Max length | 128 |
| Epochs | 2 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient accumulation | 4 |
| FP16 | True |

### 3.2 Checkpoints

- **Best Model**: `outputs/best_model/` - Model có validation loss thấp nhất
- **Checkpoint**: `outputs/checkpoint-37500/` - Checkpoint tại step 37500

---

## 4. Kết quả đánh giá

### 4.1 Metrics sử dụng

| Metric | Mô tả | Ưu điểm |
|--------|-------|---------|
| **SacreBLEU** | N-gram overlap | Tiêu chuẩn công nghiệp |
| **chrF++** | Character n-gram F-score | Language-independent |

### 4.2 Kết quả đánh giá (Thực tế)

| Cặp ngôn ngữ | BLEU Score | chrF++ | Số mẫu |
|-------------|-----------|--------|---------|
| en-vi | 20.47 | 36.96 | 2,000 |
| en-fr | 30.07 | 54.41 | 2,000 |
| de-en | 32.53 | 52.34 | 2,000 |
| **Overall** | **29.00** | **51.14** | **6,000** |

### 4.3 So sánh chiến lược decode

| Strategy | BLEU | Ổn định | Đa dạng |
|----------|------|---------|---------|
| Greedy | Cao | Tốt | Thấp |
| Beam Search | Cao nhất | Tốt | Thấp |
| Top-k | Trung bình | Trung bình | Cao |
| Top-p | Trung bình | Trung bình | Cao |
| Top-k+p | Trung bình | Tốt | Cao |

---

## 5. Cấu trúc project

```
NLP/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Load OPUS-100
│   │   └── preprocessor.py    # Tokenize, padding
│   ├── models/
│   │   ├── __init__.py
│   │   └── mbart.py           # mBART wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # BalancedLossTrainer
│   │   └── config.py          # Training configs
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # BLEU, chrF++
│   │   └── evaluator.py       # NMTEvaluator class
│   └── inference/
│       ├── __init__.py
│       └── translator.py       # MultilingualTranslator
├── outputs/
│   ├── best_model/            # Best checkpoint
│   ├── checkpoint-37500/       # Intermediate checkpoint
│   └── eval/                  # Evaluation results
├── notebook/
│   └── Fine-tuning.ipynb
├── app.py                      # Streamlit demo
├── .agent/
│   └── CLAUDE.md
├── .cursor/
│   └── rules/
├── requirements.txt
└── README.md
```

---

## 6. Hướng phát triển (Advanced Topics)

### 6.1 Back-translation
```python
# 1. Dịch ngược: en -> de
de_text = model.translate(en_text, "en", "de")

# 2. Dịch xuôi: de -> en
en_pseudo = model.translate(de_text, "de", "en")

# 3. Trộn vào training: (en_text, en_pseudo) = parallel
```

### 6.2 Curriculum Learning
```python
# Ưu tiên cặp dễ trước, khó sau
easy_pairs = ["en-fr", "de-en"]
medium_pairs = ["en-vi"]
hard_pairs = ["en-vi"]  # với less data

# Epoch 1: easy_pairs
# Epoch 2: easy + medium
# Epoch 3: all pairs
```

### 6.3 Document-level Translation
- Thêm ngữ cảnh xung quanh (previous/next sentences)
- Đảm bảo nhất quán xuyên câu (pronouns, terminology)

---

## 7. Kết luận

### 7.1 Đạt được
- Hiểu và triển khai mBART-50 cho dịch đa ngôn ngữ
- Áp dụng Balanced Loss giải quyết mất cân bằng dữ liệu
- So sánh và đánh giá các chiến lược decode (Top-k, Top-p)
- Xây dựng pipeline hoàn chỉnh: Data -> Model -> Train -> Evaluate
- Demo ứng dụng với Streamlit

### 7.2 Hạn chế
- Chưa tích hợp COMET cho đánh giá semantic
- Chưa áp dụng Back-translation
- Chưa có curriculum learning

### 7.3 Đề xuất cải tiến
1. Tích hợp COMET/WMT metrics cho đánh giá toàn diện hơn
2. Áp dụng Back-translation cho cặp ít dữ liệu
3. Triển khai Curriculum Learning theo độ khó cặp ngôn ngữ
4. Thử nghiệm Knowledge Distillation cho inference nhanh hơn

---
