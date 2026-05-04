# Multilingual Neural Machine Translation

> Xây dựng hệ thống dịch máy đa ngôn ngữ sử dụng mBART-50

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.36+-orange.svg)

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Cấu trúc Project](#cấu-trúc-project)
- [Sử dụng](#sử-dụng)
- [Dataset](#dataset)
- [Mô hình](#mô-hình)
- [Kết quả](#kết-quả)

---

## Giới thiệu

Dự án NLP - Xây dựng hệ thống dịch máy đa ngôn ngữ (Multilingual Neural Machine Translation) sử dụng mô hình mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`).

Mô hình có khả năng dịch giữa 3 cặp ngôn ngữ:
- English -> Vietnamese (en-vi)
- English -> French (en-fr)
- German -> English (de-en)

## Tính năng

### Đã hoàn thành
- [x] Fine-tune mBART-50 trên dataset OPUS-100
- [x] Balanced Loss theo cặp ngôn ngữ
- [x] So sánh chiến lược decode (Greedy, Beam Search, Top-k, Top-p)
- [x] Đánh giá bằng SacreBLEU và chrF++

### Đang phát triển
- [ ] Tích hợp COMET cho đánh giá semantic
- [ ] Back-translation cho cặp ít dữ liệu
- [ ] Curriculum Learning
- [ ] Knowledge Distillation

---

## Cài đặt

### Yêu cầu
- Python 3.9+
- PyTorch 2.0+
- CUDA (khuyến nghị cho training)

### Các bước cài đặt

```bash
# Clone repository
git clone <repo-url>
cd NLP

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt trực tiếp
pip install torch transformers datasets accelerate evaluate sacrebleu matplotlib pandas
```

### Dataset

Sử dụng **OPUS-100** từ Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-vi")
```

---

## Cấu trúc Project

```
NLP/
├── src/                          # Source code Python
│   ├── data/
│   │   └── loader.py            # Load dataset từ OPUS-100
│   ├── models/
│   │   └── mbart.py             # mBART-50 wrapper
│   ├── training/
│   │   ├── trainer.py           # BalancedLossSeq2SeqTrainer
│   │   └── config.py            # Training configs
│   ├── evaluation/
│   │   ├── metrics.py           # BLEU, chrF++
│   │   └── evaluator.py         # NMTEvaluator class
│   └── inference/
│       └── translator.py         # MultilingualTranslator, OpusMTTranslator
├── scripts/
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
├── outputs/                      # Training outputs, checkpoints
│   ├── best_model/              # Best model checkpoint
│   ├── checkpoint-37500/        # Intermediate checkpoint (step 37500)
│   └── eval/                    # Evaluation results
├── notebook/
│   └── Fine-tuning.ipynb        # Fine-tuning notebook
├── app.py                        # Streamlit demo app
├── .agent/
│   └── CLAUDE.md                # Agent instructions
├── .cursor/
│   └── rules/                   # Cursor IDE rules
├── requirements.txt
└── README.md                    # (file này)
```

---

## Sử dụng

### 1. Huấn luyện mô hình

**Sử dụng Notebook:**
```bash
jupyter notebook notebook/Fine-tuning.ipynb
```

### 2. Dịch văn bản

**Python API:**
```python
from src.inference.translator import MultilingualTranslator

# Khởi tạo translator
translator = MultilingualTranslator("outputs/best_model")

# Dịch đơn
result = translator.translate("Hello, how are you?", src_lang="en", tgt_lang="vi")
print(result)

# Dịch nhiều câu
translations = translator.translate_batch(
    ["Hello", "Goodbye"],
    src_lang="en",
    tgt_lang="fr"
)
```

**Sử dụng Streamlit Demo:**
```bash
streamlit run app.py
```

### 3. Đánh giá mô hình

```python
from src.evaluation.evaluator import NMTEvaluator

evaluator = NMTEvaluator(model, tokenizer)
results = evaluator.evaluate(test_dataset)
print(results.summary())
```

---

## Dataset

### OPUS-100
- **Nguồn**: Helsinki-NLP/opus-100
- **Ngôn ngữ**: 100 ngôn ngữ
- **Cặp**: 200+ cặp song ngữ
- **Nguồn gốc**: Open Subtitles, EU documents, GNOME, Ubuntu, Wikipedia

### Thống kê dataset sử dụng

| Cặp ngôn ngữ | Số mẫu train | Số mẫu val | Số mẫu test |
|-------------|------------|------------|------------|
| en-vi       | 50,000     | 2,000      | 2,000      |
| en-fr       | 50,000     | 2,000      | 2,000      |
| de-en       | 50,000     | 2,000      | 2,000      |
| **Tổng**    | **150,000**| **6,000**  | **6,000**  |

---

## Mô hình

### mBART-50
- **Kiến trúc**: Transformer Encoder-Decoder
- **Tham số**: ~610 triệu
- **Ngôn ngữ**: 50 ngôn ngữ
- **Pre-training**: Denoising autoencoder trên CCRL
- **Checkpoint**: `facebook/mbart-large-50-many-to-many-mmt`

### Balanced Loss

Giải quyết vấn đề mất cân bằng dữ liệu giữa các cặp ngôn ngữ:

```python
# Công thức:
L_balanced = Σ(w_pair × L_sample) / Σ(w_pair)

# Trọng số:
w_pair = √(N_total / N_pair)
```

### Chiến lược Decode

| Chiến lược | Mô tả | Ưu điểm |
|------------|-------|---------|
| Greedy | Chọn token có xác suất cao nhất | Nhanh |
| Beam Search | Giữ top-k beams | Chất lượng cao |
| Top-k | Lấy mẫu từ k token cao nhất | Đa dạng |
| Top-p | Lấy mẫu từ nucleus tokens | Cân bằng |

### Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Learning rate | 3e-5 |
| Batch size | 8 |
| Max length | 128 |
| Epochs | 2 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient accumulation | 4 |

---

## Kết quả

### SacreBLEU & chrF++ Scores

| Cặp ngôn ngữ | BLEU Score | chrF++ | Số mẫu test |
|-------------|-----------|--------|-------------|
| en-vi       | 20.47     | 36.96  | 2,000       |
| en-fr       | 30.07     | 54.41  | 2,000       |
| de-en       | 32.53     | 52.34  | 2,000       |
| **Overall** | **29.00** | **51.14** | **6,000** |

### So sánh chiến lược Decode

| Strategy | BLEU | Ổn định | Tốc độ |
|----------|------|---------|--------|
| Greedy | Cao | Tốt | Nhanh |
| Beam (k=4) | Cao nhất | Tốt | Trung bình |
| Top-k (k=50) | Trung bình | Trung bình | Chậm |
| Top-p (p=0.9) | Trung bình | Trung bình | Chậm |
| Top-k+p | Trung bình | Tốt | Chậm |

---

## Tài liệu tham khảo

1. **mBART**: Multilingual Denoising Pre-training for Neural Machine Translation
   - Liu et al., 2020

2. **BART**: Denoising Sequence-to-Sequence Pre-training
   - Lewis et al., 2019

3. **T5**: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
   - Raffel et al., 2019

4. **OPUS-100**: 100 Languages, 1 Model, 200+ Language Pairs
   - Tumanyan et al., 2023
