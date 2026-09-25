# Task: FedICL-MQA — Federated In-Context Learning cho MedQA với Qwen3-0.6B

## 1. Mục tiêu
Hiện thực framework **FedICL-MQA** theo sơ đồ kiến trúc (`1-FedICL-MQA-Overview of the proposed method-Jun7.png`) và so sánh nó với các baseline Centralized / non-ICL. Model là **Qwen3-0.6B**, fine-tune bằng LoRA adapter trên **MedQA**. Mọi tham số điều khiển qua file `.yaml`, không hard-code.

**Ánh xạ kiến trúc → plan:**

| Bước trong kiến trúc | Hiện thực trong plan | Mục |
|----------------------|----------------------|-----|
| 1. Question Encoding | Encoder câu hỏi y khoa → vector `h_q` | 3.6 |
| 2. Adaptive Demonstration Retrieval (từ Local Repository `E_i`) | Semantic + Clinical Relevance + Diversity (MMR), Top-K = 3, chỉ trong repository cục bộ | 3.2, 3.6 |
| 3. In-Context Prompt Construction | Prompt builder dùng chung cho train và eval | 4.1 |
| 4. SLM Reasoning (Local Inference) → `â_i` | Qwen3-0.6B + LoRA sinh đáp án dạng text, sau đó so khớp với option | 5.1 |
| Local Training Objective `L_i = L_QA + L_ICL + L_REG` | Loss 3 thành phần | 4.2 |
| Privacy-aware Retrieval | Demo (cả lúc train lẫn lúc eval) chỉ lấy từ dữ liệu của chính client | 3.2 |
| 5. Federated Aggregation Server | FedAvg trên trọng số LoRA, phân phối model global | 4 |

**Ngoài phạm vi (ghi rõ trong phần Limitations của bài):**
- Các SLM khác (Phi-3, Gemma, TinyLlama). Dùng một model duy nhất để có kết quả nhanh.
- PubMedQA.
- Differential Privacy, Secure Aggregation, Encrypted Channel. FL chạy dạng mô phỏng.
- Tiêu chí Recency/Authority, vì MedQA không có timestamp hay nguồn cho từng câu.
- BLEU, ROUGE, BERTScore.

## 2. Ma trận thí nghiệm

| Chiều        | Giá trị                                                        |
|--------------|----------------------------------------------------------------|
| Dataset      | MedQA (USMLE, 4 options, English)                              |
| Model        | `Qwen/Qwen3-0.6B` (duy nhất)                                   |
| Setting      | `centralized`, `federated`                                     |
| Prompt mode  | `icl` (k_shot = 3, adaptive retrieval), `non_icl` (k_shot = 0) |
| Checkpoint   | Centralized: epoch 1 → 3; FL: round 1 → 3 (cả hai có early stopping) |

→ 1 model × 2 setting × 2 mode = **4 arm**. Mỗi arm dùng các thành phần như sau:

| Arm | Vai trò trong bài | Truy xuất demo | Loss | Aggregation |
|-----|-------------------|----------------|------|-------------|
| `centralized_non_icl` | Baseline: Centralized SLM | — | `L_QA` | — |
| `centralized_icl` | Ablation: FedICL-MQA w/o FL | adaptive, pool = `train` | `L_QA + λ_ICL·L_ICL` | — |
| `federated_non_icl` | Ablation: FedICL-MQA w/o ICL | — | `L_QA + λ_REG·L_REG` | FedAvg |
| `federated_icl` | **Phương pháp đề xuất: FedICL-MQA** | adaptive, pool = `client_i` | `L_QA + λ_ICL·L_ICL + λ_REG·L_REG` | FedAvg |

Mỗi arm có tối đa 3 checkpoint (epoch hoặc round) cộng 1 checkpoint `best`. Centralized và FL có cùng ngân sách train: tối đa 3 lượt qua `train`.

**Nguyên tắc "train sao eval vậy":** run ICL thì train *và* eval đều dùng prompt có 3 demo lấy bằng cùng một chiến lược truy xuất. Run non-ICL thì cả train và eval đều zero-shot. Không trộn format.

**Ablation có sẵn trong config nhưng không chạy mặc định:**
- `retrieval.strategy`: `random` | `semantic` | `semantic_clinical` | `adaptive`, ứng với mục "Impact of Retrieval Strategy".
- `icl.k_shot`, ứng với mục "Impact of Demonstration Number".
- `loss.icl.weight = 0` và `loss.reg.mu = 0`, để đo đóng góp của từng thành phần loss.

**Lưu ý model:** Qwen3-0.6B là model hybrid-thinking (không có bản `-Instruct` riêng). Gọi chat template với `enable_thinking=False` ở cả train và eval để model trả thẳng nội dung đáp án (dạng text, xem 5.1), không sinh `<think>`.

## 3. Dữ liệu

### 3.1 Chuẩn hoá (generate lại structure)
Có hai tầng data, cùng 3 split `train` / `validation` / `test` của MedQA gốc (không tự chia lại):

- **`raw/`**: MedQA gốc chỉ đổi định dạng sang CSV, không sửa nội dung. Có tất cả các cột dưới đây trừ `q_hash`.
- **`centralized/`**: bản chuẩn hoá mà mọi run đọc. Được tạo từ `raw/` bằng cách thêm `q_hash` và kiểm tra tính hợp lệ: `answer` thuộc A–D, không option nào rỗng, `id` là duy nhất. Dòng lỗi bị loại và được đếm trong `dataset_summary.csv`. Câu trùng *không* bị xoá; chúng được xử lý ở 3.3.

Cột của `centralized/*.csv` (cũng là cột của mọi file client):

| Cột | Ví dụ | Ghi chú |
|-----|-------|---------|
| `id` | `medqa-train-00001` | duy nhất trong toàn dataset |
| `question` | `A 23-year-old ...` | |
| `option_A` … `option_D` | `...` | mỗi lựa chọn 1 cột, không nhét dict/JSON vào 1 ô |
| `answer` | `C` | chữ cái của option đúng *trong file*; chỉ dùng để lấy `gold_text`, không bao giờ đưa vào prompt |
| `meta` | `step1` | step1 / step2&3 |
| `q_hash` | `sha1(normalize(question))` | phát hiện câu trùng giữa query và demo (xem 3.3) |

- `normalize` = lowercase, gộp khoảng trắng, bỏ dấu câu ở hai đầu.
- Ghi CSV bằng UTF-8, có quoting chuẩn (câu hỏi MedQA chứa dấu phẩy, ngoặc kép và xuống dòng).
- Đọc CSV với `dtype=str, keep_default_na=False`. Nếu không, pandas sẽ biến các option có nội dung `"None"` / `"NA"` thành `NaN`.

### 3.2 Nguồn demo ICL: Local Demonstration Repository
Không tách pool riêng. Demo được truy xuất từ dữ liệu train, và toàn bộ train vẫn được dùng để fine-tune (cả run ICL lẫn non-ICL). **Mỗi client chỉ truy xuất từ repository của chính nó, cả lúc train lẫn lúc eval** (Privacy-aware Retrieval).

| Lúc | Setting | Câu query | Repository lấy demo |
|-----|---------|-----------|---------------------|
| Train | Centralized | mỗi câu trong `train` | `train` |
| Train | Federated | mỗi câu trong `client_i` | `client_i` |
| Eval  | Centralized | mỗi câu trong `validation` / `test` | `train` |
| Eval  | Federated | mỗi câu trong `validation` / `test`, **lặp lại cho từng client i** | `client_i` |

- **FL eval theo từng client:** model global được eval `k` lần, mỗi lần giả lập góc nhìn của một bệnh viện, tức demo lấy từ `client_i`. Báo cáo accuracy của từng client, trung bình macro và client kém nhất (5.1).
- **Chênh lệch giữa Centralized ICL và FedICL-MQA** gồm hai phần: model được train khác nhau, và repository nhỏ hơn / lệch hơn. Phần sau là "cái giá của quyền riêng tư", và được phân tích trong bài.
- **Chẩn đoán (tuỳ chọn, không phải kết quả FL):** `eval.fl_global_pool_diagnostic: true` eval thêm model global của FL với repository = toàn bộ `train`, để tách hai phần trên. Kết quả ghi với `demo_pool = train_diagnostic` và không bao giờ được báo cáo như kết quả của FL.

### 3.3 Chống trùng và rò rỉ đáp án giữa query và demo
"Trùng chính xác" được xác định theo `q_hash`, không theo `id`, vì MedQA có câu hỏi trùng mang id khác nhau. Truy xuất theo độ tương đồng lại **ưu tiên** chọn đúng các câu gần trùng, nên cần thêm bộ lọc gần trùng.

Quy tắc bắt buộc cho mỗi prompt:
1. Không demo nào có `q_hash` trùng với câu query (loại cả chính câu đó lẫn các bản sao của nó).
2. Không demo nào có `sem(q, d) ≥ retrieval.near_dup_threshold` (0.95). Đây là các câu viết lại gần như y hệt; nếu để lại, model có thể chép đáp án từ demo.
3. 3 demo có `q_hash` đôi một khác nhau.
4. Demo chỉ nằm trong repository đúng theo bảng 3.2. Với FL, demo của query thuộc client_i chỉ đến từ `client_i`.

**Assertion** (chạy sau khi gán demo, fail thì dừng pipeline): duyệt toàn bộ `demo_assignment_*.json` và kiểm tra 4 quy tắc trên. Kết quả ghi vào `processed_data/MedQA/centralized/demo_check_report.json`, gồm:
- số query đã kiểm tra;
- số demo bị loại theo từng quy tắc;
- số câu train trùng với validation/test;
- kích thước của từng repository;
- số câu không trích được thực thể y khoa nào (khi đó điểm clinical = 0).

### 3.4 Centralized
- Fine-tune trên toàn bộ `train`. Demo lấy theo bảng ở 3.2.
- Chọn checkpoint/tuning trên `validation`, báo cáo trên `test`.

### 3.5 Federated
- **Thí nghiệm chính:** chia `centralized/train.csv` thành **3 client** theo **Dirichlet non-IID, α = 0.5** → dùng `federated_noniid/alpha_0.5/k3/`.
- **Partition sinh sẵn:** script partition sinh luôn toàn bộ lưới dưới đây (rẻ, chỉ vài MB) để sau này ablation mà không phải chạy lại bước data:
  - `federated_iid/k{2,3,4,5}/`: xáo trộn ngẫu nhiên rồi chia đều.
  - `federated_noniid/alpha_{0.5,0.1}/k{2,3,4,5}/`: Dirichlet theo `data_prep.label_key` (mặc định `answer`; xem mục 7). α càng nhỏ thì càng lệch.
- Mọi partition dùng seed cố định (ghi trong `config.json`). Ở mỗi partition, hợp các client = toàn bộ `train` và các client rời nhau. Script assert cả hai điều này.
- `validation`/`test` không chia cho client. Model global được eval trên `centralized/validation.csv` và `centralized/test.csv`, với demo lấy từ repository của từng client (3.2).
- Mỗi thư mục `k{n}/` có `stats.csv` (1 dòng / client: `client_id, n_samples, answer_A..D, meta_step1, meta_step2&3`). Script partition cũng in bảng này ra console.

### 3.6 Adaptive Demonstration Retrieval (Top-K = 3)
Hiện thực bước 1 và bước 2 của kiến trúc. Mọi encoder đều **đóng băng**, và truy xuất là tất định, nên demo được **tính một lần** trước khi train và lưu ra file. Nhờ vậy demo cố định qua các epoch/round và tái lập được.

**Bước 1: Question Encoding.**
- Encoder: `NeuML/pubmedbert-base-embeddings` (sentence-transformers, mean pooling, tối đa 512 token), vector được chuẩn hoá L2.
- Đầu vào là phần `question` (ca bệnh), không kèm option.
- **Không dùng MedCPT-Query-Encoder** vì nó chỉ nhận 64 token và sẽ cắt mất phần lớn một ca bệnh MedQA.
- **BioBERT/ClinicalBERT** (như ghi trong hình) là model MLM thô, không được train để so sánh câu. Chúng chỉ được giữ làm lựa chọn ablation qua `retrieval.question_encoder`.
- Độ tương đồng ngữ nghĩa: `sem(q, d) = cos(h_q, h_d)`.

**Bước 2a: Clinical Relevance.**
- Trích thực thể y khoa của mỗi câu hỏi bằng scispaCy (`en_core_sci_sm`).
- Mã hoá từng thực thể bằng SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, lấy vector `[CLS]`, tối đa 25 token). SapBERT được train để gom các từ đồng nghĩa trong UMLS về gần nhau.
- `clin(q, d)` là F1 mềm giữa hai tập thực thể:
  - `R` = trung bình, trên mỗi thực thể của q, của cosine lớn nhất với một thực thể của d;
  - `P` = trung bình, trên mỗi thực thể của d, của cosine lớn nhất với một thực thể của q;
  - `clin = 2PR / (P + R)`.
- Nếu một trong hai câu không có thực thể nào thì `clin = 0`.
- Điểm liên quan tổng hợp: `rel(q, d) = w_sem·sem(q, d) + w_clin·clin(q, d)`, mặc định `w_sem = 0.7`, `w_clin = 0.3`.

**Bước 2b: Top-K Diverse Demonstrations.**
1. Lọc repository theo các quy tắc ở 3.3.
2. Lấy `candidate_top_n` = 50 ứng viên có `sem` cao nhất.
3. Tính `rel` cho 50 ứng viên đó.
4. Chọn lần lượt 3 demo bằng **MMR**: mỗi bước chọn `d* = argmax_{d ∉ S} [ λ_mmr·rel(q, d) − (1 − λ_mmr)·max_{s ∈ S} sem(d, s) ]`, với `λ_mmr = 0.7`. Nhờ vậy 3 demo vừa liên quan tới câu hỏi vừa không lặp lại nhau.
5. Nếu điểm bằng nhau thì chọn theo `rng(seed, query_id)`.

**Demo thay thế cho L_ICL** (chỉ lúc train, xem 4.2): với mỗi query train, lấy thêm `alt_demo_ids`, gồm 3 demo **ngẫu nhiên** từ cùng repository, cùng các quy tắc 3.3, không trùng với demo chính, cố định theo `rng(seed, query_id, "alt")`.

**Privacy trong FL:** client i chỉ xây index từ `client_i`. Khi mô phỏng, embedding của toàn bộ `train` được tính một lần rồi cắt theo client. Kết quả giống hệt việc mỗi client tự tính, vì encoder đóng băng và mã hoá từng câu độc lập. Tuy vậy, việc chọn ứng viên và chạy MMR luôn chỉ diễn ra trong repository của client đó.

**Chiến lược ablation** (`retrieval.strategy`):
- `random`: chọn ngẫu nhiên, vẫn áp dụng các quy tắc 3.3.
- `semantic`: top-3 theo `sem`.
- `semantic_clinical`: top-3 theo `rel`.
- `adaptive`: `rel` + MMR. Đây là mặc định.

**Output:**
- File demo nằm cạnh file data mà nó phục vụ (cấu trúc ở 3.7).
- Mỗi query có một record: `{demo_ids, alt_demo_ids (chỉ khi train), sem, clin, rel, mmr_score}`.
- Predictions lưu thêm `demo_ids` và `demo_pool`.

### 3.7 Cấu trúc thư mục data
Mỗi dataset có một thư mục riêng dưới `processed_data/`, để sau này thêm dataset text khác mà không phải đổi code. Bài toán là tạo sinh text thuần, không có ảnh.
```
processed_data/
│
├── MedQA/
│   ├── dataset_summary.csv                    # 1 dòng / (tầng × split), xem dưới
│   ├── config.json                            # metadata + seed + sha1 các file, xem dưới
│   │
│   ├── raw/                                   # MedQA gốc → CSV, không sửa nội dung
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   ├── centralized/                           # bản chuẩn hoá (thêm q_hash) mà mọi run đọc
│   │   ├── train.csv
│   │   ├── validation.csv                     # global eval set, dùng cho cả FL
│   │   ├── test.csv                           # global eval set, dùng cho cả FL
│   │   ├── features/{encoder_tag}/            # tính một lần, encoder đóng băng (3.6)
│   │   │   ├── question_emb_{train,validation,test}.npy   # + question_ids_{split}.csv (thứ tự dòng)
│   │   │   ├── entities_{train,validation,test}.jsonl     # id → danh sách thực thể scispaCy
│   │   │   └── entity_vocab.csv + entity_emb.npy          # SapBERT embedding của mỗi thực thể duy nhất
│   │   ├── demo_assignment_train.json         # ICL train Centralized: repo = train (có alt_demo_ids)
│   │   ├── demo_assignment_validation.json    # ICL eval Centralized: repo = train
│   │   ├── demo_assignment_test.json
│   │   └── demo_check_report.json
│   │
│   ├── federated_iid/                         # chia đều ngẫu nhiên từ centralized/train.csv
│   │   ├── k2/
│   │   │   ├── client_1.csv
│   │   │   ├── client_2.csv
│   │   │   └── stats.csv
│   │   ├── k3/
│   │   ├── k4/
│   │   └── k5/
│   │
│   ├── federated_noniid/                      # Dirichlet theo data_prep.label_key
│   │   ├── alpha_0.5/
│   │   │   ├── k2/
│   │   │   ├── k3/                            # ← partition của thí nghiệm chính
│   │   │   │   ├── client_1.csv
│   │   │   │   ├── client_2.csv
│   │   │   │   ├── client_3.csv
│   │   │   │   ├── demo_assignment_client_{1,2,3}_train.json        # repo = client_i (có alt_demo_ids)
│   │   │   │   ├── demo_assignment_client_{1,2,3}_validation.json   # eval global model, repo = client_i
│   │   │   │   ├── demo_assignment_client_{1,2,3}_test.json
│   │   │   │   └── stats.csv
│   │   │   ├── k4/
│   │   │   └── k5/
│   │   └── alpha_0.1/
│   │       ├── k2/
│   │       ├── k3/
│   │       ├── k4/
│   │       └── k5/
│   │
│   └── visualizations/
│       ├── split_distribution.png             # phân phối answer / meta theo split
│       ├── prompt_length_hist.png             # số token prompt ICL vs non-ICL → chọn max_seq_len
│       ├── retrieval_similarity.png           # phân phối sem/rel của demo được chọn: repo train vs từng client
│       ├── iid_k{2..5}_clients.png            # stacked bar phân phối nhãn theo client
│       └── noniid_alpha{0.5,0.1}_k{2..5}_clients.png
```

**`config.json`**: metadata cố định của dataset, khác với config thí nghiệm `configs/*.yaml`. Gồm:
- `name`, `source` (nguồn gốc và version của MedQA gốc), `language`;
- `num_options`, `option_letters`, `splits` (tên split → file), `columns`;
- `normalize_rule`, `hash_algo`;
- `partition` (`seed`, `label_key`, danh sách `k` và `alpha` đã sinh);
- `retrieval` (tên và revision của các encoder, `encoder_tag`);
- `created_at`, và `sha1` của từng file CSV để phát hiện data bị sửa sau khi đã train.

**`dataset_summary.csv`**: cột `layer, split, n_samples, n_dropped_invalid, n_unique_q_hash, n_dup_within_split, n_overlap_with_train, answer_A, answer_B, answer_C, answer_D, meta_step1, meta_step2&3, meta_unknown, avg_question_tokens, max_option_tokens, est_max_prompt_tokens_icl`, với `layer` là `raw` hoặc `centralized`. Các cột `*_tokens` đếm bằng tokenizer của Qwen3: `max_option_tokens` dùng để chọn `max_new_tokens`; `est_max_prompt_tokens_icl` là ước lượng trần (system + 4 × p99 độ dài một câu hỏi), còn độ dài prompt ICL thực tế (sau khi gán demo) nằm trong `visualizations/prompt_length_hist.png`. Thống kê thực thể (`avg_entities_per_question`, số câu không có thực thể) nằm trong `features/{encoder_tag}/feature_stats.json`.

**Nguồn dữ liệu:** `GBaker/MedQA-USMLE-4-options-hf` (split gốc 10178 / 1272 / 1273). Bản này không có `meta`; `meta` được ghép theo nội dung câu hỏi từ `GBaker/MedQA-USMLE-4-options`, vốn không có split validation, nên validation có `meta = unknown` (chỉ ảnh hưởng thống kê).

Quy ước:

- Mọi file CSV (`centralized/*.csv`, `client_*.csv`) có chung schema ở 3.1. Nhờ vậy một hàm `load_split(path)` đọc được tất cả.
- Mỗi script chỉ ghi vào phần của mình:
  - Script chuẩn hoá: `raw/`, `centralized/*.csv`, `config.json`, `dataset_summary.csv`.
  - Script partition: `federated_iid/`, `federated_noniid/`.
  - Script features: `centralized/features/`.
  - Script retrieval: các file `demo_assignment_*.json` và `demo_check_report.json`.
  - Script visualize: `visualizations/`.

  Không script nào sửa `raw/` hay `centralized/*.csv` sau khi đã tạo.
- Client đánh số từ 1 (`client_1` … `client_k`), nhất quán từ data tới outputs.
- Dữ liệu tải về gốc (HF hoặc file zip) không nằm trong `processed_data/`, mà để ở cache hoặc `downloads/`.

## 4. Training
- PEFT/LoRA; base model đóng băng, chỉ train và lưu adapter. Các encoder truy xuất (3.6) cũng đóng băng.
- **Centralized:** train tối đa **3 epoch** trên toàn bộ `train`, có early stopping (mục 4.3).
- **Federated:**
  - 3 client, tất cả tham gia mỗi round.
  - Mỗi client tối ưu `L_i` (4.2) trên dữ liệu cục bộ, xuất phát từ adapter global của round đó.
  - Server tổng hợp bằng **FedAvg** trên trọng số LoRA, có trọng số theo số mẫu của client, rồi phân phối adapter global mới cho mọi client.
  - Train tối đa **3 communication round**, `local_epochs` = 1 (1 round = 1 lượt qua dữ liệu, tương đương 1 epoch centralized). Có early stopping theo round (mục 4.4).
- Cố định seed (python/numpy/torch) cho mọi run.

### 4.1 In-Context Prompt Construction
Prompt builder dùng chung cho train và eval, cho cả 4 arm:
```
[system]  You are a helpful medical assistant. Answer the question by writing the full text of the correct option.
[user]    Example 1
          Question: {demo_1.question}
          Options:
          - {option}
          - {option}
          - {option}
          - {option}
          Answer: {demo_1.gold_text}
          ... (Example 2, 3)
          Target Question: {q.question}
          Options:
          - ...
          Answer:
[assistant] {q.gold_text}<EOS>          ← target lúc train, là thứ được sinh lúc eval
```
- **Option không có nhãn A–D** (`prompt.option_style: bullet`). Model không có chữ cái nào để chọn, nên buộc phải sinh nội dung đáp án. Việc này loại bỏ bias về chữ cái (5.1).
- **Thứ tự option được xáo trộn** theo `rng(seed, id)` (`prompt.shuffle_options: true`), áp dụng cho cả demo lẫn query. Hoán vị cố định cho từng câu qua mọi epoch/round, và được lưu vào predictions (`option_order`). Mục đích là loại bỏ bias về vị trí, vì MedQA gốc có phân phối vị trí đáp án không hoàn toàn đều. Bước so khớp dựa trên text nên xáo trộn không ảnh hưởng gì tới việc chấm điểm.
- Arm non-ICL dùng đúng prompt trên nhưng không có phần Example.

### 4.2 Local Training Objective
`L_i = L_QA + λ_ICL·L_ICL + λ_REG·L_REG`. Arm nào dùng thành phần nào xem bảng ở mục 2.

**`L_QA`: Answer Generation Loss.**
- Cross-entropy trên token của target = nội dung text của option đúng + EOS. Mask toàn bộ system, demo và question.
- Target không phải chữ cái. Có EOS thì model học được lúc nào dừng sinh.

**`L_ICL`: Context Consistency Loss.** Chỉ áp dụng cho arm ICL, mặc định `λ_ICL = 0.5`.
- Với mỗi mẫu train, chạy 2 forward pass cùng teacher forcing trên đáp án đúng:
  - Ngữ cảnh chính dùng `demo_ids` (demo được truy xuất). Forward này cũng dùng để tính `L_QA`.
  - Ngữ cảnh thay thế dùng `alt_demo_ids` (demo ngẫu nhiên, 3.6).
- `L_ICL` = trung bình trên các token đáp án của `KL( stopgrad(p_θ(·|D, q, a_<t)) ‖ p_θ(·|D_alt, q, a_<t) )`.
- Ý nghĩa: model được dạy trả lời *nhất quán* kể cả khi demo kém liên quan. Điều này quan trọng với FL non-IID, vì repository cục bộ nhỏ và lệch nên demo truy xuất được thường kém liên quan hơn so với repository toàn cục (RQ3).
- Biến thể thay thế (`loss.icl.variant: context_distill`): ngữ cảnh thay thế = không có demo. Rẻ hơn vì prompt ngắn hơn.
- **Chi phí:** arm ICL tốn khoảng 2 lần compute so với arm non-ICL (xem mục 7).

**`L_REG`: Regularization Loss.** Chỉ áp dụng cho arm FL, mặc định `μ = 0.01`.
- Proximal term kiểu FedProx: `L_REG = (μ/2)·‖θ_LoRA − θ_LoRA^(t)‖²`, trong đó `θ^(t)` là adapter global mà client nhận ở đầu round t.
- Tác dụng: giữ cập nhật cục bộ không trôi quá xa khỏi model global khi dữ liệu non-IID.
- Centralized không có model global nên `L_REG = 0`. Weight decay của optimizer thì giống nhau ở mọi arm.

**Log:** ghi riêng `loss_qa`, `loss_icl`, `loss_reg` và `loss_total` ở mỗi bước.

### 4.3 Early stopping: Centralized
- **Eval trên `validation` giữa epoch**, mỗi `eval_every` = 0.25 epoch (3 epoch cho ra 12 điểm kiểm tra). Chỉ kiểm tra ở cuối epoch thì quá thưa để bắt overfit.
- **Metric theo dõi:** `val_loss`, chỉ gồm `L_QA` trên token đáp án (không cộng `L_ICL` hay `L_REG`, để so sánh được giữa các arm), cùng prompt format với arm. Log thêm `val_accuracy` và `train_loss` ở mỗi điểm kiểm tra.
- **Điều kiện dừng:** `val_loss` không giảm quá `min_delta` = 0.001 so với giá trị tốt nhất trong `patience` = 2 lần eval liên tiếp. Dấu hiệu overfit điển hình là `train_loss` vẫn giảm trong khi `val_loss` tăng.
- Adapter cuối mỗi epoch đã hoàn thành vẫn được lưu và eval như bình thường. Nếu dừng giữa epoch thì không tạo checkpoint cho epoch đó, nhưng checkpoint giữa epoch có `val_loss` tốt nhất vẫn được lưu làm `best`.

### 4.4 Early stopping: Federated
- Model global chỉ tồn tại sau bước aggregate, nên **eval trên `validation` sau mỗi round**.
- **Metric theo dõi:**
  - Arm ICL: `val_loss` = trung bình macro, trên các client, của `val_loss` khi eval với demo của `client_i` (3.2).
  - Arm non-ICL: eval một lần duy nhất.
  - Log thêm `val_accuracy` (macro và của từng client) và `train_loss` trung bình có trọng số của các client.
- **Điều kiện dừng:** `val_loss` không cải thiện quá `min_delta` trong `patience` = 1 round. Với tối đa 3 round, patience = 2 thì gần như không bao giờ dừng sớm.

### 4.5 Quy tắc chung cho cả hai setting
- **Khi dừng hoặc hết ngân sách:**
  - Khôi phục checkpoint có `val_loss` tốt nhất và lưu vào `best/`.
  - Eval `best/` trên `validation` và `test`.
  - Ghi `early_stop.json` gồm `stopped`, `reason` (`overfit` hoặc `max_budget_reached`), `stop_at`, `best_at`, `best_val_loss`.
- **Nếu hết 3 epoch hoặc 3 round mà `val_loss` vẫn đang giảm:** ghi `reason: "max_budget_reached"` và in cảnh báo. Có thể resume từ checkpoint cuối để train tiếp tới `max_extend` = 5 mà không cần train lại từ đầu (mục 5.3).
- Tập `validation` chỉ dùng để chọn checkpoint. Kết quả báo cáo cuối cùng lấy trên `test`.
- Mọi tham số early stopping nằm trong yaml (`train.early_stopping`, `federated.early_stopping`).

## 5. Evaluation, lưu trữ & tái sử dụng kết quả

### 5.1 Metric
**Accuracy theo kiểu tạo sinh:**
1. Model sinh đáp án dưới dạng text tự do.
2. Text đó được so với nội dung của 4 option.
3. **Option giống nhất** được coi là lựa chọn của model.
4. So option đó với đáp án đúng.

Model không bao giờ được chọn A/B/C/D, nên không có bias về chữ cái. Vị trí option cũng đã được xáo trộn (4.1).

**Sinh:** greedy (`do_sample=False`), `max_new_tokens` = 64 (kiểm tra bằng cột `max_option_tokens` của `dataset_summary.csv`). Dừng ở EOS hoặc dòng mới đầu tiên.

**Ánh xạ text → option** (`match_answer(gen_text, options) -> (option_idx | None, match_type, score, margin)`):

1. **Chuẩn hoá** cả `gen_text` và text của từng option theo cùng một cách: lowercase, gộp khoảng trắng, bỏ dấu câu ở hai đầu, bỏ các tiền tố như `answer:`, `the answer is`.
2. **`exact`**: text đã chuẩn hoá trùng khớp với đúng 1 option → chọn option đó.
3. **`contains`**: text chứa trọn nội dung của ≥ 1 option. Bỏ các option là chuỗi con của option khác cũng được chứa (tránh trường hợp `aspirin` khớp nhầm khi đáp án là `aspirin and clopidogrel`). Nếu còn đúng 1 option → chọn option đó. Nếu còn ≥ 2 option không lồng nhau (vd `aspirin or heparin`) → câu trả lời nước đôi: chuyển sang bước `nearest` và luôn gắn `low_confidence`, không bao giờ chọn theo vị trí.
4. **`nearest`**: với mỗi option j, tính `score_j = w_lex·tokenF1(gen, opt_j) + w_sem·cos(SapBERT(gen), SapBERT(opt_j))`, mặc định `w_lex = w_sem = 0.5`. **Luôn chọn `argmax_j`**, tức option giống nhất.
   - Dùng hai tín hiệu vì chúng bù cho nhau. Token-F1 phân biệt được `hypothyroidism` với `hyperthyroidism`, là cặp mà embedding hay nhầm. SapBERT nhận ra từ đồng nghĩa không có từ nào chung, ví dụ `low thyroid hormone` với `hypothyroidism`.
5. **`empty`**: text rỗng sau khi chuẩn hoá → `pred = None`, tính là **sai**.

Cờ **`low_confidence`**: đặt khi `nearest` có `score < 0.5` hoặc cách biệt với option thứ hai `< 0.05`. Cờ này chỉ dùng để chẩn đoán, không làm thay đổi `pred`.

**Metric:**

- `accuracy` (**metric chính**) = số câu có `pred == gold` / **tổng số câu**.
- `accuracy_strict` = chỉ tính đúng các câu khớp ở bước `exact` / `contains`; câu `nearest` bị tính sai. Đây là kiểm tra độ vững của metric: nếu hai con số chênh nhau nhiều thì model hay "nói gần đúng" chứ chưa chép đúng đáp án.
- Log thêm tỉ lệ của từng `match_type` (`exact_rate`, `contains_rate`, `nearest_rate`, `empty_rate`) và `low_conf_rate`.
- **FL arm ICL:** báo cáo `accuracy` của từng client (`demo_pool = client_i`), `macro_mean` (**số chính của FL**), `min` (client kém nhất) và `std`.
- Cấu hình `eval.match` giống nhau ở mọi arm và được ghi vào `run_meta.json`. Script eval-only (5.3) có thể chấm lại predictions đã lưu với cấu hình khác mà không cần sinh lại.
- Mỗi checkpoint (epoch, round, `best`) đều được eval trên cả `validation` và `test`, và kết quả được lưu ngay sau khi eval.

### 5.2 Cấu trúc lưu trữ
**Nguyên tắc:** mọi thứ cần thiết để eval lại, phân tích lại hoặc train tiếp đều được ghi ra đĩa ngay khi tạo xong. Không có kết quả nào chỉ nằm trong RAM hoặc log console.

```
outputs/qwen3-0.6b/seed{seed}/
  runs_index.csv                        # 1 dòng/arm: arm, status, path, best_at, best_test_acc, started_at, finished_at
  summary.csv                           # gộp result.csv của mọi arm
  {arm}/                                # centralized_icl | centralized_non_icl | federated_icl | federated_non_icl
    config.yaml                         # config đã resolve của arm
    run_meta.json                       # status (running|completed|failed), config_hash, seed, git_commit,
                                        #   versions (torch/transformers/peft), GPU, thời gian bắt đầu/kết thúc,
                                        #   eval.match, sha1 của các file demo_assignment đã dùng
    result.csv                          # 1 dòng / (checkpoint × split × demo_pool)
    early_stop.json
    logs/
      train_log.csv                     # step, epoch|round, client_id, loss_qa, loss_icl, loss_reg, loss_total, lr, grad_norm
      val_curve.csv                     # mỗi điểm eval validation: step|round, demo_pool, val_loss, val_accuracy, train_loss
    # --- Centralized ---
    checkpoints/
      epoch_{1..3}/
        adapter/                        # adapter_model.safetensors + adapter_config.json
        trainer_state/                  # optimizer, scheduler, RNG, global_step → để resume
        predictions_{validation,test}.jsonl
        metrics.json
      step_{n}/                         # checkpoint giữa epoch (chỉ giữ cái đang best + K cái gần nhất)
    # --- Federated ---
    rounds/
      round_{1..3}/
        global_adapter/                 # adapter global sau khi aggregate
        clients/client_{1,2,3}/
          adapter/                      # adapter cục bộ TRƯỚC khi aggregate
          train_log.csv
          metrics.json                  # n_samples, loss_qa/icl/reg, thời gian train
        aggregation.json                # trọng số FedAvg của từng client, số mẫu
        server_state/                   # trạng thái server + RNG → để resume round kế tiếp
        predictions_{validation,test}_client_{1,2,3}.jsonl   # arm ICL: eval global model với repo client_i
        predictions_{validation,test}.jsonl                  # arm non-ICL: eval một lần
        metrics.json
    best/
      adapter/                          # bản copy của checkpoint tốt nhất
      pointer.json                      # checkpoint gốc (vd "checkpoints/epoch_2" hoặc "rounds/round_2")
      predictions_*.jsonl
      metrics.json
```
- Mỗi dòng `predictions_*.jsonl` gồm `id, raw_output, pred, pred_text, match_type, match_score, margin, low_confidence, gold, gold_text, correct, demo_ids, demo_pool, option_order`.
- Cột của `result.csv`: `arm, model, setting, mode, k_shot, retrieval_strategy, seed, checkpoint, epoch_or_round, split, demo_pool, accuracy, accuracy_strict, val_loss, n_samples, exact_rate, contains_rate, nearest_rate, empty_rate, low_conf_rate, is_best, adapter_path, timestamp`.
  - `demo_pool` ∈ `train` | `client_{i}` | `macro_mean` | `min` | `none` (non-ICL) | `train_diagnostic`.
  - `checkpoint` có dạng `epoch_2`, `round_3` hoặc `best`.
  - `adapter_path` là đường dẫn tương đối tới adapter để load lại trực tiếp.
- **Không bao giờ xoá adapter** (mỗi adapter LoRA chỉ vài MB). Riêng `trainer_state/` và `server_state/` (chứa optimizer, dung lượng lớn) được giữ cho `save.keep_last_k_states` = 2 checkpoint gần nhất cùng checkpoint best. Adapter, predictions và metrics luôn được giữ đủ.
- Ghi file theo kiểu atomic: ghi ra `*.tmp` rồi rename, để nếu run bị kill giữa chừng thì không để lại file hỏng.

### 5.3 Resume, không ghi đè, và dùng lại kết quả
- **Không ghi đè:** nếu arm đã có `run_meta.json` với `status: completed` thì bỏ qua arm đó. Chỉ chạy lại khi có `--overwrite`, và khi đó thư mục cũ được chuyển sang `_archive/{arm}_{timestamp}/`, không bị xoá.
- **Resume:** nếu `status: running` hoặc `failed`, pipeline tiếp tục từ epoch hoặc round cuối cùng đã lưu đủ. Checkpoint đã có thì không train lại.
- **Kiểm tra config:** khi resume, so `config_hash` với config hiện tại. Nếu khác thì dừng và báo lỗi, trừ khi có `--allow_config_change`.
- **Chỉ eval:** `python -m fedicl.eval --arm {arm} --checkpoint {epoch_2|round_3|best}` (hoặc `--adapter_path ...`) load adapter có sẵn, eval lại mà không train, và thêm dòng mới vào `result.csv`.
- **Chấm lại không sinh lại:** `python -m fedicl.rescore --arm {arm}` chạy lại `match_answer()` trên `raw_output` đã lưu với cấu hình `eval.match` mới.
- **Train tiếp khi chưa overfit:** `--resume --extend_to 5`.
- **Tổng hợp:** `python -m fedicl.summarize` quét mọi `{arm}/result.csv` để tạo `summary.csv`, cập nhật `runs_index.csv`, và vẽ biểu đồ accuracy và `val_loss` theo epoch/round cho 4 arm. Với FL, vẽ thêm accuracy của từng client.
- Thêm helper `load_adapter(arm, checkpoint)` trả về model đã gắn adapter, để dùng lại trong notebook hoặc script khác.

## 6. Config (`configs/*.yaml`)
Một file base và override theo từng arm (hoặc CLI override kiểu Hydra/OmegaConf). Ví dụ:
```yaml
seed: 42
model: {name: Qwen/Qwen3-0.6B, short_name: qwen3-0.6b, max_seq_len: 2048, enable_thinking: false}
data:
  root: processed_data
  dataset: MedQA
  dir: ${data.root}/${data.dataset}                 # processed_data/MedQA
  centralized_dir: ${data.dir}/centralized          # {train,validation,test}.csv + features/ + demo_assignment_*.json
data_prep:                    # chỉ script partition đọc; sinh cả lưới partition (mục 3.5)
  partition_seed: 42
  label_key: answer           # key Dirichlet (xem mục 7)
  iid_k: [2, 3, 4, 5]
  noniid_alphas: [0.5, 0.1]
  noniid_k: [2, 3, 4, 5]
setting: federated            # centralized | federated
prompt:
  system: "You are a helpful medical assistant. Answer the question by writing the full text of the correct option."
  option_style: bullet        # không có nhãn A–D → không có bias về chữ cái
  shuffle_options: true       # hoán vị cố định theo rng(seed, id) → không có bias về vị trí
icl:
  enabled: true               # false → non_icl
  k_shot: 3
  dedup_key: q_hash
  fl_eval_pool: local         # eval FL: demo từ repository của từng client (3.2)
retrieval:                    # mục 3.6
  strategy: adaptive          # random | semantic | semantic_clinical | adaptive
  question_encoder: {name: NeuML/pubmedbert-base-embeddings, pooling: mean, max_length: 512, fields: [question]}
  entity_extractor: en_core_sci_sm                  # scispaCy
  entity_encoder: {name: cambridgeltl/SapBERT-from-PubMedBERT-fulltext, pooling: cls, max_length: 25}
  weights: {semantic: 0.7, clinical: 0.3}
  candidate_top_n: 50
  mmr_lambda: 0.7
  near_dup_threshold: 0.95    # loại demo có sem ≥ ngưỡng (chống chép đáp án)
loss:                         # mục 4.2
  qa: {weight: 1.0}
  icl: {weight: 0.5, variant: demo_consistency}     # demo_consistency | context_distill; chỉ arm ICL
  reg: {mu: 0.01, type: fedprox}                    # chỉ arm FL
eval:
  generation: {do_sample: false, max_new_tokens: 64, stop_at_newline: true}
  match:                      # mục 5.1
    lexical: token_f1
    semantic_encoder: {name: cambridgeltl/SapBERT-from-PubMedBERT-fulltext, pooling: cls, max_length: 64}
    weights: {lexical: 0.5, semantic: 0.5}
    low_conf: {min_score: 0.5, min_margin: 0.05}    # chỉ gắn cờ, không đổi pred
  fl_global_pool_diagnostic: false
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj]}
train:
  epochs: 3                   # centralized
  max_extend: 5               # trần khi resume nếu chưa overfit (áp dụng cho cả epoch và round)
  lr: 2e-4
  batch_size: 4
  grad_accum: 4
  early_stopping:             # centralized
    enabled: true
    monitor: val_loss
    mode: min
    eval_every: 0.25          # đơn vị: epoch
    patience: 2               # số lần eval liên tiếp không cải thiện
    min_delta: 0.001
    restore_best: true
federated:
  num_clients: 3              # → thư mục k3/
  partition: {type: noniid, alpha: 0.5}   # iid → federated_iid/k{n}/ ; noniid → federated_noniid/alpha_{α}/k{n}/
                                          # thư mục chưa tồn tại → báo lỗi "chạy script partition trước", không tự sinh
  rounds: 3
  local_epochs: 1
  aggregation: fedavg
  early_stopping: {enabled: true, monitor: val_loss, mode: min, patience: 1, min_delta: 0.001, restore_best: true}
save:
  root: outputs/${model.short_name}/seed${seed}
  arm: ${setting}_${icl_mode}
  save_client_adapters: true  # FL: lưu adapter cục bộ của từng client mỗi round
  keep_last_k_states: 2       # optimizer/server state; adapter thì luôn giữ hết
  overwrite: false
```
Chạy cả 4 arm bằng một script (`scripts/run_all.sh` hoặc sweep). Script tự bỏ qua arm đã hoàn thành và tự resume arm đang dở. Thứ tự chạy: data → partition → features → retrieval → 4 arm.

## 7. Cần chốt trước khi code (hỏi lại nếu chưa rõ)
1. **Key Dirichlet:** chia theo `answer` có thể không tạo được non-IID thực sự, vì sau khi xáo trộn option thì chữ cái đáp án càng không mang ý nghĩa gì. Cân nhắc chia theo `meta` (step1 / step2&3) hoặc theo topic.
2. **Phần cứng (GPU/VRAM):** quyết định batch size, `max_seq_len` và có dùng QLoRA 4-bit hay không. Việc này giờ quan trọng hơn trước, vì `L_ICL` cần 2 forward pass trên prompt 3-shot.
3. **Hệ số loss và trọng số truy xuất** (`λ_ICL` = 0.5, `μ` = 0.01, `w_sem/w_clin` = 0.7/0.3, `λ_mmr` = 0.7) hiện là mặc định ước lượng. Hai lựa chọn: giữ cố định để ra kết quả nhanh, hoặc chạy một grid nhỏ trên `validation`, chỉ cho `centralized_icl`.
4. **Biến thể `L_ICL`:** `demo_consistency` (mặc định, bám sát "context consistency" trong hình) hay `context_distill` (rẻ hơn).
5. **Hình kiến trúc cần cập nhật để khớp plan:** encoder đổi từ BioBERT/ClinicalBERT sang PubMedBERT-embeddings; prompt có liệt kê option; bỏ Recency/Authority. Outline cần đổi tên mục 3.3 "Multimodal Representation Learning".

## 8. Deliverables & tiêu chí hoàn thành
- [ ] Script chuẩn hoá: MedQA gốc → `raw/*.csv` → `centralized/*.csv` (thêm `q_hash`, loại dòng lỗi), kèm `config.json` và `dataset_summary.csv`.
- [ ] Script partition trên `centralized/train.csv` → `federated_iid/k{2..5}/` và `federated_noniid/alpha_{0.5,0.1}/k{2..5}/` (`client_*.csv` + `stats.csv`), có assert hợp = train và các client rời nhau.
- [ ] Script features: embedding câu hỏi (PubMedBERT-embeddings), thực thể scispaCy và embedding thực thể (SapBERT) → `centralized/features/`.
- [ ] Script retrieval: Adaptive Demonstration Retrieval theo 3.6 (4 chiến lược), lọc theo 3.3, demo chính + `alt_demo_ids`, repository đúng theo bảng 3.2 (FL: cục bộ cả lúc train lẫn eval), kèm assertion và `demo_check_report.json`.
- [ ] Script visualize → `visualizations/` (phân phối split, độ dài prompt, độ tương đồng của demo theo repository, phân phối nhãn theo client).
- [ ] Prompt builder theo 4.1 (option không nhãn, xáo trộn tất định), dùng chung cho train và eval; target = text của option đúng + EOS.
- [ ] Loss module theo 4.2: `L_QA`, `L_ICL` (2 forward, KL có stop-grad trên token đáp án), `L_REG` (proximal FedProx), log riêng từng thành phần.
- [ ] `match_answer()` theo 5.1, kèm unit test cho các ca: exact; option là chuỗi con của option khác; output có tiền tố `The answer is`; từ đồng nghĩa không có từ chung (`low thyroid hormone` → `hypothyroidism`); cặp hypo/hyper phải chọn đúng; output rỗng → `empty`; output rác vẫn chọn argmax nhưng gắn `low_confidence`.
- [ ] Trainer centralized (tối đa 3 epoch) và simulator FL FedAvg (tối đa 3 round), cả hai có early stopping theo `val_loss` và restore best; FL eval theo từng client.
- [ ] Lưu đủ theo mục 5.2: adapter của mọi epoch/round, adapter từng client mỗi round, `best/`, predictions (theo từng client với FL ICL), metrics, log, state để resume.
- [ ] Resume, không ghi đè (archive), eval-only, rescore, `load_adapter()` hoạt động đúng như mục 5.3.
- [ ] Toàn bộ tham số nằm trong yaml; chạy lại cùng seed cho cùng kết quả.
- [ ] `summary.csv` và `runs_index.csv` tổng hợp 4 arm, kèm biểu đồ (FL có thêm biểu đồ theo từng client).
- [ ] README ngắn: cách chạy, cách resume/eval lại, cấu trúc thư mục, ánh xạ kiến trúc → code, cách chia data centralized/FL và nguồn demo ICL.
