# Runbook: chạy FedICL-MQA trên Windows Server 2025 (RTX A5000)

Tài liệu này đi từng bước, kèm **lệnh kiểm tra** và **kết quả mong đợi**. Chỉ sang bước tiếp
theo khi bước hiện tại đạt mọi ô ✅. Máy Ubuntu: xem [UBUNTU.md](UBUNTU.md) (cùng trình tự,
lệnh `bash scripts/*.sh`).

Mọi lệnh chạy trong **PowerShell 7** (`pwsh`), từ **thư mục gốc của repo**. Không có `pwsh` thì
thay `pwsh` bằng `powershell` (Windows PowerShell 5.1 cũng chạy được).

**Chọn GPU:** mọi script nhận tham số `-Gpu 0` hoặc `-Gpu 1` (số thứ tự GPU như trong
`nvidia-smi`). Không ghi `-Gpu` thì dùng GPU 0. Mọi bước bên trong script (chuẩn bị dữ liệu,
train, đánh giá) đều chạy trên GPU đã chọn. Runbook này chạy **một GPU tại một thời điểm**; các
ví dụ dùng `-Gpu 0`, đổi thành `-Gpu 1` nếu GPU 0 đang bận.

Thời gian thực tế đo trên A5000 (log ngày 2026-09-26):

| Arm | Thời gian |
|-----|-----------|
| `centralized_non_icl` | ~1.0 giờ |
| `federated_non_icl` | ~1.7 giờ |
| `centralized_icl` | ~2.4 giờ |
| `federated_icl` | ~13 giờ |
| **Tổng, chạy lần lượt trên 1 GPU** | **~18 giờ** |

---

## Bước 0: Trên máy local, trước khi push

```bash
git status                        # xem còn file nào chưa commit
python -m pytest -q tests         # trong .venv (hoặc: source scripts/env.sh)
git add -A && git commit -m "..." && git push
```

- [ ] ✅ `pytest` báo `N passed`, không có `failed`.
- [ ] ✅ `git status` sạch, `git log -1` là commit vừa push.
- [ ] ✅ **Không** có `HF_Access_Token*`, `processed_data*`, `outputs*` trong commit (đã nằm
      trong `.gitignore`; kiểm tra lại bằng `git show --stat HEAD`).

## Bước 1: Kiểm tra server

```powershell
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvidia-smi | Select-String "CUDA Version"
pwsh --version; git --version
"{0:N0} GB trống trên D:" -f ((Get-PSDrive D).Free / 1GB)
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled
```

- [ ] ✅ Thấy GPU sẽ dùng (`0` hoặc `1`) là `NVIDIA RTX A5000`, khoảng 24 GB.
- [ ] ✅ GPU đó đang rảnh: `nvidia-smi` không có tiến trình nào chiếm bộ nhớ trên GPU đó.
- [ ] ✅ `CUDA Version` ≥ **12.4** (driver ≥ 550). Thấp hơn: cập nhật driver, hoặc ở Bước 4
      dùng `-Cuda cu118`.
- [ ] ✅ Có `pwsh` 7.x và `git`.
- [ ] ✅ Ổ chứa repo còn trống **≥ 20 GB** (môi trường ~7 GB, dữ liệu + model ~3 GB, checkpoint vài GB).
- [ ] ✅ `LongPathsEnabled` = `1`. Nếu chưa, chạy một lần trong PowerShell **Administrator**:

  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

- [ ] (Khuyến nghị, cần admin) Loại thư mục repo khỏi quét real-time của Defender để cài đặt và
      ghi checkpoint nhanh hơn: `Add-MpPreference -ExclusionPath "D:\phungthan\fedicl-medqa-v3"`.

## Bước 2: Lấy code

```powershell
# lần đầu
git clone git@github.com:nguyenhoangvuthan/fedicl-medqa-v3.git D:\phungthan\fedicl-medqa-v3
# các lần sau
cd D:\phungthan\fedicl-medqa-v3
git pull
git log --oneline -1
```

- [ ] ✅ `git log --oneline -1` trùng với commit vừa push ở Bước 0.
- [ ] ✅ `git status` không báo file bị sửa trên server (nếu có: `git stash` hoặc hỏi lại trước khi xoá).

## Bước 3: Token Hugging Face (tuỳ chọn)

Tạo token loại **Read** tại <https://huggingface.co/settings/tokens>, bấm nút copy, dán vào
`D:\phungthan\fedicl-medqa-v3\HF_Access_Token.txt` (một dòng `hf_...`). Kiểm tra mà không in token:

```powershell
$t = (Get-Content .\HF_Access_Token.txt -Raw).Trim(); "length: $($t.Length)"; try { "user: " + (Invoke-RestMethod https://huggingface.co/api/whoami-v2 -Headers @{Authorization = "Bearer $t"}).name } catch { $_.Exception.Message }; Remove-Variable t
```

- [ ] ✅ `length: 37` và `user: <tên tài khoản>`.
- [ ] ⚠️ Báo 401: token sai, đã bị xoá hoặc copy thiếu. Script vẫn chạy được ở chế độ ẩn danh,
      chỉ tải chậm hơn.

## Bước 4: Cài môi trường (một lần, ~10 phút, tải ~6 GB)

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\windows\setup_env.ps1
```

(Bước này không dùng GPU nên không cần `-Gpu`.)

- [ ] ✅ Dòng `[HF token] OK: user '...'` (hoặc cảnh báo ẩn danh nếu bỏ qua Bước 3).
- [ ] ✅ Dòng `torch 2.6.0+cu124 cuda True` kèm tên GPU, và dòng `environment OK`.
- [ ] ✅ `uv cache:` và `HF cache:` trỏ vào thư mục repo, không phải `C:\Users\...`.

## Bước 5: Unit test

```powershell
. .\scripts\windows\env.ps1
Invoke-Py -m pytest -q tests
```

- [ ] ✅ `N passed`, không có `failed`.

## Bước 6: Smoke test (tuỳ chọn, ~10–15 phút, dữ liệu 120/24/24 câu)

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1 -Gpu 0
Import-Csv outputs_smoke\qwen3-0.6b\seed42\runs_index.csv | Format-Table arm, status, best_at
```

- [ ] ✅ Đầu log có `[GPU] using GPU 0 (nvidia-smi index)`.
- [ ] ✅ Cả 4 arm `status = completed`.
- [ ] ✅ Cuối log có `PASSED: every federated ICL demo comes from the client's own local data`.
- Con số accuracy của smoke test **không có ý nghĩa** (quá ít dữ liệu); bước này chỉ để kiểm tra
  pipeline.

## Bước 7: Chuẩn bị dữ liệu thật

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\windows\prepare_data.ps1 -Gpu 0 *>&1 | Tee-Object prepare_data.log
```

Kiểm tra:

```powershell
Import-Csv processed_data\MedQA\dataset_summary.csv | Where-Object layer -eq centralized | Format-Table split, n_samples, n_dropped_invalid, n_overlap_with_train
Import-Csv processed_data\MedQA\federated_noniid\alpha_0.5\k3\stats.csv | Format-Table client_id, n_samples
$r = Get-Content processed_data\MedQA\centralized\demo_check_report.json | ConvertFrom-Json
$r.scopes.PSObject.Properties | ForEach-Object { "{0,-55} passed={1} short_of_k={2}" -f $_.Name, $_.Value.check.passed, $_.Value.short_of_k }
```

- [ ] ✅ `train` = **10178**, `validation` = **1272**, `test` = **1273**, `n_dropped_invalid` = 0.
- [ ] ✅ 3 client: **6040 / 2798 / 1340** câu (cộng lại = 10178).
- [ ] ✅ Mọi dòng `passed=True`, `short_of_k=0`. Có ít nhất 12 dòng: 3 dòng `centralized/...` và
      9 dòng `federated_noniid/alpha_0.5/k3/client_{1,2,3}/...` (nếu đã chạy ablation k2 ở Bước 10
      thì có thêm 6 dòng `.../k2/...`).
- [ ] ✅ Cuối log: `ICL prompt tokens: ... max=` **< 2048** và `PASSED: every federated ICL demo ...`.

## Bước 8: Train 4 arm trên một GPU (~18 giờ)

Một lệnh chạy lần lượt cả 4 arm (non-ICL trước để có baseline sớm):

```powershell
cd D:\phungthan\fedicl-medqa-v3
pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Gpu 0 *>&1 | Tee-Object run_all.log
```

Muốn chia thành nhiều đợt (ví dụ GPU chỉ rảnh vào ban đêm), chạy từng nhóm arm, mỗi đợt có thể
chọn GPU khác nhau:

```powershell
$env:FEDICL_ARMS = "centralized_non_icl,federated_non_icl,centralized_icl"   # ~5 giờ
pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Gpu 0 *>&1 | Tee-Object run_part1.log

$env:FEDICL_ARMS = "federated_icl"                                           # ~13 giờ
pwsh -ExecutionPolicy Bypass -File scripts\windows\run_all.ps1 -Gpu 1 *>&1 | Tee-Object run_part2.log
Remove-Item Env:FEDICL_ARMS                                                  # trở lại "cả 4 arm"
```

Theo dõi (mở một cửa sổ khác):

```powershell
nvidia-smi                                                              # GPU đã chọn đang bận
Import-Csv outputs\qwen3-0.6b\seed42\runs_index.csv | Format-Table arm, status, best_at, best_test_acc
Get-Content outputs\qwen3-0.6b\seed42\federated_icl\logs\run.log -Tail 5 -Wait   # Ctrl+C để thoát
```

- [ ] ✅ Đầu log: `[GPU] using GPU 0 (nvidia-smi index)` và `arms: ...` đúng danh sách mong muốn.
- [ ] ✅ `nvidia-smi` cho thấy một tiến trình `python` trên đúng GPU đã chọn.
- [ ] ✅ Khi xong: cả 4 arm `status = completed` trong `runs_index.csv`.
- [ ] ✅ Cuối log có `PASSED: every federated ICL demo ...`.

Quan trọng:

- **Ngắt Remote Desktop thì được, Sign out thì tiến trình bị giết.**
- Bị dừng giữa chừng (mất điện, lỡ đóng cửa sổ, cần nhường GPU): chạy lại **đúng lệnh cũ**, có
  thể đổi `-Gpu`. Arm đã xong được bỏ qua, arm đang dở chạy tiếp từ epoch/round cuối đã lưu.
- `-Gpu` chỉ nhận `0` hoặc `1`; GPU không tồn tại thì script dừng ngay với thông báo
  `GPU 1 requested ... but CUDA sees no device`, không chạy âm thầm bằng CPU.

## Bước 9: Tổng hợp, kiểm định, audit

```powershell
. .\scripts\windows\env.ps1
Invoke-Py -m fedicl.summarize
Invoke-Py -m fedicl.compare
Invoke-Py -m fedicl.compare --checkpoint round_3
Invoke-Py -m fedicl.audit
```

- [ ] ✅ Có các file trong `outputs\qwen3-0.6b\seed42\`: `headline.csv`, `summary.csv`,
      `curves.png`, `fl_per_client.png`, `compare_test_best.csv`, `compare_test_round_3.csv`.
- [ ] ✅ `audit` báo `PASSED`.

Cách đọc `compare`: cột `gain_pp` là chênh lệch accuracy (điểm %), `ci95_low_pp`–`ci95_high_pp`
là khoảng tin cậy 95%, `significant_5pct = True` nghĩa là chênh lệch có ý nghĩa thống kê.
Số chính của FL là `macro_mean` (trung bình đều các client); `min` là client kém nhất.

## Bước 10 (tuỳ chọn): Ablation kiểm chứng gain ICL trong FL

Kết quả ghi vào thư mục riêng, **không đụng** kết quả chính. Chạy lần lượt trên một GPU:

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\windows\run_ablation.ps1 -Ablation no_licl -Gpu 0 *>&1 | Tee-Object ablation_no_licl.log   # ~8 giờ
pwsh -ExecutionPolicy Bypass -File scripts\windows\run_ablation.ps1 -Ablation k2 -Gpu 0      *>&1 | Tee-Object ablation_k2.log        # ~15 giờ
```

- [ ] ✅ Có `outputs\ablation_no_licl\qwen3-0.6b\seed42\compare_best.csv` và
      `outputs\ablation_k2\qwen3-0.6b\seed42\compare_best.csv`.

## Bước 11: Mang kết quả về

`outputs\` nằm trong `.gitignore` (adapter rất nặng). Chỉ gom file nhỏ cần phân tích:

```powershell
$dst = "results\$(Get-Date -Format yyyyMMdd)"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item outputs\qwen3-0.6b\seed42\*.csv, outputs\qwen3-0.6b\seed42\*.png $dst
Get-ChildItem outputs\qwen3-0.6b\seed42 -Directory | ForEach-Object { Copy-Item "$($_.FullName)\result.csv" "$dst\$($_.Name)_result.csv" -ErrorAction SilentlyContinue }
Copy-Item run_all.log, run_part*.log, prepare_data.log, ablation_*.log $dst -ErrorAction SilentlyContinue
git add $dst; git commit -m "exp: results $(Get-Date -Format yyyy-MM-dd)"; git push
```

- [ ] ✅ Trước khi push, `git show --stat HEAD` chỉ gồm các file `.csv`, `.png`, `.log` trong `results\`.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `The argument 'scripts\windows\...' to the -File parameter does not exist` | Đang đứng sai thư mục | `cd D:\phungthan\fedicl-medqa-v3` rồi chạy lại |
| `[HF token] rejected by huggingface.co (401)` | Token sai/hết hạn/copy thiếu | Tạo token Read mới, dán lại (Bước 3). Không chặn việc chạy |
| `[HF token] add a line '...' to .gitignore` | `.gitignore` cũ | `git pull` |
| `[HF token] both HF_Access_Token and HF_Access_Token.txt exist` | Có cả hai file | Xoá một file |
| `GPU 1 requested ... but CUDA sees no device` | Sai số GPU hoặc driver | Đối chiếu `nvidia-smi`, chạy lại với `-Gpu` đúng |
| `Cannot validate argument on parameter 'Gpu'` | `-Gpu` không phải `0`/`1` | Dùng `-Gpu 0` hoặc `-Gpu 1` |
| `CUDA out of memory` | GPU đang có tiến trình khác | `nvidia-smi` xem tiến trình nào chiếm GPU; chuyển sang GPU kia bằng `-Gpu`, hoặc chờ GPU rảnh |
| `config changed since this run started` | Đổi config giữa chừng | Giữ config cũ; hoặc `--overwrite` (bản cũ chuyển vào `_archive\`, không bị xoá) |
| `no .venv` / `.venv not found` | Chưa cài môi trường | Bước 4 |
| Lỗi đường dẫn khi cài torch | Long paths chưa bật | Bước 1 (`LongPathsEnabled`) |
| Thanh tiến trình hiện ký tự lạ | Bản code cũ | `git pull` (đã chuyển sang ASCII) |
