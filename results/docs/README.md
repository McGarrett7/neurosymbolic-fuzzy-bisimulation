# Hướng dẫn xem results

## Bản đồ các file

Thư mục `results/` có 4 nhóm kết quả; mỗi thực nghiệm có 1 file `.md` (bảng đưa vào báo cáo) và 1 file `.json` (số liệu đầy đủ từng seed, dùng để vẽ hình).

| Nhóm | File | Ứng với |
| --- | --- | --- |
| Kết quả chính | `main_fb15k-237.md/.json`, `main_wn18rr.md/.json`, `main_yago3-10.md/.json` | Table V, VI của paper |
| Kết quả chính | `main_<dataset>_zeroshot.json` (3 dataset) | Baseline không train trong Table V (LLM Similarity) |
| Ablation | `ablation_fb15k-237.md/.json` | Table VIII |
| Phần IV | `alpha_fb15k-237.md/.json` | Độ nhạy theo trọng số trộn α |
| Phần IV | `encoders_fb15k-237.md/.json` | So sánh bộ mã hóa tên entity |
| Phần IV | `sampling_fb15k-237.md/.json` | Table VII: tỉ lệ và cách lấy mẫu; Fig. 2 |
| Reviewer | `reference_fb15k-237.md/.json` | R1, R2: khả năng mở rộng; Assumption 1 |
| Reviewer | `latency_fb15k-237.md/.json` | R1, R2: "constant-time inference" |
| Reviewer | `stability_fb15k-237.md/.json` | R1, R2: Theorem 2 |
| Reviewer | `semantics_fb15k-237.md/.json` | R1: hạn chế của ngữ nghĩa Gödel |

## Thuật ngữ dùng chung

Các bảng kết quả đều dùng chung những khái niệm dưới đây; đọc mục này trước khi xem từng file.

| Thuật ngữ | Nghĩa | Đọc thế nào |
| --- | --- | --- |
| Nhãn B\* | Độ giống nhau về hành vi giữa 2 entity, tính chính xác bằng fuzzy bisimulation, từ 0 đến 1 | Đáp án mà mô hình phải dự đoán |
| B̂ | Giá trị mô hình dự đoán | Càng gần B\* càng tốt |
| uniform pairs | 5,000 cặp entity test chọn ngẫu nhiên | Phản ánh phân bố thật: gần như toàn nhãn 0 |
| stratified pairs | Cặp test chọn đều theo các khoảng giá trị nhãn | Có nhiều cặp nhãn cao hơn; phù hợp để đo khả năng phân biệt |
| MAE | Sai số tuyệt đối trung bình \|B̂ − B\*\| | Thấp là tốt; 0.5 = lệch trung bình nửa thang đo |
| RMSE | Căn bậc hai sai số bình phương | Thấp là tốt; phạt nặng các lần sai lớn |
| F1 | Đổi thành bài toán có/không với ngưỡng τ = 0.7 (B\* ≥ 0.7 là "giống") | Cao là tốt; không ổn định khi có rất ít cặp "giống" |
| Spearman | Độ tương quan thứ hạng giữa B̂ và B\*, từ −1 đến 1 | Cao là tốt; đo việc xếp hạng đúng, không cần đúng giá trị |
| 0.464 ± 0.015 | Trung bình ± độ lệch chuẩn qua các seed | ± nhỏ = ổn định giữa các lần chạy |
| p (BH) | p-value của paired t-test, hiệu chỉnh Benjamini-Hochberg | p < 0.05 = khác biệt có ý nghĩa thống kê |
| seed | Một lần chạy với cách lấy mẫu ngẫu nhiên khác | main FB15k-237, WN18RR: 3 seed; main YAGO3-10: 1 seed; ablation: 2 seed; Phần IV: 1 seed |

## Kết quả chính: main\_\*

`main_<dataset>` so sánh mô hình đề xuất với 5 baseline có train; ứng với Table V và Table VI. FB15k-237 và WN18RR chạy 3 seed; YAGO3-10 chạy 1 seed (đồ thị lớn gấp ~4 lần), nên `main_yago3-10.md` không có ± và không có p-value.

**File `.md`** gồm 2 bảng (uniform pairs, stratified pairs), mỗi bảng có cột MAE, RMSE, F1, Spearman, và cuối file là danh sách p-value. Các hàng:

| Hàng | Mô hình | Ý nghĩa khi so sánh |
| --- | --- | --- |
| Proposed (R-GCN + semantic, a=0.7) | Mô hình của paper: R-GCN + embedding tên entity, trộn 70/30 | Mốc so sánh cho mọi hàng khác |
| R-GCN | Chỉ cấu trúc, không semantic (α = 1) | Proposed ≈ R-GCN nghĩa là semantic không giúp |
| GCN | GNN không phân biệt loại quan hệ | Kiểm tra vai trò của việc tách theo relation |
| GraphSAGE | GNN gộp hàng xóm kiểu khác | Baseline GNN thứ hai |
| MLP on structural features | Không truyền tin trên đồ thị, chỉ dùng đặc trưng bậc/trọng số | Kiểm tra GNN có cần hay không |
| Semantic only (trained projection) | Chỉ dùng embedding tên (α = 0) | Kiểm tra riêng kênh LLM |

Dòng p-value dạng `GCN: mae p=3.68e-02, ...` là kết quả so Proposed với GCN trên stratified pairs; p < 0.05 là khác biệt có ý nghĩa.

**File `.json`** có 4 khóa:

| Khóa | Nội dung |
| --- | --- |
| `rows` | Số liệu từng seed cho mỗi mô hình: `rows[mô hình][uniform\|stratified][mae\|rmse\|f1\|spearman]` = danh sách 3 giá trị |
| `tests` | p-value thô và đã hiệu chỉnh (`p_bh`) cho từng cặp so sánh |
| `extras` | `alpha` (trọng số trộn) và `cost` (chi phí) của seed đầu |
| `results` | Chi tiết từng lần chạy: đường dẫn `checkpoint`, cấu hình nhãn, kết quả `test` và `val`, thống kê nhãn train (`train_reference_stats`), thông tin tham chiếu test |

Trường `cost` dùng cho cột Runtime của Table V và cho Fig. 2:

| Trường | Nghĩa |
| --- | --- |
| `offline_seconds` | Thời gian từng bước offline: lấy mẫu, tính nhãn train, mã hóa semantic, train, tính embedding |
| `offline_total` | Tổng thời gian offline (giây) |
| `query_seconds` | Thời gian mỗi truy vấn sau khi train: `batched` (theo lô) và `single` (từng cặp) |
| `exact_reference` | Chi phí tính bisimulation chính xác cho tập test: số giây, số cặp, thời gian mỗi cặp |
| `break_even_queries` | Sau bao nhiêu truy vấn thì chi phí offline được bù lại so với tính chính xác từng lần |

**File `main_<dataset>_zeroshot.json`**: 3 baseline không cần train, tính trên seed 0.

| Baseline | Cách tính |
| --- | --- |
| `feature_cosine` | Cosine giữa đặc trưng cấu trúc của 2 entity |
| `semantic_cosine[st:all-mpnet-base-v2]` | Cosine giữa embedding tên entity ("LLM Similarity" trong Table V) |
| `neighbour_jaccard` | Tỉ lệ hàng xóm chung |

Mỗi baseline có `test` và `val`, chia `uniform`/`stratified`, với các trường: `n` (số cặp), `mae`, `rmse`, `f1`, `precision`, `recall`, `pos_rate` (tỉ lệ cặp có nhãn ≥ 0.7), `confusion` (tp, fp, fn, tn), `pearson`, `spearman`. Ví dụ `pos_rate` = 0.0088 trên WN18RR nghĩa là chỉ 15 trên 1,706 cặp là "giống".

**Kết quả YAGO3-10 (1 seed).** Giống hai dataset kia: mọi mô hình có MAE ≈ 0.50 (uniform) và ≈ 0.43–0.44 (stratified), trong khi `neighbour_jaccard` chỉ 0.021 (uniform). Khác biệt: stratified có nhiều cặp "giống" hơn (`pos_rate` 0.14, 1,962 cặp) nên Spearman cao hơn (≈ 0.62 cho mọi mô hình) và F1 0.70–0.86. Các mô hình gần như ngang nhau; Semantic only có F1 cao nhất (0.862). Nhãn train: 92% bằng 0, mean 0.056. Chi phí offline của Proposed: 1,894 s (mã hóa MPNet 499 s, train 1,393 s).

## Ablation: ablation\_fb15k-237

`ablation_fb15k-237` bỏ hoặc thay từng thành phần của mô hình đề xuất để xem thành phần đó đóng góp bao nhiêu; chạy trên FB15k-237, 2 seed, ứng với Table VIII.

Cấu trúc file giống hệt `main_*` (2 bảng uniform/stratified, p-value so với Full model; `.json` có `rows`, `tests`, `extras`, `results`). Khác biệt nằm ở các hàng:

| Hàng | Thay đổi so với Full model | Trong Table VIII | Nếu kém hơn Full model nhiều thì nghĩa là |
| --- | --- | --- | --- |
| Full model | Không đổi (giống Proposed trong main) | Full Model | — |
| w/o semantic (a=1) | Bỏ kênh LLM | w/o LLM | LLM có ích |
| w/o graph learning (a=0) | Bỏ GNN, chỉ dùng embedding tên | w/o Graph Learning | GNN có ích |
| w/o symbolic supervision (Jaccard target) | Train bằng nhãn Jaccard hàng xóm thay cho nhãn bisimulation | w/o Symbolic Supervision | Nhãn bisimulation có ích |
| random sampling | Lấy mẫu entity ngẫu nhiên thay vì theo bậc | Random Sampling | Lấy mẫu theo bậc có ích |
| with fuzzy edge weights in messages | Nhân trọng số fuzzy vào thông điệp GNN | Không có | (thử thêm) |
| paper's final ReLU (full model) | Thêm ReLU ở lớp GNN cuối, đúng công thức paper | Không có | (code mặc định bỏ ReLU này) |
| paper's final ReLU, w/o semantic (a=1) | ReLU lớp cuối và bỏ LLM | Không có | (thử thêm) |

Cách đọc: so mỗi hàng với Full model trên bảng stratified, rồi xem p-value cuối file. Với 2 seed, p-value thường lớn, nên chênh lệch nhỏ chưa đủ kết luận.

## Phần IV: alpha, encoders, sampling

Ba thực nghiệm phân tích độ nhạy của mô hình đề xuất trên FB15k-237. Cấu trúc file giống `main_*` (bảng uniform/stratified, `.json` có `rows`, `extras`, `results`), nhưng **chỉ chạy 1 seed**: không có ± và không có p-value, nên chênh lệch nhỏ (khoảng < 0.02) chưa đủ để kết luận.

Mọi hàng đều dùng chung cấu hình của Proposed (R-GCN, MPNet, α = 0.7, lấy mẫu degree 10%, γ = 0.7) và chỉ thay đúng một yếu tố.

### alpha\_fb15k-237: trọng số trộn α

Embedding z = α · GNN(e) + (1 − α) · LLM\_proj(e).

| Hàng | Nghĩa |
| --- | --- |
| alpha=0.0 | Chỉ dùng embedding tên (giống "Semantic only") |
| alpha=0.1 … 0.9 | Trộn với tỉ lệ tăng dần về phía GNN; 0.7 là giá trị của paper |
| alpha=1.0 | Chỉ dùng GNN (giống "R-GCN") |
| alpha=learned | α là tham số học được (qua sigmoid); giá trị học được nằm trong `extras["alpha=learned"]["alpha"]` |

Cách đọc: nếu đường kết quả có đỉnh rõ quanh 0.7 thì α của paper là hợp lý; nếu nằm ngang thì một trong hai kênh không ảnh hưởng tới kết quả.

Kết quả hiện tại: Spearman stratified ≈ 0.436 với **mọi α từ 0.1 đến 1.0**, α học được (≈ 0.51) cũng vậy; chỉ α = 0 giảm xuống 0.288. Kênh LLM gần như không đóng góp; chỉ cần có GNN là đủ.

### encoders\_fb15k-237: bộ mã hóa tên entity

| Hàng | Bộ mã hóa | Chiều |
| --- | --- | --- |
| tfidf | TF-IDF trên tên entity, không dùng mạng nơ-ron | 768 (giảm chiều) |
| st:all-MiniLM-L6-v2 | Sentence-Transformer nhỏ | 384 |
| st:all-mpnet-base-v2 | Sentence-Transformer lớn, mặc định của project | 768 |

Cách đọc: nếu LLM có ích thì MPNet > MiniLM > TF-IDF.

Kết quả hiện tại: cả 3 cho Spearman stratified = 0.435. TF-IDF ngang LLM, khớp với kết luận ở phần alpha.

### sampling\_fb15k-237: lấy mẫu subgraph train

Ứng với Table VII. Nhãn train B\*\_s được tính trên subgraph sinh từ các entity lấy mẫu, nên cách lấy mẫu quyết định chất lượng nhãn (xem thêm Bảng 2 của `reference_*`).

| Hàng | Nghĩa |
| --- | --- |
| `<r>% random` | Lấy r% entity ngẫu nhiên đều |
| `<r>% degree` | Lấy r% entity với xác suất p(e) ∝ deg(e)^γ, γ = 0.7 |
| `10% degree gamma=<γ>` | Giữ 10%, đổi γ; γ = 0 tương đương random, γ = 1 tỉ lệ thuận với bậc |

Cách đọc: so random với degree ở cùng tỉ lệ; so các mức γ với hàng `10% degree` (γ = 0.7).

Kết quả hiện tại:
- Degree cho F1 và RMSE tốt hơn random ở mọi tỉ lệ; 20% degree tốt nhất (F1 0.886).
- `5% random` có Spearman cao nhất (0.544) nhưng F1/RMSE kém, nhiều khả năng do nhiễu của 1 seed.
- γ = 0.7 tốt nhất trong các mức đã thử, nhưng kết quả không đơn điệu theo γ.

`extras` của file này chứa `cost` cho từng biến thể; đây là nguồn dữ liệu của Fig. 2 (`complexity_tradeoff`).

## Thực nghiệm cho reviewer

Bốn file này không so sánh mô hình mà kiểm chứng từng khẳng định reviewer đã hỏi; đều chạy trên FB15k-237.

### reference\_fb15k-237 — tham chiếu local có chính xác không

Trả lời R1, R2 (phải tính B\* toàn đồ thị mới đánh giá được). Dòng đầu `Full-graph Algorithm 1: 7.70s, 8 iterations, n=14541` là thời gian tính bisimulation chính xác trên toàn đồ thị.

Bảng 1, local reference theo độ sâu:

| Cột | Nghĩa |
| --- | --- |
| local depth | Bán kính k: chỉ tính trên các node cách entity test tối đa k bước |
| ball nodes | Số node trong vùng đó (toàn đồ thị: 14,541) |
| exact B\* | Vùng đó có đóng kín hoàn toàn không (True = chắc chắn bằng B\* toàn đồ thị) |
| seconds | Thời gian tính |
| max / mean abs diff to B\* | Sai số lớn nhất / trung bình so với B\* toàn đồ thị; 0.0000 = khớp hoàn toàn |

Bảng 2, Assumption 1 (subgraph lấy mẫu có đại diện không): `sampling` (random/degree), `ratio` (5/10/20%), `n_s` (số entity lấy mẫu), `E abs(B*_s - B*)` (sai lệch trung bình giữa nhãn trên subgraph và nhãn thật, chính là ε\_s), `mean B*_s` và `mean B*` (giá trị trung bình mỗi bên). `mean B*_s` lớn hơn `mean B*` nhiều nghĩa là subgraph làm nhãn bị thổi phồng.

JSON: `full` (seconds, iterations, nodes), `local` (bảng 1), `induced_gap` (bảng 2).

### latency\_fb15k-237 — thời gian truy vấn

Trả lời R1, R2 ("constant-time" là gây hiểu lầm). Cột `d` là số chiều embedding, `ns / query` là nano giây cho một lần tính cosine giữa 2 embedding có sẵn. Thời gian tăng gần gấp đôi khi d gấp đôi, tức O(d). Phép đo dùng vector ngẫu nhiên nên không phụ thuộc dataset. JSON: `rows` với `dim` và `seconds_per_query`.

### stability\_fb15k-237 — Theorem 2

Trả lời R1, R2 (giả định lý thuyết chưa kiểm chứng). Train 1 mô hình, sau đó cộng nhiễu vào trọng số cạnh và xem dự đoán đổi bao nhiêu.

| Cột | Nghĩa |
| --- | --- |
| delta | Mức nhiễu δ cộng vào mỗi trọng số cạnh |
| mean / max abs change | Mức thay đổi trung bình / lớn nhất của B̂ |
| max / delta | Ước lượng hằng số C trong Theorem 2 (\|ΔB̂\| ≤ Cδ); nhỏ và ổn định = định lý được ủng hộ |

JSON: `rows` với `delta`, `mean`, `max`.

### semantics\_fb15k-237 — phân bố nhãn theo t-norm

Dùng cho phần hạn chế (R1). Không train; chỉ tính nhãn theo 8 cấu hình.

| Cột | Nghĩa |
| --- | --- |
| semantics | Cách tính: godel (paper), lukasiewicz, product |
| discount | λ; 1.0 = không chiết khấu (paper) |
| mean, std | Trung bình và độ trải của nhãn test (uniform); std nhỏ = nhãn gần như bằng nhau |
| frac zero | Tỉ lệ cặp có nhãn đúng bằng 0 |
| frac >= 0.7 | Tỉ lệ cặp "giống" theo ngưỡng F1 |
| iterations, seconds | Số vòng lặp và thời gian tính |

JSON: mỗi cấu hình (ví dụ `godel/1.0`) có `test_uniform`, `train` (thống kê nhãn) và `info`.

## Thư mục figures/


| Hình | Vẽ từ | Thể hiện |
| --- | --- | --- |
| `training_loss_fb15k-237`, `training_loss_wn18rr`, `training_loss_yago3-10` | Checkpoint của Proposed, seed 0 | Fig. 1: train MSE và val MSE theo epoch; đường đứt là epoch tốt nhất |
| `main_fb15k-237`, `main_wn18rr`, `main_yago3-10` | `main_*.json` | Cột ngang MAE, RMSE, F1, Spearman cho 6 mô hình; xanh = uniform, cam = stratified; vạch đen = ± độ lệch chuẩn |
| `ablation_fb15k-237` | `ablation_fb15k-237.json` | Cùng kiểu, cho 8 biến thể ablation |
| `reference_fb15k-237` | `reference_fb15k-237.json` | Trái: sai số local so với B\* theo độ sâu (thang log). Phải: thời gian local so với full-graph |
| `latency` | `latency_fb15k-237.json` | Thời gian truy vấn theo d, thang log-log; đường thẳng = O(d) |
| `stability_fb15k-237` | `stability_fb15k-237.json` | Thay đổi của B̂ theo δ; đường đứt y = δ để so; nằm dưới đường đó = C < 1 |
| `semantics_fb15k-237` | `semantics_fb15k-237.json` | 4 cột (mean, std, tỉ lệ = 0, tỉ lệ ≥ 0.7) cho từng cấu hình t-norm/λ |
| `alpha_fb15k-237` | `alpha_fb15k-237.json` | Trái: MAE theo α; phải: Spearman theo α; ★ = α học được, đường chấm = α = 0.7 |
| `encoders_fb15k-237` | `encoders_fb15k-237.json` | Cột ngang như `main`, cho 3 bộ mã hóa |
| `sampling_fb15k-237` | `sampling_fb15k-237.json` | MAE theo tỉ lệ lấy mẫu, mỗi đường một cách lấy mẫu |
| `complexity_tradeoff_fb15k-237` | `sampling_fb15k-237.json` (+ `reference_*`) | Fig. 2: chi phí offline (giây) so với 1 − MAE trên stratified pairs; đường đứt = thời gian tính B\* chính xác toàn đồ thị (≈ 8 s) |

Đọc Fig. 2: điểm nào nằm bên phải đường đứt là tốn thời gian offline hơn cả việc tính chính xác. Trên FB15k-237 mọi biến thể đều tốn 290–545 s, nên lợi thế chi phí của mô hình chỉ có thể xuất hiện trên đồ thị lớn hơn nhiều.
