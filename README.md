# Demo luật kết hợp – Nhóm 6: hướng dẫn chạy và kịch bản

## 1. Cài đặt (làm một lần, trước buổi báo cáo)

1. Giải nén thư mục `Demo_Nhom6_App`. Bên trong phải có: `app.py`, `groceries.csv`, `requirements.txt`, thư mục `.streamlit`.
2. Mở thư mục này bằng VS Code → Terminal → New Terminal.
3. Gõ lệnh cài thư viện:

```
pip install -r requirements.txt
```

4. Chạy app:

```
streamlit run app.py
```

Trình duyệt tự mở trang `http://localhost:8501`. Không cần Internet khi chạy.

Tắt app: bấm `Ctrl + C` trong Terminal.

## 2. Cách dùng

- **Thanh bên trái:** chọn bộ dữ liệu, sửa bảng hóa đơn (thêm/xóa dòng, mặt hàng cách nhau bằng dấu phẩy), kéo minsup và minconf. Đổi bất kỳ thứ gì thì các tab tự quay về bước 1.
- **Tab ① ② ③:** bấm **Bước tiếp ▶** để thuật toán đi thêm một bước. Bên phải có mã giả, dòng đang chạy được tô vàng.
- **Tab ④:** đối chiếu 3 thuật toán với thư viện mlxtend, sinh luật, chọn một luật để xem cách tính confidence và lift.
- **Tab ⑤:** bộ Groceries 9.835 hóa đơn: đo thời gian 3 thuật toán và gợi ý mua kèm.

## 3. Kịch bản demo (khoảng 7 phút)

| Thời gian | Tab | Làm gì | Nói gì (ý chính) |
|---|---|---|---|
| 0:00 | ① Apriori | Bấm từng bước đến hết | Nối chỉ ghép tập chung k−2 phần tử đầu; cắt tỉa trước khi đếm; quét CSDL 3 lần, ra 8 tập (khớp slide 13–14) |
| 1:30 | Thanh bên | Kéo minsup về **2**, bấm lại tab ① đến bước “Cắt tỉa C3” | {Bánh mì, Bơ, Trứng} bị cắt tỉa vì {Bơ, Trứng} không phổ biến, **không cần đếm** |
| 2:15 | Thanh bên | Trả minsup về 3 | |
| 2:20 | ② FP-Growth | Bấm qua 6 lần chèn hóa đơn, dừng ở T3 và T4 | Tiền tố chung đi chung nhánh (vàng), nút mới (xanh); cây 7 nút sau 2 lần quét |
| 3:40 | ② FP-Growth | Bấm đến “Khai phá mục Sữa” | Đường tiền tố {Bánh mì, Bơ}:3 → cây điều kiện → {Sữa, Bánh mì}, {Sữa, Bơ}, {Bánh mì, Bơ, Sữa} |
| 4:20 | ③ ECLAT | Bấm 3–4 bước | Mỗi bước chỉ là một phép giao TID-list, không quét lại CSDL |
| 5:00 | ④ Đối chiếu | Cuộn xem bảng ✓ và luật | Cả 3 thuật toán = mlxtend: 8 tập; 10 luật mạnh, lọc lift > 1 còn 9 |
| 5:40 | ⑤ Groceries | Bấm “Chạy đo thời gian” | Hạ minsup 1% → 0,1%: Apriori chậm đi nhiều nhất, ECLAT nhanh nhất |
| 6:20 | ⑤ Groceries | Giỏ {sữa tươi, sữa chua}, rồi giỏ chỉ có {nước ngọt} | Gợi ý rau khác, trái cây; luật nước ngọt → sữa tươi bị bỏ vì lift 0,90 |

**Tương tác với lớp:** sau bước Apriori, mời thầy hoặc một bạn đề xuất minsup, hoặc thêm một hóa đơn (ví dụ `T7 = Bơ, Trứng`) rồi chạy lại.

**Nếu thầy hỏi “ví dụ trong bài giảng thì sao?”:** chọn bộ dữ liệu **“Ví dụ ACTW trong bài giảng (Chương 3)”**, minsup = 3. Apriori sẽ cắt tỉa {C, D, T} vì {D, T} không phổ biến, ra 19 tập phổ biến.

## 4. Kết quả để đối chiếu (6 hóa đơn, minsup = 3)

| | Kết quả |
|---|---|
| Apriori | 9 bước; quét CSDL 3 lần; 8 tập phổ biến |
| FP-Growth | F-list: Bánh mì (5) → Bơ (4) → Sữa (4) → Trứng (3); cây 7 nút; 8 tập |
| ECLAT | 7 phép giao; 8 tập |
| Sinh luật (minconf 70%) | 12 luật ứng viên, 10 luật mạnh, 9 luật có lift > 1 |
| minsup = 2 | Apriori cắt tỉa {Bánh mì, Bơ, Trứng} |
| minsup = 5 hoặc 6 | Không sinh được luật nào (app báo, không lỗi) |

Thời gian ở tab ⑤ phụ thuộc máy, nên con số có thể khác slide, nhưng thứ tự Apriori chậm nhất, ECLAT nhanh nhất vẫn giữ.

## 5. Dự phòng

Nếu máy không chạy được app: mở notebook `Demo_Nhom6_LuatKetHop_LOCAL.ipynb` hoặc file `Demo_Nhom6_KetQua_DuPhong.html`.
