# -*- coding: utf-8 -*-
"""
Demo luật kết hợp – Nhóm 6 (Học máy, HUIT)
Chạy:  streamlit run app.py
Xem từng bước Apriori, FP-Growth, ECLAT chạy trên dữ liệu nhỏ (có thể sửa),
rồi thử trên bộ Groceries 9.835 hóa đơn.
"""
import itertools
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demo luật kết hợp – Nhóm 6", page_icon="🛒", layout="wide")

NAVY, GOLD, GREEN, RED, GRAY = "#1F3864", "#C99B3D", "#2E7D4F", "#B03A2E", "#5B6B84"
OK_BG, BAD_BG, WARN_BG = "#E7F3EC", "#FBEAE8", "#FFF4D6"

st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
h1, h2, h3 {{color: {NAVY};}}
.step-title {{font-size: 1.25rem; font-weight: 700; color: {NAVY}; margin: .2rem 0 .4rem 0;}}
.explain {{background: #EEF3FB; border-left: 4px solid {NAVY}; padding: .7rem 1rem; border-radius: 6px;
          font-size: 1.02rem; line-height: 1.55; margin-bottom: .8rem;}}
.pseudo {{font-size: .95rem; background: #F7F8FB;
          border: 1px solid #D5DCE8; border-radius: 6px; padding: .5rem .2rem;}}
.pseudo div {{padding: .18rem .7rem; white-space: pre-wrap;}}
.pseudo .cur {{background: {WARN_BG}; border-left: 4px solid {GOLD}; font-weight: 700; color: {NAVY};}}
.small-note {{color: {GRAY}; font-size: .9rem;}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 1. DỮ LIỆU
# ============================================================
PRESETS = {
    "6 hóa đơn của nhóm (dùng trong slide)": [
        ("T1", "Sữa, Bánh mì, Bơ"),
        ("T2", "Bánh mì, Bơ"),
        ("T3", "Sữa, Bánh mì, Bơ, Trứng"),
        ("T4", "Sữa, Trứng"),
        ("T5", "Bánh mì, Trứng"),
        ("T6", "Sữa, Bánh mì, Bơ"),
    ],
    "Ví dụ ACTW trong bài giảng (Chương 3)": [
        ("1", "A, C, T, W"),
        ("2", "C, D, W"),
        ("3", "A, C, T, W"),
        ("4", "A, C, D, W"),
        ("5", "A, C, D, T, W"),
        ("6", "C, D, T"),
    ],
}


def fmt(s):
    return "{" + ", ".join(sorted(s)) + "}"


def tids_of(itemset, trans):
    return [tid for tid, t in trans if itemset <= t]


def tid_str(tids):
    return "{" + ", ".join(tids) + "}" if tids else "∅"


def color_rows(df, col):
    def f(row):
        v = str(row[col])
        if v.startswith("Giữ") or v.startswith("✓"):
            bg = OK_BG
        elif v.startswith("Loại") or v.startswith("Cắt") or v.startswith("✗"):
            bg = BAD_BG
        elif v.startswith("⚠"):
            bg = WARN_BG
        else:
            bg = ""
        return [f"background-color:{bg}" if bg else ""] * len(row)

    return df.style.apply(f, axis=1)


def show_df(df, status_col=None, height=None):
    obj = color_rows(df, status_col) if status_col else df
    kw = {"hide_index": True, "width": "stretch"}
    if height:
        kw["height"] = height
    st.dataframe(obj, **kw)


def pseudo(lines, cur):
    html = "".join(
        f'<div class="{"cur" if i == cur else ""}">{"▶ " if i == cur else "  "}{l}</div>' for i, l in enumerate(lines)
    )
    st.markdown(f'<div class="pseudo">{html}</div>', unsafe_allow_html=True)


def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


# ============================================================
# 2. TRÌNH PHÁT TỪNG BƯỚC
# ============================================================
def player(key, steps, render):
    n = len(steps)
    idx = min(st.session_state.get(key, 0), n - 1)

    def go(v):
        st.session_state[key] = max(0, min(n - 1, v))

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.3, 1, 3])
    c1.button("⏮ Đầu", key=key + "_first", on_click=go, args=(0,), width="stretch")
    c2.button("◀ Lùi", key=key + "_prev", on_click=go, args=(idx - 1,), width="stretch", disabled=idx == 0)
    c3.button("Bước tiếp ▶", key=key + "_next", on_click=go, args=(idx + 1,), width="stretch",
              type="primary", disabled=idx == n - 1)
    c4.button("Cuối ⏭", key=key + "_last", on_click=go, args=(n - 1,), width="stretch")
    c5.progress((idx + 1) / n, text=f"Bước {idx + 1} / {n}")
    st.markdown(f'<div class="step-title">{steps[idx]["title"]}</div>', unsafe_allow_html=True)
    render(steps[idx])


# ============================================================
# 3. APRIORI – sinh các bước
# ============================================================
APRIORI_CODE = [
    "L1 ← quét CSDL, đếm từng mục, giữ mục có support ≥ minsup",
    "lặp với k = 2, 3, …",
    "   NỐI: ghép 2 tập trong L(k−1) có chung k−2 phần tử đầu → Ck",
    "   CẮT TỈA: bỏ ứng viên có tập con (k−1) phần tử ∉ L(k−1)",
    "   QUÉT CSDL: đếm support các ứng viên còn lại → Lk",
    "dừng khi không còn ứng viên; kết quả = L1 ∪ L2 ∪ …",
]


def apriori_steps(trans, ms):
    steps = []
    stat = {"scan": 0, "counted": 0, "pruned": 0, "freq": 0}
    cnt = Counter(i for _, t in trans for i in t)
    items = sorted(cnt)
    stat["scan"] = 1
    stat["counted"] = len(items)
    rows = []
    L = {}
    for i in items:
        ok = cnt[i] >= ms
        rows.append({"Ứng viên": fmt({i}), "Có trong hóa đơn": tid_str(tids_of(frozenset([i]), trans)),
                     "Đếm": cnt[i], "Kết quả": "Giữ ✓" if ok else f"Loại ✗ (< {ms})"})
        if ok:
            L[frozenset([i])] = cnt[i]
    stat["freq"] = len(L)
    all_freq = dict(L)
    steps.append(dict(
        title="k = 1 · Quét CSDL lần 1: đếm từng mặt hàng → L1", line=0, stat=dict(stat),
        text=f"Đếm số hóa đơn chứa từng mặt hàng. Mặt hàng nào có trong ít nhất <b>{ms}</b> hóa đơn thì giữ lại. "
             f"→ <b>L1 có {len(L)} tập</b>: {', '.join(fmt(s) for s in sorted(L, key=sorted))}.",
        tables=[("C1 → L1", pd.DataFrame(rows), "Kết quả")]))
    k = 2
    while L:
        prev = sorted(tuple(sorted(s)) for s in L)
        joins = []
        for a, b in itertools.combinations(prev, 2):
            if a[:-1] == b[:-1]:
                joins.append((a, b, frozenset(a) | frozenset(b)))
        m = len({i for s in L for i in s})
        brute = len(list(itertools.combinations(range(m), k))) if m >= k else 0
        if not joins:
            steps.append(dict(
                title=f"k = {k} · Nối: không ghép được ứng viên nào → dừng", line=5, stat=dict(stat),
                text=f"L{k-1} chỉ có {len(L)} tập và không có hai tập nào chung {k-2} phần tử đầu, "
                     f"nên không sinh được C{k}. Thuật toán dừng.", tables=[]))
            break
        jrows = [{"Tập thứ nhất": fmt(a), "Tập thứ hai": fmt(b),
                  "Chung (k−2) phần tử đầu": fmt(a[:-1]) if k > 2 else "— (k = 2: ghép mọi cặp)",
                  "→ Ứng viên": fmt(c)} for a, b, c in joins]
        steps.append(dict(
            title=f"k = {k} · Nối L{k-1} với chính nó → C{k}", line=2, stat=dict(stat),
            text=f"Chỉ ghép hai tập trong L{k-1} có chung {k-2} phần tử đầu (đã sắp theo thứ tự chữ cái). "
                 f"Được <b>{len(joins)} ứng viên</b>. Nếu vét cạn thì phải xét <b>{brute}</b> tập {k} phần tử "
                 f"từ {m} mặt hàng còn lại.",
            tables=[(f"Phép nối sinh C{k}", pd.DataFrame(jrows), None)]))
        prows, kept = [], []
        for a, b, c in joins:
            subs = [frozenset(s) for s in itertools.combinations(sorted(c), k - 1)]
            bad = [s for s in subs if s not in L]
            prows.append({"Ứng viên": fmt(c),
                          f"Các tập con {k-1} phần tử": "   ".join(fmt(s) + (" ✓" if s in L else " ✗") for s in subs),
                          "Kết quả": f"Cắt tỉa ✂ vì {fmt(bad[0])} không phổ biến" if bad else "Giữ để đếm"})
            if not bad:
                kept.append(c)
        npr = len(joins) - len(kept)
        stat["pruned"] += npr
        msg = (f"<b>{npr} ứng viên bị cắt tỉa</b> mà không cần quét CSDL: tập con của nó đã không phổ biến thì nó "
               f"chắc chắn không phổ biến (tính chất Apriori)." if npr else
               "Mọi tập con của các ứng viên đều nằm trong L" + str(k - 1) + " → không ứng viên nào bị cắt tỉa ở mức này.")
        steps.append(dict(
            title=f"k = {k} · Cắt tỉa C{k} bằng tính chất Apriori", line=3, stat=dict(stat),
            text=f"Kiểm tra mọi tập con {k-1} phần tử của từng ứng viên. {msg}",
            tables=[(f"Cắt tỉa C{k}", pd.DataFrame(prows).replace({"Giữ để đếm": "Giữ để đếm"}), "Kết quả")]))
        if not kept:
            steps.append(dict(
                title=f"k = {k} · Không còn ứng viên để đếm → dừng", line=5, stat=dict(stat),
                text="Mọi ứng viên đều đã bị cắt tỉa nên không cần quét CSDL nữa.", tables=[]))
            break
        stat["scan"] += 1
        stat["counted"] += len(kept)
        crows, newL = [], {}
        for c in sorted(kept, key=sorted):
            ts = tids_of(c, trans)
            ok = len(ts) >= ms
            crows.append({"Ứng viên": fmt(c), "Có trong hóa đơn": tid_str(ts), "Đếm": len(ts),
                          "Kết quả": "Giữ ✓" if ok else f"Loại ✗ (< {ms})"})
            if ok:
                newL[c] = len(ts)
        stat["freq"] += len(newL)
        steps.append(dict(
            title=f"k = {k} · Quét CSDL lần {stat['scan']}: đếm C{k} → L{k}", line=4, stat=dict(stat),
            text=f"Chỉ đếm {len(kept)} ứng viên còn lại. → <b>L{k} có {len(newL)} tập</b>"
                 + (f": {', '.join(fmt(s) for s in sorted(newL, key=sorted))}." if newL else
                    ". L" + str(k) + " rỗng nên thuật toán dừng."),
            tables=[(f"C{k} → L{k}", pd.DataFrame(crows), "Kết quả")]))
        all_freq.update(newL)
        L = newL
        k += 1
    m_all = len(cnt)
    srows = [{"Tập phổ biến": fmt(s), "k": len(s), "Support": f"{v}/{len(trans)} = {v/len(trans):.0%}"}
             for s, v in sorted(all_freq.items(), key=lambda x: (len(x[0]), sorted(x[0])))]
    steps.append(dict(
        title=f"Kết quả: {len(all_freq)} tập phổ biến", line=5, stat=dict(stat),
        text=f"Apriori quét CSDL <b>{stat['scan']} lần</b>, chỉ đếm <b>{stat['counted']} ứng viên</b> "
             f"và cắt tỉa <b>{stat['pruned']}</b> ứng viên không cần đếm. "
             f"Vét cạn phải xét 2<sup>{m_all}</sup> − 1 = <b>{2**m_all - 1}</b> tập con.",
        tables=[("Tất cả tập phổ biến", pd.DataFrame(srows), None)]))
    return steps, all_freq


def render_apriori(step):
    left, right = st.columns([3, 2], gap="large")
    with left:
        explain(step["text"])
        for name, df, col in step["tables"]:
            st.markdown(f"**{name}**")
            show_df(df, col)
    with right:
        s = step["stat"]
        a, b = st.columns(2)
        a.metric("Lần quét CSDL", s["scan"])
        b.metric("Ứng viên đã đếm", s["counted"])
        a.metric("Bị cắt tỉa (không đếm)", s["pruned"])
        b.metric("Tập phổ biến tìm được", s["freq"])
        st.markdown("**Mã giả – dòng đang chạy**")
        pseudo(APRIORI_CODE, step["line"])


# ============================================================
# 4. FP-GROWTH – sinh các bước
# ============================================================
FP_CODE = [
    "Quét lần 1: đếm từng mục, bỏ mục < minsup, sắp giảm dần → F-list",
    "Quét lần 2: với mỗi hóa đơn, sắp theo F-list rồi chèn vào FP-tree",
    "   (tiền tố trùng thì đi chung nhánh, tăng số đếm)",
    "Duyệt F-list từ dưới lên, với mỗi mục x:",
    "   gom các đường tiền tố dẫn tới x → cơ sở mẫu điều kiện",
    "   đếm gộp, bỏ mục < minsup → cây điều kiện của x",
    "   khai phá đệ quy cây điều kiện, ghép thêm x",
    "Kết quả = mọi tập sinh ra trong quá trình đệ quy",
]


class FPNode:
    _id = 0

    def __init__(self, item, parent):
        FPNode._id += 1
        self.id = f"n{FPNode._id}"
        self.item, self.parent, self.count, self.children = item, parent, 0, {}


def fp_nodes(root):
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(reversed(list(n.children.values())))
    return out


def tree_dot(root, new=(), inc=(), focus=(), path=(), links=False, flist=None, title=None):
    lines = ['digraph G {', 'graph [rankdir=TB, nodesep=0.35, ranksep=0.45, bgcolor="transparent"];',
             f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="{NAVY}", '
             f'fillcolor="#EEF2F8", fontcolor="{NAVY}", penwidth=1.2];', f'edge [color="{NAVY}", arrowsize=0.6];']
    if title:
        lines.append(f'labelloc="t"; label="{title}"; fontname="Helvetica"; fontsize=14; fontcolor="{NAVY}";')
    for n in fp_nodes(root):
        if n.item is None:
            lines.append(f'{n.id} [label="∅ (gốc)", fillcolor="white"];')
            continue
        fill, pen = "#EEF2F8", "1.2"
        if n.id in path:
            fill = "#D6E4FB"
        if n.id in inc:
            fill, pen = "#FBEFC9", "2.4"
        if n.id in new:
            fill, pen = "#CDEBD9", "2.4"
        if n.id in focus:
            fill, pen = "#FBEFC9", "2.6"
        lines.append(f'{n.id} [label="{n.item} : {n.count}", fillcolor="{fill}", penwidth={pen}];')
        lines.append(f'{n.parent.id} -> {n.id};')
    if links and flist:
        by = defaultdict(list)
        for n in fp_nodes(root):
            if n.item is not None:
                by[n.item].append(n)
        for it in flist:
            for a, b in zip(by[it], by[it][1:]):
                lines.append(f'{a.id} -> {b.id} [style=dashed, color="{GOLD}", constraint=false, arrowsize=0.5];')
    lines.append("}")
    return "\n".join(lines)


def build_tree(weighted, ms):
    cnt = Counter()
    for t, w in weighted:
        for i in t:
            cnt[i] += w
    flist = [i for i, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if c >= ms]
    rank = {i: r for r, i in enumerate(flist)}
    root = FPNode(None, None)
    header = {i: [] for i in flist}
    for t, w in weighted:
        node = root
        for i in sorted((i for i in t if i in rank), key=rank.get):
            if i not in node.children:
                node.children[i] = FPNode(i, node)
                header[i].append(node.children[i])
            node = node.children[i]
            node.count += w
    return root, header, flist, cnt


def fp_mine(weighted, ms, suffix, out):
    root, header, flist, cnt = build_tree(weighted, ms)
    for item in reversed(flist):
        new = frozenset(suffix | {item})
        out[new] = cnt[item]
        base = []
        for node in header[item]:
            p, path = node.parent, []
            while p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                base.append((path[::-1], node.count))
        if base:
            fp_mine(base, ms, set(new), out)


def fpgrowth_steps(trans, ms):
    steps = []
    cnt = Counter(i for _, t in trans for i in t)
    flist = [i for i, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if c >= ms]
    rank = {i: r for r, i in enumerate(flist)}
    rows = [{"Mục": i, "Support": c, "Kết quả": "Giữ ✓" if c >= ms else f"Loại ✗ (< {ms})"}
            for i, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))]
    root = FPNode(None, None)
    steps.append(dict(
        title="Quét CSDL lần 1: đếm từng mục → F-list", line=0, dot=tree_dot(root),
        text=f"Đếm support từng mục, bỏ mục dưới minsup = {ms}, sắp giảm dần theo support. "
             f"<b>F-list: {' → '.join(f'{i} ({cnt[i]})' for i in flist) or '∅'}</b>. "
             f"Mọi hóa đơn sẽ được sắp lại theo thứ tự này trước khi chèn vào cây.",
        tables=[("Đếm support và F-list", pd.DataFrame(rows), "Kết quả")]))
    header = {i: [] for i in flist}
    for tid, t in trans:
        items = sorted((i for i in t if i in rank), key=rank.get)
        node, new, inc = root, [], []
        for i in items:
            if i not in node.children:
                node.children[i] = FPNode(i, node)
                header[i].append(node.children[i])
                new.append(node.children[i].id)
                desc_new = True
            else:
                inc.append(node.children[i].id)
            node = node.children[i]
            node.count += 1
        shared = [n for n in items[:len(inc)]]
        created = items[len(inc):]
        parts = []
        if shared:
            parts.append(f"đi chung nhánh có sẵn <b>{' → '.join(shared)}</b> (mỗi nút +1, tô vàng)")
        if created:
            parts.append(f"tạo nút mới <b>{' → '.join(created)}</b> (tô xanh)")
        dropped = sorted(i for i in t if i not in rank)
        steps.append(dict(
            title=f"Quét CSDL lần 2: chèn hóa đơn {tid} vào FP-tree", line=1 if created else 2,
            dot=tree_dot(root, new=set(new), inc=set(inc)),
            text=f"<b>{tid}</b> = {fmt(t)} → sắp theo F-list: <b>{' → '.join(items) or '∅'}</b>"
                 + (f" (bỏ {', '.join(dropped)} vì không phổ biến)" if dropped else "") + ". "
                 + ("; ".join(parts) + "." if parts else "Không còn mục nào để chèn."),
            tables=[]))
    nnodes = len(fp_nodes(root)) - 1
    hrows = [{"Mục": i, "Support": cnt[i], "Số nút trên cây": len(header[i]),
              "Kiểm tra": " + ".join(str(n.count) for n in header[i]) + f" = {sum(n.count for n in header[i])}"}
             for i in flist]
    steps.append(dict(
        title=f"FP-tree hoàn chỉnh: {nnodes} nút (không tính gốc)", line=2,
        dot=tree_dot(root, links=True, flist=flist),
        text=f"Cả CSDL được nén vào cây {nnodes} nút sau đúng <b>2 lần quét</b>. Đường nét đứt màu vàng là "
             f"<i>node-link</i> nối các nút cùng tên, giúp tìm nhanh mọi vị trí của một mục. "
             f"Từ đây <b>không cần quét CSDL nữa</b>.",
        tables=[("Header table", pd.DataFrame(hrows), None)]))
    total = {}
    for item in reversed(flist):
        base, focus, path_ids = [], set(), set()
        for node in header[item]:
            focus.add(node.id)
            p, path = node.parent, []
            while p.item is not None:
                path.append(p.item)
                path_ids.add(p.id)
                p = p.parent
            base.append((path[::-1], node.count))
        cond = Counter()
        for pth, w in base:
            for i in pth:
                cond[i] += w
        crow = [{"Mục": i, "Đếm gộp": c, "Kết quả": "Giữ ✓" if c >= ms else f"Loại ✗ (< {ms})"}
                for i, c in sorted(cond.items(), key=lambda x: (-x[1], x[0]))]
        found = {frozenset([item]): cnt[item]}
        nonempty = [(p, w) for p, w in base if p]
        if nonempty:
            fp_mine(nonempty, ms, {item}, found)
        total.update(found)
        croot, _, cflist, _ = build_tree(nonempty, ms) if nonempty else (FPNode(None, None), {}, [], {})
        cdot = tree_dot(croot, title=f"Cây điều kiện của {item}") if cflist else None
        base_txt = "; ".join(f"{fmt(p) if p else '∅ (ngay dưới gốc)'} : {w}" for p, w in base)
        steps.append(dict(
            title=f"Khai phá mục “{item}” (support {cnt[item]})", line=4,
            dot=tree_dot(root, focus=focus, path=path_ids, links=True, flist=flist), cdot=cdot,
            text=f"Các nút <b>{item}</b> tô vàng, đường tiền tố tô xanh nhạt. "
                 f"Cơ sở mẫu điều kiện: <b>{base_txt}</b>. "
                 + (f"Cây điều kiện gồm {', '.join(cflist)} → khai phá đệ quy. " if cflist else
                    "Không mục nào đạt minsup → cây điều kiện rỗng. ")
                 + f"<br>→ Tập phổ biến thu được: <b>{', '.join(fmt(s) for s in sorted(found, key=lambda s: (len(s), sorted(s))))}</b>",
            tables=[("Đếm gộp trên cơ sở mẫu", pd.DataFrame(crow), "Kết quả")] if crow else []))
    srows = [{"Tập phổ biến": fmt(s), "k": len(s), "Support": f"{v}/{len(trans)} = {v/len(trans):.0%}"}
             for s, v in sorted(total.items(), key=lambda x: (len(x[0]), sorted(x[0])))]
    steps.append(dict(
        title=f"Kết quả: {len(total)} tập phổ biến", line=7, dot=tree_dot(root, links=True, flist=flist),
        text=f"FP-Growth chỉ quét CSDL <b>2 lần</b> và <b>không sinh tập ứng viên</b> nào: mọi tập phổ biến đều "
             f"được đọc ra từ cây.", tables=[("Tất cả tập phổ biến", pd.DataFrame(srows), None)]))
    return steps, total


def render_fp(step):
    left, right = st.columns([2, 3], gap="large")
    with left:
        explain(step["text"])
        for name, df, col in step["tables"]:
            st.markdown(f"**{name}**")
            show_df(df, col)
        st.markdown("**Mã giả – dòng đang chạy**")
        pseudo(FP_CODE, step["line"])
    with right:
        st.graphviz_chart(step["dot"], width="content")
        if step.get("cdot"):
            st.graphviz_chart(step["cdot"], width="content")
        st.markdown('<span class="small-note">Xanh lá: nút mới tạo · Vàng: nút được tăng đếm / mục đang khai phá · '
                    'Xanh nhạt: đường tiền tố · Nét đứt: node-link</span>', unsafe_allow_html=True)


# ============================================================
# 5. ECLAT – sinh các bước
# ============================================================
ECLAT_CODE = [
    "Quét CSDL 1 lần: chuyển sang dạng dọc, mỗi mục x có TID-list t(x)",
    "giữ các mục có |t(x)| ≥ minsup, sắp theo thứ tự",
    "duyệt theo chiều sâu (DFS), với mỗi cặp X, Y cùng tiền tố:",
    "   t(X ∪ Y) = t(X) ∩ t(Y),  support = |t(X ∪ Y)|",
    "   nếu support ≥ minsup: giữ, tiếp tục đi sâu",
    "Kết quả = mọi nút được giữ trên cây duyệt",
]


def eclat_dot(nodes, edges, cur=None):
    lines = ['digraph E {', 'graph [rankdir=LR, nodesep=0.18, ranksep=0.5, bgcolor="transparent"];',
             f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="{NAVY}", fontcolor="{NAVY}"];',
             f'edge [color="{GRAY}", arrowsize=0.6];', 'root [label="∅", fillcolor="white"];']
    for nid, (label, ok) in nodes.items():
        fill = "#E7F3EC" if ok else "#FBEAE8"
        style = '"rounded,filled"' if ok else '"rounded,filled,dashed"'
        pen = "3" if nid == cur else "1.2"
        col = GOLD if nid == cur else (NAVY if ok else RED)
        lines.append(f'{nid} [label="{label}", fillcolor="{fill}", style={style}, penwidth={pen}, color="{col}"];')
    for a, b in edges:
        lines.append(f"{a} -> {b};")
    lines.append("}")
    return "\n".join(lines)


def eclat_steps(trans, ms):
    steps = []
    tid = defaultdict(list)
    for name, t in trans:
        for i in t:
            tid[i].append(name)
    allitems = sorted(tid)
    rows = [{"Mục": i, "TID-list t(x)": tid_str(tid[i]), "Support": len(tid[i]),
             "Kết quả": "Giữ ✓" if len(tid[i]) >= ms else f"Loại ✗ (< {ms})"} for i in allitems]
    nodes, edges = {}, []
    ids = {}

    def nid(s):
        if s not in ids:
            ids[s] = f"e{len(ids) + 1}"
        return ids[s]

    start = [(i, tid[i]) for i in allitems if len(tid[i]) >= ms]
    result = {}
    for i, ts in start:
        s = frozenset([i])
        nodes[nid(s)] = (f"{i}\\n{tid_str(ts)} · {len(ts)}", True)
        edges.append(("root", nid(s)))
    vrows = pd.DataFrame(rows)
    steps.append(dict(
        title="Quét CSDL 1 lần: chuyển sang dạng dọc (TID-list)", line=0,
        dot=eclat_dot(dict(nodes), list(edges)),
        text="Thay vì lưu “mỗi hóa đơn có những món gì”, ECLAT lưu “mỗi món nằm ở những hóa đơn nào”. "
             f"Support của một mục chính là độ dài TID-list. Có <b>{len(start)}</b> mục đạt minsup = {ms}.",
        tables=[("Dạng dọc", vrows, "Kết quả")]))

    def dfs(prefix, cands):
        for idx, (item, ts) in enumerate(cands):
            new = frozenset(prefix | {item})
            result[new] = len(ts)
            ext = []
            for item2, ts2 in cands[idx + 1:]:
                inter = [x for x in ts if x in set(ts2)]
                ok = len(inter) >= ms
                child = frozenset(new | {item2})
                nodes[nid(child)] = (f"{', '.join(sorted(child))}\\n{tid_str(inter)} · {len(inter)}", ok)
                edges.append((nid(new), nid(child)))
                left_s = fmt(new)
                right_s = fmt(prefix | {item2})
                steps.append(dict(
                    title=f"Giao TID-list: {left_s} ∩ {right_s}", line=4 if ok else 3,
                    dot=eclat_dot(dict(nodes), list(edges), cur=nid(child)),
                    text=f"t({left_s}) ∩ t({right_s}) = {tid_str(ts)} ∩ {tid_str(ts2)} = <b>{tid_str(inter)}</b> "
                         f"→ support({fmt(child)}) = <b>{len(inter)}</b> "
                         + (f"≥ {ms} → <b>giữ</b>, sẽ đi sâu tiếp từ nút này." if ok else
                            f"< {ms} → <b>loại</b> (khung đỏ nét đứt), không đi tiếp nhánh này.")
                         + " Không cần quét lại CSDL, chỉ giao hai danh sách.",
                    tables=[]))
                if ok:
                    ext.append((item2, inter))
            if ext:
                dfs(set(new), ext)

    dfs(set(), start)
    srows = [{"Tập phổ biến": fmt(s), "k": len(s), "Support": f"{v}/{len(trans)} = {v/len(trans):.0%}"}
             for s, v in sorted(result.items(), key=lambda x: (len(x[0]), sorted(x[0])))]
    n_inter = sum(1 for s in steps if s["title"].startswith("Giao"))
    steps.append(dict(
        title=f"Kết quả: {len(result)} tập phổ biến", line=5, dot=eclat_dot(dict(nodes), list(edges)),
        text=f"ECLAT quét CSDL <b>1 lần</b>, sau đó chỉ thực hiện <b>{n_inter} phép giao</b> TID-list theo chiều sâu. "
             "Các nút xanh trên cây là tập phổ biến.",
        tables=[("Tất cả tập phổ biến", pd.DataFrame(srows), None)]))
    return steps, result


def render_eclat(step):
    left, right = st.columns([2, 3], gap="large")
    with left:
        explain(step["text"])
        for name, df, col in step["tables"]:
            st.markdown(f"**{name}**")
            show_df(df, col)
        st.markdown("**Mã giả – dòng đang chạy**")
        pseudo(ECLAT_CODE, step["line"])
    with right:
        st.graphviz_chart(step["dot"], width="content")
        st.markdown('<span class="small-note">Mỗi nút: tập mục · TID-list · support. Xanh: giữ · Đỏ nét đứt: loại · '
                    'Viền vàng: phép giao đang xét</span>', unsafe_allow_html=True)


# ============================================================
# 6. SINH LUẬT
# ============================================================
def gen_rules(freq, n, minconf):
    rows = []
    for L, nL in freq.items():
        if len(L) < 2:
            continue
        for r in range(1, len(L)):
            for A in itertools.combinations(sorted(L), r):
                A = frozenset(A)
                Y = L - A
                if A not in freq or Y not in freq:
                    continue
                conf = nL / freq[A]
                lift = conf / (freq[Y] / n)
                rows.append(dict(A=A, Y=Y, L=L, nL=nL, nA=freq[A], nY=freq[Y], support=nL / n,
                                 confidence=conf, lift=lift, strong=conf >= minconf))
    return rows


# ============================================================
# 7. GROCERIES – phiên bản chạy nhanh (không ghi bước)
# ============================================================
def apriori_fast(transactions, ms):
    counts = Counter(i for t in transactions for i in t)
    L = {frozenset([i]): c for i, c in counts.items() if c >= ms}
    allf = dict(L)
    k, ncand = 2, len(counts)
    while L:
        prev = sorted(tuple(sorted(s)) for s in L)
        prev_set = set(L)
        cands = set()
        for a, b in itertools.combinations(prev, 2):
            if a[:-1] == b[:-1]:
                c = frozenset(a) | frozenset(b)
                ncand += 1
                if all(frozenset(s) in prev_set for s in itertools.combinations(sorted(c), k - 1)):
                    cands.add(c)
        if not cands:
            break
        cnt = Counter()
        ci = set().union(*cands)
        for t in transactions:
            for c in itertools.combinations(sorted(i for i in t if i in ci), k):
                fc = frozenset(c)
                if fc in cands:
                    cnt[fc] += 1
        L = {c: n for c, n in cnt.items() if n >= ms}
        allf.update(L)
        k += 1
    return allf, ncand


def fpgrowth_fast(transactions, ms):
    out = {}
    fp_mine([(t, 1) for t in transactions], ms, set(), out)
    return out


def eclat_fast(transactions, ms):
    tid = defaultdict(set)
    for k, t in enumerate(transactions):
        for i in t:
            tid[i].add(k)
    res = {}

    def dfs(prefix, cands):
        for idx, (item, ts) in enumerate(cands):
            new = prefix | {item}
            res[frozenset(new)] = len(ts)
            ext = [(i2, ts & t2) for i2, t2 in cands[idx + 1:] if len(ts & t2) >= ms]
            if ext:
                dfs(new, ext)

    dfs(set(), sorted(((i, s) for i, s in tid.items() if len(s) >= ms), key=lambda x: x[0]))
    return res


VI = {"whole milk": "sữa tươi", "other vegetables": "rau khác", "rolls/buns": "bánh mì cuộn", "soda": "nước ngọt",
      "yogurt": "sữa chua", "bottled water": "nước đóng chai", "root vegetables": "rau củ", "tropical fruit": "trái cây nhiệt đới",
      "shopping bags": "túi mua sắm", "sausage": "xúc xích", "pastry": "bánh ngọt", "citrus fruit": "cam quýt",
      "bottled beer": "bia chai", "newspapers": "báo", "canned beer": "bia lon", "pip fruit": "táo, lê",
      "fruit/vegetable juice": "nước ép", "whipped/sour cream": "kem tươi", "brown bread": "bánh mì đen",
      "domestic eggs": "trứng", "frankfurter": "xúc xích Đức", "margarine": "bơ thực vật", "coffee": "cà phê",
      "pork": "thịt heo", "butter": "bơ", "curd": "sữa đông", "beef": "thịt bò", "napkins": "khăn giấy",
      "chocolate": "sô-cô-la", "frozen vegetables": "rau đông lạnh", "chicken": "thịt gà", "white bread": "bánh mì trắng",
      "cream cheese": "phô mai kem", "waffles": "bánh quế", "salty snack": "snack mặn", "dessert": "tráng miệng",
      "sugar": "đường", "UHT-milk": "sữa tiệt trùng", "hamburger meat": "thịt xay", "berries": "quả mọng",
      "onions": "hành tây", "candy": "kẹo", "hygiene articles": "đồ vệ sinh", "specialty chocolate": "sô-cô-la đặc biệt",
      "long life bakery product": "bánh để lâu"}


def vi(x):
    return f"{VI[x]} ({x})" if x in VI else x


def vis(s):
    return "{" + ", ".join(VI.get(i, i) for i in sorted(s)) + "}"


@st.cache_data(show_spinner=False)
def load_groceries(text):
    return [frozenset(x.strip() for x in line.split(",") if x.strip()) for line in text.splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def time_all(text, ms_frac):
    G = load_groceries(text)
    mc = int(ms_frac * len(G) + 0.999)
    out = {}
    t0 = time.perf_counter(); fa, ncand = apriori_fast(G, mc); out["Apriori"] = time.perf_counter() - t0
    t0 = time.perf_counter(); ff = fpgrowth_fast(G, mc); out["FP-Growth"] = time.perf_counter() - t0
    t0 = time.perf_counter(); fe = eclat_fast(G, mc); out["ECLAT"] = time.perf_counter() - t0
    same = fa == ff == fe
    return out, len(fe), ncand, same


@st.cache_data(show_spinner=False)
def rec_rules(text, ms_frac=0.005, minconf=0.2):
    G = load_groceries(text)
    freq = eclat_fast(G, int(ms_frac * len(G) + 0.999))
    return gen_rules(freq, len(G), minconf), len(G)


# ============================================================
# 8. GIAO DIỆN
# ============================================================
st.title("🛒 Luật kết hợp: xem thuật toán chạy từng bước")
st.caption("Nhóm 6 · Học máy · Apriori · FP-Growth · ECLAT — chọn dữ liệu và minsup ở thanh bên trái, "
           "rồi bấm “Bước tiếp ▶” trong từng tab.")

with st.sidebar:
    st.header("⚙️ Dữ liệu và ngưỡng")
    preset = st.selectbox("Bộ dữ liệu", list(PRESETS))
    st.markdown('<span class="small-note">Có thể sửa, thêm hoặc xóa dòng trực tiếp trong bảng '
                '(mặt hàng cách nhau bằng dấu phẩy).</span>', unsafe_allow_html=True)
    base_df = pd.DataFrame(PRESETS[preset], columns=["TID", "Mặt hàng"])
    edited = st.data_editor(base_df, num_rows="dynamic", hide_index=True, width="stretch",
                            key="editor_" + preset)
    trans = []
    for _, r in edited.iterrows():
        tid_v = str(r["TID"]).strip() if pd.notna(r["TID"]) else ""
        items_v = frozenset(x.strip() for x in str(r["Mặt hàng"] if pd.notna(r["Mặt hàng"]) else "").split(",") if x.strip())
        if tid_v and items_v:
            trans.append((tid_v, items_v))
    n = len(trans)
    if n == 0:
        st.error("Bảng dữ liệu đang trống.")
        st.stop()
    ms = st.slider("minsup (số hóa đơn tối thiểu)", 1, n, min(3, n))
    st.markdown(f"→ minsup = {ms}/{n} = **{ms / n:.0%}**")
    minconf = st.slider("minconf (dùng ở tab Sinh luật)", 0.1, 1.0, 0.7, 0.05)

sig = (preset, tuple((t, tuple(sorted(s))) for t, s in trans), ms)
if st.session_state.get("sig") != sig:
    st.session_state["sig"] = sig
    for k in ("ap", "fp", "ec"):
        st.session_state[k] = 0

ap_steps, f_ap = apriori_steps(trans, ms)
FPNode._id = 0
fp_steps, f_fp = fpgrowth_steps(trans, ms)
ec_steps, f_ec = eclat_steps(trans, ms)

tabs = st.tabs(["① Apriori", "② FP-Growth", "③ ECLAT", "④ Đối chiếu & sinh luật", "⑤ Dữ liệu thật: Groceries"])

with tabs[0]:
    player("ap", ap_steps, render_apriori)
with tabs[1]:
    player("fp", fp_steps, render_fp)
with tabs[2]:
    player("ec", ec_steps, render_eclat)

with tabs[3]:
    st.subheader("Ba thuật toán có cho cùng kết quả không?")
    allsets = sorted(set(f_ap) | set(f_fp) | set(f_ec), key=lambda s: (len(s), sorted(s)))
    try:
        from mlxtend.frequent_patterns import fpgrowth as mx_fp
        items_all = sorted({i for _, t in trans for i in t})
        oh = pd.DataFrame([[i in t for i in items_all] for _, t in trans], columns=items_all)
        mx = mx_fp(oh, min_support=ms / n - 1e-9, use_colnames=True)
        f_mx = {frozenset(s) for s in mx["itemsets"]}
    except Exception:
        f_mx = None
    rows = []
    for s in allsets:
        r = {"Tập phổ biến": fmt(s), "Support": f_ap.get(s, f_fp.get(s, f_ec.get(s))),
             "Apriori": "✓" if s in f_ap else "✗", "FP-Growth": "✓" if s in f_fp else "✗",
             "ECLAT": "✓" if s in f_ec else "✗"}
        if f_mx is not None:
            r["mlxtend (thư viện)"] = "✓" if s in f_mx else "✗"
        rows.append(r)
    same = f_ap == f_fp == f_ec and (f_mx is None or f_mx == set(f_ap))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if rows:
            show_df(pd.DataFrame(rows))
        else:
            st.info("Không có tập phổ biến nào với minsup này.")
    with c2:
        st.metric("Số tập phổ biến", len(f_ap))
        if same:
            st.success("Cả ba thuật toán nhóm tự cài đặt" + (" và thư viện mlxtend" if f_mx is not None else "")
                       + " cho **cùng một kết quả**. Khác nhau ở cách làm, không khác ở đáp án.")
        else:
            st.error("Kết quả không khớp!")
        longest = max((len(s) for s in f_ap), default=0)
        st.metric("Tập phổ biến dài nhất (k)", longest)

    st.subheader("Sinh luật từ tập phổ biến: A → (L \\ A)")
    rules = gen_rules(f_ap, n, minconf)
    if not rules:
        st.warning("Không còn tập phổ biến nào có từ 2 mục trở lên → **không sinh được luật nào**. "
                   "Thử giảm minsup.")
    else:
        strong = [r for r in rules if r["strong"]]
        good = [r for r in strong if r["lift"] > 1]
        a, b, c = st.columns(3)
        a.metric("Luật ứng viên", len(rules))
        b.metric(f"Luật mạnh (confidence ≥ {minconf:.0%})", len(strong))
        c.metric("Luật mạnh có lift > 1", len(good))
        rrows = []
        for r in sorted(rules, key=lambda r: (-r["strong"], -r["confidence"], -r["lift"])):
            if not r["strong"]:
                verdict = f"Loại ✗ (confidence < {minconf:.0%})"
            elif r["lift"] <= 1:
                verdict = "⚠ Mạnh nhưng lift ≤ 1: bỏ"
            else:
                verdict = "✓ Giữ"
            rrows.append({"Luật": f"{fmt(r['A'])} → {fmt(r['Y'])}", "Support": f"{r['support']:.0%}",
                          "Confidence": f"{r['nL']}/{r['nA']} = {r['confidence']:.0%}",
                          "Lift": f"{r['lift']:.2f}".replace(".", ","), "Kết luận": verdict})
        show_df(pd.DataFrame(rrows), "Kết luận")
        pick = st.selectbox("Giải thích cách tính một luật", [x["Luật"] for x in rrows])
        r = next(r for r in rules if f"{fmt(r['A'])} → {fmt(r['Y'])}" == pick)
        explain(
            f"<b>{pick}</b><br>"
            f"confidence = support({fmt(r['L'])}) / support({fmt(r['A'])}) = {r['nL']}/{r['nA']} = <b>{r['confidence']:.0%}</b>"
            f" — trong {r['nA']} hóa đơn có {fmt(r['A'])}, {r['nL']} hóa đơn có thêm {fmt(r['Y'])}.<br>"
            f"Khách bất kỳ mua {fmt(r['Y'])}: {r['nY']}/{n} = {r['nY']/n:.0%}.<br>"
            f"lift = {r['confidence']:.0%} / {r['nY']/n:.0%} = <b>{format(r['lift'], '.2f').replace('.', ',')}</b> → "
            + ("mua vế trái làm khả năng mua vế phải <b>tăng</b>." if r["lift"] > 1 else
               "mua vế trái <b>không</b> làm tăng khả năng mua vế phải → không nên dùng luật này."))

with tabs[4]:
    gpath = Path(__file__).with_name("groceries.csv")
    text = None
    if gpath.exists():
        text = gpath.read_text(encoding="utf-8")
    else:
        up = st.file_uploader("Không thấy groceries.csv cạnh app.py — chọn file để tải lên", type="csv")
        if up:
            text = up.read().decode("utf-8")
    if text is None:
        st.stop()
    G = load_groceries(text)
    a, b, c = st.columns(3)
    a.metric("Số hóa đơn", f"{len(G):,}".replace(",", "."))
    b.metric("Số mặt hàng", len({i for t in G for i in t}))
    c.metric("Trung bình món / hóa đơn", f"{sum(map(len, G)) / len(G):.2f}".replace(".", ","))

    st.subheader("Tốc độ: hạ minsup thì thuật toán nào chậm đi nhiều nhất?")
    choices = {"1%": 0.01, "0,5%": 0.005, "0,2%": 0.002, "0,1%": 0.001}
    sel = st.multiselect("Các mức minsup cần đo", list(choices), default=list(choices))
    if st.button("⏱ Chạy đo thời gian", type="primary"):
        st.session_state["timed"] = sel
    if st.session_state.get("timed"):
        trows = []
        with st.spinner("Đang chạy 3 thuật toán trên 9.835 hóa đơn…"):
            for lab in st.session_state["timed"]:
                t, nf, nc, same = time_all(text, choices[lab])
                trows.append({"minsup": lab, "Số tập phổ biến": nf, "Ứng viên Apriori phải xét": nc,
                              "Apriori (s)": round(t["Apriori"], 2), "FP-Growth (s)": round(t["FP-Growth"], 2),
                              "ECLAT (s)": round(t["ECLAT"], 2), "Cùng kết quả": "✓" if same else "✗"})
        tdf = pd.DataFrame(trows)
        import altair as alt
        order = list(tdf["minsup"])
        long = tdf.melt(id_vars="minsup", value_vars=["Apriori (s)", "FP-Growth (s)", "ECLAT (s)"],
                        var_name="Thuật toán", value_name="Giây")
        long["Thuật toán"] = long["Thuật toán"].str.replace(" (s)", "", regex=False)
        algos = ["Apriori", "FP-Growth", "ECLAT"]
        base_ch = alt.Chart(long).encode(
            x=alt.X("minsup:N", sort=order, title="minsup", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Thuật toán:N", sort=algos),
            y=alt.Y("Giây:Q", title="giây"),
            color=alt.Color("Thuật toán:N", sort=algos, scale=alt.Scale(domain=algos, range=[RED, GREEN, NAVY]),
                            legend=alt.Legend(orient="top", title=None)))
        chart = base_ch.mark_bar() + base_ch.mark_text(dy=-7, fontSize=11).encode(text=alt.Text("Giây:Q", format=".2f"))
        st.altair_chart(chart.properties(height=320), width="stretch")
        show_df(tdf)
        st.markdown('<span class="small-note">Ba thuật toán luôn cho cùng tập kết quả (cột cuối). '
                    'Kết quả đo được lưu lại, bấm lần sau hiện ngay.</span>', unsafe_allow_html=True)

    st.subheader("Gợi ý mua kèm (luật lọc theo lift)")
    rules_g, nG = rec_rules(text)
    freq_items = [i for i, _ in Counter(i for t in G for i in t).most_common()]
    gio = st.multiselect("Khách đã bỏ vào giỏ", freq_items, default=["whole milk", "yogurt"], format_func=vi)
    if gio:
        gs = set(gio)
        match = [r for r in rules_g if r["strong"] and r["A"] <= gs and not (r["Y"] & gs)]
        keep = sorted([r for r in match if r["lift"] > 1], key=lambda r: (-r["confidence"], -r["lift"]))
        drop = sorted([r for r in match if r["lift"] <= 1], key=lambda r: r["lift"])
        out, seen = [], set()
        for r in keep:
            if r["Y"] in seen:
                continue
            seen.add(r["Y"])
            out.append({"Gợi ý mua thêm": vis(r["Y"]), "Dựa trên luật": f"{vis(r['A'])} → {vis(r['Y'])}",
                        "Confidence": f"{r['confidence']:.0%}", "Lift": f"{r['lift']:.2f}".replace(".", ",")})
            if len(out) == 5:
                break
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            if out:
                show_df(pd.DataFrame(out))
            else:
                st.info("Không có luật phù hợp với giỏ này — thử thêm hoặc đổi món.")
        with c2:
            if drop:
                st.markdown("**Luật bị bỏ vì lift ≤ 1**")
                for r in drop[:3]:
                    base = r["nY"] / nG
                    explain(f"{vis(r['A'])} → {vis(r['Y'])}: confidence {r['confidence']:.0%}, nhưng khách bất kỳ "
                            f"đã mua {vis(r['Y'])} {base:.1%} → lift <b>{r['lift']:.2f}</b>, bỏ.".replace(".", ",")
                            .replace("bỏ,", "bỏ."))
            else:
                st.markdown('<span class="small-note">Không có luật nào bị loại vì lift ≤ 1 với giỏ này. '
                            'Thử giỏ chỉ có “nước ngọt (soda)”.</span>', unsafe_allow_html=True)
        st.markdown('<span class="small-note">Luật sinh với minsup 0,5% và minconf 20% trên toàn bộ Groceries; '
                    'chỉ giữ luật có vế trái nằm trong giỏ, vế phải chưa có trong giỏ và lift &gt; 1.</span>',
                    unsafe_allow_html=True)
