# -*- coding: utf-8 -*-
"""會議 ASR 評分 harness —— 拿黃金段量任意一份逐字稿，吐六個指標。

## 為什麼要有這支

`three-way-meeting-asr-bench-2026-07-16.md` §6：**無 ground truth → 無 WER/CER**。
06-30 / 07-08 / 07-16 三份 bench 的所有數字都是**相對**指標（覆蓋率、分歧率、
marker 密度），「誰比較準」一直無法回答，而 `（內部紀錄）plans/arkiv/2026-06-09-stt-bench-plan.md`
定的「**≥2% 絕對 CER 才換引擎**」門檻從未被量測。

黃金段（`golden.py`）給了 ref，這支給尺。

完整規格 → `（內部評測紀錄）`

## 指標

⚠️ 這張表是指標清單的 **SSOT** —— README 不複製它。
2026-09-11 之前 README 有一份自己的清單，停在第一個 commit 的狀態。

| 指標 | 怎麼算 | 為什麼是這樣算 |
|---|---|---|
| `cer_content` | ref＝`zh` 軌，**逐段串接**後算編輯距離 | 串接而非逐句比對，是為了不被切段風格懲罰（06-30 量到 whisper 2966 段 vs QwenASR 3567 段，逐句對齊會把切段差異讀成錯誤） |
| `verbosity` | 視窗內 hyp 字數 ÷ ref 字數 | 🔴 **CER 必須跟它並列讀**。對齊走自由端點，視窗內多餘輸出不算插入錯誤（刻意的取捨），而分工的另一半 `hallucination` 會在 cue 太粗時回 `None` ⇒ 一顆大 cue 可同時拿到「多餘文字零成本」與「幻覺率 n/a」。實測 1618 字垃圾包住 18 字答案 → CER 0.0000、膨脹 89.89x。接近 1 ⇒ CER 可引用；遠大於 1 ⇒ 那是「最佳子字串」不是「這份稿有多準」 |
| `taigi_retention_per10k` | 人標為台語的句子時間範圍內，引擎輸出含台語專屬字形的每萬字命中數 | **語言忠實度的主指標，不需要台語標準答案**。初版要人填 `native` 再相減，那條路作廢（標註者 不打台語中打，而 AI 回譯＝編造 ground truth）。抹平的引擎趨近 0、忠實的明顯 > 0；⚠️ 絕對值無意義，只能同一批句子跨引擎比 |
| `cer_native` | 同上但 ref＝`native` 軌。**次要** | `native` 是選填 ⇒ 沒填就回 `None`，不回一個假的 0 |
| `taigi_flatten_gap` | `cer_native - cer_content`。**次要** | 07-08「Qwen 忠實、Whisper 抹平」那個質性判讀的量化版。只在真的有人填了 `native` 時才有值 —— 主指標已改成 `taigi_retention` |
| `coverage` | 每句收**所有**有時間重疊的 hyp 段、串起來，該句子字串對齊 CER ≤0.5 才算涵蓋 | 三份 bench 一直在報覆蓋率，但**從來沒有真分母**（拿引擎自己的輸出當分母）。這裡分母是人工標的語音秒數 |
| `hallucination` | 視窗內與任何黃金句零重疊的 hyp **窗內**秒數 ÷ 視窗內 hyp 總秒數。判不動時回 `None`，理由記在 `hallucination_blocked_by` | **whisper-guard 真正的 KPI。** 三種判不動：①`cue_too_coarse`（中位 cue > 黃金句中位的 4 倍 —— 一顆 296 秒的 cue 重疊到段內每一句，永遠算不出幻覺，回 0 會被讀成「這家不幻覺」）②`end_estimated`（來源只給起點、end 是我們補的 ⇒ 分子分母都建在自編時間上）③`no_hyp_in_window`（分母 0）。🔴 **秒數一律夾到視窗內、判不動的段不進 overall 分母** —— 兩者都是 2026-09-11 審計修的，偏差方向固定：cue 越粗讀數越靠近 0 |
| 型號 `precision`／`recall` | canonical 正規化後的多重集比對 | 0716 只量了 recall，而**只量命中會鼓勵亂猜** |

`rtf` 不在這裡算 —— 它必須**當場量**（0716：0.093 → 實測 0.142；mlx 名目 0.126、
記憶體吃緊時 0.509）。給了 `asr_sec` 才會填。

## 0716 的教訓直接寫進正規化層

那次「型號命中率」做失敗了：拿正本型號清單直接字串比對，三線命中數全趨近 0，
因為各引擎的**數字書寫慣例不同**（中文數字↔阿拉伯數字、省連字號）
→ **量到的是書寫格式，不是準確率**。所以：

1. 正規化順序**寫死**（NFKC → 中文數字 → 大寫 → 剝標點 → 選擇性去填充詞）。
   順序會改變結果，所以不可由呼叫端決定。
2. 型號比對前兩邊都過同一支 `normalize()`，`MX-K`／`MX K`／`ｍｘ－ｋ` 同一形。
3. 型號偵測**長→短排序 ＋ 遮罩已匹配區間**，否則 `204` 會匹配進 `204-D` 裡面
   （跟 `term_dict.py` 的「長→短排序」是同一條紀律）。

## 🔴 對齊用近似子字串，不用時間分桶（2026-09-10 實測後改）

初版按「cue 中點落在黃金段視窗內」收 hyp，然後整串比 Levenshtein。
**兩個假設都不成立**：雅婷的 cue 平均 12 秒、而蓋住 G3 的那顆長達 **296.8 秒**
→ 中點落在窗外，G3 收到 0 字被判 CER 100%；改成「有重疊就收」→
收到 1043 字比 228 字的 ref，CER 判 187%。
**兩種時間分桶量的都是切段風格，不是準確率。**

現行：hyp 收「有重疊」的全部 cue，ref 對它做 `levenshtein_substring`
（hyp 兩端自由）。視窗外的多餘輸出不算插入錯誤，那部分由 `hallucination` 另計。

## 刻意不吃 rapidfuzz

0716 用了 `rapidfuzz.distance.Levenshtein`。這支改用 stdlib DP：5 分鐘黃金段規模
（~2000 字）是毫秒級，而**少一個依賴＝任何機器都跑得起來、測試永遠會跑**
（對照 `test_meeting_chunking.py` 檔頭的同一個理由）。兩套實作＝會漂。

## 🔴 用它下結論前，五條負向測試必須先全綠

量尺沒過負向測試就評分，等於用沒校準的尺量東西
（per memory `feedback_verifier_needs_its_own_negative_test`）。
測試在 `tests/acoustic/test_meeting_score.py`，其中兩條是這次新加的：
**時間戳平移 +30s 覆蓋率必須大幅下降**（不然覆蓋率根本沒在用時間軸）、
**型號換成別的必須得 0**（不然正規化過度歸一、把不同型號當同一個）。
"""
import argparse
import json
import os
import re
import sys
import unicodedata

try:
    from golden import load_golden, ref_text
except ImportError:  # 被當套件 import 時
    from .golden import load_golden, ref_text  # type: ignore

# 單一中文數字。〇 與 零 都收（ASR 兩種都會吐）；兩 也收（`兩萬` 不收會變成 `兩0`）。
# ⚠️ 已知副作用：`兩邊` → `2邊`、`零件` → `0件`。**兩邊都過同一支正規化，所以 CER 不受影響**，
# 而收了才接得住「ASR 寫 2 萬、人打兩萬」這種真實差異。
_CJK_DIGITS = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "兩": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}
_CJK_UNITS = {"十": 10, "百": 100, "千": 1000}
_CJK_NUM_RE = re.compile("[" + "".join(list(_CJK_DIGITS) + list(_CJK_UNITS) + ["萬"]) + "]+")

# 保守的填充詞集合。**預設不去**（見 plan §1.3：各引擎處理差異大，本身就是要量的維度）。
# 刻意不收 啊／喔／啦／嘛／吧 —— 那些同時是承載語意的句末助詞，去掉會改內容。
FILLERS = ("呃", "嗯", "欸", "唔", "哦")

# 台語**專屬**字形（教育部推薦漢字裡在國語散文中幾乎不出現的那些）。
# 用途：`taigi_retention` —— **不需要台語標準答案**的忠實度指標。
#
# 🔴 為什麼需要一個不需參考答案的指標：`native` 軌原本要由人填台語原形，
#    但 標註者 不會打台語中打，而由 CC 從國語回譯 ＝ 自己製造 ground truth
#    （引擎寫出另一個同樣正確的台語形會被判錯）。改成只需要人標出「哪幾句是台語」。
#
# ⚠️ **絕對值沒有意義，只能拿來比引擎。** 這是啟發式字表，不是語言學判準。
# 刻意排除的三類（都會製造假命中）：
#   ① 國台通用字：較／講／嘛／無／食／伊 —— 命中上百次也不構成證據
#   ② 國語裡也用得到的：遮（遮蔽）／予（給予）／咧 —— 2026-09-10 實測這兩個正是唯一命中
#   ③ **簡體同形**：个（＝簡體「個」）—— Whisper 未過 opencc 前吐簡體，會整份假陽性
TAIGI_EXCLUSIVE = (
    "迄", "佇", "毋", "袂", "恁", "阮", "囡", "攏", "捌", "佮", "拄", "蹛", "媠", "濟",
    "逐家", "按怎", "這馬", "啥物", "歹勢", "彼个", "頭家", "無彩", "拍拚", "猶原", "敢若",
)


# ---------------------------------------------------------------- 正規化

def _cjk_run_to_arabic(run):
    """一串中文數字轉阿拉伯。

    沒有位數字（十百千萬）時走**逐字映射**：`二零四` → `204`（型號的口語讀法）。
    有位數字時走標準節解析：`二百零四` → `204`、`六十` → `60`、`十` → `10`。
    """
    if not any(c in _CJK_UNITS or c == "萬" for c in run):
        return "".join(_CJK_DIGITS.get(c, c) for c in run)
    total = 0
    section = 0
    num = 0
    for c in run:
        if c in _CJK_DIGITS:
            num = int(_CJK_DIGITS[c])
        elif c in _CJK_UNITS:
            section += (num or 1) * _CJK_UNITS[c]
            num = 0
        elif c == "萬":
            total += (section + num) * 10000
            section = 0
            num = 0
    return str(total + section + num)


def normalize(text, drop_fillers=False):
    """把文字壓成可比對形。**順序寫死，不給呼叫端調**（見檔頭）。"""
    if not text:
        return ""
    # 1. NFKC：全形英數／全形標點 → 半形，並攤平相容字
    out = unicodedata.normalize("NFKC", text)
    # 2. 中文數字 → 阿拉伯（在剝標點之前做，讓「一、二」不會先被黏成一串）
    out = _CJK_NUM_RE.sub(lambda m: _cjk_run_to_arabic(m.group(0)), out)
    # 3. 英文統一大寫
    out = out.upper()
    # 4. 剝標點（Unicode P*）、分隔（Z*）、控制字元（C*）與殘留空白
    kept = []
    for ch in out:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "Z", "C"):
            continue
        if ch.isspace():
            continue
        kept.append(ch)
    out = "".join(kept)
    # 5. 選擇性去填充詞
    if drop_fillers:
        for f in FILLERS:
            out = out.replace(f, "")
    return out


def normalize_model(token):
    """型號 canonical 形。刻意複用 `normalize()`，保證與稿面走同一條路。"""
    return normalize(token, drop_fillers=False)


# ---------------------------------------------------------------- 距離

def levenshtein(a, b):
    """編輯距離。滾動一列的 DP，O(len(a)) 記憶體。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,            # 刪
                           cur[j - 1] + 1,          # 插
                           prev[j - 1] + (ca != cb)))  # 替
        prev = cur
    return prev[-1]


def levenshtein_substring(ref, hyp):
    """ref 對上 hyp 之中**最接近的一段**的編輯距離（hyp 兩端自由）。

    標準 Levenshtein 的變形：首列全填 0（允許在 hyp 任意位置起頭），
    答案取末列最小值（允許在任意位置收尾）。

    🔴 為什麼需要它（2026-09-10 實測踩到）：雅婷的 cue 平均 12 秒，
    但蓋住 G3 的那一顆長達 **296.8 秒**。用「cue 中點落在視窗內」篩 → G3 收到 0 字、
    CER 判 100%；改用「有重疊就收」→ 收到 1043 字要比 228 字的 ref、CER 判 187%。
    **兩種時間分桶都在量切段風格，不是在量準確率** —— 與 0716
    「量到的是書寫格式不是準確率」同型。

    取捨（要講清楚）：hyp 在黃金段之外的多餘輸出**不會**被算成插入錯誤。
    那部分由 `hallucination` 另計，不重複罰。
    """
    if not ref:
        return 0
    if not hyp:
        return len(ref)
    prev = [0] * (len(hyp) + 1)          # 首列全 0 ＝ 起點自由
    for ca in ref:
        cur = [prev[0] + 1]
        for j, cb in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1,
                           cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev)                     # 末列取最小 ＝ 終點自由


def levenshtein_substring_per_ref(ref, hyp):
    """跟 `levenshtein_substring` 同一個對齊，但把誤差**歸因回 ref 的每個位置**。

    回 (distance, costs)：`costs` 長度 ＝ len(ref)，第 i 項是「ref 第 i 個字
    在最佳對齊路徑上分攤到的編輯成本」。`sum(costs) == distance` 永遠成立
    （測試鎖這條）。

    ## 🔴 為什麼不是「逐句各自對齊」

    b6 要 per-line 是為了做 bootstrap（n=64 >> n=5 才有檢定力）。但**照字面
    做「每句各自抓自己的 hyp 再算 CER」會把閘 6／閘 8 剛擋掉的缺陷放回來** ——
    那又是一次時間分桶，量到的會是引擎的切段風格（雅婷有一顆 296.8 秒的 cue，
    逐句分桶時它要嘛整顆丟掉、要嘛整段重複計入每一句）。

    所以這裡的做法是：**對齊仍然在「段」的層級做一次**（對 hyp 的切法免疫），
    只是把對齊路徑上的錯誤攤回 ref 的字元位置，再依各句在串接字串裡的
    起訖切開。統計單位變成句，對齊單位仍然是段。

    ## 代價（引用前要知道）

插入（hyp 多出來的字）的歸屬**取決於對齊路徑，方向不固定**。實測兩例：

        ref=ABCD, hyp=ABXXCD  →  costs=[0, 0, 1, 1]   （攤到後面兩個字）
        ref=ABCD, hyp=AYBCD   →  costs=[1, 0, 0, 0]   （攤到前一個字）

    ⚠️ 2026-09-11 更正：原本這裡寫「歸給它插在哪個 ref 位置**之前**那一格、
    邊界上的插入算在**後一句**頭上」，而上面第二例的方向正好相反。
    偏移模型寫錯會讓引用 per-line CER 的人誤判，而 per-line CER 是要餵
    bootstrap 權重的。

    保證的只有兩件事（有測試守）：`sum(costs) == distance`、
    `len(costs) == len(ref)`。**單句 CER 不宜當精確值看**，它是權重不是量測值；
    段層級的總和不受影響。
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, []
    if m == 0:
        return n, [1] * n
    # dp[i][j]：ref 前 i 字對上 hyp 前 j 字（hyp 起點自由 → 第 0 列全 0）
    prev = [0] * (m + 1)
    table = [prev]
    for i in range(1, n + 1):
        cur = [prev[0] + 1]
        ca = ref[i - 1]
        for j in range(1, m + 1):
            cur.append(min(prev[j] + 1,                       # 刪（ref 有 hyp 無）
                           cur[j - 1] + 1,                    # 插（hyp 多）
                           prev[j - 1] + (ca != hyp[j - 1]))) # 對/替
        table.append(cur)
        prev = cur
    end_j = min(range(m + 1), key=lambda j: table[n][j])       # 終點自由
    dist = table[n][end_j]

    costs = [0] * n
    i, j = n, end_j
    while i > 0:
        cur, up = table[i], table[i - 1]
        if j > 0 and cur[j] == cur[j - 1] + 1:                 # 插入
            costs[i - 1] += 1
            j -= 1
        elif j > 0 and cur[j] == up[j - 1] + (ref[i - 1] != hyp[j - 1]):
            costs[i - 1] += (ref[i - 1] != hyp[j - 1])         # 對齊或替換
            i -= 1
            j -= 1
        else:                                                  # 刪除
            costs[i - 1] += 1
            i -= 1
    return dist, costs


def cer(ref, hyp):
    """CER ＝ **近似子字串**編輯距離 ÷ ref 長度。

    ref 空的邊界**定義為**：hyp 也空 → 0.0；hyp 有東西 → 1.0（不是 NaN 也不是除零）。
    負向測試 #3 鎖這條。

    ⚠️ 2026-09-10 由整串比對改為子字串對齊，理由見 `levenshtein_substring`。
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_substring(ref, hyp) / float(len(ref))


# ---------------------------------------------------------------- 型號偵測

def find_models(text_norm, vocab_norm):
    """在正規化後的稿面找型號，回 {canonical: 次數}。

    **長→短排序 ＋ 遮罩已匹配區間**：否則 `204` 會匹配進 `204D` 裡面，
    量出來的命中數是假的（跟 `term_dict.py` 的長→短排序同一條紀律）。
    """
    found = {}
    if not text_norm:
        return found
    mask = [False] * len(text_norm)
    # 🔴 `key=len, reverse=True` 之下，同長度的先後由 `set` 的迭代序決定，
    # 而字串 hash 每個 process 隨機 ⇒ 兩個等長且在稿面重疊的型號，
    # 同一份輸入會給不同答案（實測 PYTHONHASHSEED 1–3 與 4–5 結果不同）。
    # 一支決定要不要換引擎的量尺不可以這樣。改成全序：長度優先、同長按字典序。
    for m in sorted(set(v for v in vocab_norm if v),
                    key=lambda v: (-len(v), v)):
        start = 0
        while True:
            i = text_norm.find(m, start)
            if i < 0:
                break
            if any(mask[i:i + len(m)]):
                start = i + 1
                continue
            found[m] = found.get(m, 0) + 1
            for k in range(i, i + len(m)):
                mask[k] = True
            start = i + len(m)
    return found


# ---------------------------------------------------------------- hyp 載入

def load_hyp(path, fmt=None):
    """讀一份逐字稿 → [{'start','end','text', 'speaker'?}]。吃四種形狀。

    1. `.raw.json`／`.json`：list of {start, end, text, …}（我們三線的落檔格式）
    2. `.json`：{'segments': [...]}（whisper 原生 dump）
    3. `.srt`：字幕檔（雅婷等雲端競品給這個；`語者N:` 前綴會被抽進 `speaker`）
    4. `.md`：**Recapp 會記**的匯出（`**說話者 N**（分:秒）` ＋ 內文）

    `fmt` 給了就不 sniff（`srt`／`json`／`recapp`）。
    """
    fmt = (fmt or "").lower() or None
    low = path.lower()
    if fmt == "srt" or (fmt is None and low.endswith(".srt")):
        with open(path, "r", encoding="utf-8-sig") as fh:
            return parse_srt(fh.read())
    if fmt == "recapp" or (fmt is None and low.endswith(".md")):
        with open(path, "r", encoding="utf-8-sig") as fh:
            return parse_recapp_md(fh.read())
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("segments") or []
    out = []
    for s in data:
        if s.get("start") is None or s.get("end") is None:
            continue
        out.append({"start": float(s["start"]), "end": float(s["end"]),
                    "text": s.get("text") or ""})
    return out


# Recapp 會記匯出的 markdown：`**說話者 N**（分:秒）` 之後接內文。
_RECAPP_CUE = re.compile(r"^\*\*([^*\n]{1,20})\*\*\s*[（(]\s*([\d:]+)\s*[）)]\s*$")


def _hms_to_sec(tok):
    """`0:07`／`12:34`／`1:33:27` 都吃。分鐘可以超過 59（Recapp 會寫 93:27）。"""
    parts = [int(x) for x in tok.split(":") if x != ""]
    sec = 0
    for v in parts:
        sec = sec * 60 + v
    return float(sec)


def parse_recapp_md(raw):
    """Recapp 會記的 `.md` → [{'start','end','text','speaker'}]。

    只讀 `## 逐字稿` 之後的內容 —— 前面的摘要／重點／待辦是**模型生成的二次產物**，
    不是轉錄，混進去會把 LLM 的改寫算成 ASR 的輸出。

    只給起點不給終點 → 終點用**下一句的起點**補；最後一句用其餘句子的
    中位長度補（沒得推時退回 5 秒）。

    🔴 **每一顆 cue 的 end 都是我們補的，不只最後一句。** 原本的 docstring
    寫「只有最後一句是估的、只影響幻覺率的分母」，兩句都錯：

      - `zip(cues, cues[1:])` 套用在全部 cue 上
      - 被補出來的長度 ＝ 講者之間的靜默，不是說話時長。實測兩個字的「短句」
        因為下一位講者 46 秒後才開口，被補成 **46 秒**
      - 於是中位 cue 長度被灌大 ⇒ 直接決定 `too_coarse` 要不要回 None；
        cue 被拉長後也可能跨進／跨出黃金句 ⇒ 連幻覺的**分子**都換掉

    ⇒ 每顆 cue 帶 `end_estimated: True`，`score()` 看到就把幻覺率回 `None`。
    **這種來源與有真 end 的來源之間，幻覺率不可比。**
    CER 不受影響（對齊走自由端點子字串，不吃 cue 邊界）。
    """
    idx = raw.find("## 逐字稿")
    body = raw[idx:] if idx >= 0 else raw
    cues = []
    lines = body.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _RECAPP_CUE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        spk, tok = m.group(1).strip(), m.group(2)
        i += 1
        buf = []
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or _RECAPP_CUE.match(nxt) or nxt.startswith("## "):
                break
            buf.append(nxt)
            i += 1
        cues.append({"start": _hms_to_sec(tok), "end": None,
                     "text": " ".join(buf), "speaker": spk,
                     # 這個來源的 end 全部是推估的 —— 見 docstring
                     "end_estimated": True})
    if not cues:
        return []
    for a, b in zip(cues, cues[1:]):
        a["end"] = max(a["start"], b["start"])
    durs = [c["end"] - c["start"] for c in cues[:-1] if c["end"] is not None]
    durs = [d for d in durs if d > 0]
    tail = sorted(durs)[len(durs) // 2] if durs else 5.0
    cues[-1]["end"] = cues[-1]["start"] + tail
    return cues


_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def parse_srt(raw):
    out = []
    block_lines = []
    for line in raw.replace("\r\n", "\n").split("\n") + [""]:
        if line.strip():
            block_lines.append(line)
            continue
        if block_lines:
            out.extend(_srt_block(block_lines))
            block_lines = []
    return out


def _srt_block(block_lines):
    for i, line in enumerate(block_lines):
        m = _SRT_TIME.search(line)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        text = " ".join(block_lines[i + 1:])
        return [{"start": start, "end": end, "text": text}]
    return []


# ---------------------------------------------------------------- 評分

def taigi_retention(golden_lines, hyp, drop_fillers=False):
    """台語保留率 —— **不需要台語標準答案**。

    只吃「哪幾句是台語」（`lang == "nan"`，人標的），去看引擎在**那些句子的時間範圍內**
    吐出來的文字含不含台語專屬字形。

    回 (per_10k, hits, chars, taigi_sec)：
      per_10k  每萬字命中數（None ＝ 這批沒有台語句、或那些句子引擎完全沒輸出）
      taigi_sec 人標的台語總秒數 —— **先看這個**：它是 0 就代表沒有訊號可談，
                不是「引擎抹平了」。

    抹平的引擎 → 趨近 0；忠實的引擎 → 明顯 > 0。
    ⚠️ 絕對值無意義，只能同一批句子跨引擎比。

    🔴 **每顆 cue 只能計一次。** 原本的寫法在 per-line 迴圈裡 `extend`，
    一顆橫跨兩句台語句的 cue 會被串進分母兩次，`hits` 與 `chars` 同時被乘上
    「它重疊了幾句」。實測：hyp 實際文字是 `迄甲`（2 字），卻回 3 字、
    每萬 6666.7（應為 5000）。而 cue 會不會跨句取決於**引擎的切段風格** ⇒
    量到的又是切法（0716「量到的是書寫格式不是準確率」的同一個形狀，
    這是今天第四次）。
    """
    nan_lines = [ln for ln in golden_lines if ln.get("lang") == "nan"]
    taigi_sec = sum(ln["t_end"] - ln["t_start"] for ln in nan_lines)
    if not nan_lines:
        return None, 0, 0, 0.0
    # 先收成「碰到任一句台語句的 cue 集合」（去重），再依時序串一次。
    seen = set()
    ov_all = []
    for ln in nan_lines:
        for h in hyp:
            if id(h) not in seen and _touches(h, ln["t_start"], ln["t_end"]):
                seen.add(id(h))
                ov_all.append(h)
    text = "".join(normalize(h["text"], drop_fillers)
                   for h in sorted(ov_all, key=lambda h: h["start"]))
    if not text:
        return None, 0, 0, taigi_sec
    hits = sum(text.count(m) for m in TAIGI_EXCLUSIVE)
    return hits * 10000.0 / len(text), hits, len(text), taigi_sec


def _median(xs):
    if not xs:
        return None
    t = sorted(xs)
    n = len(t)
    return t[n // 2] if n % 2 else (t[n // 2 - 1] + t[n // 2]) / 2.0


def window_hyp(hyp, t_start, t_end):
    """取落在 [t_start, t_end] 視窗內的 hyp 段（依 start 排序）。

    **這是視窗選取的唯一定義** —— `score()` 與任何外部分析都必須走這裡。
    2026-09-11 抽出來的原因：一支臨時寫的地板分析自己手寫了一份選取邏輯，
    複製到的是**修好之前的中點篩**，於是同一份資料算出兩組不同的 CER
    （雅婷 G3：29.8% vs 100.0%）。錯的是那份副本，但真正該消掉的是
    「能有第二份實作」這件事本身。

    **有重疊就收，不用中點篩。** 中點篩會整顆丟掉跨窗的長 cue ——
    雅婷實測有一顆 296.8 秒的 cue，中點落在窗外，G3 因此收到 0 字被判 CER 100%。
    多收進來的部分由 `levenshtein_substring` 的自由端點吸收，不算成錯誤。

    **零長度 cue 的處理在 `_touches()` 裡**，不在這裡重寫一份。
    """
    return sorted([h for h in hyp if _touches(h, t_start, t_end)],
                  key=lambda h: h["start"])


def window_hyp_text(hyp, t_start, t_end, drop_fillers=False):
    """視窗內的 hyp 正規化文字。外部分析要比對文字時用這支，不要自己串。"""
    return "".join(normalize(h["text"], drop_fillers)
                   for h in window_hyp(hyp, t_start, t_end))


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _touches(h, a, b):
    """cue `h` 有沒有碰到區間 `[a, b]` —— **這是「碰到」的唯一定義**。

    2026-09-11 審計抓到的：這個判定原本有**四份**實作 —— `window_hyp` 的
    `_inside`、`coverage` 的迴圈、`taigi_retention` 的迴圈、
    `hallucination_spans` 的 `touched`。只有第一份帶了零長度 cue 的特判，
    另外三份都是 `_overlap(...) > 0`。後果是同一顆零長度 cue：

      - CER 收下它（`window_hyp` 有特判）
      - 覆蓋率、台語保留率丟掉它
      - `hallucination_spans` 裡 `touched` 恆 False ⇒ **一律判成幻覺**，
        即使它落在黃金句正中間、文字完全正確

    最後一條最貴：`adjudicate.py` 吃的就是 `hallucination_spans`，
    那些 cue 每一顆都會變成一列假的幻覺候選。

    零長度 cue（`end <= start`）不是罕見邊角：某些來源的時間戳只有秒精度，
    兩人同秒交替就撞成零長度（實測 1268 段裡 92 段、247 字）。
    丟它或誤判它，**只影響時間戳精度差的那一家** ⇒ 跨引擎比較的偏差方向固定。

    ⚠️ 這支存在的理由跟 `window_hyp` 的 docstring 是同一條：
    能有第二份實作，就一定會漂。那條紀律上次寫在這個檔裡，
    然後往下三個函式就是第二份實作 —— 所以這次是把判定本身收成一支。
    """
    if h["end"] <= h["start"]:
        return a <= h["start"] <= b
    return _overlap(h["start"], h["end"], a, b) > 0


def hallucination_spans(seg_lines, hyp, t_start, t_end):
    """回視窗內「與任何黃金句零重疊」的 hyp 段（＝被判成幻覺的那些）。

    **這是幻覺判定的唯一定義**，`score()` 與外部複聽工具都走這裡
    （同 `window_hyp` 的理由：第二份實作會漂）。

    ⚠️ 語義要講準：它量的是「**人沒寫字的地方引擎有輸出**」。那裡面包含
    真幻覺，**也包含人聽不清楚而沒打的真實語音** —— 黃金段不是逐秒全錄
    （G4 那 60 秒只打了 45.5 秒）。所以由它算出的幻覺率是**上限**，
    要坐實成幻覺率必須人工複聽。
    """
    out = []
    for h in window_hyp(hyp, t_start, t_end):
        touched = any(_touches(h, ln["t_start"], ln["t_end"])
                      for ln in seg_lines)
        if not touched:
            out.append(h)
    return out


def _mid(seg):
    return (seg["start"] + seg["end"]) / 2.0


def score(manifest, lines, hyp, vocab=None, drop_fillers=False, asr_sec=None):
    """回一份 report dict。純函式：不讀檔、不印東西。"""
    segs = manifest.get("segments") or []
    by_seg = {}

    # 型號詞彙表。**預設只有黃金段裡出現過的型號** —— 這樣算得出的 precision
    # 只反映「有沒有把 A 段的型號寫成 B 段的型號」。要更嚴的 precision，
    # 呼叫端要傳進正本型號全表（`--vocab`），否則引擎亂寫一個不在表內的型號抓不到。
    if vocab is None:
        vocab = [m for ln in lines for m in (ln.get("models") or [])]
    vocab_norm = [normalize_model(v) for v in vocab]

    per_line = []
    tot = {"ref_zh": 0, "dist_zh": 0, "ref_nat": 0, "dist_nat": 0,
           "tp": 0, "fp": 0, "fn": 0,
           "cov_sec": 0.0, "gold_sec": 0.0,
           "hyp_sec": 0.0, "halluc_sec": 0.0, "halluc_segs": 0,
           "ref_empty_segs": 0, "hyp_chars": 0}

    for seg in segs:
        sid, lo, hi = seg["id"], seg["t_start"], seg["t_end"]
        seg_lines = sorted([ln for ln in lines if ln.get("seg") == sid],
                           key=lambda l: l["t_start"])
        hyp_in = window_hyp(hyp, lo, hi)

        ref_zh = "".join(normalize(ref_text(ln, "zh"), drop_fillers) for ln in seg_lines)
        ref_nat = "".join(normalize(ref_text(ln, "native"), drop_fillers) for ln in seg_lines)
        hyp_txt = "".join(normalize(h["text"], drop_fillers) for h in hyp_in)

        # per-line 歸因：對齊仍在段層級做一次（對 hyp 切法免疫），
        # 只把路徑上的錯誤攤回 ref 字元位置，再依各句在串接字串裡的起訖切開。
        spans = []
        _pos = 0
        for ln in seg_lines:
            _t = normalize(ref_text(ln, "zh"), drop_fillers)
            spans.append((ln, _pos, _pos + len(_t)))
            _pos += len(_t)
        if ref_zh:
            d_zh, _costs = levenshtein_substring_per_ref(ref_zh, hyp_txt)
        else:
            # 這一段沒有黃金句。逐段 `cer()` 回的是**比值** 1.0，而這裡加的是
            # **絕對距離** 1 —— 兩種幣別。加進 `tot["dist_zh"]` 卻不加分母
            # （`len(ref_zh)` 是 0）⇒ overall CER 被偷偷墊高。
            # 實測：S1 完美 ＋ S2 無黃金句卻有輸出 → overall 0.04（應為 0.0）。
            # ⇒ 無黃金句的段不參與 overall，只在 `ref_empty_segments` 裡報數。
            d_zh, _costs = 0, []
        for ln, a, b in spans:
            per_line.append({
                "id": ln.get("id"), "seg": sid, "lang": ln.get("lang"),
                "spk": ln.get("spk"),
                "ref_len": b - a,
                "dist": sum(_costs[a:b]) if _costs else (b - a),
                "sec": round(ln["t_end"] - ln["t_start"], 2),
            })
        d_nat = (levenshtein_substring(ref_nat, hyp_txt) if ref_nat
                 else (0 if not hyp_txt else 1))

        # 覆蓋率：逐句判，分母是人工標的語音秒數。
        # ⚠️ S3 交疊段裡重疊的句子會各自計入分母（每一句發言都是要抓到的單位）。
        cov_sec = 0.0
        gold_sec = 0.0
        for ln in seg_lines:
            dur = ln["t_end"] - ln["t_start"]
            gold_sec += dur
            # 🔴 2026-09-10 第三次修同一個病：初版要求**單一** hyp 段覆蓋該句 ≥50%。
            # Whisper 的 cue 中位 2.0 秒且段間有真空隙 → 一句 6 秒的黃金句被三顆 2 秒的
            # cue 蓋住時，沒有任何一顆單獨達到 3 秒 → 判成沒覆蓋。實測覆蓋率 24.3%，
            # 而同一份稿的 CER 只有 26.5%（兩者矛盾 ＝ 指標在量別的東西）。
            # 另一條線中位同樣 2.0 秒卻拿三倍以上，只因它的 cue 是連續的
            # （`end` 用下一句 start 補）⇒ **量到的是切段風格不是覆蓋**。
            # 改成：所有有重疊的 cue 全收、依時序串起來，再用子字串對齊判。
            ov = [h for h in hyp if _touches(h, ln["t_start"], ln["t_end"])]
            if not ov:
                continue
            txt = "".join(normalize(h["text"], drop_fillers)
                          for h in sorted(ov, key=lambda h: h["start"]))
            if cer(normalize(ref_text(ln, "zh"), drop_fillers), txt) <= 0.5:
                cov_sec += dur

        # 幻覺：視窗內、與任何黃金句零重疊的 hyp 段。
        # ⚠️ 幻覺率只有在「引擎切段夠細」時才判得動：一顆 296 秒的 cue 會重疊到
        # 段內每一句，永遠算不出幻覺 —— 回 0 會被誤讀成「這家不幻覺」。
        # 判準：hyp 的中位 cue 長度 > 黃金句中位長度的 4 倍 → 判不了，回 None。
        med_line = _median([ln["t_end"] - ln["t_start"] for ln in seg_lines]) or 0.0
        # 零長度 cue 不進中位數。它們不代表「這家切得細」，是時間戳精度不足的
        # 產物；混進來會把中位拉到 0 ⇒ `too_coarse` 恆 False ⇒ 一顆 60 秒的 cue
        # 也照樣算出幻覺率（實測 durs=[60, 0, 0] → 中位 0.0 → 幻覺率 0.0）。
        med_cue = _median([h["end"] - h["start"] for h in hyp_in
                           if h["end"] > h["start"]]) or 0.0
        too_coarse = bool(med_line and med_cue > med_line * 4)
        # 🔴 來源的 end 本身是推估的（只給起點的匯出格式，見 `parse_recapp_md`）
        # ⇒ 幻覺率的分子與分母都建立在我們自己編的時間上，不可算。
        # 這跟 `too_coarse` 是兩個不同的不可判定理由，要分開記。
        end_estimated = any(h.get("end_estimated") for h in hyp_in)
        hyp_sec = 0.0
        halluc_sec = 0.0
        _halluc = hallucination_spans(seg_lines, hyp, lo, hi)
        _halluc_ids = set(id(h) for h in _halluc)
        for h in hyp_in:
            # 🔴 秒數一律夾到視窗內。不夾的話一顆 296.8 秒的 cue 會對一個 60 秒
            # 視窗貢獻 296.8 秒，而且跨兩窗時被算兩次 ⇒ 分母被窗外的秒數灌大，
            # 讀數往 0 壓。**偏差方向是固定的：cue 越粗 → 幻覺率越接近 0**，
            # 正好獎勵 `too_coarse` 要抓的那一類引擎。
            # 實測：同一個 10 秒幻覺事實，別處多一顆 498 秒的 cue
            # 就讓 overall 從 29.4% 變 1.9%（15.6 倍）。
            dur = _overlap(h["start"], h["end"], lo, hi)
            hyp_sec += dur
            if id(h) in _halluc_ids:
                halluc_sec += dur
        # 窗內一秒 hyp 都沒有 ⇒ 分母是 0，那不是「沒幻覺」而是**沒東西可判**。
        # 回 0.0 會被讀成「這家不幻覺」＝把量不到講成優點；而 `coverage` 與
        # `_ratio` 早就是「分母 0 → None」，只有幻覺率例外（2026-09-11 收斂）。
        judgeable = bool(hyp_sec) and not too_coarse and not end_estimated

        # 型號多重集比對
        ref_models = {}
        for ln in seg_lines:
            for m in (ln.get("models") or []):
                k = normalize_model(m)
                ref_models[k] = ref_models.get(k, 0) + 1
        hyp_models = find_models(hyp_txt, vocab_norm)
        keys = set(ref_models) | set(hyp_models)
        tp = sum(min(ref_models.get(k, 0), hyp_models.get(k, 0)) for k in keys)
        fp = sum(max(0, hyp_models.get(k, 0) - ref_models.get(k, 0)) for k in keys)
        fn = sum(max(0, ref_models.get(k, 0) - hyp_models.get(k, 0)) for k in keys)

        # 台語：主指標是**不需參考答案**的保留率；`cer_native` 只在真的有人填了
        # native 時才有意義（native 選填），沒填就回 None 而不是回一個假的 0 gap。
        t_per10k, t_hits, t_chars, t_sec = taigi_retention(seg_lines, hyp, drop_fillers)
        has_native = any((ln.get("native") or "").strip() for ln in seg_lines)

        by_seg[sid] = {
            "reason": seg.get("reason", ""),
            "lines": len(seg_lines),
            "hyp_segments": len(hyp_in),
            "cer_content": cer(ref_zh, hyp_txt),
            # 🔴 CER 走自由端點子字串對齊 ⇒ **視窗內多餘的輸出不算插入錯誤**
            # （那是刻意的取捨，見 `levenshtein_substring` 的 docstring）。
            # 分工的另一半本來是 `hallucination`，但 `too_coarse` 會把它變 None
            # ⇒ 一顆大 cue 可以同時拿到「多餘文字零成本」與「幻覺率 n/a」。
            # 實測：1618 字垃圾包住 18 字正確答案 → CER 0.0000、覆蓋率 100%。
            # 所以 CER 旁邊必須有這個數字：接近 1 代表鬆量沒被吃掉、CER 可引用；
            # 遠大於 1 代表 CER 是「最佳子字串」而不是「這份稿有多準」。
            # （0908 四條線實測 0.96–1.01，所以那批數字站得住。）
            "verbosity": (len(hyp_txt) / float(len(ref_zh))) if ref_zh else None,
            "ref_chars": len(ref_zh),
            "hyp_chars": len(hyp_txt),
            "cer_native": cer(ref_nat, hyp_txt) if has_native else None,
            "taigi_flatten_gap": (cer(ref_nat, hyp_txt) - cer(ref_zh, hyp_txt))
                                 if has_native else None,
            "taigi_retention_per10k": t_per10k,
            "taigi_hits": t_hits,
            "taigi_hyp_chars": t_chars,
            "taigi_sec": round(t_sec, 1),
            "coverage": (cov_sec / gold_sec) if gold_sec else None,
            "hallucination": None if not judgeable
                             else ((halluc_sec / hyp_sec) if hyp_sec else 0.0),
            # 為什麼判不動 —— 讀報表的人要分得出「沒幻覺」與「量不到」
            "hallucination_blocked_by": (
                None if judgeable
                else "end_estimated" if end_estimated
                else "cue_too_coarse" if too_coarse
                else "no_hyp_in_window"),
            "hyp_median_cue_sec": round(med_cue, 1),
            "model_precision": _ratio(tp, tp + fp),
            "model_recall": _ratio(tp, tp + fn),
            "model_counts": {"tp": tp, "fp": fp, "fn": fn},
        }

        if not ref_zh:
            tot["ref_empty_segs"] += 1
        tot["ref_zh"] += len(ref_zh)
        tot["dist_zh"] += d_zh
        tot["hyp_chars"] += len(hyp_txt)
        tot["ref_nat"] += len(ref_nat)
        tot["dist_nat"] += d_nat
        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn
        tot["cov_sec"] += cov_sec
        tot["gold_sec"] += gold_sec
        # 🔴 判不動的段不進 overall 的分母。原本無條件累加 ⇒ overall 會用一個
        # 「我們上一行才剛宣告不可判定」的段算出一個數字，而 by_segment 回 None。
        # `adjudicate.py` 與 `verification.md` §Negative Result 的同一條紀律：
        # 不可判定不准折成任何一邊 —— 這裡折了。
        if judgeable:
            tot["hyp_sec"] += hyp_sec
            tot["halluc_sec"] += halluc_sec
            tot["halluc_segs"] += 1

    cer_c = (tot["dist_zh"] / float(tot["ref_zh"])) if tot["ref_zh"] else 0.0
    cer_n = (tot["dist_nat"] / float(tot["ref_nat"])) if tot["ref_nat"] else 0.0
    any_native = any((ln.get("native") or "").strip() for ln in lines)
    o_per10k, o_hits, o_chars, o_sec = taigi_retention(lines, hyp, drop_fillers)
    overall = {
        "cer_content": cer_c,
        # 見 by_segment 的 `verbosity` 註解：CER 旁邊要並列這個數字。
        "verbosity": ((tot["hyp_chars"] / float(tot["ref_zh"]))
                      if tot["ref_zh"] else None),
        "ref_chars": tot["ref_zh"],
        "hyp_chars": tot["hyp_chars"],
        # 沒有黃金句的段數 —— 它們不參與 overall CER（分母是 0，沒得算）
        "ref_empty_segments": tot["ref_empty_segs"],
        "cer_native": cer_n if any_native else None,
        "taigi_flatten_gap": (cer_n - cer_c) if any_native else None,
        "taigi_retention_per10k": o_per10k,
        "taigi_hits": o_hits,
        "taigi_hyp_chars": o_chars,
        "taigi_sec": round(o_sec, 1),
        "coverage": (tot["cov_sec"] / tot["gold_sec"]) if tot["gold_sec"] else None,
        # 只用判得動的段算。一段都判不動 → None（不是 0.0）。
        # 原本的條件是「**全部**段都判不動才回 None」，於是「一段判不動、
        # 一段判得動」時，判不動那段的秒數照樣進了分母。
        # `halluc_segs > 0` 時 `hyp_sec` 必然 > 0（judgeable 已含這個條件），
        # 所以這裡不需要「分母 0 → 0.0」那個分支。
        "hallucination": None if not tot["halluc_segs"]
                         else tot["halluc_sec"] / tot["hyp_sec"],
        # 這個數字是幾段算出來的 —— 引用前必看，5 段裡只有 1 段判得動的
        # 「幻覺率」跟 5 段都判得動的不是同一種東西。
        "hallucination_judged_segments": tot["halluc_segs"],
        "hallucination_total_segments": len(by_seg),
        "hyp_median_cue_sec": round(_median([h["end"] - h["start"]
                                             for h in hyp]) or 0.0, 1),
        "model_precision": _ratio(tot["tp"], tot["tp"] + tot["fp"]),
        "model_recall": _ratio(tot["tp"], tot["tp"] + tot["fn"]),
        "golden_sec": tot["gold_sec"],
    }
    if asr_sec is not None and manifest.get("duration_sec"):
        overall["rtf"] = asr_sec / float(manifest["duration_sec"])
    return {"set_id": manifest.get("set_id"), "overall": overall, "by_segment": by_seg, "per_line": per_line,
            "drop_fillers": drop_fillers, "vocab_size": len(set(vocab_norm))}


def _ratio(num, den):
    """分母 0 回 None（＝這段沒得算），不回 0.0 —— 0.0 會被讀成「全錯」。"""
    return (num / float(den)) if den else None


# ---------------------------------------------------------------- CLI

def _fmt(v):
    if v is None:
        return "  n/a"
    return "%5.1f%%" % (v * 100)


def main(argv=None):
    ap = argparse.ArgumentParser(description="用黃金段評一份逐字稿")
    ap.add_argument("--golden", required=True, help="黃金段目錄（含 manifest.json + lines.jsonl）")
    ap.add_argument("--hyp", required=True, help="逐字稿：.raw.json / .json / .srt")
    ap.add_argument("--vocab", help="型號正本全表（一行一個），給了 precision 才嚴")
    ap.add_argument("--drop-fillers", action="store_true",
                    help="評分前去填充詞（預設不去，見 plan §1.3）")
    ap.add_argument("--asr-sec", type=float, help="這次 ASR 耗時（秒），給了才算 RTF")
    ap.add_argument("--json", action="store_true", help="輸出 JSON 而非人看的表")
    args = ap.parse_args(argv)

    manifest, lines = load_golden(args.golden)
    hyp = load_hyp(args.hyp)
    vocab = None
    if args.vocab:
        with open(args.vocab, "r", encoding="utf-8") as fh:
            vocab = [l.strip() for l in fh if l.strip()]
    rep = score(manifest, lines, hyp, vocab=vocab,
                drop_fillers=args.drop_fillers, asr_sec=args.asr_sec)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    o = rep["overall"]
    print("黃金段 %s ／ hyp %s" % (rep["set_id"], os.path.basename(args.hyp)))
    print("")
    hdr = ("段", "內容CER", "覆蓋率", "幻覺率", "型號P", "型號R", "台語保留", "台語秒")
    fmtline = "%-4s %8s %8s %8s %8s %8s %10s %8s"
    print(fmtline % hdr)

    def _taigi(s):
        v = s["taigi_retention_per10k"]
        return "  n/a" if v is None else "%6.1f/萬" % v

    for sid in sorted(rep["by_segment"]):
        s = rep["by_segment"][sid]
        print(fmtline % (sid, _fmt(s["cer_content"]), _fmt(s["coverage"]),
                         _fmt(s["hallucination"]), _fmt(s["model_precision"]),
                         _fmt(s["model_recall"]), _taigi(s), "%.0fs" % s["taigi_sec"]))
    print(fmtline % ("全體", _fmt(o["cer_content"]), _fmt(o["coverage"]),
                     _fmt(o["hallucination"]), _fmt(o["model_precision"]),
                     _fmt(o["model_recall"]), _taigi(o), "%.0fs" % o["taigi_sec"]))
    print("")
    print("hyp 中位 cue 長度 %.1fs（黃金句多在 2–6s；差太多時幻覺率判不動、顯示 n/a）"
          % o.get("hyp_median_cue_sec", 0.0))
    v = o.get("verbosity")
    if v is not None:
        print("稿面膨脹 %.2fx（視窗內 hyp %d 字 ÷ ref %d 字）—— **CER 要跟它並列讀**："
              % (v, o.get("hyp_chars", 0), o.get("ref_chars", 0)))
        print("  對齊走自由端點，視窗內多餘輸出不算錯 ⇒ 膨脹遠大於 1 時，"
              "CER 是「最佳子字串」不是「這份稿有多準」。")
    if o.get("ref_empty_segments"):
        print("")
        print("⚠️ 有 %d 段沒有黃金句 —— 它們不參與全體 CER（分母是 0）"
              % o["ref_empty_segments"])

    # 幻覺率判不動時，一定要說出是哪一段、因為什麼 —— 否則讀表的人只看到 n/a，
    # 得靠旁邊的手寫註解才知道發生什麼事（那是散文 backstop，不是指標）。
    _WHY = {"cue_too_coarse": "cue 太粗（中位超過黃金句的 4 倍）",
            "end_estimated": "來源只給起點，end 是我們補的",
            "no_hyp_in_window": "視窗內引擎零輸出（分母 0）"}
    blocked = [(sid, rep["by_segment"][sid]["hallucination_blocked_by"])
               for sid in sorted(rep["by_segment"])
               if rep["by_segment"][sid]["hallucination_blocked_by"]]
    njudged = o.get("hallucination_judged_segments", 0)
    ntotal = o.get("hallucination_total_segments", 0)
    if blocked:
        print("")
        print("幻覺率判不動的段（%d/%d）：" % (len(blocked), ntotal))
        for sid, why in blocked:
            print("  %-4s %s" % (sid, _WHY.get(why, why)))
        if njudged:
            print("⚠️ 全體幻覺率只由 %d/%d 段算出 —— 與 %d 段都判得動的數字不可並列比較。"
                  % (njudged, ntotal, ntotal))
        else:
            print("⚠️ 沒有任何一段判得動 ⇒ 全體幻覺率 n/a。"
                  "**這條線的幻覺率不可引用**，不是「幻覺率低」。")
    if o["taigi_sec"] == 0:
        print("")
        print("⚠️ 台語秒數為 0 —— 這批黃金段沒有標到台語句，"
              "台語保留率無訊號可談（不是「引擎抹平了」）")
    if o["cer_native"] is not None:
        print("")
        print("（有人填了 native 軌）台語CER %s ／ 抹平gap %s"
              % (_fmt(o["cer_native"]), _fmt(o["taigi_flatten_gap"])))
    if "rtf" in o:
        print("")
        print("RTF %.3f（⚠️ 當場量的值，不可跨場次沿用 —— 見 README §RTF）" % o["rtf"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
