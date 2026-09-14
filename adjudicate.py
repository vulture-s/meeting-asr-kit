# -*- coding: utf-8 -*-
"""把「真幻覺」與「人沒打到的真實語音」分開 —— 用聲學，不用人耳。

## 為什麼要有這支

`score.py` 的 `hallucination` 量的是「與任何黃金句零重疊的 hyp 輸出」。
**在非窮盡的黃金段上，那個數字必然混進兩種東西**：

  (a) 真幻覺 —— 那幾秒其實是靜音／噪音，引擎憑空生字
  (b) 人沒打到的真實語音 —— 那幾秒有人在講，只是沒被聽打進去

黃金段不是逐秒全錄（0908 的 G4 那 60 秒只打了 45.5 秒）⇒ (b) 一定存在。

**2026-09-11 實測，這不是理論疑慮**：Whisper 被判幻覺的 15.3 秒裡，
13.3 秒落在 G4 那個 14.0 秒的空白裡，語音密度 75–88%（全檔 60 秒窗中位 41.7%），
而且語意與空白前後連貫成同一段論述。當時差一點被寫成
「甲線幻覺率比乙線高一個數量級」的結論。

判別子不必用人耳：**看那幾秒的語音密度**，用 `scan_levels` 的同一把尺。

## 🔴 已知不可靠處：短段

判定基於 0.25 秒格。**2 秒的段只有 8 格**，二項變異大到 12% 與 25% 分不開。
所以短段回 `None`（不判），不要硬給答案 —— 那正是這支工具存在的理由：
把「不可判定」明確標出來，而不是折成某一邊
（`verification.md` §Negative Result「不可判定不准折成通過」同型）。

## 跑法

```
python adjudicate.py --golden <黃金段目錄> --hyp <逐字稿> --audio <原始音檔>
```
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden as golden_mod  # noqa: E402
import scan_levels as sl  # noqa: E402
import score as score_mod  # noqa: E402

# 低於這個密度才算「那幾秒沒人在講話」。全檔 60 秒窗密度中位實測約 42%，
# 15% 是保守下限（寧可判成「有人講」也不要誤指幻覺）。
SILENCE_DENSITY = 0.15
# 少於這麼多格就不判 —— 見檔頭「短段」。8 格（2 秒）已經在不可靠區。
MIN_BINS = 12


def speech_density(bins, t0, t1, floor_db, bin_sec=sl.BIN_SEC,
                   margin_db=sl.SPEECH_MARGIN_DB):
    """回 (density, n_bins)。`density=None` ＝ 格數不足以判定。

    `bins` 是 `scan_levels.scan(..., window_sec=bin_sec)` 回的逐格統計。
    ⚠️ `bin_sec` 預設吃 `scan_levels.BIN_SEC` —— 原本這裡寫死 0.25，
    是第二份真相；生產者改了格長，這支會安靜地讀錯時間範圍。

    🔴 **區間要含所有與 [t0, t1] 相交的格**。原本 `i1 = int(t1 / bin_sec)`
    在切片是 exclusive 的前提下，把尾端那個**部分重疊**的格整個丟掉：
    0–3.1 秒只取 12 格（0–3.0），第 12 格（3.00–3.25）與 3.1 相交卻沒收。
    實測那一格是熱的時，密度從 2/13 = 15.4%（→「人沒打到的語音」）
    變成 1/12 = 8.3%（→「真幻覺」）—— **跨過了 15% 門檻，判定翻面**。
    """
    thresh = floor_db + margin_db
    i0 = max(0, int(math.floor(t0 / bin_sec)))
    i1 = max(i0 + 1, int(math.ceil(t1 / bin_sec)))
    sel = bins[i0:i1]
    if len(sel) < MIN_BINS:
        return None, len(sel)
    hot = sum(1 for w in sel if w["rms_db"] > thresh)
    return hot / float(len(sel)), len(sel)


def classify(density):
    """density → 判定字串。None 一律回「不可判定」，不折成任何一邊。"""
    if density is None:
        return "不可判定(格數不足)"
    return "真幻覺" if density < SILENCE_DENSITY else "人沒打到的語音"


# 相鄰未對應段之間小於這個間隔就視為同一塊。ASR 的 cue 邊界是切段風格的產物，
# 判定單位應該是「連續的未對應區段」而不是單顆 cue —— 2026-09-11 實測：
# G4 那 12 秒是六顆 2 秒 cue 連在一起，逐顆判每顆都只有 8 格、全部「不可判定」，
# 合起來 48 格就判得出來（密度 75–88%）。逐顆判會把看得出來的事實丟掉。
MERGE_GAP_SEC = 0.5


def _blocks_merge(occupied, a_end, b_start):
    """兩塊之間的空隙裡有沒有已對應的語音（黃金句）。

    🔴 沒有這道檢查時，`merge_adjacent` 只看 cue 間距 ——
    它看不到空隙裡其實有一句已經被轉出來的話。實測：兩段各 1.25 秒的靜音
    未對應 cue（0–1.25、1.75–3.0）各自只有 5 格、都回 `None`；
    在中間 1.25–1.75 放一句熱的黃金句，兩段就併成 12 格、密度 2/12 = 16.7%
    ⇒ 被判成「人沒打到的語音」，**而那個密度完全來自已經打出來的那句話**。
    """
    for lo, hi in occupied:
        if hi > a_end and lo < b_start:
            return True
    return False


def merge_adjacent(spans, occupied, gap=MERGE_GAP_SEC):
    """把同段內相鄰（間隔 <= gap）的未對應區段併成一塊。回併好的 list。

    `occupied` 是「已對應語音」的時間區間 list —— 空隙裡有它就不併
    （見 `_blocks_merge`）。**刻意做成必填**：給預設值等於把陷阱寫進 docstring
    而不是拆掉它，呼叫端忘記傳時會靜默失去保護。沒有已對應語音就傳 `()`。

    回傳的每塊帶兩個秒數，**刻意分開**：

      - `cue_sec` 各 cue 自己的長度加總 ＝ 與 `score.py` 的 `halluc_sec` 同幣別
      - `span_sec` 合併後的跨距（含 cue 之間的空隙）＝ `speech_density` 量的範圍

    實測 3 顆 2.0 秒 cue、間隔 0.4 秒：cue_sec 6.0、span_sec 6.8（+13%）。
    兩者混用的後果：case-study 的「15.3 秒裡有 13.3 秒」是兩種單位相減。
    """
    out = []
    for h in sorted(spans, key=lambda x: (str(x.get("seg") or ""), x["start"])):
        cur = dict(h)
        cur.setdefault("text", "")
        cur["n_cues"] = 1
        cur["cue_sec"] = max(0.0, cur["end"] - cur["start"])
        prev = out[-1] if out else None
        if (prev is not None and prev.get("seg") == cur.get("seg")
                and cur["start"] - prev["end"] <= gap
                and not _blocks_merge(occupied, prev["end"], cur["start"])):
            prev["end"] = max(prev["end"], cur["end"])
            prev["text"] = prev["text"] + cur["text"]
            prev["n_cues"] += 1
            prev["cue_sec"] += cur["cue_sec"]
        else:
            out.append(cur)
    for h in out:
        h["span_sec"] = max(0.0, h["end"] - h["start"])
    return out


def adjudicate(manifest, lines, hyp, bins, floor_db):
    """回 [{seg, start, end, sec, text, density, n_bins, verdict}]。"""
    raw = []
    occupied = []
    for seg in manifest["segments"]:
        seg_lines = [l for l in lines if l.get("seg") == seg["id"]]
        # 已對應語音的區間 —— 合併時不可跨過它們（見 `_blocks_merge`）
        occupied.extend((l["t_start"], l["t_end"]) for l in seg_lines)
        for h in score_mod.hallucination_spans(seg_lines, hyp,
                                               seg["t_start"], seg["t_end"]):
            raw.append({"seg": seg["id"], "start": h["start"], "end": h["end"],
                        "text": h.get("text", "")})
    out = []
    for h in merge_adjacent(raw, occupied):
        d, n = speech_density(bins, h["start"], h["end"], floor_db)
        out.append({
            "seg": h["seg"], "start": h["start"], "end": h["end"],
            # cue_sec 與 score.py 的 halluc_sec 同幣別；span_sec 是密度量的範圍
            "cue_sec": round(h["cue_sec"], 2),
            "span_sec": round(h["span_sec"], 2),
            "n_cues": h.get("n_cues", 1),
            "text": score_mod.normalize(h["text"]),
            "density": d, "n_bins": n, "verdict": classify(d),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把被判成幻覺的輸出，用語音密度分成真幻覺／人沒打到的語音")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--audio", required=True)
    a = ap.parse_args(argv)

    man, lines = golden_mod.load_golden(a.golden)
    hyp = score_mod.load_hyp(a.hyp)
    bins, floor_db, meta = sl.scan(a.audio, bin_sec=0.25, window_sec=0.25)
    rows = adjudicate(man, lines, hyp, bins, floor_db)

    print("噪音地板 %.2f dB／語音門檻 %.2f dB" % (floor_db, floor_db + sl.SPEECH_MARGIN_DB))
    print("")
    print("%-5s %15s %6s %6s %8s %6s %5s  %-20s %s"
          % ("段", "時間", "cue秒", "跨距", "語音密度", "格數", "cue數",
             "引擎寫了什麼", "判"))
    tot = {}
    for r in rows:
        tot[r["verdict"]] = tot.get(r["verdict"], 0.0) + r["cue_sec"]
        print("%-5s %6.1f–%6.1f %5.1fs %5.1fs %7s %6d %5d  %-20s %s"
              % (r["seg"], r["start"], r["end"], r["cue_sec"], r["span_sec"],
                 "n/a" if r["density"] is None else "%.0f%%" % (r["density"] * 100),
                 r["n_bins"], r["n_cues"], r["text"][:20], r["verdict"]))
    print("")
    # 合計一律用 cue 秒數 —— 與 score.py 的 halluc_sec 同幣別，才減得起來。
    # 跨距含 cue 之間的空隙，只用來界定 speech_density 量的範圍。
    for k in sorted(tot):
        print("  %s %.1fs（cue 秒數）" % (k, tot[k]))
    if not rows:
        print("  （沒有被判成幻覺的輸出）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
