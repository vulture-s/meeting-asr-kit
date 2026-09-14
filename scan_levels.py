# -*- coding: utf-8 -*-
"""黃金段選段輔助 —— 掃全檔電平與語音密度，逐 60 秒窗列表。

## 為什麼要有這支

黃金段的選段紀律是「**不看任何 ASR 稿來選**」
（`（內部評測紀錄）` §1.2）
—— 看了會選到「引擎已經出錯的地方」，量尺被偏誤放大，而**選段偏誤沒有事後補救的辦法**。

那紀律留下一個實際問題：不看稿要憑什麼選？plan 寫的是「波形電平統計 ＋ 對會議流程的記憶」。
這支負責前半 —— **把客觀聲學特徵攤出來**，後半（誰在講、講什麼、哪裡在搶話）由人補。

## 🔴 它不會幫你分類 S1–S5，這是刻意的

五個段各打一個失敗模式，但**只有 S5（低電平／噪音）是聲學上判得出來的**。
S1 單人 vs S3 多人交疊、S2 台語密集、S4 型號密集 —— 這些**光看電平分不出來**，
硬猜就是在造假客觀性。所以輸出只給：

- 每個 60 秒窗的 RMS 電平、峰值、語音密度、動態範圍
- 兩份**誠實的**排行：語音密度最高的窗、電平最低（但仍有語音）的窗 ← S5 候選

其餘四段請照 plan 的規定，用你對那場會議的記憶選，並把理由寫進 manifest 的 `reason`。

## 量法

一次 ffmpeg pass，讓 C 那邊做量測（不吃 numpy，不自己在 Python 迴圈裡跑 9 千萬個樣本）：

```
aresample=16000 → mono → asetnsamples=4000（＝0.25s 一格）→ astats(reset=1) → ametadata print
```

0.25 秒一格，60 秒窗 ＝ 240 格。窗統計由格統計在 Python 聚合（純函式、可單測）。

**語音密度**＝該窗內「RMS 高於全檔噪音地板 + `SPEECH_MARGIN_DB`」的格數比例。
噪音地板取全檔格 RMS 的第 10 百分位 —— 不是絕對門檻，因為 0714／0728 兩場已經證明
**交來的電平隨錄音裝置變動**（0714 比 0630 低 21dB），寫死門檻會在下一場失準。

⚠️ 密度是**粗指標**：0.25 秒的格會把短停頓算成語音。它用來相對比較窗與窗，不是 VAD。
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

BIN_SEC = 0.25          # 一格 0.25 秒（16kHz → 4000 samples）
WINDOW_SEC = 60.0       # 黃金段一段 60 秒（plan §1.2）
SPEECH_MARGIN_DB = 12.0  # 高於噪音地板多少 dB 才算語音格
FLOOR_PERCENTILE = 10   # 噪音地板取第幾百分位

_META_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?inf|-?\d+(?:\.\d+)?)")
_PEAK_RE = re.compile(r"lavfi\.astats\.Overall\.Peak_level=(-?inf|-?\d+(?:\.\d+)?)")


class ScanError(RuntimeError):
    """ffmpeg 掛了或吐不出東西。**不靜默回空** —— 回空會被讀成「這檔真的沒聲音」。"""


# ---------------------------------------------------------------- 純函式層

def _to_db(tok):
    """astats 對全靜音格吐 `-inf`。轉成一個很低但有限的值，讓後面的算術不炸。"""
    if tok in ("-inf", "inf", "-inf"):
        return -120.0
    try:
        v = float(tok)
    except ValueError:
        return -120.0
    if math.isinf(v) or math.isnan(v):
        return -120.0
    return max(v, -120.0)


def parse_astats(text):
    """把 ametadata 的輸出解析成 [(rms_db, peak_db), ...]，一格一筆。

    peak 缺席時用 rms 補（有些 ffmpeg build 不吐 Peak_level）。
    """
    rms = [_to_db(m) for m in _META_RE.findall(text)]
    peak = [_to_db(m) for m in _PEAK_RE.findall(text)]
    if not rms:
        raise ScanError("ffmpeg 沒吐出任何 RMS_level —— 檢查 ffmpeg 是否有該格式的 decoder")
    if len(peak) != len(rms):
        peak = list(rms)
    return list(zip(rms, peak))


def percentile(values, pct):
    """第 pct 百分位（線性插值）。空 list 回 None。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean_db(dbs):
    """dB 平均要在**線性功率域**做，不能直接平均 dB 值。"""
    if not dbs:
        return -120.0
    p = sum(10.0 ** (d / 10.0) for d in dbs) / len(dbs)
    if p <= 0:
        return -120.0
    return 10.0 * math.log10(p)


def bins_to_windows(bins, bin_sec=BIN_SEC, window_sec=WINDOW_SEC,
                    margin_db=SPEECH_MARGIN_DB, floor_pct=FLOOR_PERCENTILE):
    """格統計 → 窗統計。純函式，`scan()` 之外可單獨測。

    回 (windows, floor_db)。windows 每筆：
      t_start / t_end / rms_db / peak_db / speech_frac / dyn_range_db / bins
    """
    if not bins:
        return [], None
    all_rms = [b[0] for b in bins]
    floor_db = percentile(all_rms, floor_pct)
    thresh = floor_db + margin_db

    per_window = max(1, int(round(window_sec / bin_sec)))
    out = []
    for i in range(0, len(bins), per_window):
        chunk = bins[i:i + per_window]
        rms = [c[0] for c in chunk]
        peaks = [c[1] for c in chunk]
        speech = [r for r in rms if r > thresh]
        out.append({
            "t_start": round(i * bin_sec, 2),
            "t_end": round(min((i + len(chunk)) * bin_sec, len(bins) * bin_sec), 2),
            "rms_db": round(_mean_db(rms), 2),
            "speech_rms_db": round(_mean_db(speech), 2) if speech else None,
            "peak_db": round(max(peaks), 2),
            "speech_frac": round(len(speech) / float(len(chunk)), 3),
            "dyn_range_db": round(max(rms) - min(rms), 2),
            "bins": len(chunk),
        })
    return out, floor_db


# ---------------------------------------------------------------- ffmpeg 層

def scan(path, bin_sec=BIN_SEC, window_sec=WINDOW_SEC, ffmpeg=None):
    """跑一次 ffmpeg，回 (windows, floor_db, meta)。"""
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise ScanError("PATH 上找不到 ffmpeg")
    if not os.path.exists(path):
        raise ScanError("音檔不存在：%s" % path)

    nsamples = int(round(16000 * bin_sec))
    af = ("aresample=16000,aformat=channel_layouts=mono,"
          "asetnsamples=n=%d:p=0,astats=metadata=1:reset=1,"
          "ametadata=print:file=-" % nsamples)
    cmd = [exe, "-hide_banner", "-nostats", "-v", "error",
           "-i", path, "-af", af, "-f", "null", os.devnull]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # 不靜默降級：ffmpeg 缺 decoder（NAS 上實際踩過）跟「檔案沒聲音」長得一樣
        raise ScanError("ffmpeg 失敗 rc=%d：%s"
                        % (proc.returncode, proc.stderr.decode("utf-8", "replace")[-500:]))
    text = proc.stdout.decode("utf-8", "replace")
    bins = parse_astats(text)
    windows, floor_db = bins_to_windows(bins, bin_sec, window_sec)
    meta = {"path": path, "bin_sec": bin_sec, "window_sec": window_sec,
            "bins": len(bins), "duration_sec": round(len(bins) * bin_sec, 2),
            "floor_db": round(floor_db, 2) if floor_db is not None else None,
            "speech_threshold_db": round(floor_db + SPEECH_MARGIN_DB, 2)
            if floor_db is not None else None}
    return windows, floor_db, meta


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="掃全檔電平與語音密度，逐 60 秒窗列出（黃金段選段輔助）")
    ap.add_argument("audio")
    ap.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    ap.add_argument("--bin-sec", type=float, default=BIN_SEC)
    ap.add_argument("--top", type=int, default=8, help="兩份排行各列幾筆")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    windows, floor_db, meta = scan(args.audio, args.bin_sec, args.window_sec)

    if args.json:
        print(json.dumps({"meta": meta, "windows": windows},
                         ensure_ascii=False, indent=1))
        return 0

    print("檔案 %s" % os.path.basename(meta["path"]))
    print("時長 %.1fs／格 %.2fs／窗 %.0fs　噪音地板 %.2f dB　語音門檻 %.2f dB"
          % (meta["duration_sec"], meta["bin_sec"], meta["window_sec"],
             meta["floor_db"], meta["speech_threshold_db"]))
    print("")
    print("%-14s %9s %9s %9s %9s" % ("窗（時:分:秒）", "RMS dB", "峰值 dB", "語音密度", "動態 dB"))
    for w in windows:
        print("%-14s %9.2f %9.2f %8.1f%% %9.2f"
              % (_hms(w["t_start"]) + "–" + _hms(w["t_end"]),
                 w["rms_db"], w["peak_db"], w["speech_frac"] * 100, w["dyn_range_db"]))

    # 只給聲學上判得出來的兩份排行（見檔頭）
    dense = sorted(windows, key=lambda w: -w["speech_frac"])[:args.top]
    quiet = [w for w in windows if w["speech_frac"] >= 0.30]
    quiet = sorted(quiet, key=lambda w: w["rms_db"])[:args.top]

    print("")
    print("語音密度最高（講話最滿的窗）：")
    for w in dense:
        print("  %s  密度 %.1f%%  RMS %.2f dB"
              % (_hms(w["t_start"]), w["speech_frac"] * 100, w["rms_db"]))
    print("")
    print("電平最低但仍有語音的窗 ← S5（低電平／噪音）候選：")
    for w in quiet:
        print("  %s  RMS %.2f dB  密度 %.1f%%"
              % (_hms(w["t_start"]), w["rms_db"], w["speech_frac"] * 100))
    print("")
    print("⚠️ S1 單人／S2 台語／S3 交疊／S4 型號 **聲學上分不出來** —— 照 plan §1.2 用你對")
    print("   那場會議的記憶選，並把理由寫進 manifest 的 reason。不要看 ASR 稿。")
    return 0


def _hms(sec):
    s = int(round(sec))
    return "%d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


if __name__ == "__main__":
    sys.exit(main())
