# -*- coding: utf-8 -*-
"""RD 會議轉錄 — 長音檔靜音感知切段（三線可共用，目前只接 whisper 線）。

## 為什麼要有這支

長音檔一直是**手動**切的：`ffmpeg` 固定秒數切兩三段、各自前景跑、再手動併時間軸
（見 2026-07-16 三線 bench §1「PC 端長音檔的可行跑法」）。兩個問題：

1. **固定秒數會切在句子中間** —— 切點落在講話中途，兩邊各拿到半句，
   而 Whisper 的 30s 視窗在段首缺前文時最容易出錯。
2. **手動步驟不留痕** —— 切了幾段、offset 多少，只存在當次的 shell history。

本模組把切點改成**靜音感知**：在目標切點附近找自然停頓，切在那裡。
出處是 2026-08-30 拆解 `github.com/myyang19770915/MOSS-ASR` 的
`audio_processor.py::prepare_chunks`（見
`（內部紀錄）case-studies/model-eval/moss-asr-repo-teardown-2026-09-02.md` §7）。
**只借切點演算法，沒有借它的後處理**——那一層會把「謝謝」改成「謝」。

## 跟原版刻意不同的四點（都是本 repo 踩過的坑）

1. **規劃與執行分開。** `plan_cuts()` 是純函式（吃 duration + 靜音點，吐 span），
   不碰 ffmpeg → 切點邏輯可離線單測。原版把探測／偵測／規劃／切檔揉成一支
   90 行函式，所以它的切點邏輯**一條測試都沒有**。
2. **預設切片 18 分鐘，不是原版的 30 分鐘。** 依 README §RTF 那條實測教訓反推：
   前景 Bash 上限 10 分鐘、RTF 要抓上緣 **0.5 級**（2026-07-28 WH 場次量到 0.509，
   是短切片校準值 0.136 的 3.7 倍）→ 10min / 0.5 ≈ 20min 音訊，退一步取 18。
   ⚠️ 這個數字綁的是**當時那台 PC 的 RTF 上緣**，換機器要重量，別當常數沿用。
3. **ffmpeg 失敗要出聲。** 原版 `detect_silence_points` 不看 returncode，
   失敗就回空 list → 靜默退化成固定秒數硬切，而且畫面跟「這段音檔真的沒有靜音」
   完全一樣。這正是 `arkiv opencc 靜默降級` 那個形狀，本模組改成 raise。
4. **尾段守衛。** 切點若落在離結尾不到 `MIN_CHUNK_SEC` 的地方，直接併進最後一段，
   不留幾秒的碎片。原版沒有這個守衛。

## 不做重疊（與原版行為相同，但原版 README 講錯）

切片是**連續不重疊**的（`cur = end`），所以不會產生重複文字，也不需要去重。
原版 README §3.2 宣稱「若無檢測到靜音點，則採用平滑 Overlap 視窗切片」——
**code 裡沒有這回事**，`chunk_overlap_seconds` 是宣告了從未被讀取的死設定。

## 已知取捨

切段會斷掉 `condition_on_previous_text` 的跨段上下文：每一段都是重新開始，
段首拿不到前一段的文字。所以**短到不需要切的檔就不要切**（`plan_cuts` 在
`duration <= max_chunk_sec` 時回單一 span，呼叫端等於沒切）。
"""
import math
import os
import re
import shutil
import subprocess

# 🔴 2026-09-10：subprocess 讀 ffmpeg 輸出一律寫死 UTF-8
# text=True 會用系統 locale（本機 cp950）解 ffmpeg 的 UTF-8 stderr，
# reader thread 噴 UnicodeDecodeError 而 returncode 仍為 0 →
# 靜音點回空、靜默退化成固定秒數硬切。實測切點壓在黃金段邊界上。
import sys
from collections import namedtuple
from pathlib import Path

# 依 README §RTF 上緣 0.5 級 × 前景 10 分鐘上限反推（見檔頭第 2 點）
DEFAULT_MAX_CHUNK_SEC = 1080.0    # 18 min
SILENCE_NOISE_DB = -30.0
MIN_SILENCE_SEC = 0.5
SEARCH_BACK_SEC = 45.0            # 目標切點往前找多遠（偏早切，寧可短一點）
SEARCH_FWD_SEC = 15.0             # 往後找多遠
MIN_CHUNK_SEC = 60.0              # 切片下限，同時是尾段守衛的門檻

# start/duration 是「相對原檔」的秒數；path 為 None 代表尚未實際切檔
Chunk = namedtuple("Chunk", "index path start_offset duration")


def _require(exe):
    found = shutil.which(exe)
    if not found:
        sys.exit(
            "[FAIL] 找不到 %s — 靜音感知切段需要它。\n"
            "       PC: winget install Gyan.FFmpeg ／ Mac: brew install ffmpeg" % exe
        )
    return found


def probe_duration(path):
    """回傳音檔秒數。讀不到就 raise，不猜、不回 0。"""
    _require("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(
            "ffprobe 讀不到時長（exit %s）: %s\n%s"
            % (res.returncode, path, (res.stderr or "").strip()[-500:])
        )
    try:
        duration = float((res.stdout or "").strip())
    except ValueError:
        raise RuntimeError("ffprobe 回了無法解析的時長 %r: %s" % (res.stdout, path))
    if duration <= 0:
        raise RuntimeError("音檔時長為 %s，不是有效音訊: %s" % (duration, path))
    return duration


def parse_silence_log(stderr):
    """從 ffmpeg silencedetect 的 stderr 抽出每段靜音的**中點**。

    拆成獨立函式是為了能拿固定字串單測，不必真的跑 ffmpeg。
    只有成對的 start/end 才算；音檔在靜音中結束時 ffmpeg 會留一個沒有 end 的
    start，那筆捨棄（切在檔尾沒有意義）。
    """
    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*(-?[\d.]+)", stderr)]
    points = []
    for s, e in zip(starts, ends):
        if e > s:
            points.append((s + e) / 2.0)
    return sorted(points)


def detect_silence_points(audio_path, noise_db=SILENCE_NOISE_DB, min_silence=MIN_SILENCE_SEC):
    """跑 ffmpeg silencedetect，回傳靜音中點清單。

    ⚠️ ffmpeg 失敗一律 raise。回空 list **只代表真的沒偵到靜音**，
    不代表偵測壞掉 —— 兩者不可同形（原版正是在這裡靜默降級）。
    """
    _require("ffmpeg")
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", "silencedetect=noise=%sdB:d=%s" % (noise_db, min_silence),
        "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(
            "ffmpeg silencedetect 失敗（exit %s）: %s\n%s"
            % (res.returncode, audio_path, (res.stderr or "").strip()[-500:])
        )
    return parse_silence_log(res.stderr or "")


def plan_cuts(total_duration, silence_points, max_chunk_sec=DEFAULT_MAX_CHUNK_SEC,
              search_back=SEARCH_BACK_SEC, search_fwd=SEARCH_FWD_SEC,
              min_chunk=MIN_CHUNK_SEC):
    """規劃切段，回傳 [(start, end), ...]。

    純函式：不碰檔案系統、不跑 ffmpeg，給定同樣輸入永遠回同樣輸出。

    保證（每一條都有對應的 example test **與** property-based test）：
      * span 連續且不重疊，第一段從 0 起、最後一段到 total_duration 止
      * 每個切點若窗內有靜音點，取離目標最近的那個
      * **所有** span 都不短於 min_chunk（不只尾段），除非 total_duration 本身就更短
      * total_duration <= max_chunk_sec 時回單一 span（等於不切）
      * 永遠終止；非有限輸入 raise ValueError 而不是回一個看起來像答案的東西

    ⚠️ 前兩版的「保證」寫得比實際做到的寬（宣稱尾段不短於 min_chunk，但硬切
    fallback 會切出短於下限的**非尾段**），且沒有終止性保證。2026-09-02 雙軌審計
    抓到，同輪補上 property-based test —— `verification.md` §Property-Based Testing
    硬規則本來就要求「輸入切段」類必附，第一版漏了。
    """
    # 🔴 非有限值必須擋在門口。不擋的話 nan 會讓每個比較都是 False：
    #    plan_cuts(nan, …) 悄悄回 []、plan_cuts(300, …, max_chunk_sec=nan) 回 [(0, nan)]、
    #    total_duration=inf 則永遠跑不完 —— 三種都不會出聲。（2026-09-02 審計 F1）
    for _name, _v in (("total_duration", total_duration), ("max_chunk_sec", max_chunk_sec),
                      ("search_back", search_back), ("search_fwd", search_fwd),
                      ("min_chunk", min_chunk)):
        if not math.isfinite(_v):
            raise ValueError("%s 必須是有限數值，收到 %r" % (_name, _v))
    if max_chunk_sec <= 0:
        raise ValueError("max_chunk_sec 必須大於 0，收到 %r" % (max_chunk_sec,))
    if total_duration <= 0:
        raise ValueError("total_duration 必須大於 0，收到 %r" % (total_duration,))
    if min_chunk < 0 or search_back < 0 or search_fwd < 0:
        raise ValueError("min_chunk / search_back / search_fwd 不可為負")

    if total_duration <= max_chunk_sec:
        return [(0.0, float(total_duration))]

    points = sorted(float(p) for p in silence_points if math.isfinite(p))
    spans = []
    cur = 0.0
    while cur < total_duration:
        target = cur + max_chunk_sec
        if target >= total_duration:
            spans.append((cur, float(total_duration)))
            break

        lo = max(cur + min_chunk, target - search_back)
        hi = target + search_fwd
        cands = [p for p in points if lo <= p <= hi]
        end = min(cands, key=lambda p: abs(p - target)) if cands else target

        # 🔴 min_chunk 是**所有**切片的下限，不只尾段。硬切 fallback 走的是 target，
        #    而 target 可能早於 cur + min_chunk（`max_chunk_sec < min_chunk` 時必然如此）
        #    → 會切出短於下限的**非尾段**。（2026-09-02 審計 F2；hypothesis 最小反例
        #    total=3, max_chunk=1, min_chunk=2 → [(0.0, 1.0)]）
        end = max(end, min(cur + min_chunk, float(total_duration)))

        # 尾段守衛：剩下的不夠一個 min_chunk 就不要再切，直接吃到底。
        if total_duration - end < min_chunk:
            spans.append((cur, float(total_duration)))
            break

        # 🔴 進度保證。上面每一條都可能把 end 推回 cur（例如靜音點正好等於 cur 且
        #    min_chunk=0）→ `cur = end` 就原地打轉，永遠不終止。（2026-09-02 審計 F1）
        if end <= cur:
            spans.append((cur, float(total_duration)))
            break

        spans.append((cur, end))
        cur = end

    return spans


def split_audio(src, spans, outdir, sample_rate=16000):
    """依 spans 實際切檔，回傳 Chunk list。spans 長度為 1 時不切，直接指向原檔。"""
    ffmpeg = _require("ffmpeg")
    src = Path(src)
    if len(spans) <= 1:
        start, end = spans[0]
        return [Chunk(index=0, path=src, start_offset=0.0, duration=end - start)]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i, (start, end) in enumerate(spans):
        dst = outdir / ("%s.part%02d.wav" % (src.stem, i))
        # 🔴 兩端**先各自取整到毫秒，再相減求長度**。分別取整 start 與 (end-start)
        #    會讓某一段的實效結束點跟下一段的起點差到 1ms —— 音訊出現 gap 或 overlap，
        #    而規劃層的 span 明明是首尾相接的。（2026-09-02 審計 F4）
        #    相鄰段共用同一個取整後的邊界值，才真的接得起來。
        r_start = round(start, 3)
        r_end = round(end, 3)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "%.3f" % r_start,
            "-i", str(src),
            "-t", "%.3f" % (r_end - r_start),
            "-vn", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(sample_rate),
            str(dst),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(
                "ffmpeg 切段失敗（exit %s, part %d）: %s\n%s"
                % (res.returncode, i, src, (res.stderr or "").strip()[-500:])
            )
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError("切段產出空檔: %s" % dst)
        chunks.append(Chunk(index=i, path=dst, start_offset=start, duration=end - start))
    return chunks


def prepare_chunks(src, outdir, max_chunk_sec=DEFAULT_MAX_CHUNK_SEC,
                   search_back=SEARCH_BACK_SEC, search_fwd=SEARCH_FWD_SEC,
                   min_chunk=MIN_CHUNK_SEC):
    """便利包裝：probe → 偵測靜音 → 規劃 → 切檔。

    短於 max_chunk_sec 的檔會走單一 span 路徑，等於完全不切、也不產生暫存檔。

    ⚠️ 調參一律從這裡往下傳，別讓呼叫端只能改 max_chunk_sec —— 那樣
    `max_chunk_sec < min_chunk` 時會**靜默**退回單段（實測踩過：64s 檔配
    預設 min_chunk=60 就這樣，畫面上完全看不出為什麼沒切）。
    """
    duration = probe_duration(src)
    if duration <= max_chunk_sec:
        return [Chunk(index=0, path=Path(src), start_offset=0.0, duration=duration)], duration

    points = detect_silence_points(src)
    spans = plan_cuts(duration, points, max_chunk_sec=max_chunk_sec,
                      search_back=search_back, search_fwd=search_fwd,
                      min_chunk=min_chunk)
    print("[chunk] %.1f min -> %d 段（靜音點 %d 個）"
          % (duration / 60.0, len(spans), len(points)), flush=True)
    if len(spans) == 1:
        # 到得了這裡代表 duration > max_chunk_sec 卻仍只有一段 —— 說清楚為什麼，
        # 不要讓「切不動」跟「不需要切」在畫面上同形。
        print("        ⚠️ 未切：%.0fs 的檔配 min_chunk=%.0fs 切不出兩段有效切片。"
              "  要真的切請把 max_chunk_sec 拉到至少 %.0fs，或調低 min_chunk。"
              % (duration, min_chunk, min_chunk * 2), flush=True)
    for i, (s, e) in enumerate(spans):
        print("        part%02d  %7.1fs → %7.1fs  (%.1f min)"
              % (i, s, e, (e - s) / 60.0), flush=True)
    return split_audio(src, spans, outdir), duration


def shift_segments(segments, start_offset):
    """把一段切片的轉錄結果時間戳平移回原檔座標。

    segments 為 dict list，需含 'start' / 'end'。回傳新 list，不就地改。
    """
    if not start_offset:
        return list(segments)
    shifted = []
    for s in segments:
        item = dict(s)
        item["start"] = s["start"] + start_offset
        item["end"] = s["end"] + start_offset
        shifted.append(item)
    return shifted


def cleanup_chunks(chunks, src):
    """刪掉切出來的暫存檔；`path` 就是原檔的那筆（未切）不動。"""
    src = Path(src)
    for c in chunks:
        p = Path(c.path)
        if p != src and p.exists():
            try:
                os.unlink(p)
            except OSError:
                pass
