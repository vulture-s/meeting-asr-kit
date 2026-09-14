# -*- coding: utf-8 -*-
"""meeting-transcribe/chunking.py 靜音感知切段的護欄測試。

## 為什麼這個檔存在

切點演算法搬自 MOSS-ASR（見
`（內部紀錄）case-studies/model-eval/moss-asr-repo-teardown-2026-09-02.md` §7），
而**原版那份的切點邏輯一條測試都沒有** —— 它把探測／偵測／規劃／切檔揉進一支
90 行函式，沒有可單測的接縫。搬過來時特意把規劃拆成純函式 `plan_cuts()`，
這個檔就是那個決定的兌現。

放 `tests/`（repo 根層）而非 `apps/.../tests/`：CI 只跑 `pytest tests/`
（沿用 `tests/acoustic/test_deid.py` 檔頭記的同一個理由）。被測模組
`chunking.py` 刻意只依賴 stdlib + ffmpeg subprocess，不 import faster_whisper，
否則測試會在 import 就 skip、等於沒有保護。

## 鎖住的東西

純函式層（不需要 ffmpeg，永遠會跑）：
1. span 連續、不重疊、完整覆蓋 [0, duration] —— 漏一段就是逐字稿少一塊
2. 窗內有靜音點就切在最近的那個；窗外的不算
3. 尾段守衛：不留短於 min_chunk 的碎片
4. min_chunk 下限：不會切在 cur + min_chunk 之前
5. 短檔回單一 span（等於不切）→ 不會白白斷掉 condition_on_previous_text
6. 壞輸入 raise，不靜默回空

ffmpeg 層（無 ffmpeg 時 skip）：
7. 合成音檔（20s 音 / 2s 靜音 / 20s 音 / 2s 靜音 / 20s 音）真的切在靜音中點
8. 切出來的檔案時長與規劃相符

跑法：`python -m pytest tests/acoustic/test_meeting_chunking.py -q`
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 工具目錄。**不寫死版面** —— 這批測試會出現在兩種地方：本 repo
# （`apps/acoustic-lab/tools/meeting-transcribe/`）與消毒後的獨立 kit（扁平）。
# 判準是「哪個目錄裡有 chunking.py」，不是路徑長什麼樣。
def _tool_dir():
    cands = [os.path.join(REPO_ROOT, "apps", "acoustic-lab", "tools",
                          "meeting-transcribe"),
             REPO_ROOT,
             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for d in cands:
        if os.path.exists(os.path.join(d, "chunking.py")):
            return d
    return cands[0]


sys.path.insert(0, _tool_dir())
import chunking  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg/ffprobe")


# ── 純函式層 ────────────────────────────────────────────────────────────

def _assert_contiguous(spans, total):
    """span 必須首尾相接、從 0 起、到 total 止 —— 任何縫隙都是逐字稿的洞。"""
    assert spans, "不可回空"
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(total)
    for (s, e) in spans:
        assert e > s, "span 必須有正長度: %r" % ((s, e),)
    for (prev, nxt) in zip(spans, spans[1:]):
        assert prev[1] == nxt[0], "span 之間有縫或重疊: %r -> %r" % (prev, nxt)


def test_short_file_returns_single_span():
    """短於上限就不切 —— 切了只會白白斷掉跨段上下文。"""
    spans = chunking.plan_cuts(600.0, [100.0, 200.0], max_chunk_sec=1080.0)
    assert spans == [(0.0, 600.0)]


def test_exactly_at_limit_not_split():
    spans = chunking.plan_cuts(1080.0, [500.0], max_chunk_sec=1080.0)
    assert spans == [(0.0, 1080.0)]


def test_cuts_at_nearest_silence_in_window():
    """窗內有多個靜音點，取離目標最近的那個。"""
    # target = 100；窗 = [max(0+10, 100-45), 100+15] = [55, 115]
    silences = [58.0, 97.0, 112.0]
    spans = chunking.plan_cuts(
        300.0, silences, max_chunk_sec=100.0,
        search_back=45.0, search_fwd=15.0, min_chunk=10.0,
    )
    assert spans[0] == (0.0, 97.0), "應選離 100 最近的 97，實得 %r" % (spans[0],)
    _assert_contiguous(spans, 300.0)


def test_ignores_silence_outside_window():
    """窗外的靜音點不可採用 —— 否則切片長度會失控。"""
    # target = 100，窗 = [55, 115]；40 與 200 都在窗外 → 應退回硬切 100
    spans = chunking.plan_cuts(
        300.0, [40.0, 200.0], max_chunk_sec=100.0,
        search_back=45.0, search_fwd=15.0, min_chunk=10.0,
    )
    assert spans[0] == (0.0, 100.0)
    _assert_contiguous(spans, 300.0)


def test_no_silence_falls_back_to_hard_cut():
    spans = chunking.plan_cuts(250.0, [], max_chunk_sec=100.0, min_chunk=10.0)
    assert spans[0] == (0.0, 100.0)
    assert spans[1] == (100.0, 200.0)
    _assert_contiguous(spans, 250.0)


def test_tail_guard_no_tiny_final_chunk():
    """切點離結尾太近時要併進尾段，不留碎片。

    total=205、max=100、無靜音 → 若無守衛會切出 (200, 205) 這種 5 秒碎片。
    """
    spans = chunking.plan_cuts(205.0, [], max_chunk_sec=100.0, min_chunk=60.0)
    assert spans[-1][1] - spans[-1][0] >= 60.0, "尾段碎片未被守衛吃掉: %r" % (spans,)
    _assert_contiguous(spans, 205.0)


def test_tail_guard_with_silence_point_near_end():
    """靜音點落在離結尾不到 min_chunk 處，同樣要併進尾段。"""
    spans = chunking.plan_cuts(
        260.0, [255.0], max_chunk_sec=250.0,
        search_back=45.0, search_fwd=15.0, min_chunk=60.0,
    )
    assert spans == [(0.0, 260.0)], "255 那個切點會留下 5 秒碎片，應被吃掉: %r" % (spans,)


def test_min_chunk_floor_blocks_early_silence():
    """不可切在 cur + min_chunk 之前，即使那裡有靜音點。"""
    # target=100，min_chunk=80 → 窗下緣 = max(80, 55) = 80；70 應被擋掉
    spans = chunking.plan_cuts(
        400.0, [70.0], max_chunk_sec=100.0,
        search_back=45.0, search_fwd=15.0, min_chunk=80.0,
    )
    assert spans[0] == (0.0, 100.0), "70 在 min_chunk 下限之前，不該被選: %r" % (spans[0],)


def test_multi_chunk_all_land_on_silence():
    """連續多段都應各自吃到自己窗內的靜音點。"""
    silences = [95.0, 190.0, 290.0]
    spans = chunking.plan_cuts(
        350.0, silences, max_chunk_sec=100.0,
        search_back=45.0, search_fwd=15.0, min_chunk=10.0,
    )
    assert [s[1] for s in spans[:-1]] == [95.0, 190.0, 290.0]
    _assert_contiguous(spans, 350.0)


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_invalid_max_chunk_raises(bad):
    with pytest.raises(ValueError):
        chunking.plan_cuts(100.0, [], max_chunk_sec=bad)


@pytest.mark.parametrize("bad", [0, -10])
def test_invalid_duration_raises(bad):
    with pytest.raises(ValueError):
        chunking.plan_cuts(bad, [], max_chunk_sec=100.0)


# ── silencedetect log 解析 ──────────────────────────────────────────────

def test_parse_silence_log_midpoints():
    log = (
        "[silencedetect @ 0x1] silence_start: 20\n"
        "[silencedetect @ 0x1] silence_end: 22.000063 | silence_duration: 2.000062\n"
        "[silencedetect @ 0x1] silence_start: 42\n"
        "[silencedetect @ 0x1] silence_end: 44.000062 | silence_duration: 2.000062\n"
    )
    points = chunking.parse_silence_log(log)
    assert points == pytest.approx([21.000031, 43.000031], abs=1e-4)


def test_parse_silence_log_drops_unclosed_trailing_start():
    """音檔在靜音中結束 → 最後一個 start 沒有 end，切在檔尾沒有意義。"""
    log = (
        "silence_start: 10\nsilence_end: 12 | silence_duration: 2\n"
        "silence_start: 50\n"
    )
    assert chunking.parse_silence_log(log) == [11.0]


def test_parse_silence_log_empty():
    assert chunking.parse_silence_log("") == []
    assert chunking.parse_silence_log("no silence here at all") == []


def test_parse_silence_log_ignores_duration_field():
    """silence_duration 不可被誤讀成 start/end。"""
    log = "silence_start: 1\nsilence_end: 3 | silence_duration: 2\n"
    assert chunking.parse_silence_log(log) == [2.0]


# ── 時間戳平移 ──────────────────────────────────────────────────────────

def test_shift_segments_applies_offset_without_mutating():
    segs = [{"start": 1.0, "end": 2.0, "text": "a"}]
    out = chunking.shift_segments(segs, 100.0)
    assert out[0]["start"] == 101.0 and out[0]["end"] == 102.0
    assert out[0]["text"] == "a"
    assert segs[0]["start"] == 1.0, "原 list 不可被就地改動"


def test_shift_segments_zero_offset_is_copy():
    segs = [{"start": 1.0, "end": 2.0}]
    out = chunking.shift_segments(segs, 0.0)
    assert out == segs
    assert out is not segs


# ── ffmpeg 實跑層 ──────────────────────────────────────────────────────

def _synth_audio(path):
    """20s 音 / 2s 靜音 / 20s 音 / 2s 靜音 / 20s 音 = 64s，靜音中點在 21 與 43。"""
    expr = "if(between(t,20,22)+between(t,42,44),0,0.5*sin(2*PI*440*t))"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "aevalsrc='%s':s=16000:d=64" % expr,
         "-ac", "1", str(path)],
        check=True,
    )


@needs_ffmpeg
def test_end_to_end_cuts_land_in_real_silence():
    tmpdir = tempfile.mkdtemp(prefix="chunktest-")
    try:
        src = os.path.join(tmpdir, "synth.wav")
        _synth_audio(src)

        assert chunking.probe_duration(src) == pytest.approx(64.0, abs=0.2)

        points = chunking.detect_silence_points(src)
        assert points == pytest.approx([21.0, 43.0], abs=0.1), (
            "偵測到的靜音中點不對: %r" % (points,)
        )

        spans = chunking.plan_cuts(
            64.0, points, max_chunk_sec=25.0,
            search_back=15.0, search_fwd=5.0, min_chunk=5.0,
        )
        assert len(spans) == 3
        assert spans[0][1] == pytest.approx(21.0, abs=0.1)
        assert spans[1][1] == pytest.approx(43.0, abs=0.1)
        _assert_contiguous(spans, 64.0)

        chunks = chunking.split_audio(src, spans, os.path.join(tmpdir, "parts"))
        assert len(chunks) == 3
        assert [c.start_offset for c in chunks] == pytest.approx([0.0, 21.0, 43.0], abs=0.1)
        for c in chunks:
            actual = chunking.probe_duration(c.path)
            assert actual == pytest.approx(c.duration, abs=0.15), (
                "part%d 實際時長 %.3f 與規劃 %.3f 不符" % (c.index, actual, c.duration)
            )
        # 切片總長須等於原檔 —— 少一秒就是逐字稿少一秒
        assert sum(c.duration for c in chunks) == pytest.approx(64.0, abs=0.2)

        chunking.cleanup_chunks(chunks, src)
        assert all(not os.path.exists(c.path) for c in chunks)
        assert os.path.exists(src), "cleanup 不可刪到原檔"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@needs_ffmpeg
def test_prepare_chunks_short_file_does_not_split():
    tmpdir = tempfile.mkdtemp(prefix="chunktest-")
    try:
        src = os.path.join(tmpdir, "synth.wav")
        _synth_audio(src)
        chunks, duration = chunking.prepare_chunks(
            src, os.path.join(tmpdir, "parts"), max_chunk_sec=600.0
        )
        assert len(chunks) == 1
        assert chunks[0].start_offset == 0.0
        assert str(chunks[0].path) == src, "未切時應直接指向原檔，不產生暫存檔"
        assert duration == pytest.approx(64.0, abs=0.2)
        assert not os.path.exists(os.path.join(tmpdir, "parts"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@needs_ffmpeg
def test_prepare_chunks_forwards_tuning_params():
    """min_chunk 等參數必須能從 prepare_chunks 傳下去。

    這條鎖的是實際踩過的坑：原本 prepare_chunks 只轉 max_chunk_sec，
    於是 `max_chunk_sec < min_chunk` 時會**靜默**退回單段 —— 64s 檔配
    預設 min_chunk=60 就是這樣，畫面上完全看不出為什麼沒切。
    """
    tmpdir = tempfile.mkdtemp(prefix="chunktest-")
    try:
        src = os.path.join(tmpdir, "synth.wav")
        _synth_audio(src)  # 64s，靜音中點 21 / 43

        # 生產預設（min_chunk=60）→ 這麼短的檔切不出兩段有效切片 → 單段
        chunks, _ = chunking.prepare_chunks(
            src, os.path.join(tmpdir, "a"), max_chunk_sec=25.0)
        assert len(chunks) == 1, "min_chunk=60 時 64s 檔不該切開"

        # 傳下調小的 min_chunk → 應該真的切成 3 段
        chunks, _ = chunking.prepare_chunks(
            src, os.path.join(tmpdir, "b"), max_chunk_sec=25.0,
            search_back=15.0, search_fwd=5.0, min_chunk=5.0)
        assert len(chunks) == 3, "調參沒有被傳下去: %r" % (chunks,)
        assert [round(c.start_offset) for c in chunks] == [0, 21, 43]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@needs_ffmpeg
def test_probe_duration_raises_on_garbage():
    """讀不到就 raise，不可回 0 讓後面靜默走單段路徑。"""
    tmpdir = tempfile.mkdtemp(prefix="chunktest-")
    try:
        bad = os.path.join(tmpdir, "not-audio.wav")
        with open(bad, "w") as fh:
            fh.write("this is not a wav file")
        with pytest.raises(RuntimeError):
            chunking.probe_duration(bad)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@needs_ffmpeg
def test_detect_silence_raises_on_garbage():
    """ffmpeg 失敗必須出聲 —— 靜默回空會退化成硬切且畫面跟『真的沒靜音』同形。"""
    tmpdir = tempfile.mkdtemp(prefix="chunktest-")
    try:
        bad = os.path.join(tmpdir, "not-audio.wav")
        with open(bad, "w") as fh:
            fh.write("this is not a wav file")
        with pytest.raises(RuntimeError):
            chunking.detect_silence_points(bad)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Property-based（hypothesis）─────────────────────────────────────────
#
# `verification.md` §Property-Based Testing 是**硬規則**：AI agent 產出的
# pure-function matcher / parser / classifier —— 明列「**輸入切段**」—— merge 前
# 必附 hypothesis property test 針對其宣稱的 invariant。`plan_cuts()` 正是那一類，
# 而第一版只有 example test 就送出去了，2026-09-02 雙軌審計抓到。
#
# 補這批的當下，它們在**未修的** `chunking.py` 上是紅的（hypothesis 自己縮到最小
# 反例 total=3.0 / max_chunk=1.0 / min_chunk=2.0 → [(0.0, 1.0)]），修完才轉綠 ——
# 照同節第 27 行的紅綠紀律。
#
# 值得記的是這條規則的來源案例（`gmail_triage._chunk_by_lines` 對超長單行破
# `len(c)<=limit`）跟這次抓到的缺陷幾乎同型：**切段函式破自己宣稱的長度下限**。

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st  # noqa: E402

_FIN = dict(allow_nan=False, allow_infinity=False)


@settings(deadline=None, max_examples=150)
@given(
    total=st.floats(min_value=0.1, max_value=100000.0, **_FIN),
    pts=st.lists(st.floats(min_value=-1000.0, max_value=110000.0, **_FIN), max_size=30),
    max_chunk=st.floats(min_value=0.1, max_value=5000.0, **_FIN),
    back=st.floats(min_value=0.0, max_value=500.0, **_FIN),
    fwd=st.floats(min_value=0.0, max_value=500.0, **_FIN),
    min_chunk=st.floats(min_value=0.0, max_value=500.0, **_FIN),
)
def test_prop_spans_are_a_partition(total, pts, max_chunk, back, fwd, min_chunk):
    """spans 必須是 [0, total] 的一個分割：連續、不重疊、正長度、完整覆蓋。

    也順帶保證終止 —— 不終止的話這個測試會逾時而不是失敗。
    """
    spans = chunking.plan_cuts(total, pts, max_chunk_sec=max_chunk,
                               search_back=back, search_fwd=fwd, min_chunk=min_chunk)
    assert spans, "不可回空"
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(total)
    for s, e in spans:
        assert math.isfinite(s) and math.isfinite(e), "端點必須有限: %r" % ((s, e),)
        assert e > s, "span 必須有正長度: %r" % ((s, e),)
    for prev, nxt in zip(spans, spans[1:]):
        assert prev[1] == nxt[0], "span 之間有縫或重疊: %r -> %r" % (prev, nxt)


@settings(deadline=None, max_examples=150)
@given(
    total=st.floats(min_value=0.1, max_value=100000.0, **_FIN),
    pts=st.lists(st.floats(min_value=0.0, max_value=100000.0, **_FIN), max_size=30),
    max_chunk=st.floats(min_value=0.1, max_value=5000.0, **_FIN),
    min_chunk=st.floats(min_value=0.0, max_value=500.0, **_FIN),
)
def test_prop_no_span_shorter_than_min_chunk(total, pts, max_chunk, min_chunk):
    """min_chunk 是**所有**切片的下限，不只尾段。

    唯一例外是整支音檔本身就比 min_chunk 短（那時只會有一段）。
    """
    assume(total > min_chunk)
    spans = chunking.plan_cuts(total, pts, max_chunk_sec=max_chunk, min_chunk=min_chunk)
    if len(spans) == 1:
        return
    # ⚠️ 要留浮點容差。`cur` 是累加出來的（0.1 → 0.2 → 0.30000000000000004 …），
    #    所以 `(cur + min_chunk) - cur` 可能比 min_chunk 少個 2e-17。
    #    hypothesis 找到的第一個「反例」就是這個，那不是碼的缺陷、是斷言太嚴：
    #    total=1.0 / max_chunk=0.1 / min_chunk=0.1 → (0.30000000000000004, 0.4)。
    #    實務上 min_chunk 是 60 秒級，1e-9 秒的鬆動沒有意義。
    tol = 1e-9
    short = [(s, e) for (s, e) in spans if (e - s) < min_chunk - tol]
    assert not short, "有 span 短於 min_chunk=%r: %r" % (min_chunk, short)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_raise(bad):
    """非有限輸入必須 raise —— 不可靜默回一個看起來像答案的東西。

    未修前：total=nan 回 []、max_chunk_sec=nan 回 [(0.0, nan)]、total=inf 不終止。
    """
    with pytest.raises(ValueError):
        chunking.plan_cuts(bad, [], max_chunk_sec=100.0)
    with pytest.raises(ValueError):
        chunking.plan_cuts(300.0, [], max_chunk_sec=bad)
    with pytest.raises(ValueError):
        chunking.plan_cuts(300.0, [], max_chunk_sec=100.0, min_chunk=bad)


def test_terminates_when_silence_point_equals_cursor():
    """靜音點正好等於當前游標且 min_chunk=0 → 未修前會原地打轉不終止。"""
    spans = chunking.plan_cuts(300.0, [0.0], max_chunk_sec=100.0,
                               search_back=100.0, search_fwd=0.0, min_chunk=0.0)
    _assert_contiguous(spans, 300.0)


def test_split_boundaries_round_consistently():
    """相鄰切片的 ffmpeg 參數必須共用同一個取整後的邊界，否則音訊會有 1ms gap/overlap。"""
    for a, b in [(97.0004, 190.0008), (97.0006, 190.0004), (12.3455, 45.6789)]:
        r_a, r_b = round(a, 3), round(b, 3)
        eff_end = float("%.3f" % r_a) + float("%.3f" % (r_b - r_a))
        next_start = float("%.3f" % r_b)
        assert eff_end == pytest.approx(next_start, abs=1e-9), (
            "邊界 %r/%r 取整後對不上：實效結束 %r vs 下段起點 %r"
            % (a, b, eff_end, next_start))
