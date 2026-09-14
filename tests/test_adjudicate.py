# -*- coding: utf-8 -*-
"""`adjudicate.py` 的護欄測試 —— 把真幻覺與「人沒打到的語音」分開。

## 為什麼這個檔存在

2026-09-11 差一點把「甲線幻覺率比乙線高一個數量級」寫成結論。
實際拆開：Whisper 被判幻覺的 15.3 秒裡有 13.3 秒落在黃金段的 14.0 秒空白裡，
語音密度 75–88%，語意與前後連貫成同一段話 —— **那是人沒打到的真實語音，不是幻覺**。

⇒ 在非窮盡的黃金段上，`hallucination` 是**上限**不是幻覺率。
這支把上限拆開，而拆法必須守住一條：**不可判定要明說，不准折成任何一邊**
（`verification.md` §Negative Result 同型）。

## 鎖住的東西

1. 高密度 → 判「人沒打到的語音」；近靜音 → 判「真幻覺」
2. **格數不足 → `None` ＋「不可判定」**，不可硬給答案（2 秒只有 8 格，
   二項變異大到 12% 與 25% 分不開）
3. 門檻是相對噪音地板算的，不是絕對 dB —— 收音電平逐場而異
   （0714 比 0630 低 21dB），寫死會在下一場失準
"""
import os
import sys

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
import adjudicate as adj  # noqa: E402


def bins(pattern):
    """pattern 是每格的 rms_db。"""
    return [{"rms_db": db, "peak_db": db} for db in pattern]


FLOOR = -48.0          # 門檻 = -48 + 12 = -36 dB


class TestSpeechDensity:
    def test_all_loud_is_full_density(self):
        d, n = adj.speech_density(bins([-20.0] * 40), 0.0, 10.0, FLOOR)
        assert d == 1.0 and n == 40

    def test_all_floor_is_zero_density(self):
        d, _ = adj.speech_density(bins([-48.0] * 40), 0.0, 10.0, FLOOR)
        assert d == 0.0

    def test_half_and_half(self):
        d, _ = adj.speech_density(bins([-20.0] * 20 + [-60.0] * 20), 0.0, 10.0, FLOOR)
        assert d == pytest.approx(0.5)

    def test_threshold_is_relative_to_floor(self):
        """整份平移 30dB，密度不可變 —— 收音電平逐場而異，絕對門檻會失準。"""
        a, _ = adj.speech_density(bins([-20.0] * 20 + [-60.0] * 20), 0.0, 10.0, FLOOR)
        b, _ = adj.speech_density(bins([-50.0] * 20 + [-90.0] * 20), 0.0, 10.0,
                                  FLOOR - 30)
        assert a == b


class TestShortSpansAreUnjudgeable:
    """**本檔最重要的一條。** 短段不准硬給答案。"""

    def test_two_second_span_returns_none(self):
        # 2 秒 = 8 格 < MIN_BINS
        d, n = adj.speech_density(bins([-20.0] * 8), 0.0, 2.0, FLOOR)
        assert d is None and n == 8

    def test_none_classifies_as_unjudgeable_not_hallucination(self):
        v = adj.classify(None)
        assert "不可判定" in v
        assert "幻覺" not in v.replace("不可判定(格數不足)", "")

    def test_just_enough_bins_is_judged(self):
        d, n = adj.speech_density(bins([-20.0] * adj.MIN_BINS), 0.0, 3.0, FLOOR)
        assert d is not None and n == adj.MIN_BINS


class TestClassify:
    def test_dense_is_real_speech(self):
        assert adj.classify(0.80) == "人沒打到的語音"

    def test_silent_is_hallucination(self):
        assert adj.classify(0.02) == "真幻覺"

    def test_boundary_is_inclusive_toward_speech(self):
        """邊界偏向「有人講」—— 誤指幻覺的代價比漏判高（會冤枉引擎）。"""
        assert adj.classify(adj.SILENCE_DENSITY) == "人沒打到的語音"
        assert adj.classify(adj.SILENCE_DENSITY - 0.001) == "真幻覺"


class TestAdjudicateEndToEnd:
    def _fixture(self):
        man = {"set_id": "T", "source_audio": "x", "source_md5": "0" * 32,
               "duration_sec": 100.0,
               "segments": [{"id": "G1", "t_start": 0.0, "t_end": 60.0,
                             "reason": "測試"}]}
        lines = [{"id": "G1-001", "seg": "G1", "t_start": 0.0, "t_end": 10.0,
                  "spk": "S1", "lang": "zho", "zh": "有打到的句子", "models": []}]
        return man, lines

    def test_span_in_golden_gap_with_speech_is_not_hallucination(self):
        """黃金段的空白 ＋ 高密度 ＝ 人沒打到的語音（0908 G4 那個形狀）。"""
        man, lines = self._fixture()
        hyp = [{"start": 20.0, "end": 30.0, "text": "空白裡的真實發言"}]
        rows = adj.adjudicate(man, lines, hyp, bins([-20.0] * 400), FLOOR)
        assert len(rows) == 1 and rows[0]["verdict"] == "人沒打到的語音"

    def test_span_in_silence_is_hallucination(self):
        man, lines = self._fixture()
        hyp = [{"start": 20.0, "end": 30.0, "text": "謝謝觀看下次再見"}]
        rows = adj.adjudicate(man, lines, hyp, bins([-48.0] * 400), FLOOR)
        assert len(rows) == 1 and rows[0]["verdict"] == "真幻覺"

    def test_span_overlapping_golden_is_not_reported(self):
        """有對到黃金句的輸出不是幻覺候選，不該出現在清單裡。"""
        man, lines = self._fixture()
        hyp = [{"start": 2.0, "end": 8.0, "text": "有打到的句子"}]
        assert adj.adjudicate(man, lines, hyp, bins([-20.0] * 400), FLOOR) == []

class TestMergeAdjacent:
    """判定單位是「連續的未對應區段」，不是單顆 cue。

    cue 邊界是引擎切段風格的產物。2026-09-11 實測：G4 那 12 秒是六顆 2 秒 cue
    連在一起，**逐顆判每顆只有 8 格、全部「不可判定」**，合起來 48 格就判得出來
    （密度 52%）。逐顆判會把看得出來的事實丟掉 —— 那也是一種量錯單位。
    """

    def _s(self, seg, a, b, txt="x"):
        return {"seg": seg, "start": a, "end": b, "text": txt}

    def test_contiguous_cues_merge(self):
        spans = [self._s("G4", 10.0, 12.0, "甲"), self._s("G4", 12.0, 14.0, "乙"),
                 self._s("G4", 14.0, 16.0, "丙")]
        got = adj.merge_adjacent(spans, ())
        assert len(got) == 1
        assert (got[0]["start"], got[0]["end"]) == (10.0, 16.0)
        assert got[0]["n_cues"] == 3 and got[0]["text"] == "甲乙丙"

    def test_gap_larger_than_threshold_does_not_merge(self):
        spans = [self._s("G4", 10.0, 12.0), self._s("G4", 20.0, 22.0)]
        assert len(adj.merge_adjacent(spans, ())) == 2

    def test_different_segments_never_merge(self):
        """不同段之間即使時間相鄰也不可併 —— 它們屬於不同的判定脈絡。"""
        spans = [self._s("G1", 10.0, 12.0), self._s("G2", 12.0, 14.0)]
        assert len(adj.merge_adjacent(spans, ())) == 2

    def test_merge_refuses_to_cross_transcribed_speech(self):
        """空隙裡有已對應的黃金句 ⇒ 不可併（Codex #11）。

        原本 `merge_adjacent` 只看 cue 間距，看不到空隙裡其實有一句已經被
        轉出來的話。實測：兩段各 1.25 秒的靜音 cue 各自只有 5 格、都回 `None`；
        在中間 1.25–1.75 放一句熱的黃金句，兩段就併成 12 格、密度 2/12 = 16.7%
        ⇒ 判成「人沒打到的語音」，**而那個密度完全來自已經打出來的那句話**。
        """
        spans = [self._s("G1", 0.0, 1.25), self._s("G1", 1.75, 3.0)]
        assert len(adj.merge_adjacent(spans, ())) == 1, "沒有黃金句時本來該併"
        assert len(adj.merge_adjacent(spans, [(1.25, 1.75)])) == 2, (
            "跨過已對應語音併起來了 —— 密度會來自已經轉出來的那句話")

    def test_occupied_is_required_not_defaulted(self):
        """忘記傳 `occupied` 要 TypeError，不可靜默失去保護。

        給預設值等於把陷阱寫進 docstring 而不是拆掉它。
        """
        spans = [self._s("G1", 0.0, 1.0)]
        with pytest.raises(TypeError):
            adj.merge_adjacent(spans)

    def test_cue_seconds_and_span_seconds_are_separate(self):
        """合併後要分開回報兩種秒數 —— 混用就是不同幣別相減。

        `cue_sec` 與 `score.py` 的 `halluc_sec` 同幣別；
        `span_sec` 含 cue 之間的空隙，是 `speech_density` 量的範圍。
        case-study 的「15.3 秒裡有 13.3 秒」正是兩種單位相減。
        """
        spans = [self._s("G1", 0.0, 2.0), self._s("G1", 2.4, 4.4),
                 self._s("G1", 4.8, 6.8)]
        got = adj.merge_adjacent(spans, ())
        assert len(got) == 1
        assert got[0]["cue_sec"] == pytest.approx(6.0)
        assert got[0]["span_sec"] == pytest.approx(6.8)

    def test_missing_text_and_none_seg_do_not_crash(self):
        """部分輸入原本會炸：`seg=None` → TypeError、缺 `text` → KeyError。"""
        spans = [{"seg": None, "start": 0.0, "end": 1.0},
                 {"seg": "G1", "start": 2.0, "end": 3.0, "text": "x"}]
        got = adj.merge_adjacent(spans, ())
        assert len(got) == 2
        assert all("text" in h for h in got)

    def test_empty_input_is_safe(self):
        assert adj.merge_adjacent([], ()) == []

    def test_merge_adjacent_does_not_mutate_input(self):
        """回傳新 dict，不可改到呼叫端的資料。"""
        spans = [self._s("G1", 0.0, 1.0, "甲"), self._s("G1", 1.0, 2.0, "乙")]
        snapshot = [dict(s) for s in spans]
        adj.merge_adjacent(spans, ())
        assert spans == snapshot

    def test_merged_span_becomes_judgeable(self):
        """六顆 2 秒（各 8 格，不可判定）併成 12 秒（48 格）後判得出來。"""
        man = {"set_id": "T", "source_audio": "x", "source_md5": "0" * 32,
               "duration_sec": 100.0,
               "segments": [{"id": "G1", "t_start": 0.0, "t_end": 60.0,
                             "reason": "測試"}]}
        lines = [{"id": "G1-001", "seg": "G1", "t_start": 0.0, "t_end": 5.0,
                  "spk": "S1", "lang": "zho", "zh": "有打到", "models": []}]
        hyp = [{"start": 20.0 + 2 * i, "end": 22.0 + 2 * i, "text": "話"}
               for i in range(6)]
        rows = adj.adjudicate(man, lines, hyp, bins([-20.0] * 400), FLOOR)
        assert len(rows) == 1, "六顆相鄰 cue 沒有被併起來"
        assert rows[0]["n_cues"] == 6 and rows[0]["n_bins"] == 48
        assert rows[0]["verdict"] == "人沒打到的語音"


class TestSpeechDensityBinBoundaries:
    """格的選取：與 [t0, t1] **相交**的格全要算（Codex #10）。

    原本 `i1 = int(t1 / bin_sec)` 在切片 exclusive 的前提下，把尾端那個
    部分重疊的格整個丟掉。實測那一格是熱的時，密度 2/13 = 15.4%
    變成 1/12 = 8.3% —— **跨過 15% 門檻，判定從「人沒打到的語音」翻成「真幻覺」**。
    """

    def _bins(self, n, hot=()):
        out = [{"rms_db": -80.0} for _ in range(n)]
        for i in hot:
            out[i]["rms_db"] = -10.0
        return out

    def test_partially_overlapping_tail_bin_is_included(self):
        d, n = adj.speech_density(self._bins(13, hot=(0, 12)), 0.0, 3.1, FLOOR)
        assert n == 13, "尾端那個相交的格被 floor 掉了"
        assert d == pytest.approx(2.0 / 13)
        assert adj.classify(d) == "人沒打到的語音"

    def test_dropping_the_tail_bin_flips_the_verdict(self):
        """反向：證明這個差別真的會翻面，不是無關的小數點。"""
        bins = self._bins(13, hot=(0, 12))
        floored = 1.0 / 12                      # 舊行為
        assert adj.classify(floored) == "真幻覺"
        d, _ = adj.speech_density(bins, 0.0, 3.1, FLOOR)
        assert adj.classify(d) != adj.classify(floored)

    def test_grid_aligned_range_is_unchanged(self):
        """反向：剛好對齊格線時行為不可被這次改動動到。"""
        _, n = adj.speech_density(self._bins(20, hot=(0,)), 0.0, 3.0, FLOOR)
        assert n == 12

    def test_negative_start_is_clamped(self):
        """t0 < 0 不可讓切片變成從尾巴倒數（Python 負索引）。"""
        _, n = adj.speech_density(self._bins(20, hot=(0,)), -1.0, 3.0, FLOOR)
        assert n == 12

    def test_bin_sec_defaults_to_the_producer(self):
        """格長只有一份真相 —— 生產者是 `scan_levels.BIN_SEC`。

        原本這裡寫死 0.25，是第二份真相：生產者改了格長，這支會安靜地
        讀錯時間範圍，而所有測試照綠（fixture 自己手造 bins）。
        """
        import scan_levels as sl
        assert adj.speech_density.__defaults__[0] == sl.BIN_SEC

    def test_real_producer_output_feeds_this_consumer(self):
        """把 `bins_to_windows()` 的**真實輸出**餵進來 —— 索引換算要接得上。

        原本沒有任何測試把生產者接回消費者：fixture 直接手造
        `[{"rms_db": ...}]`，於是「index == 時間 / 0.25」這條換算一旦漂掉，
        adjudicate 的測試全綠而工具讀的是錯的時間範圍。
        那正是「fixture 與被測對象同形」的形狀，而它守的是最貴的那個判讀。
        """
        import scan_levels as sl
        # 40 格 = 10 秒；讓 20–24 格（5.0–6.0 秒）是熱的
        raw = [{"t": i * sl.BIN_SEC, "rms_db": (-10.0 if 20 <= i < 24 else -80.0)}
               for i in range(40)]
        d, n = adj.speech_density(raw, 5.0, 6.0, FLOOR)
        assert n == 4, "時間 5.0–6.0 應該對到 4 格，實得 %d" % n
        assert d is None or d == pytest.approx(1.0)   # 4 格 < MIN_BINS ⇒ None
        d2, n2 = adj.speech_density(raw, 5.0, 9.0, FLOOR)
        assert n2 == 16
        assert d2 == pytest.approx(4.0 / 16)
