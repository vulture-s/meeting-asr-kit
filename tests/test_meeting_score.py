# -*- coding: utf-8 -*-
"""會議 ASR 評分 harness（`score.py` + `golden.py`）的護欄測試。

## 為什麼這個檔存在

這支 harness 的產出會被用來下「換不換引擎」的結論
（`（內部紀錄）plans/arkiv/2026-06-09-stt-bench-plan.md` 的「≥2% 絕對 CER」門檻）。
**量尺沒過負向測試就評分，等於用沒校準的尺量東西**
（memory `feedback_verifier_needs_its_own_negative_test`）。

而 0716 那次「型號命中率」失敗正是這個形狀：比對方式看起來合理、跑起來有數字，
但量到的是**書寫格式**不是準確率。那次沒有任何測試會紅。

## 校準閘（`TestCalibrationGates`）—— 全綠才准引用任何 CER

1. hyp ＝ golden 自己 → 內容 CER 0.0／覆蓋率 100%／幻覺率 0%
2. golden 刪掉 10% 字元 → CER 落在 8–12%
3. 空稿 → CER 100%／覆蓋率 0%／幻覺率**定義為 0 而非 NaN**
4. **hyp 時間戳全平移 +30s → 覆蓋率大幅下降**（不然覆蓋率根本沒在用時間軸）
5. **每個型號換成別的型號 → precision 與 recall 皆為 0**（不然正規化過度歸一）

6. **hyp 併成一顆巨大 cue → CER 不可變**（不然量的是引擎的切段風格）
8. **hyp 切更細 → 覆蓋率不可降**（第 6 條的反方向；真實資料上先炸的是這一邊）
7. 一顆蓋住全段的 cue → **幻覺率必須回 None 不可回 0**（回 0 ＝ 假陰性）

第 4、5 條是 2026-09-10 新加的：前三條只驗「算得對」，驗不到「量的是不是對的維度」。
**第 6、7 條同日再加**，因為前五條**沒抓到一個真實缺陷** —— 它們的 hyp 都跟 golden
同一種切法，「時間分桶」的錯誤永遠不會現形；真實資料一送進來（雅婷有一顆 296.8 秒的
cue）就爆掉。教訓：**校準閘的 fixture 若與真實資料形狀不同，它守不到真實的失敗。**

## 其餘鎖住的東西

- 正規化順序與冪等性；中文數字兩種讀法（`二零四` / `二百零四` 都要是 204）
- 型號偵測長→短遮罩：`204` 不可匹配進 `204-D` 裡面
- `taigi_retention`（**不需參考答案**的台語忠實度）：抹平的引擎 ≈ 0、忠實的 > 0；
  國台通用字與簡體同形字**不可算命中**（否則整份假陽性）
- `taigi_flatten_gap`：只在真的有人填 native 時才有值，否則 None（不可回假的 0）
- schema 驗證器**必須**擋：缺欄位／lang 不合法／t_end ≤ t_start／id 重複／
  `native` 與 `zh` 同字（不構成第二軌）／manifest 段缺 reason
  ⚠️ **`lang=nan` 不再要求 native**（2026-09-10 改，見 `golden.py` 該處註解）
- schema 驗證器**不可**擋：同段內時間重疊的句子（S3 交疊段的重疊是要量的東西）

被測模組只依賴 stdlib，測試永遠會跑（同 `test_meeting_chunking.py` 檔頭的理由）。

跑法：`python -m pytest tests/acoustic/test_meeting_score.py -q`
"""
import json
import os
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
import golden  # noqa: E402
import score  # noqa: E402


# ── fixture：一份最小但形狀完整的黃金段 ────────────────────────────────

def _manifest():
    return {
        "set_id": "T-test",
        "source_audio": "test.m4a",
        "source_md5": "0" * 32,
        "duration_sec": 600.0,
        "segments": [
            {"id": "S1", "t_start": 0.0, "t_end": 60.0,
             "reason": "乾淨國語單人（測試用）", "post_edit": True},
            {"id": "S2", "t_start": 100.0, "t_end": 160.0,
             "reason": "台語密集（測試用）", "post_edit": False},
        ],
    }


def _lines():
    return [
        {"id": "S1-01", "seg": "S1", "t_start": 2.0, "t_end": 8.0, "spk": "S1",
         "lang": "zho", "zh": "這批MX-K的庫存還有九個", "models": ["MX-K"]},
        {"id": "S1-02", "seg": "S1", "t_start": 10.0, "t_end": 16.0, "spk": "S2",
         "lang": "zho", "zh": "二零四D的開模下週才會好", "models": ["204-D"]},
        {"id": "S2-01", "seg": "S2", "t_start": 102.0, "t_end": 108.0, "spk": "S1",
         "lang": "nan", "zh": "網路線沒有這個問題",
         "native": "網路線無迄個問題", "models": []},
        {"id": "S2-02", "seg": "S2", "t_start": 110.0, "t_end": 116.0, "spk": "S3",
         "lang": "nan", "zh": "那個插座要先送去電鍍",
         "native": "彼个插座愛先送去電鍍", "models": []},
    ]


def _hyp_from_lines(lines, track="zh", shift=0.0, text_map=None):
    """把黃金段本身當成一份完美的 hyp（各條校準閘的起點）。"""
    out = []
    for ln in lines:
        txt = golden.ref_text(ln, track)
        if text_map:
            txt = text_map(ln, txt)
        out.append({"start": ln["t_start"] + shift,
                    "end": ln["t_end"] + shift,
                    "text": txt})
    return out


def _split_three_with_gaps(hyp, gap=0.3):
    """把每顆 cue 切成三份、彼此之間留真空隙 —— 閘 8 的 fixture。

    **刻意不用對半切。** 對半切時每一半剛好佔黃金句 50%，壓在舊規則
    `>= 50%` 的邊界內側，於是舊 bug 拿滿分、閘恆綠（harness M1 實測）。
    切三份 ⇒ 每份約 33%，低於 50%；加空隙 ⇒ 同時模擬「段間有真空隙」，
    那是 Whisper 真實病因的另一半。

    文字按三等分切開，串起來與原文完全相同（覆蓋率不該因切法改變）。
    """
    out = []
    for h in hyp:
        span = h["end"] - h["start"]
        # 三份加兩個空隙要塞得進原本的時間範圍
        g = min(gap, max(0.0, span / 6.0))
        piece = (span - 2 * g) / 3.0
        txt = h["text"]
        k = len(txt) // 3
        cuts = [txt[:k], txt[k:2 * k], txt[2 * k:]]
        t0 = h["start"]
        for i, part in enumerate(cuts):
            out.append({"start": t0, "end": t0 + piece, "text": part})
            t0 += piece + (g if i < 2 else 0.0)
    return out


def _score(hyp, **kw):
    return score.score(_manifest(), _lines(), hyp, **kw)


# ── 五條校準閘 ──────────────────────────────────────────────────────────

class TestCalibrationGates:
    """全綠才准引用任何 CER 數字。"""

    def test_gate1_identity_is_perfect(self):
        """#1 hyp ＝ golden 自己 → 完美分數。抓正規化層的 bug。"""
        rep = _score(_hyp_from_lines(_lines()))
        o = rep["overall"]
        assert o["cer_content"] == 0.0
        assert o["coverage"] == 1.0
        assert o["hallucination"] == 0.0
        assert o["model_recall"] == 1.0
        assert o["model_precision"] == 1.0

    def test_gate2_ten_percent_deletion_lands_near_ten_percent(self):
        """#2 刪 10% 字元 → CER 8–12%。抓距離算錯。"""
        # 計數器跨行累積，且**先正規化再刪** —— 兩者都是為了讓 10% 是對
        # 「整份可比對字元」算的。逐行各自從 0 數會漏掉短於 10 字的句子
        # （實測那樣只得 6.8%）；直接刪原文則會刪到標點或刪到被中文數字
        # 轉換合併掉的字。這條閘的重點正是「比例算得準」，所以刪法要精確。
        seen = [0]

        def drop(_ln, txt):
            kept = []
            for ch in score.normalize(txt):
                seen[0] += 1
                if seen[0] % 10 != 0:
                    kept.append(ch)
            return "".join(kept)
        rep = _score(_hyp_from_lines(_lines(), text_map=drop))
        assert 0.08 <= rep["overall"]["cer_content"] <= 0.12, rep["overall"]

    def test_gate3_empty_hyp_is_total_loss_not_nan(self):
        """#3 空稿 → CER 100%／覆蓋率 0%／幻覺率 0（不是 NaN、不是除零）。"""
        rep = _score([])
        o = rep["overall"]
        assert o["cer_content"] == 1.0
        assert o["coverage"] == 0.0
        # 🔴 2026-09-11 改判：原本這裡斷言 `hallucination == 0.0`，理由寫
        # 「沒有輸出就沒有幻覺」。但**下面兩行的註解就是反對它的理由** ——
        # 分母 0 的比值回 0.0 會被誤讀。幻覺率的誤讀方向更糟：0.0 讀起來是
        # 「這家不幻覺」，等於把「量不到」講成優點，而這份稿的 CER 是 100%。
        # `coverage` 與 `_ratio` 本來就是「分母 0 → None」，只有它例外。
        assert o["hallucination"] is None, "空稿的幻覺率是 n/a，不是 0%"
        assert o["hallucination_judged_segments"] == 0
        # 全空時型號分母也是 0 → 必須是 None（n/a），不可是 0.0（會被讀成「全錯」）
        assert o["model_precision"] is None

    def test_gate4_timestamp_shift_collapses_coverage(self):
        """#4 時間戳平移 +30s → 覆蓋率大幅下降。

        這條在抓一種**跑起來永遠是綠的**缺陷：如果覆蓋率其實只看文字、
        沒真的用時間軸對齊，那平移之後它會照樣回 100%，而我們會拿一個
        跟時間無關的數字去談「覆蓋率」。
        """
        base = _score(_hyp_from_lines(_lines()))["overall"]["coverage"]
        shifted = _score(_hyp_from_lines(_lines(), shift=30.0))["overall"]["coverage"]
        assert base == 1.0
        assert shifted < 0.2, "平移 30 秒還能有 %.2f 覆蓋率＝時間軸沒在用" % shifted

    def test_gate5_swapped_models_score_zero(self):
        """#5 每個型號換成**別的**型號 → precision 與 recall 皆 0。

        在抓正規化過度歸一：如果 `MX-K` 與 `ZR-9` 被壓成同一形，
        亂寫型號會拿到滿分，而型號正名正是這條 pipeline 的核心產出之一。
        """
        swap = {"MX-K": "ZR-9", "二零四D": "QN-S2"}

        def replace(_ln, txt):
            for a, b in swap.items():
                txt = txt.replace(a, b)
            return txt

        # vocab 要含「被換上去的」型號，否則偵測不到 → 抓不到 FP
        vocab = ["MX-K", "204-D", "ZR-9", "QN-S2"]
        rep = _score(_hyp_from_lines(_lines(), text_map=replace), vocab=vocab)
        o = rep["overall"]
        assert o["model_recall"] == 0.0, o
        assert o["model_precision"] == 0.0, o


    def test_gate6_coarse_segmentation_does_not_change_cer(self):
        """#6 把 hyp 併成**一顆巨大 cue** → CER 不可改變。

        🔴 這條是 2026-09-10 補的，補的原因是前五條**沒抓到一個真實缺陷**：
        它們的 hyp 都跟 golden 同一種切法，於是「時間分桶」的錯誤永遠不會現形。
        真實資料一送進來（雅婷有一顆 296.8 秒的 cue）就爆掉 ——
        中點篩會整顆丟掉判 CER 100%、重疊收又把多餘文字算成插入判 187%。

        量尺必須對**引擎的切段風格免疫**，否則量到的是切法不是準確率
        （0716「量到的是書寫格式不是準確率」同型）。
        """
        fine = _hyp_from_lines(_lines())
        # 併成一顆蓋住全部時間、文字順序不變的巨大 cue
        coarse = [{"start": min(h["start"] for h in fine) - 50,
                   "end": max(h["end"] for h in fine) + 50,
                   "text": "無關的開場白" * 20
                           + "".join(h["text"] for h in sorted(fine, key=lambda h: h["start"]))
                           + "無關的結尾" * 20}]
        a = _score(fine)["overall"]["cer_content"]
        b = _score(coarse)["overall"]["cer_content"]
        assert a == 0.0
        assert b == pytest.approx(a, abs=0.02), (
            "併成一顆大 cue 後 CER 從 %.3f 變 %.3f ＝ 量尺在量切段風格" % (a, b))

    def test_gate6b_coarse_cue_makes_hallucination_unjudgeable(self):
        """一顆蓋住全段的 cue **算不出幻覺** → 必須回 None，不可回 0。

        回 0 會被讀成「這家引擎不幻覺」，那是最貴的一種假陰性。

        🔴 2026-09-11 擴充：原本這條只斷言 `by_segment["S2"]`，而**報告引用的是
        `overall`**。用同一個 fixture 跑 overall 得到 0.0 —— 沒有任何斷言看它，
        所以這個閘守的維度與被讀的維度不是同一個。量對維度。
        """
        fine = _hyp_from_lines(_lines())
        coarse = [{"start": 100.0, "end": 160.0,
                   "text": "".join(h["text"] for h in fine)}]
        rep = _score(coarse)
        assert rep["by_segment"]["S2"]["hallucination"] is None
        assert rep["by_segment"]["S2"]["hallucination_blocked_by"] == "cue_too_coarse"
        assert rep["overall"]["hallucination"] is None, (
            "by_segment 判不動，overall 卻給了一個數字 —— 不可判定被折進分母了")


class TestHallucinationDenominator:
    """幻覺率的分母：夾到視窗、判不動的段不入、零長度不進中位。

    2026-09-11 雙軌審計各自抓到同一處（Codex #4/#5、harness B1）。
    共同形狀：**偏差方向固定，且偏向獎勵這個 guard 要抓的那類引擎** ——
    cue 越粗 → 讀數越接近 0 → 越像「不幻覺」。
    """

    def _man(self, *wins):
        return {"set_id": "h", "source_audio": "a.m4a", "source_md5": "e" * 32,
                "duration_sec": 400.0, "speakers": ["標註者"],
                "segments": [{"id": "G%d" % (i + 1), "t_start": a, "t_end": b,
                              "reason": "幻覺率分母"}
                             for i, (a, b) in enumerate(wins)]}

    def _line(self, seg, a, b, zh="真的有人講了這句話"):
        return {"id": "%s-01" % seg, "seg": seg, "t_start": a, "t_end": b,
                "spk": "標註者", "lang": "zho", "zh": zh}

    def test_seconds_are_clipped_to_the_window(self):
        """跨出視窗的 cue 只能貢獻窗內那一段秒數。

        不夾的話一顆 296.8 秒的 cue 會對一個 60 秒視窗貢獻 296.8 秒，
        跨兩窗時還被算兩次 ⇒ 分母被窗外秒數灌大，讀數往 0 壓。
        """
        man = self._man((0.0, 60.0))
        lines = [self._line("G1", 20.0, 30.0)]
        hyp = [{"start": -20.0, "end": 10.0, "text": "憑空生出來的一段字"},
               {"start": 20.0, "end": 30.0, "text": "真的有人講了這句話"}]
        got = score.score(man, lines, hyp)["by_segment"]["G1"]["hallucination"]
        # 窗內：幻覺 10 秒（-20~10 夾成 0~10）、對到 10 秒 ⇒ 10/20
        assert got == pytest.approx(0.5), (
            "未夾窗會得 30/(30+10)=0.75 —— 把窗外的 20 秒算進了分母")

    def test_unjudgeable_segment_stays_out_of_overall(self):
        """一段判不動、一段判得動 ⇒ overall 只能等於判得動那段。"""
        man = self._man((0.0, 60.0), (100.0, 160.0))
        lines = [self._line("G1", 10.0, 15.0, "測試句子"),
                 self._line("G2", 110.0, 115.0, "另外一句話")]
        hyp = [{"start": 0.0, "end": 60.0, "text": "測試句子"},      # G1 判不動
               {"start": 110.0, "end": 115.0, "text": "另外一句話"},
               {"start": 130.0, "end": 140.0, "text": "憑空生出來的一整段"}]
        rep = score.score(man, lines, hyp)
        g2 = rep["by_segment"]["G2"]["hallucination"]
        assert rep["by_segment"]["G1"]["hallucination"] is None
        assert rep["overall"]["hallucination"] == pytest.approx(g2), (
            "判不動的 G1 的秒數進了 overall 分母")
        assert rep["overall"]["hallucination_judged_segments"] == 1
        assert rep["overall"]["hallucination_total_segments"] == 2

    def test_judged_segment_count_is_reported(self):
        """5 段裡只有 1 段判得動的「幻覺率」與 5 段都判得動的不是同一種東西。"""
        man = self._man((0.0, 60.0), (100.0, 160.0))
        lines = [self._line("G1", 10.0, 15.0, "測試句子"),
                 self._line("G2", 110.0, 115.0, "另外一句話")]
        coarse = [{"start": 0.0, "end": 60.0, "text": "測試句子"},
                  {"start": 100.0, "end": 160.0, "text": "另外一句話"}]
        rep = score.score(man, lines, coarse)
        assert rep["overall"]["hallucination"] is None
        assert rep["overall"]["hallucination_judged_segments"] == 0

    def test_zero_length_cues_do_not_lower_median_cue(self):
        """零長度 cue 不代表切得細 —— 混進中位數會讓 too_coarse 恆 False。"""
        man = self._man((0.0, 60.0))
        lines = [self._line("G1", 10.0, 15.0, "測試句子")]
        hyp = [{"start": 0.0, "end": 60.0, "text": "測試句子"},
               {"start": 20.0, "end": 20.0, "text": ""},
               {"start": 30.0, "end": 30.0, "text": ""}]
        seg = score.score(man, lines, hyp)["by_segment"]["G1"]
        assert seg["hyp_median_cue_sec"] == pytest.approx(60.0), (
            "零長度 cue 把中位拉到 0 了")
        assert seg["hallucination"] is None

    def test_estimated_end_makes_hallucination_unjudgeable(self):
        """來源的 end 是我們補的 ⇒ 分子分母都建在自編時間上，回 None。"""
        man = self._man((0.0, 60.0))
        lines = [self._line("G1", 10.0, 15.0, "測試句子")]
        hyp = [{"start": 10.0, "end": 15.0, "text": "測試句子",
                "end_estimated": True},
               {"start": 30.0, "end": 40.0, "text": "憑空生出來的一整段",
                "end_estimated": True}]
        seg = score.score(man, lines, hyp)["by_segment"]["G1"]
        assert seg["hallucination"] is None
        assert seg["hallucination_blocked_by"] == "end_estimated"

    def test_real_end_source_is_still_judgeable(self):
        """反向：沒有 end_estimated 標記的來源不可被誤擋成不可判定。"""
        man = self._man((0.0, 60.0))
        lines = [self._line("G1", 10.0, 15.0, "測試句子")]
        hyp = [{"start": 10.0, "end": 15.0, "text": "測試句子"},
               {"start": 30.0, "end": 40.0, "text": "憑空生出來的一整段"}]
        seg = score.score(man, lines, hyp)["by_segment"]["G1"]
        assert seg["hallucination"] is not None
        assert seg["hallucination_blocked_by"] is None


class TestTaigiDenominatorCountsEachCueOnce:
    """跨兩句台語句的 cue 只能計一次 —— 否則量到的是引擎切段風格。

    2026-09-11 Codex #1。`chunks.extend()` 原本在 per-line 迴圈裡，
    一顆 cue 重疊 N 句就被串進分母 N 次。而「會不會跨句」取決於切段風格
    ⇒ 跨引擎比較時，切得粗的那家分母被放大。
    """

    def _nan(self, i, a, b, zh):
        return {"id": "N-%02d" % i, "seg": "S1", "t_start": a, "t_end": b,
                "spk": "標註者", "lang": "nan", "zh": zh}

    def test_cue_spanning_two_taigi_lines_is_counted_once(self):
        lines = [self._nan(1, 0.0, 2.0, "甲"), self._nan(2, 2.0, 4.0, "乙"),
                 self._nan(3, 5.0, 6.0, "丙")]
        hyp = [{"start": 0.5, "end": 3.5, "text": "迄"},    # 跨 N-01 與 N-02
               {"start": 5.0, "end": 6.0, "text": "佇"}]
        per10k, hits, chars, _ = score.taigi_retention(lines, hyp)
        assert chars == 2, "hyp 的實際文字是『迄佇』2 字，分母卻是 %d" % chars
        assert hits == 2
        assert per10k == pytest.approx(10000.0)

    def test_single_line_cues_unchanged(self):
        """反向：不跨句時行為不可被這次改動動到。"""
        lines = [self._nan(1, 0.0, 2.0, "甲"), self._nan(2, 5.0, 7.0, "乙")]
        hyp = [{"start": 0.5, "end": 1.5, "text": "迄"},
               {"start": 5.5, "end": 6.5, "text": "無關"}]
        _, hits, chars, _ = score.taigi_retention(lines, hyp)
        assert (hits, chars) == (1, 3)

    def test_time_order_is_preserved_after_dedupe(self):
        """去重後仍須依時序串 —— 不可變成 hyp list 的原始順序。"""
        lines = [self._nan(1, 0.0, 9.0, "甲")]
        hyp = [{"start": 5.0, "end": 6.0, "text": "後"},
               {"start": 1.0, "end": 2.0, "text": "前"}]
        _, _, chars, _ = score.taigi_retention(lines, hyp)
        assert chars == 2


class TestRecappEndsAreMarkedEstimated:
    """`parse_recapp_md` 補出來的 end 必須帶標記 —— 而那是**每一顆**不是最後一顆。

    原 docstring 寫「只有最後一句的長度是估的、只影響幻覺率的分母」，兩句都錯。
    實測：兩個字的「短句」因為下一位講者 46 秒後才開口，被補成 46 秒。
    """

    MD = (
        "## 逐字稿\n\n"
        "**說話者 1**（0:10）\n這批貨的庫存還有九個\n\n"
        "**說話者 2**（0:14）\n短句\n\n"
        "**說話者 1**（1:00）\n又一句話\n"
    )

    def test_every_cue_is_marked(self):
        cues = score.parse_recapp_md(self.MD)
        assert len(cues) == 3
        assert all(c.get("end_estimated") for c in cues), (
            "只標了一部分 —— 全部的 end 都是補的")

    def test_silence_gap_inflates_a_two_word_utterance(self):
        """這才是真正的病：補出來的長度是靜默，不是說話時長。"""
        cues = score.parse_recapp_md(self.MD)
        short = [c for c in cues if c["text"] == "短句"][0]
        assert short["end"] - short["start"] == pytest.approx(46.0)

    def test_median_cue_is_inflated_by_the_fabrication(self):
        cues = score.parse_recapp_md(self.MD)
        med = score._median([c["end"] - c["start"] for c in cues])
        assert med > 40.0, (
            "中位 cue 被灌大到 %.1f 秒 —— 這會直接決定 too_coarse 回不回 None" % med)


    def test_gate8_fine_segmentation_does_not_reduce_coverage(self):
        """#8 把 hyp **切更細** → 覆蓋率不可下降。

        🔴 2026-09-10 補。第 6 條守的是「合併成粗 cue」，這條守相反方向。
        真實資料上先炸的是這一邊：Whisper 的 cue 中位 2.0 秒、段間有真空隙，
        而初版覆蓋率要求**單一** cue 蓋住黃金句 ≥50% ⇒ 一句 6 秒被三顆 2 秒蓋住時
        判成沒覆蓋，實測覆蓋率 24.3% 對上同一份稿 26.5% 的 CER —— 自相矛盾。

        **今天同一個錯誤類別出現三次**（CER 的時間分桶、幻覺率的粗 cue、
        覆蓋率的細 cue）⇒ 量尺對切段粒度的不變性要兩個方向都守。
        """
        fine = _hyp_from_lines(_lines())
        base = _score(fine)["overall"]["coverage"]
        finer = _split_three_with_gaps(fine)
        got = _score(finer)["overall"]["coverage"]
        assert base == 1.0
        assert got == pytest.approx(base, abs=0.01), (
            "切更細之後覆蓋率從 %.2f 掉到 %.2f ＝ 量尺在量切段風格" % (base, got))

    def test_gate8_fixture_can_actually_catch_the_old_rule(self):
        """反向驗證：把**舊的 buggy 規則**實作出來，它必須紅。

        🔴 2026-09-11：原本閘 8 的 fixture 把每顆 cue **對半**切，而 cue 與黃金句
        邊界完全對齊 ⇒ 每一半剛好佔該句 3.0/6.0 = **50%**，正好壓在舊規則
        `>= 50%` 的邊界內側 ⇒ **舊 bug 在那個 fixture 上拿滿分 1.000**。
        兩半還無縫相接，而真實病因有一半是「段間有真空隙」。

        ⇒ 閘 8 對它 docstring 指名的那個缺陷是**恆綠**的。
        這違反的是同一批文件自己立的通則：
        「校準閘至少要有一條的 fixture 形狀跟被測對象**刻意不同**，
        否則它驗的是自洽不是正確」（case-study 2026-09-11）。

        現在的 fixture 切三份 ＋ 留 0.3 秒空隙，這條測試證明它抓得到。
        """
        lines = _lines()
        finer = _split_three_with_gaps(_hyp_from_lines(lines))

        def old_buggy_coverage(hyp):
            """初版：要求**單一** cue 覆蓋該黃金句 >= 50%。"""
            cov = gold = 0.0
            for ln in lines:
                dur = ln["t_end"] - ln["t_start"]
                gold += dur
                best = max((score._overlap(h["start"], h["end"],
                                           ln["t_start"], ln["t_end"])
                            for h in hyp), default=0.0)
                if best >= dur * 0.5:
                    cov += dur
            return cov / gold

        assert old_buggy_coverage(finer) < 0.2, (
            "舊規則在這個 fixture 上拿到 %.3f —— fixture 抓不到它要抓的缺陷"
            % old_buggy_coverage(finer))
        assert _score(finer)["overall"]["coverage"] == pytest.approx(1.0)

    def test_gate8_fixture_has_real_gaps(self):
        """fixture 的形狀要跟被測對象刻意不同 —— 這條驗它真的有空隙。

        沒有這條，哪天有人把 `_split_three_with_gaps` 改回無縫切，
        上面兩條還是綠的（舊規則在無縫對半切上拿滿分）。
        """
        finer = _split_three_with_gaps(_hyp_from_lines(_lines()))
        by_start = sorted(finer, key=lambda h: h["start"])
        gaps = [b["start"] - a["end"]
                for a, b in zip(by_start, by_start[1:])
                if b["start"] - a["end"] > 1e-9]
        assert gaps, "fixture 裡沒有任何空隙 —— 真實病因的一半沒被模擬到"
        assert min(gaps) >= 0.25, "空隙只有 %.3f 秒，太小了" % min(gaps)


# ── per-line 誤差歸因 ───────────────────────────────────────────────────

class TestPerLineAttribution:
    """把段層級的對齊結果攤回逐句，給 bootstrap 當統計單位（n=64 >> n=5）。

    🔴 **對齊單位仍然是段，只有統計單位是句。** 直接「每句各自抓自己的 hyp」
    會把閘 6／閘 8 剛擋掉的缺陷放回來 —— 那是又一次時間分桶，
    量到的是引擎的切段風格（雅婷那顆 296.8 秒的 cue 逐句分桶時
    要嘛整顆丟掉、要嘛重複計入每一句）。
    """

    def test_costs_sum_equals_distance(self):
        """**核心不變式**：攤回去的成本總和必須等於段層級的距離。

        不成立就代表歸因在偷加或偷減錯誤，而 bootstrap 會把那個偏差放大。
        """
        for ref, hyp in [("這批庫存還有9個", "這批庫存還有九個"),
                         ("abcdef", "xxabQdefyy"),
                         ("完全不一樣", "毫無關係的內容"),
                         ("一樣的", "一樣的")]:
            d, costs = score.levenshtein_substring_per_ref(ref, hyp)
            assert len(costs) == len(ref)
            assert sum(costs) == d, (ref, hyp, costs, d)

    def test_agrees_with_plain_substring_distance(self):
        """跟既有的 `levenshtein_substring` 必須給同一個距離 —— 兩支不可分歧。"""
        for ref, hyp in [("這批庫存還有9個", "呃這批庫存還有九個對"),
                         ("網路線沒有這個問題", "網路線無迄個問題"),
                         ("", "隨便"), ("隨便", "")]:
            assert (score.levenshtein_substring_per_ref(ref, hyp)[0]
                    == score.levenshtein_substring(ref, hyp))

    def test_identity_costs_all_zero(self):
        d, costs = score.levenshtein_substring_per_ref("完全相同", "前綴完全相同後綴")
        assert d == 0 and costs == [0, 0, 0, 0]

    def test_error_lands_on_the_wrong_character(self):
        """錯誤要攤在**出錯的那個位置**，不是平均分給整句。"""
        d, costs = score.levenshtein_substring_per_ref("ABCDE", "ABXDE")
        assert d == 1 and costs == [0, 0, 1, 0, 0]

    def test_per_line_rows_cover_every_golden_line(self):
        rep = _score(_hyp_from_lines(_lines()))
        assert len(rep["per_line"]) == len(_lines())
        assert {r["id"] for r in rep["per_line"]} == {l["id"] for l in _lines()}
        for r in rep["per_line"]:
            assert r["dist"] == 0, "hyp ＝ golden 時每句都該是 0"

    def test_per_line_dist_sums_to_segment_distance(self):
        """逐句加總 ÷ 逐句 ref 長度加總，必須還原逐段 CER。"""
        hyp = _hyp_from_lines(_lines(), text_map=lambda _l, t: t.replace("九", "9"))
        rep = score.score(_manifest(), _lines(), hyp)
        for seg in _manifest()["segments"]:
            rows = [r for r in rep["per_line"] if r["seg"] == seg["id"]]
            ref_len = sum(r["ref_len"] for r in rows)
            dist = sum(r["dist"] for r in rows)
            if ref_len:
                assert (dist / float(ref_len)) == pytest.approx(
                    rep["by_segment"][seg["id"]]["cer_content"], abs=1e-9)

    def test_per_line_carries_lang_for_taigi_subsetting(self):
        """bootstrap 要能只抽台語句，所以 lang 必須帶著走。"""
        rep = _score(_hyp_from_lines(_lines()))
        langs = {r["lang"] for r in rep["per_line"]}
        assert "nan" in langs and "zho" in langs


# ── 視窗選取只有一份定義 ────────────────────────────────────────────────

class TestWindowSelectionIsSingleSource:
    """`window_hyp` 是視窗選取的唯一定義 —— 外部分析不可自己手寫一份。

    2026-09-11 的實據：一支臨時寫的雜訊地板分析自己複製了一份選取邏輯，
    抄到的是**已被修掉的中點篩**，於是同一份資料算出兩組不同的 CER
    （雅婷 G3：29.8% vs 100.0%），差點被當成真結果報出去。
    錯的是那份副本，但該消掉的是「能有第二份實作」這件事。
    """

    def test_score_uses_window_hyp(self):
        """`score()` 取的 hyp 段數必須與 `window_hyp` 一致。

        哪天有人把 `score()` 內部改回自己篩，這條會紅。
        """
        hyp = _hyp_from_lines(_lines())
        rep = _score(hyp)
        for seg in _manifest()["segments"]:
            want = len(score.window_hyp(hyp, seg["t_start"], seg["t_end"]))
            assert rep["by_segment"][seg["id"]]["hyp_segments"] == want

    def test_long_cue_spanning_window_is_kept(self):
        """跨窗的長 cue 必須收進來 —— 中點篩會整顆丟掉。

        雅婷實測有一顆 296.8 秒的 cue 蓋住 G3，中點落在窗外。
        """
        long_cue = [{"start": 40.0, "end": 400.0, "text": "很長的一段"}]
        got = score.window_hyp(long_cue, 100.0, 160.0)
        assert len(got) == 1, "跨窗長 cue 被丟掉了（中點篩的病）"

    def test_no_overlap_is_excluded(self):
        assert score.window_hyp([{"start": 0.0, "end": 10.0, "text": "x"}],
                                100.0, 160.0) == []

    def test_window_hyp_text_matches_manual_join(self):
        hyp = _hyp_from_lines(_lines())
        txt = score.window_hyp_text(hyp, 100.0, 160.0)
        manual = "".join(score.normalize(h["text"])
                         for h in score.window_hyp(hyp, 100.0, 160.0))
        assert txt == manual


class TestTouchesIsSingleSource:
    """`_touches()` 是「cue 碰到區間」的唯一定義 —— 四個消費端必須一致。

    2026-09-11 審計實據：這個判定原本有四份實作，只有 `window_hyp` 帶了
    零長度 cue 的特判。同一顆零長度 cue 的命運因此分岔成四種：

      | 消費端                | 舊行為                          |
      |-----------------------|---------------------------------|
      | `window_hyp` / CER    | 收（文字進了 hyp_txt）          |
      | `coverage`            | 丟                              |
      | `taigi_retention`     | 丟                              |
      | `hallucination_spans` | **一律判成幻覺**（touched 恆 False）|

    最後一條最貴：它落在黃金句正中間、文字完全對，卻被記成幻覺，
    而 `adjudicate.py` 吃的就是它。

    零長度 cue 來自只有秒精度的時間戳（實測某來源 1268 段裡 92 段），
    所以偏差方向固定 —— 只罰時間戳精度差的那一家。
    """

    # 一顆零長度 cue，文字與黃金句完全相同，位置落在句子正中間。
    LINE = {"id": "Z-01", "seg": "S1", "t_start": 10.0, "t_end": 12.0,
            "spk": "標註者", "lang": "zho", "zh": "這批貨的庫存還有九個"}
    CUE = {"start": 11.0, "end": 11.0, "text": "這批貨的庫存還有九個"}

    def _man(self):
        return {"set_id": "zerolen", "source_audio": "a.m4a",
                "source_md5": "d" * 32, "duration_sec": 100.0,
                "speakers": ["標註者"],
                "segments": [{"id": "S1", "t_start": 0.0, "t_end": 60.0,
                              "reason": "零長度 cue 的邊角"}]}

    def test_window_hyp_keeps_zero_length_cue(self):
        got = score.window_hyp([self.CUE], 0.0, 60.0)
        assert len(got) == 1, "零長度 cue 被丟掉了"

    def test_coverage_counts_zero_length_cue(self):
        """舊行為：coverage 自己用 `_overlap > 0` ⇒ 0.0（明明字對了）。"""
        rep = score.score(self._man(), [self.LINE], [self.CUE])
        assert rep["overall"]["coverage"] == 1.0

    def test_taigi_retention_sees_zero_length_cue(self):
        nan_line = dict(self.LINE, lang="nan", zh="迄爾")
        _, _, chars, _ = score.taigi_retention([nan_line], [self.CUE])
        assert chars > 0, "台語保留率的分母把零長度 cue 丟掉了"

    def test_zero_length_cue_is_not_a_hallucination(self):
        """最貴的一條：字完全對、落在句子正中間，不可被記成幻覺。"""
        spans = score.hallucination_spans([self.LINE], [self.CUE], 0.0, 60.0)
        assert spans == [], "對到的 cue 被判成幻覺（adjudicate 會吃到假候選）"

    def test_all_four_consumers_agree(self):
        """四個消費端對同一顆 cue 的「碰到了嗎」必須給同一個答案。"""
        cue, line = self.CUE, self.LINE
        a, b = line["t_start"], line["t_end"]
        verdicts = {
            "window_hyp": len(score.window_hyp([cue], a, b)) == 1,
            "coverage": score.score(self._man(), [line],
                                    [cue])["overall"]["coverage"] == 1.0,
            "taigi": score.taigi_retention(
                [dict(line, lang="nan")], [cue])[2] > 0,
            "halluc": score.hallucination_spans([line], [cue], 0.0, 60.0) == [],
        }
        assert len(set(verdicts.values())) == 1, verdicts

    def test_old_overlap_rule_would_disagree(self):
        """反向驗證：把舊規則實作出來，它必須給出**不同**的答案。

        沒有這條，上面那些斷言可能只是碰巧成立（fixture 與被測對象同形）。
        """
        cue, line = self.CUE, self.LINE

        def old_rule(h, x, y):                       # 修好之前的寫法
            return score._overlap(h["start"], h["end"], x, y) > 0

        assert old_rule(cue, line["t_start"], line["t_end"]) is False
        assert score._touches(cue, line["t_start"], line["t_end"]) is True

    def test_non_zero_length_cues_unchanged(self):
        """正常 cue 的判定不可被這次改動動到。"""
        normal = {"start": 10.0, "end": 12.0, "text": "x"}
        assert score._touches(normal, 11.0, 20.0) is True
        assert score._touches(normal, 12.0, 20.0) is False   # 端點相接不算碰到
        assert score._touches(normal, 0.0, 9.0) is False

    def test_zero_length_cue_outside_window_is_excluded(self):
        outside = {"start": 90.0, "end": 90.0, "text": "x"}
        assert score._touches(outside, 0.0, 60.0) is False


# ── 正規化 ──────────────────────────────────────────────────────────────

class TestNormalize:
    def test_strips_punctuation_and_space(self):
        assert score.normalize("好，那就這樣。 OK？") == "好那就這樣OK"

    def test_fullwidth_becomes_halfwidth(self):
        assert score.normalize("ＭＸ－Ｋ") == score.normalize("MX-K") == "MXK"

    def test_case_folded_up(self):
        assert score.normalize("mx-k xcf") == "MXKXCF"

    def test_cjk_digits_two_readings_agree(self):
        """`二零四` 與 `二百零四` 都要成 204 —— 0716 栽在書寫慣例上。"""
        assert score.normalize("二零四") == "204"
        assert score.normalize("二百零四") == "204"
        assert score.normalize("204") == "204"

    @pytest.mark.parametrize("src,want", [
        ("六十", "60"),
        ("十", "10"),
        ("二十四", "24"),
        ("一萬", "10000"),
        ("兩萬", "20000"),          # 兩 沒收進數字表時這裡會得「兩0」
        ("一萬零八", "10008"),
        ("三十萬五千", "305000"),
    ])
    def test_cjk_units(self, src, want):
        assert score.normalize(src) == want

    def test_known_side_effect_of_taking_liang(self):
        """收了 `兩` 的代價：`兩邊` 會變 `2邊`。

        刻意鎖住這個行為而不是假裝沒有 —— 兩邊都過同一支正規化，CER 不受影響，
        而收了才接得住「ASR 寫 2 萬、人打兩萬」。哪天有人想改回去，這條會提醒他代價在哪。
        """
        assert score.normalize("兩邊") == "2邊"
        assert score.normalize("零件") == "0件"

    def test_idempotent(self):
        once = score.normalize("這批ＭＸ－Ｋ，庫存二零四個。")
        assert score.normalize(once) == once

    def test_fillers_only_dropped_when_asked(self):
        assert score.normalize("呃這個嗯對") == "呃這個嗯對"
        assert score.normalize("呃這個嗯對", drop_fillers=True) == "這個對"

    def test_sentence_particles_are_not_fillers(self):
        """啊／喔／啦／嘛 承載語意，去掉會改內容 → 刻意不列入 FILLERS。"""
        for ch in ("啊", "喔", "啦", "嘛", "吧"):
            assert ch not in score.FILLERS


class TestCer:
    def test_empty_ref_empty_hyp_is_zero(self):
        assert score.cer("", "") == 0.0

    def test_empty_ref_nonempty_hyp_is_one(self):
        assert score.cer("", "亂講") == 1.0

    def test_known_distance(self):
        assert score.levenshtein("abc", "abd") == 1
        assert score.cer("abcd", "abd") == 0.25


class TestFindModels:
    def test_longest_first_masking(self):
        """`204` 不可匹配進 `204D` 裡面 —— 否則命中數是假的。"""
        vocab = [score.normalize_model(v) for v in ("204", "204-D")]
        found = score.find_models(score.normalize("這批204-D要出貨"), vocab)
        assert found == {"204D": 1}, found

    def test_counts_repeats(self):
        vocab = [score.normalize_model("MX-K")]
        found = score.find_models(score.normalize("MX-K 跟 MX-K 都要"), vocab)
        assert found == {"MXK": 2}

    def test_writing_convention_variants_all_hit(self):
        vocab = [score.normalize_model("204-D")]
        for variant in ("204-D", "204 D", "二零四D", "２０４－Ｄ"):
            found = score.find_models(score.normalize(variant), vocab)
            assert found == {"204D": 1}, variant


# ── 台語保留率（不需參考答案）────────────────────────────────────────────

class TestTaigiRetention:
    """人只標「這句是台語」，忠實度靠引擎輸出裡有沒有台語專屬字形來判。

    這是 2026-09-10 取代 `taigi_flatten_gap` 當主指標的東西 —— 因為 標註者
    打不出台語漢字，而由 CC 回譯補 native ＝ 自己製造 ground truth。
    """

    def _hyp(self, text):
        # 蓋住 S2 兩句台語的時間範圍
        return [{"start": 102.0, "end": 116.0, "text": text}]

    def test_no_taigi_lines_returns_none_not_zero(self):
        """沒有台語句 → None（無訊號），**不可回 0**（那會被讀成「引擎抹平了」）。"""
        zho_only = [l for l in _lines() if l["lang"] == "zho"]
        per10k, hits, chars, sec = score.taigi_retention(zho_only, self._hyp("隨便"))
        assert per10k is None and sec == 0.0

    def test_taigi_sec_counts_marked_seconds(self):
        _, _, _, sec = score.taigi_retention(_lines(), self._hyp("x"))
        assert sec == pytest.approx(12.0)      # 兩句各 6 秒

    def test_flattening_engine_scores_zero(self):
        """引擎把台語寫成國語 → 命中 0。"""
        per10k, hits, _, _ = score.taigi_retention(
            _lines(), self._hyp("網路線沒有這個問題那個插座要先送去電鍍"))
        assert hits == 0 and per10k == 0.0

    def test_faithful_engine_scores_above_zero(self):
        per10k, hits, _, _ = score.taigi_retention(
            _lines(), self._hyp("網路線無迄個問題彼个插座愛先送去電鍍"))
        assert hits >= 2 and per10k > 0

    def test_shared_chars_do_not_count(self):
        """較／講／嘛／無／食 是國台通用 —— 命中上百次也不構成證據。"""
        per10k, hits, _, _ = score.taigi_retention(
            _lines(), self._hyp("他講較無食嘛" * 10))
        assert hits == 0, "通用字被當成台語命中了"

    def test_simplified_ge_does_not_count(self):
        """`个` 是簡體「個」。Whisper 過 opencc 之前吐簡體 —— 收它會整份假陽性。"""
        per10k, hits, _, _ = score.taigi_retention(
            _lines(), self._hyp("这个那个每个" * 10))
        assert hits == 0, "簡體『个』被當成台語命中了"

    def test_engine_silent_on_taigi_lines_returns_none(self):
        """台語那幾秒引擎完全沒輸出 → None（分母 0），不是 0.0。"""
        per10k, _, _, sec = score.taigi_retention(
            _lines(), [{"start": 5.0, "end": 8.0, "text": "別段的話"}])
        assert per10k is None and sec == pytest.approx(12.0)


# ── 台語抹平指數（次要，只在有人填 native 時才有值）──────────────────────

class TestTaigiFlattenGap:
    def test_gap_is_none_when_nobody_filled_native(self):
        """native 是選填 —— 沒填就回 None，不可回一個假的 0 gap。"""
        no_nat = [dict(l) for l in _lines()]
        for l in no_nat:
            l.pop("native", None)
        rep = score.score(_manifest(), no_nat, _hyp_from_lines(no_nat))
        assert rep["overall"]["cer_native"] is None
        assert rep["overall"]["taigi_flatten_gap"] is None

    def test_faithful_engine_has_small_gap(self):
        """照實寫台語原形的引擎：兩軌 CER 接近。"""
        rep = _score(_hyp_from_lines(_lines(), track="native"))
        s2 = rep["by_segment"]["S2"]
        assert s2["cer_native"] == 0.0
        assert s2["taigi_flatten_gap"] <= 0.0

    def test_flattening_engine_shows_positive_gap(self):
        """把台語抹平成國語的引擎：內容 CER 好、台語 CER 差 → gap 明顯為正。

        這是 07-08 bench「Qwen 忠實、Whisper 抹平」那個質性判讀的量化版。
        """
        rep = _score(_hyp_from_lines(_lines(), track="zh"))
        s2 = rep["by_segment"]["S2"]
        assert s2["cer_content"] == 0.0
        assert s2["cer_native"] > 0.2
        assert s2["taigi_flatten_gap"] > 0.2, s2


# ── 幻覺率 ──────────────────────────────────────────────────────────────

class TestHallucination:
    def test_output_in_silence_counts_as_hallucination(self):
        """黃金句之間的空檔有輸出 → 計入幻覺。這是 guard 真正的 KPI。"""
        hyp = _hyp_from_lines(_lines())
        hyp.append({"start": 30.0, "end": 40.0, "text": "謝謝觀看下次再見"})
        rep = _score(hyp)
        assert rep["overall"]["hallucination"] > 0
        assert rep["by_segment"]["S1"]["hallucination"] > 0
        assert rep["by_segment"]["S2"]["hallucination"] == 0

    def test_outside_windows_is_ignored(self):
        """視窗外的輸出既不算幻覺也不算命中 —— 黃金段只覆蓋那 5×60 秒。"""
        hyp = _hyp_from_lines(_lines())
        hyp.append({"start": 400.0, "end": 410.0, "text": "視窗外的東西"})
        rep = _score(hyp)
        assert rep["overall"]["hallucination"] == 0.0


# ── schema 驗證器 ───────────────────────────────────────────────────────

class TestSchemaValidator:
    def test_clean_fixture_passes(self):
        assert golden.validate_manifest(_manifest()) == []
        assert golden.validate_lines(_lines(), _manifest()) == []

    @pytest.mark.parametrize("mutate,needle", [
        (lambda ls: ls[0].pop("zh"), "zh"),
        (lambda ls: ls[0].update(lang="taigi"), "lang"),
        (lambda ls: ls[0].update(t_end=ls[0]["t_start"]), "t_end"),
        (lambda ls: ls[1].update(id=ls[0]["id"]), "重複"),
        (lambda ls: ls[2].update(native=ls[2]["zh"]), "完全相同"),
        (lambda ls: ls[0].update(spk="Vincent"), "spk"),
        (lambda ls: ls[0].update(seg="S9"), "S9"),
        (lambda ls: ls[0].update(models="MX-K"), "models"),
        (lambda ls: ls[0].update(t_start=500.0, t_end=520.0), "超出"),
    ])
    def test_each_mutation_is_caught(self, mutate, needle):
        """每一種壞法都要被指名抓到 —— 驗證器不可只是「有跑」。"""
        lines = _lines()
        mutate(lines)
        errs = golden.validate_lines(lines, _manifest())
        assert errs, "這個突變沒被抓到：%s" % needle
        assert any(needle in e for e in errs), errs

    def test_manifest_requires_reason(self):
        """選段依據沒寫下來 ＝ 選段偏誤無從複查（plan §1.2）。"""
        m = _manifest()
        m["segments"][0].pop("reason")
        errs = golden.validate_manifest(m)
        assert any("reason" in e for e in errs), errs

    def test_overlapping_lines_are_legal(self):
        """S3 交疊段的重疊是要量的東西，不是資料錯誤。

        這是一條**反向**測試：如果哪天有人「順手」加了不許重疊的檢查，
        它會紅，而不是等到聽打完 S3 才發現整段被判死。
        """
        lines = _lines()
        lines.append({"id": "S2-03", "seg": "S2", "t_start": 104.0, "t_end": 110.0,
                      "spk": "S2", "lang": "nan", "zh": "對啦就是那個",
                      "native": "對啦就是彼个", "models": []})
        assert golden.validate_lines(lines, _manifest()) == []


class TestVerbosityExposesOverGeneration:
    """CER 對視窗內的多餘輸出免費 —— 所以旁邊必須有膨脹率（harness M2）。

    那不是 bug 而是刻意的取捨（`levenshtein_substring` 自由端點，
    多餘輸出交給 `hallucination` 另計）。問題在於分工的另一半會被
    `too_coarse` 拿掉：一顆大 cue 同時拿到「多餘文字零成本」與「幻覺率 n/a」。

    實測：1618 字垃圾包住 18 字正確答案 → CER 0.0000、覆蓋率 100%、幻覺率 n/a。
    """

    def _man(self):
        return {"set_id": "v", "source_audio": "a.m4a", "source_md5": "f" * 32,
                "duration_sec": 200.0, "speakers": ["標註者"],
                "segments": [{"id": "G1", "t_start": 0.0, "t_end": 60.0,
                              "reason": "膨脹率"}]}

    REF = "這批貨的庫存還有九個而開模下週才會好"

    def _lines(self):
        return [{"id": "G1-01", "seg": "G1", "t_start": 10.0, "t_end": 20.0,
                 "spk": "標註者", "lang": "zho", "zh": self.REF}]

    def test_clean_transcript_has_verbosity_near_one(self):
        hyp = [{"start": 10.0, "end": 20.0, "text": self.REF}]
        o = score.score(self._man(), self._lines(), hyp)["overall"]
        assert o["cer_content"] == 0.0
        assert o["verbosity"] == pytest.approx(1.0)

    def test_junk_wrapped_answer_is_visible_in_verbosity(self):
        """CER 仍是 0（設計如此），但膨脹率必須把它揭出來。"""
        junk = "胡言亂語" * 200
        hyp = [{"start": 0.0, "end": 60.0, "text": junk + self.REF + junk}]
        o = score.score(self._man(), self._lines(), hyp)["overall"]
        assert o["cer_content"] == 0.0, "這條測試的前提是 CER 不動"
        assert o["verbosity"] > 50, (
            "膨脹率只有 %.2f —— 讀者看不出這份 CER 不可信" % o["verbosity"])

    def test_verbosity_is_none_without_reference(self):
        o = score.score(self._man(), [], [])["overall"]
        assert o["verbosity"] is None


class TestEmptyReferenceSegmentDoesNotInflateOverall:
    """無黃金句的段不可往 overall CER 的分子偷加。

    原本 `d_zh = (0 if not hyp_txt else 1)` 加的是**絕對距離** 1，
    而 `tot["ref_zh"]` 加 0 ⇒ 分子有、分母沒有。
    逐段 `cer()` 回的是**比值** 1.0 —— 兩種幣別。
    """

    def _man(self):
        return {"set_id": "e", "source_audio": "a.m4a", "source_md5": "a" * 32,
                "duration_sec": 200.0, "speakers": ["標註者"],
                "segments": [{"id": "G1", "t_start": 0.0, "t_end": 60.0,
                              "reason": "有黃金句"},
                             {"id": "G2", "t_start": 100.0, "t_end": 160.0,
                              "reason": "沒有黃金句"}]}

    def test_overall_stays_zero_when_the_graded_segment_is_perfect(self):
        ref = "這批貨的庫存還有九個"
        lines = [{"id": "G1-01", "seg": "G1", "t_start": 10.0, "t_end": 20.0,
                  "spk": "標註者", "lang": "zho", "zh": ref}]
        hyp = [{"start": 10.0, "end": 20.0, "text": ref},
               {"start": 110.0, "end": 115.0, "text": "沒有黃金句的段卻有輸出"}]
        o = score.score(self._man(), lines, hyp)["overall"]
        assert o["cer_content"] == 0.0, (
            "無黃金句的段把 overall 墊高到 %.4f" % o["cer_content"])
        assert o["ref_empty_segments"] == 1, "沒報出「有幾段沒有黃金句」"

    def test_per_segment_still_reports_one_point_zero(self):
        """逐段的 1.0 是對的（比值），不要因為這次改動連它一起改掉。"""
        lines = [{"id": "G1-01", "seg": "G1", "t_start": 10.0, "t_end": 20.0,
                  "spk": "標註者", "lang": "zho", "zh": "有句子"}]
        hyp = [{"start": 110.0, "end": 115.0, "text": "只有 G2 有輸出"}]
        rep = score.score(self._man(), lines, hyp)
        assert rep["by_segment"]["G2"]["cer_content"] == 1.0


class TestFindModelsIsDeterministic:
    """同長度型號的 tie 不可由 str hash 決定（每個 process 隨機）。

    原本 `sorted(set(...), key=len, reverse=True)` 對等長項不是全序
    ⇒ 兩個等長且在稿面重疊的型號，同一份輸入會給不同答案
    （實測 PYTHONHASHSEED 1–3 與 4–5 結果不同）。
    一支決定要不要換引擎的量尺不可以這樣。
    """

    def test_equal_length_tie_is_broken_by_lexical_order(self):
        vocab = [score.normalize_model(x) for x in ("TXDN", "MXK")]
        got = score.find_models(score.normalize("這批MXKN要出貨"), vocab)
        # 等長 ⇒ 按字典序，MXK 在 TXDN 之前
        assert got == {"MXK": 1}, got

    def test_repeated_calls_agree(self):
        vocab = [score.normalize_model(x) for x in ("TXDN", "MXK")]
        txt = score.normalize("這批MXKN要出貨")
        first = score.find_models(txt, vocab)
        for _ in range(20):
            assert score.find_models(txt, vocab) == first

    def test_longest_first_still_wins(self):
        """反向：長度優先不可被字典序蓋掉。"""
        vocab = [score.normalize_model(x) for x in ("204", "204-D")]
        got = score.find_models(score.normalize("這批204-D要出貨"), vocab)
        assert got == {"204D": 1}, got


class TestSchemaRejectsNonFiniteAndMalformed:
    """2026-09-11 審計（Codex #8/#9）：六種壞資料原本一個錯都不報。

    最貴的是 NaN —— `isinstance(NaN, float)` 是 True，而 NaN 的**所有**比較
    都回 False ⇒ `te <= ts` 不成立、`ts < lo - 0.5` 不成立
    ⇒ 兩個 NaN 時間戳完全通過，然後在 `score.py` 把 CER 算成 NaN。
    **驗證器放過的東西，下游沒有第二道防線。**
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"),
                                     float("-inf")], ids=["nan", "inf", "-inf"])
    def test_non_finite_timestamps_are_rejected(self, bad):
        lines = _lines()
        lines[0].update(t_start=bad, t_end=bad)
        errs = golden.validate_lines(lines, _manifest())
        assert errs, "%r 通過了驗證" % bad

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")],
                             ids=["nan", "inf"])
    def test_non_finite_segment_bounds_are_rejected(self, bad):
        m = _manifest()
        m["segments"][0]["t_end"] = bad
        errs = golden.validate_manifest(m)
        assert errs, "%r 通過了驗證" % bad

    def test_is_num_rejects_non_finite(self):
        """直接驗判準本身 —— 上面兩條靠它。"""
        assert golden._is_num(1.5) is True
        assert golden._is_num(0) is True
        assert golden._is_num(True) is False          # bool 是 int 的子類
        assert golden._is_num(float("nan")) is False
        assert golden._is_num(float("inf")) is False
        assert golden._is_num("1.5") is False

    def test_md5_must_look_like_md5(self):
        """`source_md5: "x"` 原本也過 —— 而它是這份黃金段的唯一 provenance。"""
        m = _manifest()
        m["source_md5"] = "x"
        errs = golden.validate_manifest(m)
        assert any("md5" in e for e in errs), errs

    def test_segment_must_be_inside_the_recording(self):
        """原本完全沒比對過 duration_sec ⇒ 60 秒的錄音可以有 [-10, 100] 的段。"""
        m = _manifest()
        m["duration_sec"] = 60.0
        m["segments"] = [{"id": "S1", "t_start": -10.0, "t_end": 100.0,
                          "reason": "測試用"}]
        errs = golden.validate_manifest(m)
        assert any("超出音檔範圍" in e for e in errs), errs

    def test_duration_sec_is_required(self):
        m = _manifest()
        m.pop("duration_sec")
        errs = golden.validate_manifest(m)
        assert any("duration_sec" in e for e in errs), errs

    def test_speakers_must_be_a_list(self):
        """名冊型別不對會讓講者檢查靜默失效 —— 字串的 `in` 是子字串比對。"""
        m = _manifest()
        m["speakers"] = "標註者"
        errs = golden.validate_manifest(m)
        assert any("speakers" in e for e in errs), errs

    def test_speakers_items_must_be_non_empty_strings(self):
        m = _manifest()
        m["speakers"] = ["標註者", ""]
        errs = golden.validate_manifest(m)
        assert any("speakers" in e for e in errs), errs

    def test_reason_must_be_a_string_not_just_truthy(self):
        """`reason: 1` 原本也過，而數字 1 不是選段依據。"""
        m = _manifest()
        m["segments"][0]["reason"] = 1
        errs = golden.validate_manifest(m)
        assert any("reason" in e for e in errs), errs

    def test_empty_line_id_is_rejected(self):
        """原本 `if ln.get("id"):` ⇒ 空 id 既不報錯也不進重複檢查。

        於是任意多行可以共用「空 id」而驗證全綠 —— 而 id 是出錯時唯一的定位手段。
        """
        lines = _lines()
        lines[0]["id"] = ""
        errs = golden.validate_lines(lines, _manifest())
        assert any("id" in e for e in errs), errs

    def test_multiple_empty_ids_are_each_reported(self):
        lines = _lines()
        lines[0]["id"] = ""
        lines[1]["id"] = ""
        errs = golden.validate_lines(lines, _manifest())
        assert len([e for e in errs if "id 必須是非空字串" in e]) == 2, errs

    def test_clean_manifest_with_speakers_still_passes(self):
        """反向：合格的 manifest（含 speakers 名冊）不可被這批收緊誤擋。"""
        m = _manifest()
        m["speakers"] = ["標註者", "Frank"]
        assert golden.validate_manifest(m) == []


# ── 載入與 IO ───────────────────────────────────────────────────────────

class TestLoad:
    def test_load_golden_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(_manifest(), fh, ensure_ascii=False)
            with open(os.path.join(d, "lines.jsonl"), "w", encoding="utf-8") as fh:
                for ln in _lines():
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            m, ls = golden.load_golden(d)
            assert m["set_id"] == "T-test"
            assert len(ls) == 4

    def test_load_golden_raises_on_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            bad = _lines()
            bad[2]["native"] = bad[2]["zh"]      # 兩欄同字 = 不構成第二軌
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(_manifest(), fh, ensure_ascii=False)
            with open(os.path.join(d, "lines.jsonl"), "w", encoding="utf-8") as fh:
                for ln in bad:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            with pytest.raises(golden.GoldenError) as exc:
                golden.load_golden(d)
            assert "完全相同" in str(exc.value)

    def test_parse_srt(self):
        raw = ("1\n00:00:02,000 --> 00:00:08,000\n這批MX-K的庫存還有九個\n\n"
               "2\n00:00:10,500 --> 00:00:16,000\n二零四D的開模下週才會好\n")
        segs = score.parse_srt(raw)
        assert len(segs) == 2
        assert segs[0]["start"] == 2.0
        assert segs[1]["end"] == 16.0
        assert "MX-K" in segs[0]["text"]

    def test_srt_hyp_scores_same_as_json(self):
        """雲端競品多半只給 SRT → 兩種入口必須量出同一個分數。"""
        raw_lines = []
        for i, ln in enumerate(_lines(), 1):
            raw_lines.append("%d\n%s --> %s\n%s\n"
                             % (i, _srt_ts(ln["t_start"]), _srt_ts(ln["t_end"]),
                                ln["zh"]))
        via_srt = _score(score.parse_srt("\n".join(raw_lines)))
        via_json = _score(_hyp_from_lines(_lines()))
        assert via_srt["overall"]["cer_content"] == via_json["overall"]["cer_content"]
        assert via_srt["overall"]["coverage"] == via_json["overall"]["coverage"]


def _srt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


# ── report 形狀 ─────────────────────────────────────────────────────────

class TestReportShape:
    def test_rtf_only_when_asr_sec_given(self):
        assert "rtf" not in _score(_hyp_from_lines(_lines()))["overall"]
        rep = _score(_hyp_from_lines(_lines()), asr_sec=60.0)
        assert rep["overall"]["rtf"] == pytest.approx(0.1)

    def test_every_segment_reports_its_reason(self):
        """選段依據要一路帶到報表 —— 看數字的人要看得到「為什麼是這 60 秒」。"""
        rep = _score(_hyp_from_lines(_lines()))
        for sid in ("S1", "S2"):
            assert rep["by_segment"][sid]["reason"]
