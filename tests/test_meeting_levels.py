# -*- coding: utf-8 -*-
"""`scan_levels.py` 純函式層的護欄測試。

## 為什麼這個檔存在

這支的輸出會被用來**選黃金段**，而 plan §1.2 寫得很白：
**選段偏誤沒有事後補救的辦法**。所以它算錯不會有人發現 —— 錯的窗被選中、
聽打完 5 分鐘、才發現量尺架在錯的地方。

三件最容易錯而且錯了看不出來的事，各鎖一條：

1. **dB 平均要在線性功率域做。** 直接平均 dB 值在數學上是錯的，
   而且錯得「看起來很合理」（數字仍在合理範圍內）。
2. **`-inf`（全靜音格）要被吃掉不能炸。** astats 對純靜音就是吐 `-inf`，
   直接 `float()` 之後拿去算術會噴 inf 汙染整份統計。
3. **ffmpeg 失敗要 raise 不能回空。** 「這檔沒聲音」跟「ffmpeg 缺 decoder」
   長得完全一樣 —— 2026-09-10 在 NAS 上真的踩到（NAS ffmpeg 無 AAC decoder）。
   這是 `chunking.py` 檔頭記的同一個「靜默降級」形狀。

被測模組只依賴 stdlib + ffmpeg subprocess，純函式層永遠會跑。

跑法：`python -m pytest tests/acoustic/test_meeting_levels.py -q`
"""
import math
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
import scan_levels as sl  # noqa: E402


def _meta_line(rms, peak=None):
    out = "frame:0 pts:0 pts_time:0\nlavfi.astats.Overall.RMS_level=%s\n" % rms
    if peak is not None:
        out += "lavfi.astats.Overall.Peak_level=%s\n" % peak
    return out


class TestParseAstats:
    def test_parses_rms_and_peak(self):
        text = _meta_line("-31.5", "-12.0") + _meta_line("-40.25", "-20.5")
        assert sl.parse_astats(text) == [(-31.5, -12.0), (-40.25, -20.5)]

    def test_peak_falls_back_to_rms_when_absent(self):
        """有些 ffmpeg build 不吐 Peak_level —— 不可因此少一半資料。"""
        text = _meta_line("-31.5") + _meta_line("-40.25")
        assert sl.parse_astats(text) == [(-31.5, -31.5), (-40.25, -40.25)]

    def test_negative_infinity_is_clamped_not_crashed(self):
        """全靜音格 astats 吐 -inf；直接 float() 會汙染後面所有算術。"""
        bins = sl.parse_astats(_meta_line("-inf") + _meta_line("-20.0"))
        assert bins[0][0] == -120.0
        assert all(math.isfinite(v) for pair in bins for v in pair)

    def test_empty_output_raises(self):
        """不可靜默回空 —— 空的會被讀成「這檔沒聲音」。"""
        with pytest.raises(sl.ScanError):
            sl.parse_astats("frame:0 pts:0 pts_time:0\n")


class TestPercentile:
    def test_known_values(self):
        assert sl.percentile([1, 2, 3, 4, 5], 50) == 3
        assert sl.percentile([1, 2, 3, 4, 5], 0) == 1
        assert sl.percentile([1, 2, 3, 4, 5], 100) == 5

    def test_interpolates(self):
        assert sl.percentile([0, 10], 25) == pytest.approx(2.5)

    def test_single_and_empty(self):
        assert sl.percentile([7], 90) == 7
        assert sl.percentile([], 50) is None


class TestDbMeanIsPowerDomain:
    def test_mean_of_equal_values_is_that_value(self):
        assert sl._mean_db([-20.0, -20.0, -20.0]) == pytest.approx(-20.0)

    def test_loud_bin_dominates_quiet_ones(self):
        """−20dB 與 −60dB 各半，功率域平均 ≈ −23.01dB。

        如果有人「順手」改成直接平均 dB 值，這裡會得 −40 —— 差 17dB，
        而 17dB 足以把一個正常窗判成低電平窗。
        """
        got = sl._mean_db([-20.0, -60.0])
        assert got == pytest.approx(-23.01, abs=0.02)
        assert got != pytest.approx(-40.0, abs=1.0), "退化成 dB 直接平均"

    def test_all_silence_stays_finite(self):
        assert sl._mean_db([-120.0, -120.0]) == pytest.approx(-120.0)
        assert sl._mean_db([]) == -120.0


class TestBinsToWindows:
    def _bins(self, pattern):
        return [(db, db) for db in pattern]

    def test_window_slicing_and_bounds(self):
        # 0.25s 一格、1s 一窗 → 每窗 4 格；9 格 → 3 窗（最後一窗只有 1 格）
        bins = self._bins([-20.0] * 9)
        wins, _ = sl.bins_to_windows(bins, bin_sec=0.25, window_sec=1.0)
        assert [w["bins"] for w in wins] == [4, 4, 1]
        assert wins[0]["t_start"] == 0.0
        assert wins[1]["t_start"] == 1.0
        assert wins[-1]["t_end"] == pytest.approx(2.25)

    def test_speech_frac_uses_relative_floor_not_absolute(self):
        """門檻是「全檔地板 + 12dB」，不是寫死的絕對值。

        0714 場比 0630 低 21dB（README §入口正規化）—— 寫死門檻會在下一場失準，
        所以這裡把整份訊號平移 30dB，密度必須不變。
        """
        pattern = [-60.0] * 6 + [-20.0] * 6
        wins_a, floor_a = sl.bins_to_windows(self._bins(pattern),
                                             bin_sec=0.25, window_sec=3.0)
        shifted = [db - 30.0 for db in pattern]
        wins_b, floor_b = sl.bins_to_windows(self._bins(shifted),
                                             bin_sec=0.25, window_sec=3.0)
        assert floor_b == pytest.approx(floor_a - 30.0)
        assert [w["speech_frac"] for w in wins_a] == [w["speech_frac"] for w in wins_b]

    def test_speech_frac_extremes(self):
        loud = self._bins([-10.0] * 4 + [-70.0] * 4)
        wins, _ = sl.bins_to_windows(loud, bin_sec=0.25, window_sec=1.0)
        assert wins[0]["speech_frac"] == 1.0   # 全部高於地板+12
        assert wins[1]["speech_frac"] == 0.0   # 全部是地板本身

    def test_speech_rms_is_none_when_no_speech_bins(self):
        """沒有語音格時回 None，不可回 -120（那會被當成「量到了、很小聲」）。"""
        wins, _ = sl.bins_to_windows(self._bins([-70.0] * 4),
                                     bin_sec=0.25, window_sec=1.0)
        assert wins[0]["speech_rms_db"] is None

    def test_dyn_range(self):
        wins, _ = sl.bins_to_windows(self._bins([-10.0, -50.0, -30.0, -20.0]),
                                     bin_sec=0.25, window_sec=1.0)
        assert wins[0]["dyn_range_db"] == pytest.approx(40.0)
        assert wins[0]["peak_db"] == pytest.approx(-10.0)

    def test_empty_input(self):
        wins, floor = sl.bins_to_windows([])
        assert wins == [] and floor is None


class TestScanFailsLoudly:
    def test_missing_file_raises(self):
        with pytest.raises(sl.ScanError):
            sl.scan(os.path.join(REPO_ROOT, "no-such-audio-file.m4a"))

    def test_missing_ffmpeg_raises(self, tmp_path, monkeypatch):
        """PATH 上沒有 ffmpeg 就要出聲 —— 靜默回空跟「檔案沒聲音」無法分辨。

        必須 monkeypatch `shutil.which`：初版只傳了空字串當 ffmpeg 路徑，
        但 `ffmpeg or shutil.which(...)` 會讓空字串退回真的 ffmpeg，
        於是測試走到「ffmpeg 存在但輸入不是音訊」那條路 ——
        **通過的理由跟它宣稱要測的不是同一件事**。
        """
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF....WAVEfmt ")   # 檔案存在，讓它過 exists() 那關
        monkeypatch.setattr(sl.shutil, "which", lambda _n: None)
        with pytest.raises(sl.ScanError) as exc:
            sl.scan(str(audio))
        assert "ffmpeg" in str(exc.value)

    def test_ffmpeg_nonzero_exit_raises(self, tmp_path):
        """ffmpeg 跑了但失敗（缺 decoder 是實際踩過的一種）→ raise，不回空。

        2026-09-10 在 NAS 上真的遇到：ffmpeg 在、但沒有 AAC decoder。
        """
        bad = tmp_path / "not-audio.m4a"
        bad.write_bytes(b"this is not an audio file at all")
        if not sl.shutil.which("ffmpeg"):
            pytest.skip("需要 ffmpeg")
        with pytest.raises(sl.ScanError):
            sl.scan(str(bad))
