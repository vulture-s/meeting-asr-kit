# -*- coding: utf-8 -*-
"""meeting-transcribe 的 subprocess 一律不得靠系統 locale 解碼。

## 為什麼這個檔存在

2026-09-10 實跑 Whisper 線時看到 `[chunk] 93.4 min -> 6 段（靜音點 0 個）`。
靜音點 0 個，於是退化成每 1080 秒硬切 —— 而黃金段 G2 的起點正好是 2160，
基準線會被切在句子中間。

根因是 `subprocess.run(cmd, capture_output=True, text=True)`：
`text=True` 用**系統 locale** 解碼（本機是 cp950），而 ffmpeg 的 stderr 帶 UTF-8 位元組
→ reader thread 噴 `UnicodeDecodeError`、**主流程收不到、returncode 仍是 0**
→ `detect_silence_points` 拿到空輸出 → 回 0 個點 → 靜默硬切。

修完實測：同一支音檔，靜音點 **0 → 1524 個**。
⇒ 該功能自 2026-09-02 上線以來，在這台 cp950 的 PC 上**一直沒真的生效**。

## 為什麼既有的 32 項切段測試抓不到

`chunking.py` 檔頭寫著它就是為了防「ffmpeg 失敗 → 靜默回空 → 無聲退化成硬切」，
而那條守衛量的是 **returncode**。這次的失敗**不經過 returncode**。
**量對維度只量了一個面向，等於沒量。**

而純函式層的測試餵的是 Python list，根本不經過 subprocess；
ffmpeg 層的測試在 CI 上多半 skip，就算跑也是在 UTF-8 locale 的環境。
⇒ 這個缺陷在任何既有測試裡都是綠的。

## 這條測試守什麼

原始碼層級：**這個目錄裡任何 `subprocess` 呼叫，只要會拿到文字，就必須自己指定編碼**。
不靠執行環境，所以在 UTF-8 的 CI 上也會紅 —— 那正是重點，
因為這個 bug 只在非 UTF-8 locale 的機器上發作，而生產機（標註者 的 PC）就是那種機器。
"""
import importlib.util
import os

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


TOOLS = _tool_dir()


def _py_files():
    return sorted(f for f in os.listdir(TOOLS) if f.endswith(".py"))


# 偵測器只有一份 —— 2026-09-11 之前這裡有第二份，而且漏了 `check_call`，
# 於是「零容忍」這支反而比全 repo 的 ratchet 還鬆（harness M5）。
def _load_lint():
    """載入共用偵測器。**候選路徑**，不寫死版面。

    這批檔會出現在兩種版面：本 repo（`tests/core/`）與消毒後的獨立 kit（扁平）。
    寫死一條路徑在後者會指到 repo 外面。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(REPO_ROOT, "tests", "core", "subprocess_encoding_lint.py"),
        os.path.join(here, "subprocess_encoding_lint.py"),
        os.path.join(os.path.dirname(here), "subprocess_encoding_lint.py"),
    ]
    for c in cands:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location(
                "subprocess_encoding_lint", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("找不到 subprocess_encoding_lint.py，找過：\n  "
                       + "\n  ".join(cands))


lint = _load_lint()


@pytest.mark.parametrize("fname", _py_files())
def test_no_locale_decoding_anywhere(fname):
    """這個目錄裡任何會拿到文字的 `subprocess` 呼叫都必須自己指定編碼。

    零容忍：不是 ratchet，這裡的違規數必須是 0。

    🔴 2026-09-11：原本這支有自己一份 AST 判定，而且漏了 `check_call`
    ⇒ 零容忍那支比全 repo 的 ratchet 還鬆。現在兩道閘走同一支偵測器
    （`tests/core/subprocess_encoding_lint.py`），繞道語料也共用。
    """
    path = os.path.join(TOOLS, fname)
    bad = lint.violations(lint.read_source(path), path)
    assert not bad, (
        "這些呼叫會用系統 locale 解碼，在非 UTF-8 的機器上會靜默失敗：\n  - "
        + "\n  - ".join("%s:%d %s —— %s" % (fname, ln, fn, why)
                        for ln, fn, why in bad)
        + "\n" + lint.FIX_HINT)


def test_the_directory_is_actually_scanned():
    """anti-blank：目錄要真的存在、真的有 .py 被掃到。

    否則「零違規」可能只是因為目錄被搬走、或副檔名篩掉了全部檔案。
    """
    assert os.path.isdir(TOOLS), "目錄不見了 —— 這道閘在守一個不存在的地方"
    assert len(_py_files()) >= 5, "只掃到 %d 支 .py，範圍可能已失效" % len(_py_files())


@pytest.mark.parametrize("name,src", lint.BYPASS_CORPUS,
                         ids=[b[0] for b in lint.BYPASS_CORPUS])
def test_detector_catches_every_bypass(name, src):
    """反向驗證：12 種重現同一個缺陷的寫法，每一種都要抓到。

    沒有這一步，上面那條可能只是「掃過去沒東西」的恆綠守衛
    （`verification.md` §Gate 設計：恆綠的守衛等於沒有守衛）。
    語料與 ratchet 共用 —— 兩道閘對同一組寫法負責。
    """
    assert lint.count_violations(src) == 1, "『%s』沒被抓到" % name


@pytest.mark.parametrize("name,src", lint.CLEAN_CORPUS,
                         ids=[o[0] for o in lint.CLEAN_CORPUS])
def test_clean_forms_are_not_flagged(name, src):
    """反向驗證的另一半：修好的寫法必須是綠的，否則守衛恆紅會被 mute 掉。"""
    assert lint.count_violations(src) == 0, "『%s』被誤判成違規" % name
