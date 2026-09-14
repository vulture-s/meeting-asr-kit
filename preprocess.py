# -*- coding: utf-8 -*-
"""RD 會議轉錄 — 共用入口前處理（電平正規化）。

三條線（whisper / qwenasr / mlx）共用。交來的音檔電平隨錄音裝置變動
（0707 iPhone／0714 DJI Mic 2），0714 比 0630 低 21dB → Silero VAD 把
1669s（28 分）判非語音沒餵進模型，QwenASR 崩到 49.3%。

正解＝入口一律正規化，不追每台裝置的增益、也不動引擎程式碼或 VAD 門檻。
參數與效果已於 2026-07-16 三線 bench 端到端驗證：
    Whisper  89.2 → 95.0%
    QwenASR  49.3 → 90.8%
    mlx      ~75%（不動；mlx 不吃 Silero VAD）
完整數據見 （內部紀錄）case-studies/model-eval/three-way-meeting-asr-bench-2026-07-16.md §3
"""
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# 2026-07-16 bench 驗證過的參數，改動前先重跑 bench
DYNAUDNORM = "dynaudnorm=f=200:g=15:p=0.9:m=30"
SAMPLE_RATE = 16000


def _require_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit(
            "[FAIL] 找不到 ffmpeg — 入口正規化是硬需求（低電平音檔會讓 VAD 吃掉整段）。\n"
            "       PC: winget install Gyan.FFmpeg ／ Mac: brew install ffmpeg\n"
            "       確定要跑未正規化的原始音檔，加 --no-normalize（結果會退化，別用在正式場次）。"
        )
    return exe


@contextmanager
def normalized_audio(src, enabled=True):
    """yield 正規化後的暫存 wav 路徑；離開 context 自動刪。

    enabled=False 時原封不動 yield 原檔（--no-normalize 逃生口）。
    ⚠️ 呼叫端的輸出檔名要用「原始」音檔算，別用這裡回的暫存路徑。
    """
    src = Path(src)
    if not enabled:
        yield src
        return

    exe = _require_ffmpeg()
    tmpdir = tempfile.mkdtemp(prefix="rd-asr-norm-")
    dst = Path(tmpdir) / (src.stem + ".norm.wav")
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-af", DYNAUDNORM,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(dst),
    ]
    print("[norm] dynaudnorm -> %s" % dst.name, flush=True)
    try:
        subprocess.run(cmd, check=True)
        if not dst.exists() or dst.stat().st_size == 0:
            sys.exit("[FAIL] 正規化產出空檔: %s" % dst)
        yield dst
    except subprocess.CalledProcessError as e:
        sys.exit("[FAIL] ffmpeg 正規化失敗（exit %s）: %s" % (e.returncode, src))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def add_norm_flag(ap):
    """給用 argparse 的腳本（mlx 線）掛逃生口。"""
    ap.add_argument(
        "--no-normalize", action="store_true",
        help="跳過入口 dynaudnorm 正規化（結果會退化，僅供 A/B 比對）",
    )
