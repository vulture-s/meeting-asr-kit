# -*- coding: utf-8 -*-
"""會議 ASR 黃金段 —— schema 定義、驗證與載入（不含評分，評分在 `score.py`）。

## 為什麼要有這支

`three-way-meeting-asr-bench-2026-07-16.md` §6 掛了兩個月的天花板是
「**無 ground truth → 無 WER/CER 正確率**」，而 06-30 / 07-08 兩份 bench 也各自寫過
「真要定論需人工聽打一段黃金標準」。那份黃金標準一直沒做，其中一個原因是
**從來沒有人定義過它長什麼樣** —— 沒有 schema，聽打就沒有交付定義，
兩個月後自己打的稿跟當初的判準會對不起來，量到的變成打字風格不是模型能力。

完整規格（選段紀律、聽打規格、指標定義、負向測試）→
`（內部評測紀錄）`

## 台語怎麼量 —— 而**不是**靠人寫台語漢字（2026-09-10 改）

台語如果只寫一欄漢字，「**把台語抹平成國語**」跟「**聽對了但寫成國語**」在量尺上
是同一個東西 —— 這正是 07-08 bench 只能做質性判讀的原因（該檔 TL;DR 第一條：
「Qwen 忠實、Whisper 抹平」，有例證但沒有數字）。

初版的解法是要人填 `native`（台語原形）當第二軌。**那條路已作廢**：
標註者 不會打台語的中打，而由 CC 從國語回譯來補那一欄
＝ **自己製造 ground truth**（引擎寫出另一個同樣正確的台語形會被判成錯，
量到的是回譯者的猜測不是事實）。

現行做法：

- `lang` —— 人只標「**這句是不是台語**」。他聽得出來，這就夠了。
- `zh`   —— 語意等價的國語，算**內容正確率**。
- `native` —— **選填**。打得出來就填（多一份證據），打不出來留空不罰。

**語言忠實度改用不需要參考答案的指標**：`score.py` 的 `taigi_retention` ——
看引擎在那些被標為台語的句子上，有沒有吐出台語專屬字形。
抹平的引擎趨近 0、忠實的明顯 > 0。`taigi_flatten_gap` 退為次要，
且只在真的有人填了 `native` 時才有值。

## 刻意不禁止的事

**同一段內時間重疊的句子是合法的。** S3 是「多人交疊／搶話」段，那裡的重疊
不是資料錯誤而是要量的東西。驗證器若把重疊當錯誤，等於把 S3 判死。

## 落檔位置

🔴 `manifest.json` 與 `lines.jsonl` **是 vault-only，永不進 git**
（這些素材是僱主 IP）。進 git 的只有本檔與 `score.py`。
"""
import json
import math
import os
import re

# lang 欄允許值。nan = 台語（ISO 639-3），沿用標準碼而不自造 "tw"／"taigi"。
LANGS = ("zho", "nan", "mixed", "eng")

# 每一行必備欄位。缺一個就 raise —— 聽打到一半改 schema 的成本遠大於一開始擋掉。
REQUIRED_LINE_FIELDS = ("id", "seg", "t_start", "t_end", "spk", "lang", "zh")

_SPK_RE = re.compile(r"^S(\d+|\?)$")
# 段 id 允許任何「字母前綴 ＋ 數字」。
# 🔴 2026-09-10 修：原本寫死 `^S\\d+$`，而實際產出的段叫 G1–G5
#    → `load_golden` 會擋掉自己產的 manifest（存檔沒受影響是因為
#    存檔只跑 validate_lines）。寫死字首就是這種形狀的坑。
_SEG_ID_RE = re.compile(r"^[A-Za-z]{1,2}\d{1,3}$")
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")


class GoldenError(ValueError):
    """schema 不合格。訊息裡一定帶行 id，否則 5 分鐘的稿無從找起。"""


def validate_manifest(manifest):
    """回傳錯誤訊息 list（空 list ＝ 通過）。不 raise，讓呼叫端一次看完所有問題。"""
    errs = []
    for key in ("set_id", "source_audio", "source_md5", "duration_sec",
                "segments"):
        if not manifest.get(key):
            errs.append("manifest 缺 %s" % key)

    # md5 原本只驗真值 ⇒ `source_md5: "x"` 也過。而它的用途是「這份黃金段對應的
    # 是哪一支音檔」—— 一個過不了形式檢查的值等於沒有 provenance。
    md5 = manifest.get("source_md5")
    if md5 and not _MD5_RE.match(str(md5)):
        errs.append("manifest.source_md5 必須是 32 位 hex，收到 %r" % (md5,))

    # duration_sec 原本既不必填也不驗 ⇒ 段的時間可以落在音檔之外而無人管。
    dur = manifest.get("duration_sec")
    if dur is not None and not (_is_num(dur) and dur > 0):
        errs.append("manifest.duration_sec 必須是正的有限數字，收到 %r" % (dur,))
        dur = None
    elif not _is_num(dur):
        dur = None

    # speakers 是講者驗證的名冊來源（validate_lines 吃它）⇒ 型別不對會讓
    # 講者檢查靜默失效：`"標註者"` 是字串，`in` 會做子字串比對而不是名冊比對。
    spk = manifest.get("speakers")
    if spk is not None:
        if not isinstance(spk, list):
            errs.append("manifest.speakers 必須是 list，收到 %s"
                        % type(spk).__name__)
        elif not all(isinstance(x, str) and x.strip() for x in spk):
            errs.append("manifest.speakers 的每一項都必須是非空字串")

    segs = manifest.get("segments") or []
    if not isinstance(segs, list):
        return errs + ["manifest.segments 必須是 list"]

    seen = set()
    for i, seg in enumerate(segs):
        where = "segments[%d]" % i
        sid = seg.get("id")
        if not sid or not _SEG_ID_RE.match(str(sid)):
            errs.append("%s.id 必須是 S1/S2/… 形式，收到 %r" % (where, sid))
            continue
        if sid in seen:
            errs.append("%s.id 重複：%s" % (where, sid))
        seen.add(sid)
        ts, te = seg.get("t_start"), seg.get("t_end")
        if not _is_num(ts) or not _is_num(te):
            errs.append("%s 的 t_start/t_end 必須是有限數字" % where)
        elif te <= ts:
            errs.append("%s t_end(%s) 必須大於 t_start(%s)" % (where, te, ts))
        elif ts < 0 or (dur is not None and te > dur):
            # 段必須真的在這支錄音裡面。原本完全沒比對過 duration_sec ⇒
            # 一個 60 秒的錄音可以有 [-10, 100] 的段，而下游會安靜地收到空視窗。
            errs.append("%s 時間 [%s, %s] 超出音檔範圍 [0, %s]"
                        % (where, ts, te, dur))
        # 選段依據是紀律的一部分：沒寫 reason 就是沒留下「為什麼選這 60 秒」，
        # 而選段偏誤沒有事後補救的辦法（見 plan §1.2）。
        # 原本只驗真值 ⇒ `reason: 1` 也過，而那不是依據。
        reason = seg.get("reason")
        if not reason:
            errs.append("%s 缺 reason —— 選段依據必須寫下來" % where)
        elif not (isinstance(reason, str) and reason.strip()):
            errs.append("%s.reason 必須是非空字串，收到 %r" % (where, reason))
    return errs


def validate_lines(lines, manifest=None):
    """回傳錯誤訊息 list（空 list ＝ 通過）。

    `manifest` 給了就一併檢查：seg 存在、時間落在該 seg 視窗內。
    """
    errs = []
    seg_bounds = {}
    if manifest:
        for seg in manifest.get("segments") or []:
            if seg.get("id") and _is_num(seg.get("t_start")) and _is_num(seg.get("t_end")):
                seg_bounds[seg["id"]] = (seg["t_start"], seg["t_end"])

    seen_ids = set()
    for i, ln in enumerate(lines):
        lid = ln.get("id") or "(第 %d 行,無 id)" % (i + 1)
        for f in REQUIRED_LINE_FIELDS:
            if f not in ln:
                errs.append("%s 缺欄位 %s" % (lid, f))
        # 原本是 `if ln.get("id"):` ⇒ **存在但為空**的 id 既不報錯、也不進
        # 重複檢查，於是任意多行都可以共用「空 id」而驗證全綠。而 id 是
        # 出錯時唯一的定位手段（5 分鐘的稿沒有 id 就無從找起，本檔檔頭寫的）。
        raw_id = ln.get("id")
        if "id" in ln and not (isinstance(raw_id, str) and raw_id.strip()):
            errs.append("%s id 必須是非空字串，收到 %r" % (lid, raw_id))
        elif raw_id:
            if raw_id in seen_ids:
                errs.append("%s id 重複" % lid)
            seen_ids.add(raw_id)

        ts, te = ln.get("t_start"), ln.get("t_end")
        if not _is_num(ts) or not _is_num(te):
            errs.append("%s t_start/t_end 必須是數字" % lid)
        elif te <= ts:
            errs.append("%s t_end(%s) 必須大於 t_start(%s)" % (lid, te, ts))
        elif ln.get("seg") in seg_bounds:
            lo, hi = seg_bounds[ln["seg"]]
            # 容許 0.5s 邊界誤差：人工標的起訖秒本來就粗到 1 秒。
            if ts < lo - 0.5 or te > hi + 0.5:
                errs.append("%s 時間 [%s, %s] 超出 %s 視窗 [%s, %s]"
                            % (lid, ts, te, ln["seg"], lo, hi))

        if manifest and ln.get("seg") not in seg_bounds:
            errs.append("%s 的 seg=%r 不在 manifest.segments 裡" % (lid, ln.get("seg")))

        lang = ln.get("lang")
        if lang not in LANGS:
            errs.append("%s lang=%r 不在 %s" % (lid, lang, list(LANGS)))

        # 講者：manifest 有名冊就用名冊（標註者 認得這些人的聲音，
        # 而要他在五段互不相連的片段之間維持 S1..S12 的一致對應並不合理）。
        # 沒名冊才退回 S1/S2/… 形式。兩種都收 "S?" ＝ 聽不出來。
        spk = ln.get("spk")
        roster = (manifest or {}).get("speakers") or []
        if roster:
            if spk not in roster and spk != "S?":
                errs.append("%s spk=%r 不在 manifest.speakers 名冊裡（聽不出來請填 S?）"
                            % (lid, spk))
        elif not spk or not _SPK_RE.match(str(spk)):
            errs.append("%s spk=%r 必須是 S1/S2/…／聽不出來標 S?" % (lid, spk))

        if not (ln.get("zh") or "").strip():
            errs.append("%s zh 不可為空" % lid)

        zh = (ln.get("zh") or "").strip()
        nat = (ln.get("native") or "").strip()

        # 🔴 `native` 是**選填**（2026-09-10 改）。
        # 原設計要求 lang=nan 必填台語原形，但 標註者「不會打台語的中打」——
        # 而由 CC 從國語回譯來補那一欄 **等於自己製造 ground truth**：
        # 引擎若寫出另一個同樣正確的台語形會被判成錯，量到的是我的猜測不是事實。
        # 改法是把「台語忠實度」換成**不需要參考答案**的指標：
        # 只要 標註者 標出哪幾句是台語（他聽得出來），就能量各家輸出在那些句子上
        # 有沒有出現台語專屬字形（`score.py` 的 taigi_retention）。
        # ⇒ native 打得出來就填（多一份證據），打不出來留空不罰。
        if nat and nat == zh:
            errs.append("%s native 與 zh 完全相同 —— 那不構成第二軌，留空即可" % lid)

        models = ln.get("models", [])
        if not isinstance(models, list) or any(not isinstance(m, str) for m in models):
            errs.append("%s models 必須是字串 list" % lid)

    # 刻意不檢查同段內時間重疊 —— S3 交疊段的重疊是要量的東西，不是錯誤（見檔頭）。
    return errs


def ref_text(line, track="zh"):
    """取一行的參考文字。track='native' 時，沒有 native 的句子退回 zh。

    退回而不是跳過，是因為 native 軌的 ref 要跟 zh 軌**覆蓋同一段音訊**，
    兩軌 CER 才可以相減（`taigi_flatten_gap`）。純國語句在兩軌本來就該相同。
    """
    if track == "native":
        return (line.get("native") or line.get("zh") or "")
    return line.get("zh") or ""


def load_golden(path):
    """讀一個黃金段目錄，回 (manifest, lines)。schema 不合格直接 raise。"""
    manifest_p = os.path.join(path, "manifest.json")
    lines_p = os.path.join(path, "lines.jsonl")
    with open(manifest_p, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    lines = []
    with open(lines_p, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except ValueError as exc:
                raise GoldenError("lines.jsonl 第 %d 行不是合法 JSON：%s" % (n, exc))

    errs = validate_manifest(manifest) + validate_lines(lines, manifest)
    if errs:
        raise GoldenError("黃金段 schema 不合格（%d 項）：\n  - %s"
                          % (len(errs), "\n  - ".join(errs)))
    return manifest, lines


def _is_num(v):
    """**有限**的實數。bool 不算（`True` 是 int 的子類）。

    🔴 NaN 與 ±inf 都是 float，`isinstance` 過得去 —— 而 NaN 的**所有**比較
    都回 False ⇒ `te <= ts` 不成立、`ts < lo - 0.5` 不成立
    ⇒ 兩個 NaN 時間戳**一個錯誤都不會報**，然後在 `score.py` 裡
    把 CER 算成 NaN。驗證器放過的東西，下游沒有第二道防線。
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)

