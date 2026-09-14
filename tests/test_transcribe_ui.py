# -*- coding: utf-8 -*-
"""聽打工具 `transcribe_ui.py` 伺服端邏輯（`Kit`）的護欄測試。

## 為什麼這個檔存在

這支工具是**唯一會產生黃金段原始資料**的地方，而黃金段是整條量尺的地基。
它壞掉的方式不會噴錯，會安靜地存出壞資料 —— 然後所有 CER 都建在上面。

三件最貴的事各鎖一條：

1. **不合 schema 的資料不准落地。** 寧可拒絕存檔，也不要存一份之後才發現
   講者填了名冊外的名字、或 `native` 與 `zh` 一模一樣的稿。
2. **覆寫前要有備份、寫入要原子。** 聽打是人力最貴的一步，
   半寫壞檔 ＝ 1–1.5 小時重來。
3. **隨包附的範例行不可被當成真資料。** `lines.jsonl` 出貨時帶兩行範例，
   如果讀回來時沒濾掉，它們會混進 ref 文本並汙染 CER。

外加一條安全：`clip/` 只能取該資料夾內的檔（素材是僱主 IP，不可被路徑穿越撈走）。

不依賴網路層，只測 `Kit` —— 起 HTTP server 的測試會又慢又脆。

跑法：`python -m pytest tests/acoustic/test_transcribe_ui.py -q`
"""
import io
import json
import os
import threading
import sys
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from urllib.parse import quote

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
import transcribe_ui as ui  # noqa: E402


MANIFEST = {
    "set_id": "T", "source_audio": "x.m4a", "source_md5": "0" * 32,
    "duration_sec": 600.0,
    "segments": [{"id": "G1", "t_start": 0.0, "t_end": 60.0,
                  "reason": "測試用", "clip": "G1.m4a"}],
}


def good_row(**kw):
    r = {"id": "G1-001", "seg": "G1", "t_start": 2.0, "t_end": 8.0, "spk": "S1",
         "lang": "zho", "zh": "這批庫存還有九個", "models": []}
    r.update(kw)
    return r


@pytest.fixture
def kit(tmp_path):
    with io.open(str(tmp_path / "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(MANIFEST, fh, ensure_ascii=False)
    return ui.Kit(str(tmp_path))


@pytest.fixture
def kit_roster(tmp_path):
    m = dict(MANIFEST, speakers=["與會者D", "與會者E", "標註者"])
    with io.open(str(tmp_path / "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False)
    return ui.Kit(str(tmp_path))


class TestSaveRefusesBadData:
    def test_valid_rows_are_written(self, kit):
        ok, errs = kit.save([good_row()])
        assert ok and errs == []
        assert kit.lines() == [good_row()]

    def test_refused_save_leaves_no_file(self, kit):
        kit.save([good_row(zh="")])
        assert not os.path.exists(kit.lines_p), "被拒絕的存檔不可留下半份檔案"

    def test_refused_save_does_not_clobber_good_file(self, kit):
        kit.save([good_row()])
        ok, _ = kit.save([good_row(id="G1-002", zh="")])
        assert not ok
        assert kit.lines() == [good_row()], "存檔失敗不可破壞已經存好的內容"

    @pytest.mark.parametrize("bad,needle", [
        ({"t_end": 1.0}, "t_end"),
        ({"lang": "taigi"}, "lang"),
        ({"spk": "Vincent"}, "spk"),
        ({"seg": "G9"}, "G9"),
        ({"t_start": 500.0, "t_end": 520.0}, "超出"),
    ])
    def test_each_bad_shape_is_caught(self, kit, bad, needle):
        ok, errs = kit.save([good_row(**bad)])
        assert not ok and any(needle in e for e in errs)


class TestNativeIsOptional:
    """🔴 2026-09-10 設計變更：`native` 由必填改選填。

    原設計要人把台語寫成漢字。作廢原因是 標註者「不會打台語的中打」，
    而由 CC 從國語回譯來補 ＝ **自己製造 ground truth**
    （引擎寫出另一個同樣正確的台語形會被判成錯）。
    改成人只標「這句是台語」，忠實度用 `score.taigi_retention`（不需參考答案）量。
    """

    def test_nan_without_native_now_passes(self, kit):
        ok, errs = kit.save([good_row(lang="nan", zh="網路線沒有這個問題")])
        assert ok, errs

    def test_native_still_accepted_when_provided(self, kit):
        ok, errs = kit.save([good_row(lang="nan", zh="網路線沒有這個問題",
                                      native="網路線無迄個問題")])
        assert ok, errs

    def test_native_identical_to_zh_is_refused(self, kit):
        """兩欄一模一樣不構成第二軌 —— 那是誤填，留空即可。"""
        ok, errs = kit.save([good_row(lang="nan", zh="就是這樣", native="就是這樣")])
        assert not ok and any("完全相同" in e for e in errs)


class TestSpeakerRoster:
    """講者用真名冊。標註者 認得這些人的聲音，但五段互不相連，
    要他維持 S1..S12 的一致對應並不合理（他 2026-09-10 直接問「我怎麼分」）。"""

    def test_roster_name_accepted(self, kit_roster):
        ok, errs = kit_roster.save([good_row(spk="與會者D")])
        assert ok, errs

    def test_unknown_marker_accepted(self, kit_roster):
        ok, errs = kit_roster.save([good_row(spk="S?")])
        assert ok, errs

    def test_name_not_on_roster_refused(self, kit_roster):
        ok, errs = kit_roster.save([good_row(spk="路人甲")])
        assert not ok and any("名冊" in e for e in errs)

    def test_old_S_form_refused_when_roster_exists(self, kit_roster):
        """有名冊還填 S1 ＝ 沒照名冊選，會讓講者標記在段與段之間對不起來。"""
        ok, errs = kit_roster.save([good_row(spk="S1")])
        assert not ok

    def test_falls_back_to_S_form_without_roster(self, kit):
        ok, errs = kit.save([good_row(spk="S1")])
        assert ok, errs


class TestTopicField:
    """`topic`（這段在討論的產品）與 `models`（這句唸出來的型號）是兩件事。

    2026-09-10 實際混用過一次：15 筆產品標記寫進了 `models`，而那些句子裡
    一個字都沒唸到型號 —— 照原樣評分，所有引擎的型號 recall 都會被判 0，
    量到的是標註方式不是引擎能力（同 0716「量到的是書寫格式不是準確率」）。

    Python 這側管不到 UI 的欄位，但可以鎖住：`topic` 必須能無損往返，
    且不會被 schema 驗證擋掉 —— 否則搬移過去的 15 筆會存不回去。
    """

    def test_topic_round_trips(self, kit):
        ok, errs = kit.save([good_row(topic="Aurora Power XCF N1")])
        assert ok, errs
        assert kit.lines()[0]["topic"] == "Aurora Power XCF N1"

    def test_topic_and_models_coexist(self, kit):
        ok, errs = kit.save([good_row(topic="Project-W2-L",
                                      models=["Project-W2-D"])])
        assert ok, errs
        r = kit.lines()[0]
        assert r["topic"] == "Project-W2-L" and r["models"] == ["Project-W2-D"]

    def test_topic_absent_is_fine(self, kit):
        ok, errs = kit.save([good_row()])
        assert ok and "topic" not in kit.lines()[0]


def _baks(kit):
    base = os.path.basename(kit.lines_p)
    return sorted(f for f in os.listdir(kit.root)
                  if f.startswith(base + ".") and f.endswith(".bak"))


class TestDurability:
    def test_backup_written_before_overwrite(self, kit):
        kit.save([good_row()])
        kit.save([good_row(), good_row(id="G1-002", t_start=10.0, t_end=16.0)])
        baks = _baks(kit)
        assert baks, "沒有備份"
        with io.open(os.path.join(kit.root, baks[-1]), encoding="utf-8") as fh:
            assert len([l for l in fh if l.strip()]) == 1, "備份應是覆寫前那一版"

    def test_backups_are_not_overwritten(self, kit):
        """🔴 原本只有一個固定的 `.bak`，每次存檔都被蓋掉。

        後果（實測）：第一發壞存檔讓 lines=0／bak=2，第二發讓 **bak=0**
        —— 原稿與備份一起滅掉，1.5 小時的人工聽打無處可回。
        """
        for i in range(4):
            kit.save([good_row(id="G1-%03d" % (i + 1),
                               t_start=1.0 + i, t_end=5.0 + i)])
        assert len(_baks(kit)) >= 3, "備份被互相覆寫了：%r" % (_baks(kit),)

    def test_backups_are_capped(self, kit):
        """保留最近 N 份，不無限長 —— 否則 vault 會被備份塞滿。"""
        for i in range(ui.BACKUP_KEEP + 5):
            kit.save([good_row(id="G1-%03d" % (i + 1),
                               t_start=1.0 + i, t_end=5.0 + i)])
        assert len(_baks(kit)) <= ui.BACKUP_KEEP, _baks(kit)

    def test_no_tmp_left_behind(self, kit):
        kit.save([good_row()])
        leftovers = [f for f in os.listdir(kit.root) if f.endswith(".tmp")]
        assert not leftovers, leftovers

    def test_failed_write_leaves_no_tmp_and_keeps_original(self, kit, monkeypatch):
        """寫入中途炸掉：原檔要完好、`.tmp` 不可殘留。"""
        kit.save([good_row()])
        before = io.open(kit.lines_p, encoding="utf-8").read()

        real = json.dumps

        def boom(obj, **kw):
            if isinstance(obj, dict) and obj.get("id") == "G1-002":
                raise RuntimeError("模擬寫入中途失敗")
            return real(obj, **kw)

        monkeypatch.setattr(ui.json, "dumps", boom)
        with pytest.raises(RuntimeError):
            kit.save([good_row(), good_row(id="G1-002", t_start=10.0, t_end=16.0)])
        monkeypatch.undo()
        assert io.open(kit.lines_p, encoding="utf-8").read() == before
        assert not [f for f in os.listdir(kit.root) if f.endswith(".tmp")]

    def test_concurrent_saves_do_not_corrupt(self, kit):
        """`ThreadingHTTPServer` 允許並發 POST，而存檔動的是共用路徑。

        原本兩個請求共用同一個 `.tmp`／`.bak` 且無鎖 —— 可以互相截斷、
        其中一個 `os.replace` 在另一個改名之後失敗。
        """
        kit.save([good_row()])
        rowsets = [[good_row(id="G1-%03d" % (i + 1),
                             t_start=1.0 + i, t_end=5.0 + i)]
                   for i in range(8)]
        errors = []

        def worker(rows):
            try:
                kit.save(rows)
            except Exception as e:             # noqa: BLE001
                errors.append(e)

        ths = [threading.Thread(target=worker, args=(r,)) for r in rowsets]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        assert not errors, errors
        # 檔案必須是某一次存檔的完整結果，不可是兩次交錯的殘骸
        rows = kit.lines()
        assert len(rows) == 1, rows
        assert not [f for f in os.listdir(kit.root) if f.endswith(".tmp")]


class TestEmptySaveIsRefused:
    """空清單不可寫入 —— `validate_lines([])` 沒有任何一行可以違規。

    harness B3 實測：`save([])` 回 `ok=True` 並把檔案清成 0 句。
    觸發路徑不是假想 —— 存檔鈕在 `boot()` 之前就綁好了，
    `GET /api/lines` 失敗（lines.jsonl 被手改壞一個字就會）時 `ROWS` 停在 `[]`，
    而畫面上的空列表跟「還沒打」長得一模一樣，一按存檔就是清檔。
    """

    def test_validator_alone_does_not_catch_it(self, kit):
        """先證明驗證器**確實**擋不住 —— 這是這條測試存在的理由。"""
        assert golden.validate_lines([], kit.manifest()) == []

    def test_save_refuses_empty_list(self, kit):
        kit.save([good_row()])
        ok, errs = kit.save([])
        assert not ok
        assert any("空清單" in e for e in errs), errs

    def test_original_survives_empty_save(self, kit):
        kit.save([good_row()])
        before = io.open(kit.lines_p, encoding="utf-8").read()
        kit.save([])
        kit.save([])
        assert io.open(kit.lines_p, encoding="utf-8").read() == before
        assert len(kit.lines()) == 1


class TestExampleRowsAreFiltered:
    def test_shipped_examples_do_not_count_as_data(self, kit):
        """出貨的 lines.jsonl 帶兩行範例；讀回時必須濾掉，否則會混進 ref 汙染 CER。"""
        with io.open(kit.lines_p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(good_row(zh="（範例，刪掉這幾行）測試"),
                                ensure_ascii=False) + "\n")
            fh.write(json.dumps(good_row(id="G1-002", zh="真的內容"),
                                ensure_ascii=False) + "\n")
        rows = kit.lines()
        assert len(rows) == 1 and rows[0]["zh"] == "真的內容"


class TestClipPathIsConfined:
    def test_serves_file_in_folder(self, kit):
        p = os.path.join(kit.root, "G1.m4a")
        io.open(p, "wb").write(b"x")
        assert kit.clip_path("G1.m4a") == p

    @pytest.mark.parametrize("attack", [
        "../manifest.json", "..\\..\\secret.txt", "/etc/passwd",
        "sub/../../escape.txt",
    ])
    def test_traversal_is_blocked(self, kit, attack):
        """素材是僱主 IP，不可被路徑穿越撈走。"""
        got = kit.clip_path(attack)
        assert got is None or os.path.dirname(os.path.abspath(got)) == kit.root


class TestMissingManifest:
    def test_exits_loudly(self, tmp_path):
        """沒有 manifest 就出聲 —— 靜默起一個空 UI 會讓人以為工具壞了。"""
        with pytest.raises(SystemExit):
            ui.Kit(str(tmp_path))


# ── HTTP 層 ─────────────────────────────────────────────────────────────
# 這一整節對應 2026-09-11 資安獨立審計實測過的 payload。起真的 server ——
# 這幾個洞都在 handler 的邊界上，mock 掉就驗不到。

@pytest.fixture
def server(kit):
    """起在臨時埠的真 server。回 (port, 送 request 的函式)。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(kit, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = ui.make_handler(kit, port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    def req(method, path, host=None, body=None, headers=None, raw_len=None):
        """回 (status, body_text)。`raw_len` 可以送出騙人的 Content-Length。"""
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        hdrs = dict(headers or {})
        hdrs.setdefault("Host", host if host is not None
                        else "127.0.0.1:%d" % port)
        payload = None
        if body is not None:
            payload = body if isinstance(body, bytes) else body.encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
            hdrs["Content-Length"] = (str(len(payload)) if raw_len is None
                                      else raw_len)
        try:
            conn.putrequest(method, path, skip_host=True,
                            skip_accept_encoding=True)
            for k, v in hdrs.items():
                conn.putheader(k, v)
            conn.endheaders()
            if payload is not None:
                conn.send(payload)
            r = conn.getresponse()
            return r.status, r.read().decode("utf-8", "replace")
        finally:
            conn.close()

    try:
        yield port, req
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


class TestHostHeaderIsValidated:
    """DNS rebinding：只綁 127.0.0.1 **擋不住**這件事。

    攻擊者把自己的網域解析到 127.0.0.1，受害者的瀏覽器就會帶著那個 Host
    連進來，而同源政策認為那是攻擊者的網域 ⇒ 放行讀取回應內容。
    實測（修前）：`Host: evil.attacker.com` 打 `GET /api/lines` → 200 ＋ 完整 JSON。
    """

    def test_foreign_host_cannot_read_lines(self, server, kit):
        kit.save([good_row()])
        code, body = server[1]("GET", "/api/lines", host="evil.attacker.com")
        assert code == 403, "外部 Host 讀到了逐字稿：%s" % body[:120]

    def test_foreign_host_cannot_write(self, server, kit):
        kit.save([good_row()])
        code, _ = server[1]("POST", "/api/lines", host="evil.attacker.com",
                            body="[]")
        assert code == 403
        assert len(kit.lines()) == 1, "外部 Host 清掉了稿子"

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST"])
    def test_local_hosts_are_accepted(self, server, kit, host):
        """反向：正常的本機 Host 不可被誤擋（否則工具自己用不了）。"""
        port, req = server
        code, _ = req("GET", "/api/lines", host="%s:%d" % (host, port))
        assert code == 200


class TestCsrfOnWrite:
    """任何網頁都能對本機埠發 POST —— 而這個 POST 會覆寫整份聽打稿。"""

    def test_foreign_origin_cannot_wipe(self, server, kit):
        kit.save([good_row()])
        code, _ = server[1]("POST", "/api/lines", body="[]",
                            headers={"Origin": "https://evil.example"})
        assert code == 403
        assert len(kit.lines()) == 1

    def test_local_origin_is_accepted(self, server, kit):
        port, req = server
        code, body = req("POST", "/api/lines",
                         body=json.dumps([good_row()], ensure_ascii=False),
                         headers={"Origin": "http://127.0.0.1:%d" % port})
        assert code == 200, body

    def test_no_origin_is_accepted(self, server, kit):
        """非瀏覽器（curl／本測試）沒有 Origin，不可因此被擋。"""
        code, body = server[1]("POST", "/api/lines",
                               body=json.dumps([good_row()],
                                               ensure_ascii=False))
        assert code == 200, body


class TestEmptyPostCannotWipe:
    """就算 Host／Origin 都合法，空清單也不可清檔（第二層防線）。"""

    def test_empty_body_is_refused_and_file_survives(self, server, kit):
        kit.save([good_row()])
        code, body = server[1]("POST", "/api/lines", body="[]")
        assert code == 422, body
        assert len(kit.lines()) == 1

    def test_twice_still_survives(self, server, kit):
        """實測（修前）：第一發清 lines、**第二發連 .bak 一起清**。"""
        kit.save([good_row()])
        for _ in range(2):
            server[1]("POST", "/api/lines", body="[]")
        assert len(kit.lines()) == 1
        assert _baks(kit) or True          # 備份可有可無，原稿必須在


class TestBodyLengthIsGuarded:
    def test_negative_content_length(self, server, kit):
        """`-1` 會讓 `read(-1)` 一路讀到 EOF。"""
        kit.save([good_row()])
        code, _ = server[1]("POST", "/api/lines", body="[]", raw_len="-1")
        assert code == 413
        assert len(kit.lines()) == 1

    def test_non_numeric_content_length(self, server, kit):
        """修前 `int()` 在 try 之外 ⇒ handler 拋例外、連線被關，
        前端 `await res.json()` reject ⇒ **UI 一個字都不顯示**。"""
        code, body = server[1]("POST", "/api/lines", body="[]", raw_len="abc")
        assert code == 400
        assert "Content-Length" in body

    def test_oversize_content_length(self, server):
        code, _ = server[1]("POST", "/api/lines", body="[]",
                            raw_len=str(ui.MAX_BODY + 1))
        assert code == 413

    def test_non_list_body(self, server):
        code, body = server[1]("POST", "/api/lines", body='{"a":1}')
        assert code == 400, body


class TestClipIsConfinedOnTheWire:
    def test_symlink_out_of_root_is_refused(self, server, kit, tmp_path):
        """`basename` 只是字面收斂 —— `open` 會跟隨 symlink（Codex #15）。"""
        secret = tmp_path.parent / "outside-secret.txt"
        secret.write_text("不該被送出去", encoding="utf-8")
        link = os.path.join(kit.root, "leak.m4a")
        try:
            os.symlink(str(secret), link)
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("這台機器不能建 symlink（Windows 需要權限）")
        assert kit.clip_path("leak.m4a") is None, "symlink 指到 root 外面卻放行了"
        code, _ = server[1]("GET", "/clip/leak.m4a")
        assert code == 404

    def test_drive_relative_path_is_refused(self, kit):
        """Windows 的 `D:secret.txt` —— basename 之後還是它自己，
        而 `os.path.join(root, "D:secret.txt")` 會被當成 D 磁碟的相對路徑。"""
        assert kit.clip_path("D:secret.txt") is None
        assert kit.clip_path("C:/Windows/win.ini") is None

    def test_percent_encoded_name_is_decoded(self, server, kit):
        """`/clip/` 原本從不 URL-decode ⇒ 非 ASCII 檔名一律 404。"""
        name = "測試音檔.m4a"
        with io.open(os.path.join(kit.root, name), "wb") as fh:
            fh.write(b"\x00" * 32)
        code, _ = server[1]("GET", "/clip/" + quote(name))
        assert code == 200, "URL-encoded 的中文檔名取不到"


class TestIdIsEscapedInHtml:
    """`id` 原本直接內插進 `<td>` 與 `data-e=` ⇒ 存起來的 XSS。

    而 `validate_lines` 接受 `<img src=x onerror=alert(1)>` 當 id ——
    驗證器不管，前端就必須管。
    """

    def test_id_goes_through_esc(self):
        html = ui.HTML
        assert "<td>${esc(r.id)}</td>" in html, "id 沒過 esc()"
        assert 'data-e="${esc(r.id)}"' in html
        assert 'data-x="${esc(r.id)}"' in html

    def test_esc_covers_quotes(self):
        """id 會被放進屬性值 ⇒ 引號也必須轉，只轉 `<>&` 不夠。"""
        i = ui.HTML.index("function esc(")
        body = ui.HTML[i:i + 400]
        for ch in ('&quot;', '&#39;', '&lt;', '&gt;', '&amp;'):
            assert ch in body, "esc() 沒處理 %s" % ch

    def test_next_id_uses_max_not_count(self):
        """刪過列之後不可撞號（harness M7）。"""
        i = ui.HTML.index("function nextId(")
        body = ui.HTML[i:i + 400]
        assert "Math.max" in body, "nextId 還在用筆數＋1，刪過列就會撞號"

    def test_save_button_is_gated_on_load(self):
        """`boot()` 沒載成功時按存檔不可清檔。"""
        assert "let LOADED=false;" in ui.HTML
        assert "if(!LOADED)" in ui.HTML
        assert "LOADED=true;" in ui.HTML
