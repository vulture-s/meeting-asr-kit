# -*- coding: utf-8 -*-
"""黃金段聽打工具 —— 本機小網頁，取代手打 JSONL。

## 為什麼要有這支

黃金段的 schema（`golden.py`）一句要填六個欄位，其中 `t_start`／`t_end` 還要對原檔座標。
5 分鐘密集多人對話手打 JSONL **不是難、是不合理** —— 而聽打是整條 pipeline 唯一
只有人做得到的一步（標註者 2026-09-10：「打進去 jsonl 對我來說有難度」）。
工具卡住的是那一步，等於卡住全部。

## 為什麼是本機網頁不是 .exe

真 `.exe` 要 PyInstaller：~40MB 二進位、常被防毒誤判、每改一次要重打。
本機網頁：`<audio>` 給的 `currentTime` 是逐句抓時間點的關鍵，鍵盤操作也直接；
**而且 Mac 也能跑**（標註者 多半在 Apple 那側聽東西）。雙擊 `.bat` 的體感與 exe 相同。

## 跑法

```
python transcribe_ui.py "<黃金段資料夾>"      # 該資料夾要有 manifest.json 與 *.m4a
```

會開 `http://127.0.0.1:8765`。**只綁 127.0.0.1** —— 素材是僱主 IP，不對外開埠。

## 存檔行為

- 每次存檔**整份重寫** `lines.jsonl`（不做 append），避免半寫壞檔。
- 寫檔前跑 `golden.validate_lines()`；不合格**拒絕寫入並回錯誤清單**，不靜默存壞資料。
- 寫檔前先把舊檔複製成**帶時間戳**的備份並保留最近 10 份
  （`lines.jsonl.<時間>.bak`）。原本只有一個固定的 `.bak`，每次存檔都被蓋掉 ⇒ 連續兩次壞存檔就把原稿與備份一起滅掉。
- **空清單一律拒絕寫入** —— `validate_lines([])` 沒有任何一行可以違規，所以驗證器不會擋；而空存檔的唯一來源是「前端沒載到資料」。
- `id` 由 `<段>-<三位流水號>` 自動產生，`t_start`／`t_end` 由播放位置 ＋ 該段 offset 自動換算。

## `models` 與 `topic` 是兩件事，不可混用

- **`models`** —— 這句話裡**真的被唸出來**的型號。它撐 `score.py` 的型號 precision／recall。
- **`topic`** —— 這一段在討論哪個產品（沒唸出來也算）。給會議記錄與 LLM in-context 潤稿用，
  **不進評分**。

混用的後果：引擎會為「根本沒講出口的詞」被判 recall 0，量到的是標註方式不是引擎能力
（同 0716「量到的是書寫格式不是準確率」）。2026-09-10 實際發生過一次，
15 筆產品標記已從 `models` 搬到 `topic`。
"""
import json
import os
import posixpath
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
MIME = {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".json": "application/json", ".jsonl": "application/json"}

# POST body 上限。64 MB 遠大於任何真實的 lines.jsonl（0908 那份 64 句約 30 KB），
# 但擋住「沒有上限」這件事本身 —— 原本 `read(n)` 的 n 完全來自 client。
MAX_BODY = 64 * 1024 * 1024

# 本機名字。允許的 Host 由 `allowed_hosts(port)` 依**實際綁定的埠**算出來 ——
# 寫死常數的話換埠跑就會被自己的防護擋掉，而且測試沒辦法用臨時埠起 server
# 驗它（一道驗不到的防護等於沒有）。
LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]", "::1")


def allowed_hosts(port):
    """回這個埠上可接受的 Host header 集合。

    **為什麼需要這道檢查**：實測 `Host: evil.attacker.com` 打
    `GET /api/lines` 會回 200 ＋ 完整逐字稿。那是 DNS rebinding ——
    攻擊者把自己的網域解析到 127.0.0.1，受害者的瀏覽器就會帶著那個 Host
    連到這個埠，而同源政策認為那是攻擊者的網域 ⇒ 放行讀取回應內容。
    **只綁 127.0.0.1 擋不住**：連線本來就是從本機發出的。
    """
    out = set(LOCAL_NAMES)
    out.update("%s:%d" % (h, port) for h in LOCAL_NAMES)
    return frozenset(out)

# 保留幾份帶時間戳的備份。原本只有一個固定的 `.bak`，每次存檔都被蓋掉
# ⇒ 連續兩次空存檔就把原稿與備份一起滅了（harness B3 實測）。
BACKUP_KEEP = 10


class Kit(object):
    """一個黃金段資料夾。所有讀寫都經過這裡，方便單測。"""

    def __init__(self, root):
        self.root = os.path.realpath(root)
        # ThreadingHTTPServer 允許並發 POST，而存檔會動共用的檔案路徑。
        self._lock = threading.Lock()
        self.manifest_p = os.path.join(self.root, "manifest.json")
        self.lines_p = os.path.join(self.root, "lines.jsonl")
        if not os.path.exists(self.manifest_p):
            raise SystemExit("找不到 manifest.json：%s" % self.manifest_p)

    def manifest(self):
        with open(self.manifest_p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def lines(self):
        if not os.path.exists(self.lines_p):
            return []
        out = []
        with open(self.lines_p, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    out.append(json.loads(raw))
        # 過濾掉隨包附的範例行（避免它們被當成真資料算進 CER）
        return [l for l in out if "（範例" not in (l.get("zh") or "")]

    def save(self, rows):
        """回 (ok, errors)。不合格就不寫 —— 靜默存壞資料比拒絕貴得多。

        🔴 2026-09-11 審計後改寫。原本三個宣稱有兩個不成立：

        1. **「不合格拒絕寫入」對最壞的 payload 不成立** ——
           `golden.validate_lines([], manifest)` 回 `[]`（空清單沒有任何一行
           可以違規），於是 `save([])` 回 `ok=True` 並把檔案清成 0 句。
           而前端的存檔鈕在 `boot()` 之前就綁好了：`GET /api/lines` 失敗
           （例如 `lines.jsonl` 被手改壞一個字）→ `ROWS` 停在 `[]` →
           使用者看到空列表（跟「還沒打」長得一樣）→ 一按存檔就是清檔。
        2. **「覆寫前備份」只有一份** —— 每次存檔 `copy2` 蓋掉同一個 `.bak`
           ⇒ 第一發空存檔讓 lines=0/bak=2，第二發讓 **bak=0，原稿全滅**。

        現在：空清單直接拒絕、備份帶時間戳保留 `BACKUP_KEEP` 份、
        存檔上鎖（`ThreadingHTTPServer` 允許並發 POST，而原本共用同一個
        `.tmp`/`.bak` 路徑且無鎖 —— 兩個請求會互相截斷）、
        `os.replace` 前 `fsync`、失敗路徑清掉 `.tmp`。
        """
        if not rows:
            # 空清單一定是狀態出錯（前端沒載到資料就被按了存檔），不是意圖。
            return False, ["空清單不寫入 —— 這會清掉整份聽打稿。"
                           "若真的要清空，請直接編輯 lines.jsonl。"]
        errs = golden.validate_lines(rows, self.manifest())
        if errs:
            return False, errs
        with self._lock:
            self._backup()
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".lines-",
                                       suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    # 沒有 fsync 的話「中途斷電不會留半份」對斷電並不成立：
                    # os.replace 只保證改名是原子的，不保證內容已經落盤。
                    os.fsync(fh.fileno())
                os.replace(tmp, self.lines_p)
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)          # 不留截斷的殘檔
                raise
        return True, []

    def _backup(self):
        """把現有檔案複製成帶時間戳的備份，並只留最近 BACKUP_KEEP 份。"""
        if not os.path.exists(self.lines_p):
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dst = "%s.%s.bak" % (self.lines_p, stamp)
        n = 1
        while os.path.exists(dst):          # 同一秒內連續存兩次
            dst = "%s.%s-%d.bak" % (self.lines_p, stamp, n)
            n += 1
        shutil.copy2(self.lines_p, dst)
        base = os.path.basename(self.lines_p)
        olds = sorted(f for f in os.listdir(self.root)
                      if f.startswith(base + ".") and f.endswith(".bak"))
        for f in olds[:-BACKUP_KEEP]:
            try:
                os.unlink(os.path.join(self.root, f))
            except OSError:
                pass                        # 刪不掉備份不該讓存檔失敗

    def clip_path(self, name):
        """只允許取本資料夾內的檔（擋路徑穿越）。

        🔴 2026-09-11 補兩個洞：

        - **`basename` 只是字面收斂**。`exists`／`getsize`／`open` 都跟隨
          symlink ⇒ 在 kit 目錄裡放一個指向外部檔案的 symlink，
          `/clip/leak.m4a` 就會把外部檔案送出去（Codex #15）。
          改成比對 `realpath`，確認實體位置真的在 root 底下。
        - **Windows 的磁碟相對路徑**：`D:secret.txt` 的 basename 還是
          `D:secret.txt`，而 `os.path.join(root, "D:secret.txt")` 在 Windows 上
          會被解讀成 D 磁碟的相對路徑 ⇒ 逃出 root。用 `os.path.splitdrive`
          擋掉帶磁碟代號的名字。
        """
        safe = posixpath.basename((name or "").replace("\\", "/"))
        if not safe or safe in (".", ".."):
            return None
        if os.path.splitdrive(safe)[0] or ":" in safe:
            return None                     # D:secret.txt 這種
        p = os.path.realpath(os.path.join(self.root, safe))
        root = self.root.rstrip(os.sep)
        if not (p == root or p.startswith(root + os.sep)):
            return None                     # symlink 指到外面
        return p if os.path.isfile(p) else None


def make_handler(kit, port=PORT):
    hosts = allowed_hosts(port)

    class H(BaseHTTPRequestHandler):
        # 🔴 必須是 1.1：1.0 之下瀏覽器不發 Range，媒體就 seek 不動。
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass                                   # 不要每個 request 洗版

        def _host_ok(self):
            """Host header 必須是本機名字 —— 擋 DNS rebinding。

            實測 `Host: evil.attacker.com` 打 `GET /api/lines` 會回 200 ＋
            完整逐字稿。只綁 127.0.0.1 擋不住：攻擊者把自己的網域解析到
            127.0.0.1，受害者的瀏覽器就會帶著那個 Host 連進來，
            而同源政策認為那是攻擊者的網域 ⇒ 放行讀取回應內容。
            """
            h = (self.headers.get("Host") or "").strip().lower()
            return h in hosts

        def _origin_ok(self):
            """寫入請求的 Origin 必須是本機 —— 擋 CSRF。

            實測任何網頁都能對 `POST /api/lines` 送 `[]` 並得到
            `{"ok": true}`；配上舊版的單一 `.bak`，兩發就把 1.5 小時的
            人工聽打稿連備份一起滅掉。
            """
            o = self.headers.get("Origin")
            if o is None:                   # 非瀏覽器（curl／測試）沒有 Origin
                return True
            try:
                u = urllib.parse.urlsplit(o)
            except ValueError:
                return False
            return (u.netloc or "").lower() in hosts

        def _deny(self, why):
            return self._send(403, json.dumps({"ok": False, "errors": [why]},
                                              ensure_ascii=False))

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._host_ok():
                return self._deny("Host 不是本機 —— 拒絕（DNS rebinding 防護）")
            path = urllib.parse.unquote(self.path.split("?")[0])
            if path == "/":
                return self._send(200, HTML, "text/html; charset=utf-8")
            if path == "/api/manifest":
                return self._send(200, json.dumps(kit.manifest(), ensure_ascii=False))
            if path == "/api/lines":
                return self._send(200, json.dumps(kit.lines(), ensure_ascii=False))
            if path.startswith("/clip/"):
                p = kit.clip_path(path[len("/clip/"):])
                if not p:
                    return self._send(404, '{"error":"not found"}')
                return self._send_media(p)
            return self._send(404, '{"error":"not found"}')

        def _send_media(self, p):
            """帶 Range 的媒體回應。

            🔴 2026-09-10 修：初版把整支檔以 200 回、且沒有 `Accept-Ranges`。
            瀏覽器在那種回應下**無法在媒體裡 seek** —— 任何 `currentTime` 指派
            都會被打回 0。症狀就是 標註者 回報的「前進後退跟重聽全部跳回開頭」。
            這是伺服器缺陷，不是前端邏輯錯。
            """
            ext = os.path.splitext(p)[1].lower()
            ctype = MIME.get(ext, "application/octet-stream")
            size = os.path.getsize(p)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            partial = False
            if rng and rng.startswith("bytes="):
                spec = rng[6:].split(",")[0].strip()
                a, _, b = spec.partition("-")
                try:
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    elif b:                       # bytes=-N ＝ 最後 N 個 byte
                        start = max(0, size - int(b))
                    partial = True
                except ValueError:
                    partial = False
            if partial and (start >= size or start > end):
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            end = min(end, size - 1)
            length = end - start + 1
            with open(p, "rb") as fh:
                fh.seek(start)
                data = fh.read(length)
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if partial:
                self.send_header("Content-Range",
                                 "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self._host_ok():
                return self._deny("Host 不是本機 —— 拒絕（DNS rebinding 防護）")
            if not self._origin_ok():
                return self._deny("Origin 不是本機 —— 拒絕（CSRF 防護）")
            if self.path != "/api/lines":
                return self._send(404, '{"error":"not found"}')
            # 🔴 `int()` 原本在 try 之外：`Content-Length: abc` 直接讓 handler
            # 拋例外、連線被關，而前端的 `await res.json()` reject ⇒
            # `say()` 沒機會跑 ⇒ **UI 一個字都不顯示**，使用者以為存好了。
            raw_len = (self.headers.get("Content-Length") or "0").strip()
            try:
                n = int(raw_len)
            except ValueError:
                return self._send(400, json.dumps(
                    {"ok": False, "errors": ["Content-Length 不是數字"]},
                    ensure_ascii=False))
            # 負數會讓 `read(-1)` 一路讀到 EOF；沒有上限則是任人塞。
            if n < 0 or n > MAX_BODY:
                return self._send(413, json.dumps(
                    {"ok": False, "errors": ["body 長度 %s 不在允許範圍" % n]},
                    ensure_ascii=False))
            try:
                rows = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                return self._send(400, json.dumps({"ok": False,
                                                   "errors": [str(exc)]},
                                                  ensure_ascii=False))
            if not isinstance(rows, list):
                return self._send(400, json.dumps(
                    {"ok": False, "errors": ["body 必須是 list"]},
                    ensure_ascii=False))
            # 存檔路徑上的任何例外都要變成一個**看得見**的回應。
            try:
                ok, errs = kit.save(rows)
            except Exception as exc:                 # noqa: BLE001
                return self._send(500, json.dumps(
                    {"ok": False, "errors": ["存檔失敗：%s" % exc]},
                    ensure_ascii=False))
            return self._send(200 if ok else 422,
                              json.dumps({"ok": ok, "errors": errs},
                                         ensure_ascii=False))
    return H


HTML = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>黃金段聽打</title><style>
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,"Segoe UI","Noto Sans TC",sans-serif;
 background:#14161a;color:#e8e6e3}
header{padding:10px 16px;background:#1c1f24;border-bottom:1px solid #2c3037;
 display:flex;gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5}
select,input,textarea,button{font:inherit;background:#22262c;color:#e8e6e3;
 border:1px solid #363b44;border-radius:6px;padding:7px 9px}
textarea{width:100%;resize:vertical;min-height:44px}
button{cursor:pointer}button:hover{border-color:#5b6472}
button.p{background:#2f6f4f;border-color:#3c8a63}
button.d{background:#5c2b2b;border-color:#7a3a3a}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:#1a1d22;border:1px solid #2c3037;border-radius:10px;padding:14px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:9px}
label{color:#9aa3ae;font-size:13px;min-width:52px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid #2c3037;padding:5px 6px;vertical-align:top;text-align:left}
tr.sel{background:#243040}
.k{color:#9aa3ae;font-size:12.5px}
kbd{background:#2b3038;border:1px solid #3c424c;border-radius:4px;padding:1px 5px;font-size:12px}
#msg{min-height:20px;font-size:13px}
.err{color:#ff8b8b}.ok{color:#7fd6a2}
.nat{display:none}.nat.on{display:block}
canvas#wave{width:100%;height:110px;display:block;margin-top:8px;background:#0f1216;
 border:1px solid #2c3037;border-radius:8px;cursor:crosshair;touch-action:none}
</style></head><body>
<header>
 <b>黃金段聽打</b>
 <select id="seg"></select>
 <button id="play" class="p">▶ 播放 / 暫停</button>
 <span class="k">速度</span><select id="rate">
  <option>0.5</option><option>0.75</option><option selected>1</option><option>1.25</option></select>
 <span id="pos" class="k">0.0s</span>
 <span id="msg"></span>
</header>
<main>
 <div class="card">
  <audio id="au" preload="auto" controls style="width:100%"></audio>
  <canvas id="wave" height="110"></canvas>
  <div class="k" id="wavehint">波形載入中…</div>
  <div class="row" style="margin-top:10px">
   <button id="mark">⏱ 標起點 <kbd>[</kbd></button>
   <button id="end">⏱ 標終點 <kbd>]</kbd></button>
   <label>起</label><input id="ts" size="7">
   <label>訖</label><input id="te" size="7">
   <button id="replay">↺ 重聽這句</button>
  </div>
  <div class="row">
   <label>講者</label><select id="spk"></select>
   <label>語言</label><select id="lang">
    <option value="zho">國語 zho</option><option value="nan">台語 nan</option>
    <option value="mixed">夾雜 mixed</option><option value="eng">英語 eng</option></select>
  </div>
  <div class="row"><label>內容</label>
   <span class="k">一律打<b>國語意思</b>。台語的句子只要把「語言」選成台語，
   <b>不用打台語字</b></span></div>
  <textarea id="zh" placeholder="聽到什麼，用國語寫下來"></textarea>
  <div id="prev" class="k" style="margin-top:6px"></div>
  <div class="row" style="margin-top:6px">
   <button id="manual">＋ 台語原形（選填，打得出來才填）</button></div>
  <div id="natwrap" class="nat">
   <textarea id="native" placeholder="打不出來就留空，不影響評分"></textarea></div>
  <div class="row" style="margin-top:8px"><label>型號</label>
   <input id="models" style="flex:1" placeholder="這句話裡**唸出來**的型號，逗號分隔"></div>
  <div class="row"><label>主題</label>
   <input id="topic" style="flex:1" placeholder="這段在討論哪個產品（沒唸出來也填這裡）"></div>
  <div class="row" style="margin-top:6px">
   <button id="add" class="p">✚ 新增這句 <kbd>Ctrl</kbd>+<kbd>Enter</kbd></button>
   <button id="upd">✎ 更新選取列</button>
   <button id="clr">清空欄位</button>
  </div>
  <p class="k">播放/暫停 <kbd>空白</kbd>　退 2 秒 <kbd>←</kbd>　進 2 秒 <kbd>→</kbd>
   微調 0.5 秒 <kbd>Shift</kbd>+方向　標起點 <kbd>[</kbd>　標終點 <kbd>]</kbd></p>
 </div>
 <div class="card">
  <div class="row"><b>已打的句子</b><span id="cnt" class="k"></span>
   <span style="flex:1"></span><button id="save" class="p">💾 存檔</button></div>
  <table><thead><tr><th>id</th><th>起–訖</th><th>講者</th><th>語言</th><th>內容</th><th></th></tr>
  </thead><tbody id="tb"></tbody></table>
 </div>
</main>
<script>
const $=s=>document.querySelector(s);
let MAN=null,SEG=null,ROWS=[],SELID=null;
const au=$('#au');

function segOf(id){return MAN.segments.find(s=>s.id===id);}
function abs(t){return +(SEG.t_start+t).toFixed(2);}
function rel(t){return +(t-SEG.t_start).toFixed(2);}

async function boot(){
 MAN=await (await fetch('/api/manifest')).json();
 ROWS=await (await fetch('/api/lines')).json();
 LOADED=true;   // 走到這裡才代表 ROWS 真的是檔案裡的內容
 $('#seg').innerHTML=MAN.segments.map(s=>`<option value="${s.id}">${s.id}　${fmt(s.t_start)}–${fmt(s.t_end)}</option>`).join('');
 // 講者用真名冊：標註者 認得這些人的聲音，而要他在五段互不相連的片段之間
 // 維持 S1..S12 的一致對應並不合理（他 2026-09-10 直接問「我怎麼分 S1~S12」）。
 const sp=(MAN.speakers&&MAN.speakers.length?MAN.speakers.slice():
           Array.from({length:12},(_,i)=>'S'+(i+1))).concat(['S?']);
 $('#spk').innerHTML=sp.map(s=>`<option>${esc(s)}</option>`).join('');
 pick(MAN.segments[0].id); render();
}
function fmt(s){s=Math.round(s);return `${Math.floor(s/3600)}:${String(Math.floor(s%3600/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;}
function pick(id){
 SEG=segOf(id);$('#seg').value=id;
 const url='/clip/'+encodeURIComponent(SEG.clip);
 au.src=url; loadWave(url); render();
}


/* ── 可拖曳的波形磁軌 ────────────────────────────────────────────────
   標註者 2026-09-10 回報「前進後退跟重聽全部跳回開頭，完全失效」。
   根因在**伺服器**（不回 Range → 瀏覽器無法 seek），已在 _send_media 修掉。
   另外初版 <audio> 忘了給 controls，所以他從頭到尾沒有播放軸可拉。
   這條磁軌是在那之上加的：看得到句子邊界，才不用靠盲聽反覆進退。 */
let PEAKS=null, WCTX=null;
const cv=$('#wave');

async function loadWave(url){
 PEAKS=null; $('#wavehint').textContent='波形載入中…'; draw();
 try{
  const buf=await (await fetch(url)).arrayBuffer();
  WCTX=WCTX||new (window.AudioContext||window.webkitAudioContext)();
  const ab=await WCTX.decodeAudioData(buf.slice(0));
  const N=1200, ch=ab.getChannelData(0), step=Math.floor(ch.length/N)||1;
  const pk=new Float32Array(N);
  for(let i=0;i<N;i++){
   let m=0; const a=i*step, b=Math.min(ch.length,a+step);
   for(let j=a;j<b;j++){const v=Math.abs(ch[j]); if(v>m)m=v;}
   pk[i]=m;
  }
  PEAKS=pk;
  $('#wavehint').innerHTML='點一下跳到那裡、按住拖曳可搜尋　'+
   '<span style="color:#7fd6a2">■</span> 已標句子　'+
   '<span style="color:#e0b34a">■</span> 這一句的起訖';
 }catch(e){
  $('#wavehint').innerHTML='<span class="err">波形畫不出來（'+e.name+
   '）—— 不影響聽打，用上面的播放列即可</span>';
 }
 draw();
}

function draw(){
 const dpr=window.devicePixelRatio||1;
 const w=cv.clientWidth||800, h=110;
 if(cv.width!==Math.round(w*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
 const c=cv.getContext('2d');
 c.setTransform(dpr,0,0,dpr,0,0);
 c.clearRect(0,0,w,h);
 const dur=(au.duration&&isFinite(au.duration))?au.duration:60;
 const X=t=>Math.max(0,Math.min(w,t/dur*w));

 // 已存的句子（本段）
 if(SEG){
  c.fillStyle='rgba(127,214,162,0.18)';
  ROWS.filter(r=>r.seg===SEG.id).forEach(r=>{
   const a=X(rel(r.t_start)), b=X(rel(r.t_end));
   c.fillRect(a,0,Math.max(1.5,b-a),h);
  });
 }
 // 目前這一句的起訖
 const ts=parseFloat($('#ts').value), te=parseFloat($('#te').value);
 if(!isNaN(ts)){
  const a=X(rel(ts)), b=isNaN(te)?a+2:X(rel(te));
  c.fillStyle='rgba(224,179,74,0.22)';
  c.fillRect(a,0,Math.max(2,b-a),h);
  c.fillStyle='#e0b34a'; c.fillRect(a-1,0,2,h);
  if(!isNaN(te)) c.fillRect(b-1,0,2,h);
 }
 // 波形
 if(PEAKS){
  c.fillStyle='#4d7fa8';
  const n=PEAKS.length;
  for(let i=0;i<n;i++){
   const x=i/n*w, bh=Math.max(1,PEAKS[i]*(h-14));
   c.fillRect(x,(h-bh)/2,Math.max(1,w/n),bh);
  }
 }else{
  c.strokeStyle='#2c3037'; c.beginPath(); c.moveTo(0,h/2); c.lineTo(w,h/2); c.stroke();
 }
 // 每 10 秒刻度（絕對秒數）
 c.fillStyle='#6b7480'; c.font='10px monospace';
 for(let t=0;t<=dur;t+=10){
  const x=X(t); c.fillRect(x,h-9,1,9);
  if(SEG) c.fillText(fmt(SEG.t_start+t).slice(2),Math.min(x+3,w-42),h-1);
 }
 // 播放頭
 const x=X(au.currentTime||0);
 c.fillStyle='#ff6b6b'; c.fillRect(x-1,0,2,h);
}

function seekFromEvent(ev){
 const r=cv.getBoundingClientRect();
 const dur=(au.duration&&isFinite(au.duration))?au.duration:60;
 const t=Math.max(0,Math.min(dur,(ev.clientX-r.left)/r.width*dur));
 au.currentTime=t; draw();
}
let DRAG=false;
cv.addEventListener('pointerdown',e=>{DRAG=true;cv.setPointerCapture(e.pointerId);seekFromEvent(e);});
cv.addEventListener('pointermove',e=>{if(DRAG)seekFromEvent(e);});
cv.addEventListener('pointerup',e=>{DRAG=false;});
cv.addEventListener('pointercancel',()=>{DRAG=false;});
window.addEventListener('resize',draw);
(function tick(){draw();requestAnimationFrame(tick);})();

$('#seg').onchange=e=>pick(e.target.value);
$('#play').onclick=()=>au.paused?au.play():au.pause();
$('#rate').onchange=e=>au.playbackRate=+e.target.value;
au.ontimeupdate=()=>$('#pos').textContent=abs(au.currentTime).toFixed(1)+'s';
$('#mark').onclick=()=>$('#ts').value=abs(au.currentTime);
$('#end').onclick=()=>$('#te').value=abs(au.currentTime);
$('#replay').onclick=()=>{const s=parseFloat($('#ts').value);if(!isNaN(s)){au.currentTime=rel(s);au.play();}};
/* 🔴 2026-09-10 改：原本要人把台語寫成「國語意思（台語原形）」再自動拆兩軌。
   作廢原因 —— 標註者「不會打台語的中打」，而由 CC 從國語回譯來補那一欄
   ＝ 自己製造 ground truth（引擎寫出另一個同樣正確的台語形會被判錯）。
   改法：人只標「這句是台語」（他聽得出來），台語忠實度改用 score.py 的
   taigi_retention —— 看引擎在那些句子上有沒有吐台語專屬字形，不需要標準答案。
   native 保留為選填：打得出來就多一份證據，打不出來不罰。 */
let MANUAL=false;
$('#manual').onclick=()=>{MANUAL=!MANUAL;$('#natwrap').classList.toggle('on',MANUAL);preview();};
function preview(){
 const lang=$('#lang').value, zh=$('#zh').value.trim(), nat=$('#native').value.trim();
 let h=zh?`${zh.length} 字`:'<i>（空）</i>';
 if(lang==='nan') h+=' 　<b>台語句</b> —— 內容打國語就好，台語字不用打';
 if(nat) h+=`　│　原形：${esc(nat)}`;
 $('#prev').innerHTML=h;
}
$('#zh').oninput=preview;
$('#native').oninput=preview;
$('#lang').onchange=preview;

// 🔴 2026-09-11：原本是「當前筆數＋1」⇒ 打了 001/002/003、刪掉 002 之後
// 剩 2 筆，下一個 id 就是 003 ＝ **撞號**。使用者的處境是連打十幾句之後
// 按存檔，被一句「id 重複」擋住，而 UI 沒有改 id 的欄位。
// 改成取該段既有 id 尾碼的最大值 +1。
function nextId(){
 let mx=0;
 for(const r of ROWS){
  if(r.seg!==SEG.id) continue;
  const m=/-(\d+)$/.exec(r.id||'');
  if(m) mx=Math.max(mx,parseInt(m[1],10));
 }
 return SEG.id+'-'+String(mx+1).padStart(3,'0');
}
function form(){
 const ts=parseFloat($('#ts').value),te=parseFloat($('#te').value);
 const lang=$('#lang').value;
 const nat=$('#native').value.trim();
 const r={id:SELID||nextId(),seg:SEG.id,t_start:ts,t_end:te,spk:$('#spk').value,
  lang:lang,zh:$('#zh').value.trim(),
  models:$('#models').value.split(/[,，]/).map(x=>x.trim()).filter(Boolean),note:''};
 if(nat) r.native=nat;
 const tp=$('#topic').value.trim();
 if(tp) r.topic=tp;
 return r;
}
function say(t,cls){$('#msg').className=cls||'';$('#msg').textContent=t;}
function localCheck(r){
 const e=[];
 if(isNaN(r.t_start)||isNaN(r.t_end))e.push('起訖要是數字（用 [ 和 ] 標）');
 else if(r.t_end<=r.t_start)e.push('訖要大於起');
 if(!r.zh)e.push('內容不可空');
 return e;
}
$('#add').onclick=()=>{
 const r=form(),e=localCheck(r);
 if(e.length)return say(e.join('；'),'err');
 r.id=nextId(); ROWS.push(r); SELID=null; clear(); render(); say('已加入 '+r.id,'ok');
};
$('#upd').onclick=()=>{
 if(!SELID)return say('沒有選取的列','err');
 const r=form(),e=localCheck(r);
 if(e.length)return say(e.join('；'),'err');
 const i=ROWS.findIndex(x=>x.id===SELID); ROWS[i]=r; SELID=null; clear(); render(); say('已更新','ok');
};
$('#clr').onclick=()=>{SELID=null;clear();render();};
function clear(){['ts','te','zh','native','models','topic'].forEach(k=>$('#'+k).value='');preview();}

function render(){
 const rs=ROWS.filter(r=>r.seg===(SEG?SEG.id:null)).sort((a,b)=>a.t_start-b.t_start);
 $('#cnt').textContent=`本段 ${rs.length} 句／全部 ${ROWS.length} 句`;
 $('#tb').innerHTML=rs.map(r=>`<tr class="${r.id===SELID?'sel':''}">
  <td>${esc(r.id)}</td><td>${r.t_start}–${r.t_end}</td><td>${esc(r.spk)}</td><td>${esc(r.lang)}</td>
  <td>${esc(r.zh)}${r.native?'<br><span class="k">'+esc(r.native)+'</span>':''}
   ${(r.models&&r.models.length)?'<br><span style="color:#e0b34a">型號 '+esc(r.models.join(', '))+'</span>':''}
   ${r.topic?'<br><span class="k">主題 '+esc(r.topic)+'</span>':''}</td>
  <td><button data-e="${esc(r.id)}">✎</button> <button class="d" data-x="${esc(r.id)}">✕</button></td></tr>`).join('');
 $('#tb').querySelectorAll('[data-e]').forEach(b=>b.onclick=()=>edit(b.dataset.e));
 $('#tb').querySelectorAll('[data-x]').forEach(b=>b.onclick=()=>{
  // 刪一列 ＝ 刪掉一句人工聽打。問一次。
  if(!confirm('刪掉 '+b.dataset.x+'？')) return;
  ROWS=ROWS.filter(r=>r.id!==b.dataset.x);render();});
}
// 內插進 HTML 的每一個欄位都要過這裡，**包含 id** —— 2026-09-11 審計：
// id 原本直接進 <td> 與 data-e=，而 validate_lines 接受
// `<img src=x onerror=...>` 當 id ⇒ 存起來的 XSS。引號也要轉，
// 因為 id 會被放進屬性值裡。
function esc(s){return String(s==null?'':s).replace(/[<>&"']/g,
 c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));}
function edit(id){
 const r=ROWS.find(x=>x.id===id);SELID=id;
 $('#ts').value=r.t_start;$('#te').value=r.t_end;$('#spk').value=r.spk;
 $('#lang').value=r.lang;
 $('#zh').value=r.zh||'';$('#native').value=r.native||'';
 // 🔴 2026-09-10 修：初版 edit() 漏了這兩行，於是按 ✎ 之後型號欄是空的，
 // 使用者沒重打就按更新 → **那列原本的型號被靜默清掉**。
 // 標註者 回報「編輯選取列新增型號加不進去」就是這個。
 $('#models').value=(r.models||[]).join(', ');
 $('#topic').value=r.topic||'';
 if(r.native&&!MANUAL){MANUAL=true;$('#natwrap').classList.add('on');}
 preview();
 au.currentTime=rel(r.t_start); render();
}
// 🔴 存檔鈕原本在 boot() 之前就綁好，而 boot() 只要在第一個 await 丟例外
// （例如 lines.jsonl 被手改壞一個字 → GET /api/lines 回 500），
// ROWS 就停在 []，而畫面上的「空列表」跟「還沒打」長得一模一樣 ⇒
// 一按存檔就清檔。LOADED 讓「還沒載成功」與「真的是空的」分得開。
let LOADED=false;
$('#save').onclick=async()=>{
 if(!LOADED) return say('資料還沒載入完成，先重新整理 —— 這時候存檔會清掉現有稿子','err');
 let j;
 try{
  const res=await fetch('/api/lines',{method:'POST',
   headers:{'Content-Type':'application/json'},body:JSON.stringify(ROWS)});
  j=await res.json();
 }catch(e){
  // 例外也要說出來。原本連線被關時 say() 根本沒機會跑，使用者以為存好了。
  return say('存檔沒有完成：'+e,'err');
 }
 say(j.ok?'已存檔 lines.jsonl（'+ROWS.length+' 句）':'存檔被擋下：'+(j.errors||[]).join('；'),
     j.ok?'ok':'err');
};
document.onkeydown=e=>{
 const typing=/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
 if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();return $('#add').onclick();}
 if(typing)return;
 const step=e.shiftKey?0.5:2;
 if(e.code==='Space'){e.preventDefault();$('#play').onclick();}
 else if(e.key==='ArrowLeft'){e.preventDefault();au.currentTime=Math.max(0,au.currentTime-step);}
 else if(e.key==='ArrowRight'){e.preventDefault();au.currentTime+=step;}
 else if(e.key==='['){$('#mark').onclick();}
 else if(e.key===']'){$('#end').onclick();}
};
boot();
</script></body></html>"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit("用法: python transcribe_ui.py \"<黃金段資料夾>\"")
    kit = Kit(argv[0])
    srv = ThreadingHTTPServer((HOST, PORT), make_handler(kit, PORT))
    url = "http://%s:%d" % (HOST, PORT)
    print("聽打工具已啟動：%s" % url)
    print("資料夾：%s" % kit.root)
    print("關閉：回到這個視窗按 Ctrl+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已關閉")
    return 0


if __name__ == "__main__":
    sys.exit(main())
