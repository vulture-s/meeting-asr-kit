# -*- coding: utf-8 -*-
"""`subprocess` 未指定編碼的偵測器 —— **這是唯一定義**。

## 為什麼要抽成一支

2026-09-11 審計抓到兩道閘各自有一份實作，而且不同步：

- `tests/core/test_subprocess_encoding_ratchet.py`（全 repo ratchet）
- `tests/acoustic/test_subprocess_encoding.py`（meeting-transcribe 零容忍）

後者漏了 `check_call`，於是**零容忍那支反而比 ratchet 鬆**。兩份實作會漂，
這是今天第三次踩到同一條（`window_hyp` → `_touches` 是前兩次）。

## 偵測的是什麼

不是「有沒有寫 `subprocess.run(..., text=True)` 這個字面形狀」，
而是「**這個呼叫會不會拿系統 locale 去解碼**」。原本的判定量錯維度，
以下寫法全部綠燈通過（兩軌審計各自實測；完整語料見本檔 `BYPASS_CORPUS`，12 種）：

    from subprocess import run  →  run(..., text=True)
    import subprocess as sp     →  sp.run(..., text=True)
    subprocess.run(..., text=True, encoding=None)
    subprocess.run(..., text=FLAG)
    subprocess.run(..., universal_newlines=True)
    subprocess.getoutput(...) / getstatusoutput(...)
    os.popen(...).read()
    subprocess.call(..., text=True)
    UTF-8 BOM 檔（ast.parse 炸 → 被 except 吞掉）
    宣告 coding: cp950 的檔（用 utf-8 讀會炸 → 同樣被吞掉）

## 為什麼這個 bug 貴：它靜默

    subprocess.run(["git","log","-1","--format=%s"], capture_output=True, text=True)
    → UnicodeDecodeError: 'cp950' codec can't decode byte 0x89   （在 reader thread）
    → r.stdout = None     （不是空字串）
    → r.returncode = 0    （照樣成功）

呼叫端看到「成功、但沒有輸出」。生產機（標註者 的 PC）正是 cp950。

手動跑：`python tests/core/subprocess_encoding_lint.py`
"""
import ast
import io
import os
import sys
import tokenize

def _find_root(start):
    """往上找 repo 根 —— 以「有 .git 或同時有 tests/ 與 apps/」為判準。

    這支會出現在兩種版面：本 repo 的 `tests/core/`，以及消毒後獨立 kit 的根目錄。
    寫死「往上三層」在後者會指到 repo 外面（`publish_kit --run-tests` 抓到的）。
    """
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(5):
        if (os.path.isdir(os.path.join(d, ".git"))
                or (os.path.isdir(os.path.join(d, "tests"))
                    and os.path.isdir(os.path.join(d, "apps")))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # 獨立 kit：這支就在根目錄旁邊
    return os.path.dirname(os.path.abspath(start))


REPO_ROOT = _find_root(__file__)
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".claude",
             "site-packages"}

# 這些呼叫**沒有 encoding 參數可給** ⇒ 一律走 locale，一律算違規。
ALWAYS_LOCALE = ("getoutput", "getstatusoutput")
# 這些可以給 encoding ⇒ 只有「走文字模式且沒給有效 encoding」才算違規。
ENCODABLE = ("run", "Popen", "check_output", "check_call", "call")

FIX_HINT = ("修法：該呼叫加 `encoding=\"utf-8\", errors=\"replace\"`；"
            "`getoutput`／`os.popen` 沒有這個參數，改用 `subprocess.run`。")


# ── AST 判定 ────────────────────────────────────────────────────────

def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def encoding_is_effective(call):
    """`encoding=` 有沒有真的給出一個編碼。

    🔴 `encoding=None` 在 AST 裡是 `ast.Constant(None)`，**不是** Python 的 `None`
    ⇒ 用 `_kw(call, "encoding") is None` 判「沒給」會把顯式的 `encoding=None`
    當成已指定而放行，而它的行為跟完全沒給一模一樣（都走 locale）。
    """
    v = _kw(call, "encoding")
    if v is None:
        return False
    if isinstance(v, ast.Constant) and v.value is None:
        return False
    return True


def text_mode_possible(call):
    """會不會走文字模式 —— **保守判定**。

    `text=True` 之外，`text=FLAG`（變數／運算式）也算：靜態讀不到它的值，
    而一道閘的預設必須偏向「可能有病就報」。只有**字面**的假值
    （`False`／`0`／`None`）才算確定不是文字模式。
    """
    for name in ("text", "universal_newlines"):
        v = _kw(call, name)
        if v is None:
            continue
        if isinstance(v, ast.Constant) and not v.value:
            continue                      # 字面 False / 0 / None
        return True
    return False


def imported_names(tree):
    """回 (subprocess 模組別名, from-import 的函式名→原名, os 模組別名)。

    原本寫死 `fu.value.id == "subprocess"` ⇒ `import subprocess as sp` 與
    `from subprocess import run` 兩種寫法完全看不見，而它們重現同一個缺陷。
    """
    mods, funcs, osmods = set(), {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "subprocess":
                    mods.add(a.asname or a.name)
                elif a.name == "os":
                    osmods.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for a in node.names:
                funcs[a.asname or a.name] = a.name
    return mods, funcs, osmods


def violations(src, filename="<string>"):
    """回 [(行號, 呼叫名, 理由)]。`src` 是原始碼字串。"""
    tree = ast.parse(src, filename=filename)
    mods, funcs, osmods = imported_names(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fu, fn = node.func, None
        if isinstance(fu, ast.Attribute) and isinstance(fu.value, ast.Name):
            if fu.value.id in mods:
                fn = fu.attr
            elif fu.value.id in osmods and fu.attr == "popen":
                fn = "popen"
        elif isinstance(fu, ast.Name):
            fn = funcs.get(fu.id)
        if fn is None:
            continue
        if fn == "popen" or fn in ALWAYS_LOCALE:
            out.append((node.lineno, fn, "恆走 locale（沒有 encoding 參數可給）"))
            continue
        if fn not in ENCODABLE:
            continue
        if text_mode_possible(node) and not encoding_is_effective(node):
            why = ("encoding=None 等於沒給" if _kw(node, "encoding") is not None
                   else "文字模式但沒給 encoding")
            out.append((node.lineno, fn, why))
    return out


def count_violations(src, filename="<string>"):
    return len(violations(src, filename))


# ── 檔案層 ──────────────────────────────────────────────────────────

def read_source(path):
    """依檔案自己宣告的編碼讀它 —— 讀不到就讓例外冒出來。

    🔴 原本兩道閘都是 `io.open(p, encoding="utf-8")` ＋
    `except (SyntaxError, UnicodeDecodeError): continue`，那是 **fail-open**：

      - UTF-8 **BOM** 檔用 `utf-8` 讀會留下 `\\ufeff`，`ast.parse` 丟 SyntaxError
        → 被吞掉。而 Python 自己的 tokenizer 認它是 `utf-8-sig`，
        那個檔**照樣跑得起來**，裡面的違規也照樣會炸。
      - 宣告 `coding: cp950` 的 `.py` 直接解碼失敗被跳過 ——
        而那正是最可能中招的那種檔。

    `tokenize.detect_encoding` 用的就是 Python 自己那套規則（BOM ＋ coding 宣告）。
    """
    with open(path, "rb") as fh:
        enc = tokenize.detect_encoding(fh.readline)[0]
    return io.open(path, encoding=enc).read()


def iter_py(root=None):
    root = root or REPO_ROOT
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in sorted(fn):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                yield p, os.path.relpath(p, root).replace("\\", "/")


def scan_repo(root=None):
    """回 {相對路徑: 違規數}。掃不動的檔另由 `scan_unscannable()` 報。"""
    out = {}
    for path, rel in iter_py(root):
        try:
            n = count_violations(read_source(path), path)
        except Exception:                            # noqa: BLE001
            continue                                 # 交給 scan_unscannable 判紅
        if n:
            out[rel] = n
    return out


def scan_unscannable(root=None):
    """回 [(相對路徑, 理由)] —— 解不開或 parse 不了的 .py。

    這個清單必須是空的。它不空代表這道閘對那些檔**沒有意見**，
    而「沒有意見」跟「沒有違規」是兩件事。
    """
    bad = []
    for path, rel in iter_py(root):
        try:
            count_violations(read_source(path), path)
        except Exception as e:                       # noqa: BLE001
            bad.append((rel, "%s: %s" % (type(e).__name__, e)))
    return bad


# ── 兩道閘共用的繞道語料 ────────────────────────────────────────────
# 放在這裡而不是各自的測試檔裡：兩道閘都要對同一組語料負責，
# 否則「零容忍那支比 ratchet 鬆」會再發生一次。

BYPASS_CORPUS = (
    ("原形（基線就是這個形狀）",
     "import subprocess\n"
     "subprocess.run(['git', 'log'], capture_output=True, text=True)\n"),
    ("模組別名",
     "import subprocess as sp\n"
     "sp.run(['git', 'log'], capture_output=True, text=True)\n"),
    ("from-import",
     "from subprocess import run\n"
     "run(['git', 'log'], capture_output=True, text=True)\n"),
    ("from-import ＋ 改名",
     "from subprocess import run as r\n"
     "r(['git', 'log'], capture_output=True, text=True)\n"),
    ("顯式 encoding=None（跟沒給一樣）",
     "import subprocess\n"
     "subprocess.run(['git'], text=True, encoding=None)\n"),
    ("text 是變數，靜態讀不到值 ⇒ 保守算違規",
     "import subprocess\n"
     "FLAG = True\n"
     "subprocess.run(['git'], text=FLAG)\n"),
    ("universal_newlines（text 的舊名）",
     "import subprocess\n"
     "subprocess.run(['git'], universal_newlines=True)\n"),
    ("getoutput（沒有 encoding 參數可給）",
     "import subprocess\n"
     "subprocess.getoutput('git log')\n"),
    ("getstatusoutput",
     "import subprocess\n"
     "subprocess.getstatusoutput('git log')\n"),
    ("os.popen().read()",
     "import os\n"
     "os.popen('git log').read()\n"),
    ("subprocess.call(text=True)",
     "import subprocess\n"
     "subprocess.call(['git'], text=True)\n"),
    ("check_call(text=True)（零容忍那支原本漏掉這個）",
     "import subprocess\n"
     "subprocess.check_call(['git'], text=True)\n"),
)

CLEAN_CORPUS = (
    ("有給 encoding",
     "import subprocess\n"
     "subprocess.run(['git'], text=True, encoding='utf-8')\n"),
    ("bytes 模式（沒給 text／universal_newlines）",
     "import subprocess\n"
     "subprocess.run(['git'], capture_output=True)\n"),
    ("字面 text=False",
     "import subprocess\n"
     "subprocess.run(['git'], text=False)\n"),
    ("別名 ＋ 有給 encoding",
     "import subprocess as sp\n"
     "sp.run(['git'], text=True, encoding='utf-8')\n"),
    ("同名但不是 subprocess 的函式",
     "from shutil import which as run\n"
     "run('git')\n"),
    ("自己寫的 run（沒 import subprocess）",
     "def run(cmd, text=True):\n"
     "    return cmd\n"
     "run(['git'], text=True)\n"),
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = argv[0] if argv else REPO_ROOT
    bad = scan_unscannable(root)
    found = scan_repo(root)
    for rel in sorted(found):
        print("%4d  %s" % (found[rel], rel))
    print("")
    print("合計 %d 檔 / %d 處" % (len(found), sum(found.values())))
    if bad:
        print("")
        print("🔴 掃不動的檔 %d 支（閘對它們沒有意見 ≠ 它們沒有違規）：" % len(bad))
        for rel, why in bad:
            print("  %s  %s" % (rel, why))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
