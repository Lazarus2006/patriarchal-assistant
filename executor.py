"""
executor.py — registry-driven tool executor for the local voice assistant.

The model emits intent JSON. This module resolves it via registry.json and
returns a confirmation string for Piper TTS. There are NO per-site or per-app
functions here: every verb is generic and reads its data from the registry.

Usage from the FastAPI router (Node 1):

    from executor import execute

    @app.post("/command")
    def command(req: Req):            # plain `def` => FastAPI runs it in a threadpool.
        intent = call_ollama(req.text)  # dict parsed from the model's JSON
        return {"speech": execute(intent)}

    # NOTE: keep this a sync endpoint (or use run_in_threadpool). execute() blocks
    # for up to a few seconds (Brave launch, yt-dlp) and Playwright's sync API
    # refuses to run inside a running asyncio loop.

CLI smoke test:
    python executor.py '{"action":"tool","tool":"open_site","args":{"site":"whatsapp"}}'

Requires (system): brave-browser, yt-dlp, pactl, gtk-launch (libgtk-3-bin), mpv, vlc, code
Optional:          pip install playwright   (ONLY for run_recipe; no `playwright install`
                   needed because we attach to Brave over CDP, not download a browser)
Optional:          wmctrl (verified installed; raises Brave on X11 + `window` verb on Xorg sessions)
Optional:          xdotool (Xorg `window minimize`), wtype or ydotool (COSMIC/Wayland `window` verb)
Python >= 3.10 (Pop!_OS 22.04 ships 3.10, 24.04 ships 3.12).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger("executor")

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = Path(os.environ.get("ASSISTANT_REGISTRY", HERE / "registry.json"))

BRAVE_BOOT_TIMEOUT_S = 10.0   # how long to wait for the CDP port after launching Brave
YTDLP_TIMEOUT_S = 25          # ASSUMPTION: the ProBook (i5-2520M) needs ~3-8 s for a yt-dlp search


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gui_env() -> dict[str, str]:
    """
    Environment for child processes. ASSUMPTION: if the router runs as a systemd
    *user* service or from SSH, DISPLAY/XDG_RUNTIME_DIR may be missing, and then
    pactl and GUI apps silently fail. Fill sane defaults without overriding.
    The executor must run on the machine whose desktop you want to control, logged
    into a graphical session.
    """
    env = os.environ.copy()
    runtime = env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")
    env.setdefault("DISPLAY", ":0")
    if "WAYLAND_DISPLAY" not in env:
        for name in ("wayland-1", "wayland-0"):
            if Path(runtime, name).exists():
                env["WAYLAND_DISPLAY"] = name
                break
    return env


def _spawn(cmd: list[str]) -> subprocess.Popen | None:
    """Start a detached GUI/background process. Never raises."""
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,      # survive router restarts, no zombie signal coupling
            env=_gui_env(),
        )
    except OSError as exc:
        log.warning("spawn failed %s: %s", cmd, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# OS detection + registry
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def detect_popos_version() -> str:
    """
    Returns "22.04" (GNOME) or "24.04" (COSMIC). Detection order:
      1. ASSISTANT_OS_VERSION env override (handy for testing)
      2. /etc/os-release VERSION_ID
      3. presence of the `cosmic-session` binary
      4. default "22.04"
    ASSUMPTION: Pop 24.04 == COSMIC session; a 24.04 install running a GNOME
    session is not handled (set ASSISTANT_OS_VERSION=22.04 to force GNOME names).
    """
    override = os.environ.get("ASSISTANT_OS_VERSION")
    if override in ("22.04", "24.04"):
        return override

    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("VERSION_ID="):
                vid = line.split("=", 1)[1].strip().strip('"')
                if vid.startswith("24"):
                    return "24.04"
                if vid.startswith("22"):
                    return "22.04"
    except OSError:
        pass

    if shutil.which("cosmic-session") or Path("/usr/bin/cosmic-session").exists():
        return "24.04"
    return "22.04"


def _norm(text: Any) -> str:
    """'YouTube Music!' -> 'youtube_music'. Used for all key/alias matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


@lru_cache(maxsize=1)
def load_registry() -> dict:
    """
    Load registry.json ONCE (cached for the process lifetime), attach:
        reg["_os"]    -> "22.04" | "24.04"
        reg["_index"] -> {section: {normalized alias: canonical key}}
    Restart the router (or call load_registry.cache_clear()) after editing the JSON.
    """
    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        reg = json.load(fh)

    reg["_os"] = detect_popos_version()
    index: dict[str, dict[str, str]] = {"sites": {}, "apps": {}, "media": {}, "window": {}, "toggles": {}}
    for section in index:
        if section == "window":
            specs = reg.get("window", {}).get("ops", {})
        else:
            specs = reg.get(section, {})
        for key, spec in specs.items():
            if key.startswith("_"):
                continue
            for alias in [key, spec.get("name", ""), *spec.get("aliases", [])]:
                n = _norm(alias)
                if n:
                    index[section].setdefault(n, key)
    reg["_index"] = index
    log.info("registry loaded (os=%s, sites=%d, apps=%d)", reg["_os"],
            len(index["sites"]), len(index["apps"]))
    return reg


def _lookup(reg: dict, section: str, text: Any) -> str | None:
    """Resolve free text from the model to a registry key, or None. Never invents."""
    n = _norm(text)
    idx = reg["_index"][section]
    if n in idx:
        return idx[n]
    if n.startswith("the_") and n[4:] in idx:
        return idx[n[4:]]
    return None


def _not_configured(raw: Any) -> str:
    name = str(raw or "").strip()[:40]
    return f"I don't have {name} configured yet." if name else "I don't have that configured yet."


def _fmt(template: str, args: dict, quote: bool) -> str:
    """Fill {placeholders}. quote=True URL-encodes values (for URLs)."""
    def sub(m: re.Match) -> str:
        val = str(args.get(m.group(1), ""))
        return urllib.parse.quote_plus(val) if quote else val
    return re.sub(r"\{(\w+)\}", sub, template)


# ──────────────────────────────────────────────────────────────────────────────
# Brave + Chrome DevTools Protocol (plain HTTP endpoints — no extra deps)
# ──────────────────────────────────────────────────────────────────────────────
# /json            list targets          /json/activate/<id>   focus a tab
# /json/new?<url>  open tab (PUT)        /json/close/<id>      close a tab

def _browser_cfg() -> dict:
    return load_registry()["browser"]


def _cdp(path: str, method: str = "GET", timeout: float = 2.0) -> Any:
    url = f"http://127.0.0.1:{_browser_cfg()['cdp_port']}{path}"
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    try:
        return json.loads(body)
    except ValueError:
        return body


def brave_cdp_alive() -> bool:
    try:
        _cdp("/json/version", timeout=1.0)
        return True
    except (OSError, ValueError):
        return False


def brave_list_tabs() -> list[dict]:
    try:
        targets = _cdp("/json")
    except (OSError, ValueError):
        return []
    return [
        t for t in targets
        if t.get("type") == "page"
        and not str(t.get("url", "")).startswith(("devtools://", "chrome-extension://"))
    ]


def brave_find_tab(url_prefix: str) -> dict | None:
    """First page-tab whose URL *starts with* url_prefix (WhatsApp Web appends paths)."""
    for tab in brave_list_tabs():
        if str(tab.get("url", "")).startswith(url_prefix):
            return tab
    return None


def brave_activate_tab(tab_id: str) -> bool:
    try:
        _cdp(f"/json/activate/{tab_id}")
        return True
    except (OSError, ValueError):
        return False


def brave_open_tab(url: str) -> dict | None:
    """Open a new tab. Chromium >= 111 requires PUT on /json/new; GET kept as fallback."""
    quoted = urllib.parse.quote(url, safe="")
    for method in ("PUT", "GET"):
        try:
            tab = _cdp("/json/new?" + quoted, method=method)
            if isinstance(tab, dict) and tab.get("id"):
                return tab
        except (OSError, ValueError):
            continue
    return None


def _raise_browser_window() -> None:
    """
    /json/activate switches the tab *inside* Brave but doesn't always raise the OS
    window. ASSUMPTION: best-effort via wmctrl (X11/XWayland). On pure Wayland
    (COSMIC) there's no generic way; the tab still switches, the window may stay behind.
    """
    if shutil.which("wmctrl"):
        try:
            subprocess.run(
                ["wmctrl", "-x", "-a", _browser_cfg().get("window_class_hint", "brave")],
                timeout=2, env=_gui_env(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            pass


def _brave_command(urls: list[str]) -> list[str] | None:
    cfg = _browser_cfg()
    binary = next((p for p in (shutil.which(b) for b in cfg["binaries"]) if p), None)
    if not binary:
        return None
    return [
        binary,
        f"--remote-debugging-port={cfg['cdp_port']}",
        f"--user-data-dir={os.path.expanduser(cfg['user_data_dir'])}",
        *cfg.get("extra_flags", []),
        *urls,
    ]


def brave_ensure_running(url: str | None = None) -> str:
    """
    The browser state machine (Python owns it, not the model):

      CDP port up?  ── yes ─► tab with this URL prefix?  ── yes ─► activate it   -> "activated"
                    │                                     └─ no  ─► open new tab  -> "opened"
                    └─ no ──► launch Brave with CDP flags (+url) and wait         -> "launched"
    Returns "activated" | "opened" | "launched" | "failed".
    With url=None it just makes sure Brave is up and focuses a tab.
    ASSUMPTION: "running" == "CDP port answers", because only the dedicated-profile
    instance is controllable; a normal Brave on the default profile is a different instance.
    """
    if brave_cdp_alive():
        if url:
            tab = brave_find_tab(url)
            if tab and brave_activate_tab(tab["id"]):
                _raise_browser_window()
                return "activated"
            if brave_open_tab(url):
                _raise_browser_window()
                return "opened"
            return "failed"
        tabs = brave_list_tabs()
        if tabs and brave_activate_tab(tabs[0]["id"]):
            _raise_browser_window()
            return "activated"
        return "opened" if brave_open_tab("about:blank") else "failed"

    cmd = _brave_command([url] if url else [])
    if not cmd:
        log.error("brave binary not found")
        return "failed"
    if _spawn(cmd) is None:
        return "failed"

    deadline = time.monotonic() + BRAVE_BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        if brave_cdp_alive():
            return "launched"
        time.sleep(0.25)
    log.error("Brave started but CDP never came up. Is another Brave using %s?",
              _browser_cfg()["user_data_dir"])
    return "failed"


def _browser_confirmation(state: str, name: str) -> str:
    return {
        "activated": f"Switched to {name}.",
        "opened": f"Opened {name}.",
        "launched": f"Opened {name}.",
    }.get(state, f"I couldn't open {name}.")


# ──────────────────────────────────────────────────────────────────────────────
# Generic recipe runner (Playwright attached to Brave over CDP)
# ──────────────────────────────────────────────────────────────────────────────

def run_recipe(name: str, args: dict) -> bool:
    """
    Execute registry.recipes[name] with {placeholders} filled from args.
    Step kinds: goto | wait_for | click | wait.  Returns True on success, False on
    any failure (never raises). Needs `pip install playwright` (optional dependency).
    """
    reg = load_registry()
    recipe = reg.get("recipes", {}).get(name)
    if not isinstance(recipe, dict):
        log.warning("recipe %r not in registry", name)
        return False
    if any(not str(args.get(k, "")).strip() for k in recipe.get("requires", [])):
        log.warning("recipe %r missing required args", name)
        return False

    try:
        from playwright.sync_api import sync_playwright   # optional dependency
    except ImportError:
        log.warning("playwright not installed; recipe %r skipped", name)
        return False

    if not brave_cdp_alive() and brave_ensure_running(None) == "failed":
        return False

    cdp_url = f"http://127.0.0.1:{_browser_cfg()['cdp_port']}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(cdp_url)
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()

            page = None
            prefix = recipe.get("reuse_prefix")
            if prefix:
                page = next((p for p in ctx.pages if p.url.startswith(prefix)), None)
            if page is None:
                page = ctx.new_page()

            for i, step in enumerate(recipe["steps"]):
                try:
                    _run_step(page, step, args)
                except Exception as exc:          # noqa: BLE001 — step-level tolerance
                    if step.get("optional"):
                        log.info("recipe %s step %d optional, skipped: %s", name, i, exc)
                        continue
                    log.warning("recipe %s step %d failed: %s", name, i, exc)
                    return False

            page.bring_to_front()
            _raise_browser_window()
            return True                           # leaving the `with` only detaches; tabs stay open
    except Exception:                             # noqa: BLE001
        log.exception("recipe %s crashed", name)
        return False


def _run_step(page: Any, step: dict, args: dict) -> None:
    timeout = int(step.get("timeout_ms", 10000))
    if "goto" in step:
        page.goto(_fmt(step["goto"], args, quote=True), wait_until="domcontentloaded", timeout=timeout)
    elif "wait_for" in step:
        page.wait_for_selector(_fmt(step["wait_for"], args, quote=False), state="visible", timeout=timeout)
    elif "click" in step:
        page.locator(_fmt(step["click"], args, quote=False)).first.click(timeout=timeout)
    elif "wait" in step:
        page.wait_for_timeout(int(step["wait"]))
    else:
        raise ValueError(f"unknown recipe step: {step}")


# ──────────────────────────────────────────────────────────────────────────────
# Verbs
# ──────────────────────────────────────────────────────────────────────────────

def _open_site(reg: dict, args: dict) -> str:
    raw = args.get("site")
    key = _lookup(reg, "sites", raw)
    if not key:
        # Model tolerance: "open brave" via open_site is really an app launch.
        if _lookup(reg, "apps", raw):
            return _open_app(reg, {"app": raw})
        return _not_configured(raw)

    site = reg["sites"][key]
    name = site.get("name", key)
    url = site["url"]

    path = str(args.get("path") or "").strip()
    if path:
        # Never let the model smuggle in an arbitrary URL via `path`.
        if re.match(r"^([a-z][a-z0-9+.\-]*:|//)", path, re.I):
            return f"I can only open pages inside {name}."
        url = url.rstrip("/") + "/" + path.lstrip("/")

    return _browser_confirmation(brave_ensure_running(url), name)


def _site_search(reg: dict, args: dict) -> str:
    raw = args.get("site")
    key = _lookup(reg, "sites", raw)
    if not key:
        return _not_configured(raw)
    query = str(args.get("query") or "").strip()
    if not query:
        return "What should I search for?"

    site = reg["sites"][key]
    name = site.get("name", key)
    template = site.get("search")
    if template:
        url = _fmt(template, {"q": query}, quote=True)
    else:
        # ASSUMPTION: site has no search template -> scoped web search via `site:` operator.
        host = urllib.parse.urlparse(site["url"]).netloc
        url = _fmt(reg["web_search"]["url"], {"q": f"site:{host} {query}"}, quote=True)

    state = brave_ensure_running(url)
    return f"Searching {name} for {query}." if state != "failed" else f"I couldn't open {name}."


def _web_search(reg: dict, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "What should I search for?"
    url = _fmt(reg["web_search"]["url"], {"q": query}, quote=True)
    state = brave_ensure_running(url)
    return f"Searching the web for {query}." if state != "failed" else "I couldn't open the browser."


@lru_cache(maxsize=256)
def _ytdlp_resolve_cached(
    search_prefix: str, query: str, watch_url: str
) -> tuple[str, str] | None:
    """Cached yt-dlp search. Returns (url, title) or None."""
    exe = shutil.which("yt-dlp")
    if not exe:
        log.warning("yt-dlp not installed")
        return None
    cmd = [
        exe, "--no-warnings", "--no-playlist", "--flat-playlist", "--skip-download",
        "--playlist-items", "1",
        "--print", "%(id)s\t%(url)s\t%(title)s",
        f"{search_prefix}{query}",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=YTDLP_TIMEOUT_S, env=_gui_env()).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("yt-dlp failed: %s", exc)
        return None
    line = out.splitlines()[0] if out else ""
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    vid, raw_url, title = parts[0], parts[1], parts[2]
    if watch_url and vid and vid != "NA":
        return watch_url.format(id=vid), title
    if raw_url.startswith("http"):
        return raw_url, title
    return None


def _ytdlp_resolve(media: dict, query: str) -> tuple[str, str] | None:
    """Thin wrapper: normalizes args so `media` (unhashable) never enters the cache key."""
    return _ytdlp_resolve_cached(
        media["search_prefix"],
        query.strip(),
        media.get("watch_url") or "",
    )


def _play_media(reg: dict, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "What would you like me to play?"

    raw_service = args.get("service") or reg["defaults"]["media_service"]
    mkey = _lookup(reg, "media", raw_service)
    if not mkey:
        return _not_configured(raw_service)
    media = reg["media"][mkey]
    name = media.get("name", mkey)
    done = f"Playing {query} on {name}."

    resolved = _ytdlp_resolve(media, query)

    # (b) mpv: audio only, no browser (opt-in via registry mode:"mpv")
    if resolved and media.get("mode") == "mpv" and shutil.which("mpv"):
        subprocess.run(["pkill", "-x", "mpv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc = _spawn(["mpv", "--force-window=no", "--ytdl-format=bestaudio",
                       *media.get("mpv_args", []), "--", resolved[0]])
        if proc:
            return done

    # (a) default: open the direct watch URL in Brave so the user sees it
    if resolved:
        if brave_ensure_running(resolved[0]) != "failed":
            return done

    # Fallback 1: click-through recipe (needs playwright)
    if media.get("recipe") and run_recipe(media["recipe"], {"q": query}):
        return done

    # Fallback 2: at least show the search results page
    site = reg["sites"].get(media.get("site", ""), {})
    if site.get("search") and brave_ensure_running(_fmt(site["search"], {"q": query}, quote=True)) != "failed":
        return f"I couldn't start it automatically, so I opened the {name} results for {query}."
    return f"I couldn't play {query}."


def _launch_entry(spec: dict) -> bool:
    """Try `gtk-launch <desktop-id>` first, then plain binaries. True if something started."""
    if shutil.which("gtk-launch"):
        for desktop_id in spec.get("desktop", []):
            proc = _spawn(["gtk-launch", desktop_id])
            if proc is None:
                break
            try:
                if proc.wait(timeout=3) == 0:
                    return True
            except subprocess.TimeoutExpired:
                return True               # still running after 3 s => the app took over
    for binary in spec.get("bins", []):
        exe = shutil.which(binary)
        if not exe:
            continue
        proc = _spawn([exe])
        if proc is None:
            continue
        time.sleep(0.4)
        if proc.poll() in (None, 0):
            return True
    return False


def _open_app(reg: dict, args: dict) -> str:
    raw = args.get("app")
    key = _lookup(reg, "apps", raw)
    if not key:
        if _lookup(reg, "sites", raw):    # model tolerance: "open_app whatsapp"
            return _open_site(reg, {"site": raw})
        return _not_configured(raw)

    app = reg["apps"][key]
    name = app.get("name", key)

    if app.get("browser"):                # Brave must carry the CDP flags -> state machine
        return _browser_confirmation(brave_ensure_running(None), name)

    launch = app.get("launch", {})
    spec = launch.get(reg["_os"]) or launch.get("all")
    if not spec:
        return f"{name} isn't set up for this version of Pop OS."
    return f"Opening {name}." if _launch_entry(spec) else f"I couldn't launch {name}."


def _close_app(reg: dict, args: dict) -> str:
    raw = args.get("app")
    key = _lookup(reg, "apps", raw)
    if not key:
        return _not_configured(raw)
    app = reg["apps"][key]
    name = app.get("name", key)
    patterns = app.get("kill", [])
    if not patterns:
        return f"I don't know how to close {name}."

    closed = False
    for pat in patterns:
        try:
            # -f: match full command line (comm is truncated to 15 chars, e.g. gnome-terminal-server)
            rc = subprocess.run(["pkill", "-f", "--", pat], timeout=3,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        except (OSError, subprocess.SubprocessError):
            continue
        closed = closed or rc == 0
    return f"Closed {name}." if closed else f"{name} isn't running."


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _to_int(value: Any) -> int | None:
    m = _NUM.search(str(value))
    return int(round(float(m.group()))) if m else None


def _current_volume() -> int | None:
    try:
        out = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                             capture_output=True, text=True, timeout=3, env=_gui_env()).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)%", out)
    return int(m.group(1)) if m else None


def _set_volume(reg: dict, args: dict) -> str:
    level, delta = args.get("level"), args.get("delta")
    if level not in (None, ""):
        target = _to_int(level)
    elif delta not in (None, ""):
        step, cur = _to_int(delta), _current_volume()
        target = None if step is None or cur is None else cur + step
    else:
        return "What volume level would you like?"
    if target is None:
        return "I couldn't work out the volume."

    target = max(0, min(100, target))          # pactl won't clamp; never blast >100%
    try:
        env = _gui_env()
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{target}%"],
                       check=True, timeout=3, env=env, capture_output=True)
        if target > 0:                         # raising volume on a muted sink should unmute
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                           timeout=3, env=env, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return "I couldn't change the volume."
    return f"Volume {target} percent."


def _session_type() -> str:
    """"x11" or "wayland". Env override ASSISTANT_SESSION_TYPE; else XDG_SESSION_TYPE; else WAYLAND_DISPLAY."""
    for var in ("ASSISTANT_SESSION_TYPE", "XDG_SESSION_TYPE"):
        val = os.environ.get(var, "").lower()
        if val in ("x11", "wayland"):
            return val
    return "wayland" if _gui_env().get("WAYLAND_DISPLAY") else "x11"


def _window(reg: dict, args: dict) -> str:
    """
    close / minimize / maximize the CURRENTLY FOCUSED window. All commands live in
    registry.window.ops.<op>.<session>; first command whose binary exists and exits 0 wins.
    """
    raw = args.get("op") or args.get("action") or args.get("do")
    key = _lookup(reg, "window", raw)
    if not key:
        return "I can close, minimize or maximize the current window."
    spec = reg["window"]["ops"][key]
    session = _session_type()

    for cmd in spec.get(session, []):
        if not shutil.which(cmd[0]):
            continue
        try:
            rc = subprocess.run(cmd, timeout=3, env=_gui_env(),
                                capture_output=True).returncode
        except (OSError, subprocess.SubprocessError):
            continue
        if rc == 0:
            return spec.get("done", "Done.")
        log.warning("window %s: %s exited %s", key, cmd[0], rc)
    return f"I can't {key} the window on this {session} session yet."


# ──────────────────────────────────────────────────────────────────────────────
# System-control verbs (brightness, toggle, media_control, lock_screen)
# ──────────────────────────────────────────────────────────────────────────────

def _set_brightness(reg: dict, args: dict) -> str:
    """Brightness via registry.brightness.{session|all}. {val} → '40%' or '+20%' / '-20%'."""
    cfg = reg.get("brightness", {})
    session = _session_type()
    backends = cfg.get(session) or cfg.get("all") or []
    if not backends:
        return "Brightness control isn't set up."

    level, delta = args.get("level"), args.get("delta")
    if level not in (None, ""):
        target = _to_int(level)
        if target is None:
            return "I couldn't work out the brightness."
        target = max(0, min(100, target))
        val, done = f"{target}%", f"Brightness {target} percent."
    elif delta not in (None, ""):
        step = _to_int(delta)
        if step is None:
            return "I couldn't work out the brightness."
        step = max(-50, min(50, step))
        if step == 0:
            return "No change."
        val = f"+{step}%" if step > 0 else f"{step}%"
        done = "Brightness increased." if step > 0 else "Brightness decreased."
    else:
        return "What brightness level would you like?"

    for backend in backends:
        if not shutil.which(backend[0]):
            continue
        cmd = [val if p == "{val}" else p for p in backend]
        try:
            if subprocess.run(cmd, timeout=3, env=_gui_env(),
                              capture_output=True).returncode == 0:
                return done
        except (OSError, subprocess.SubprocessError):
            continue
    return "I couldn't change the brightness."


def _toggle(reg: dict, args: dict) -> str:
    """Boolean feature via registry.toggles.<feature>.{on|off}.{session|all}."""
    raw = args.get("feature")
    key = _lookup(reg, "toggles", raw)
    if not key:
        return f"I don't have {raw or 'that'} as a toggle."

    feat = reg["toggles"][key]
    name = feat.get("name", key)

    on_raw = args.get("on")
    if isinstance(on_raw, bool):
        want_on = on_raw
    elif isinstance(on_raw, str):
        want_on = on_raw.lower() in ("on", "true", "1", "yes", "enable", "enabled", "chalu")
    else:
        return f"Should I turn {name} on or off?"

    spec = feat.get("on" if want_on else "off", {})
    session = _session_type()
    candidates = spec.get(session) or spec.get("all") or []
    if not candidates:
        return f"I can't toggle {name} on this {session} session."

    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            if subprocess.run(cmd, timeout=5, env=_gui_env(),
                                capture_output=True).returncode == 0:
                return f"{name} {'on' if want_on else 'off'}."
        except (OSError, subprocess.SubprocessError):
            continue
    return f"I couldn't turn {name} {'on' if want_on else 'off'}."


def _media_control(reg: dict, args: dict) -> str:
    """Play/pause/skip via playerctl."""
    action = str(args.get("action") or "").lower().strip()
    action_map = {
        "play": "play", "pause": "pause",
        "toggle": "play-pause", "play_pause": "play-pause", "playpause": "play-pause",
        "next": "next", "skip": "next", "forward": "next",
        "prev": "previous", "previous": "previous", "back": "previous",
        "stop": "stop",
    }
    pc = action_map.get(action)
    if not pc:
        return "I can play, pause, skip, go back, or stop."
    if not shutil.which("playerctl"):
        return "Media control isn't installed."

    try:
        rc = subprocess.run(["playerctl", pc], timeout=3, env=_gui_env(),
                            capture_output=True).returncode
    except (OSError, subprocess.SubprocessError):
        return "I couldn't control the media."
    if rc != 0:
        return "Nothing seems to be playing."

    return {
        "play": "Playing.", "pause": "Paused.", "play-pause": "Toggled playback.",
        "next": "Skipped.", "previous": "Went back.", "stop": "Stopped.",
    }.get(pc, "Done.")


def _lock_screen(reg: dict, args: dict) -> str:
    """Lock the current session."""
    for cmd in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"]):
        if not shutil.which(cmd[0]):
            continue
        try:
            if subprocess.run(cmd, timeout=3, env=_gui_env(),
                                capture_output=True).returncode == 0:
                return "Locking the screen."
        except (OSError, subprocess.SubprocessError):
            continue
    return "I couldn't lock the screen."

# Fixed verb table — the ONLY place verbs are enumerated. Never add site-specific entries.
TOOLS = {
    "open_site": _open_site,
    "play_media": _play_media,
    "site_search": _site_search,
    "web_search": _web_search,
    "open_app": _open_app,
    "close_app": _close_app,
    "set_volume": _set_volume,
    "window": _window,
    "set_brightness": _set_brightness,
    "toggle": _toggle,
    "set_feature": _toggle,    
    "media_control": _media_control,
    "lock_screen": _lock_screen,
}


# ──────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _run_call(reg: dict, tool: Any, args: Any) -> str:
    fn = TOOLS.get(_norm(tool))
    if fn is None:
        return "I don't know how to do that yet."
    args = args if isinstance(args, dict) else {}

    # Recovery: if play_media has volume args, split into two actions
    if _norm(tool) == "play_media" and ("delta" in args or "level" in args):
        vol = {k: v for k, v in args.items() if k in ("delta", "level")}
        play = {k: v for k, v in args.items() if k not in ("delta", "level")}
        r1 = _play_media(reg, play) if play else ""
        r2 = _set_volume(reg, vol)
        return f"{r1} {r2}".strip()

    # Recovery: strip unknown args from other verbs
    VALID = {
        "open_site": {"site", "path"}, "site_search": {"site", "query"},
        "web_search": {"query"}, "open_app": {"app"}, "close_app": {"app"},
        "set_volume": {"level", "delta"}, "set_brightness": {"level", "delta"},
        "window": {"op"}, "toggle": {"feature", "on"},
        "media_control": {"action"}, "lock_screen": set(),
    }
    if _norm(tool) in VALID:
        args = {k: v for k, v in args.items() if k in VALID[_norm(tool)]}

    return fn(reg, args)


def execute(intent: dict | str) -> str:
    """
    Model JSON in -> confirmation string for Piper out. Never raises.
        {"action":"tool","tool":"open_site","args":{"site":"whatsapp"}}
        {"action":"tools","calls":[{"tool":...,"args":{...}}, ...]}   (sequential)
        {"action":"chat","response":"..."}
    The confirmation is built from what actually happened, not from the model.
    """
    try:
        if isinstance(intent, (str, bytes)):
            intent = json.loads(intent)
        if not isinstance(intent, dict):
            return "Sorry, I didn't understand that."
        reg = load_registry()

        action = intent.get("action")
        if action == "chat":
            return str(intent.get("response") or "").strip() or "Sorry, I didn't catch that."
        if action == "tool":
            return _run_call(reg, intent.get("tool"), intent.get("args"))
        if action == "tools":
            calls = intent.get("calls")
            if not isinstance(calls, list) or not calls:
                return "Sorry, I didn't understand that."
            results = []
            for call in calls:                 # sequential: Brave must be up before the 2nd site
                if isinstance(call, dict):
                    results.append(_run_call(reg, call.get("tool"), call.get("args")))
            return " ".join(results) or "Sorry, I didn't understand that."
        return "Sorry, I didn't understand that."
    except FileNotFoundError:
        log.exception("registry missing")
        return "My registry file is missing."
    except Exception:                          # noqa: BLE001 — failures must be speech, not exceptions
        log.exception("execute() failed")
        return "Something went wrong with that one."


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(execute(sys.argv[1] if len(sys.argv) > 1 else '{"action":"chat","response":"Executor loaded."}'))
