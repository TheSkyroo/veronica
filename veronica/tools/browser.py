"""Browser control for Chrome and Safari via AppleScript + injected
JavaScript. One-time user setup: Chrome -> View > Developer > Allow
JavaScript from Apple Events; Safari -> Develop > Allow JavaScript from
Apple Events. No remote debugging / CDP (Chrome >=136 refuses it on the
default profile, which would lose the user's logins)."""
import asyncio
import json
import logging
import subprocess

from claude_agent_sdk import create_sdk_mcp_server, tool

log = logging.getLogger("veronica.tools.browser")

CHROME = "Google Chrome"
SAFARI = "Safari"
TIMEOUT_S = 20
READ_MIN = 50
READ_DEFAULT = 6000
READ_MAX = 20000
FIND_MAX_LINES = 10


class BrowserUnavailable(RuntimeError):
    pass


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None) -> dict:
    """Run argv (never a shell string) and map the result to MCP content."""
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return _err(f"{argv[0]} timed out after {TIMEOUT_S}s")
    except OSError as e:
        return _err(str(e))
    if p.returncode != 0:
        return _err(p.stderr.strip() or f"{argv[0]} failed")
    return _ok(ok_text if ok_text is not None else p.stdout.strip())


async def _osascript(script: str) -> dict:
    return await asyncio.to_thread(run, ["osascript", "-e", script])


def _q(s: str) -> str:
    """Escape for inclusion inside an AppleScript double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _wrap_js(browser: str, js: str) -> str:
    if browser == CHROME:
        return f'tell application "{CHROME}" to execute active tab of front window javascript "{_q(js)}"'
    return f'tell application "{SAFARI}" to do JavaScript "{_q(js)}" in current tab of front window'


async def target_browser() -> str:
    """The frontmost supported browser, else whichever one is running
    (Chrome preferred). Raises BrowserUnavailable if neither is open."""
    res = await _osascript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
    )
    front = res["content"][0]["text"].strip() if not res.get("is_error") else ""
    if front in (CHROME, SAFARI):
        return front
    for name in (CHROME, SAFARI):
        r = await _osascript(f'tell application "System Events" to (exists process "{name}")')
        if not r.get("is_error") and r["content"][0]["text"].strip().lower() == "true":
            return name
    raise BrowserUnavailable("No supported browser is open (Chrome or Safari).")


def _map_error(browser: str, text: str) -> dict:
    low = text.lower()
    if "allow javascript from apple events" in low or "turned off" in low:
        menu = "View > Developer" if browser == CHROME else "Develop"
        return _err(
            f"JavaScript from Apple Events is off in {browser}. Turn it on under "
            f"{menu} > Allow JavaScript from Apple Events and try again."
        )
    if "-1743" in text or "not authorized" in low:
        return _err(
            f"Veronica isn't allowed to control {browser} yet; allow it in "
            "System Settings > Privacy & Security > Automation."
        )
    return _err(text)


async def _js(js: str) -> tuple[str, dict]:
    """Run `js` in the target browser; returns (browser, result)."""
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return "", _err(str(e))
    res = await _osascript(_wrap_js(browser, js))
    if res.get("is_error"):
        return browser, _map_error(browser, res["content"][0]["text"])
    return browser, res


def _json(res: dict) -> dict | None:
    try:
        data = json.loads(res["content"][0]["text"])
    except (ValueError, KeyError, IndexError):
        return None
    return data if isinstance(data, dict) else None


def _guard(fn):
    """Wrap a handler so malformed args/unexpected failures return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            log.exception("browser tool failed")
            return _err(f"{type(exc).__name__}: {exc}")
    wrapper.__name__ = fn.__name__
    return wrapper


# -- JS snippets (each an IIFE returning a JSON string) -------------------------
# Python-level `%` formatting is applied to _JS_FIND/_JS_CLICK/_JS_TYPE only;
# keep literal `%` out of those bodies (or write `%%`).
_JS_MATCH_HELPERS = """
function norm(s){return (s||'').replace(/\\s+/g,' ').trim().toLowerCase();}
function labelsOf(el){
  var out=[el.innerText, el.getAttribute('aria-label'), el.value, el.title, el.alt,
           el.placeholder, el.name, el.id];
  if(el.labels){for(var i=0;i<el.labels.length;i++){out.push(el.labels[i].innerText);}}
  return out.map(norm).filter(Boolean);
}
function visible(el){var r=el.getBoundingClientRect();return r.width>0&&r.height>0;}
function findEl(sel, target){
  var t=norm(target), els=Array.from(document.querySelectorAll(sel)).filter(visible);
  for(var i=0;i<els.length;i++){if(labelsOf(els[i]).indexOf(t)>=0)return els[i];}
  for(var j=0;j<els.length;j++){if(labelsOf(els[j]).some(function(l){return l.indexOf(t)>=0;}))return els[j];}
  return null;
}
function describe(el){return el.tagName+' '+norm(el.innerText||el.value||el.getAttribute('aria-label')||el.placeholder||'').slice(0,60);}
"""

_JS_READ = """(function(){
  var t=(document.body&&document.body.innerText||'').replace(/[ \\t]+/g,' ').replace(/\\n{3,}/g,'\\n\\n');
  return JSON.stringify({title:document.title,url:location.href,text:t});
})()"""

_JS_FIND = """(function(){%s
  var q=norm(%s), lines=(document.body&&document.body.innerText||'').split('\\n'), out=[];
  for(var i=0;i<lines.length&&out.length<%d;i++){var l=lines[i].trim(); if(l&&norm(l).indexOf(q)>=0)out.push([i+1,l.slice(0,160)]);}
  return JSON.stringify({lines:out});
})()"""

_JS_CLICK = """(function(){%s
  var el=findEl('a,button,input[type=submit],input[type=button],[role=button],[role=link],[onclick],summary,label', %s);
  if(!el)return JSON.stringify({clicked:null});
  el.scrollIntoView({block:'center'}); el.click();
  return JSON.stringify({clicked:describe(el)});
})()"""

_JS_TYPE = """(function(){%s
  var el=findEl('input:not([type=hidden]):not([type=submit]):not([type=button]),textarea,[contenteditable=true],[role=textbox]', %s);
  if(!el)return JSON.stringify({typed:null});
  el.scrollIntoView({block:'center'}); el.focus();
  var v=%s;
  if(el.isContentEditable){el.textContent=v;}else{
    var setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el),'value');
    if(setter&&setter.set){setter.set.call(el,v);}else{el.value=v;}
  }
  el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true}));
  if(%s){
    var opts={key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true};
    el.dispatchEvent(new KeyboardEvent('keydown',opts)); el.dispatchEvent(new KeyboardEvent('keypress',opts));
    el.dispatchEvent(new KeyboardEvent('keyup',opts));
    if(el.form&&document.activeElement===el){ if(el.form.requestSubmit){el.form.requestSubmit();} else {el.form.submit();} }
  }
  return JSON.stringify({typed:describe(el)});
})()"""

_JS_SCROLL = {
    "down": "window.scrollBy(0, Math.round(window.innerHeight*0.8)); 'ok'",
    "up": "window.scrollBy(0, -Math.round(window.innerHeight*0.8)); 'ok'",
    "top": "window.scrollTo(0, 0); 'ok'",
    "bottom": "window.scrollTo(0, document.body.scrollHeight); 'ok'",
}
_JS_BACK = "history.back(); 'ok'"


# -- tools -----------------------------------------------------------------------
@tool("browser_tabs", "List the open tabs of the front window of Chrome or Safari (current tab marked *)", {})
@_guard
async def browser_tabs(args: dict) -> dict:
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return _err(str(e))
    if browser == CHROME:
        script = (
            f'tell application "{CHROME}"\n'
            'set w to front window\nset out to (active tab index of w as text) & linefeed\n'
            'repeat with t in tabs of w\nset out to out & (title of t) & tab & (URL of t) & linefeed\nend repeat\n'
            'return out\nend tell'
        )
    else:
        script = (
            f'tell application "{SAFARI}"\n'
            'set w to front window\nset out to (index of current tab of w as text) & linefeed\n'
            'repeat with t in tabs of w\nset out to out & (name of t) & tab & (URL of t) & linefeed\nend repeat\n'
            'return out\nend tell'
        )
    res = await _osascript(script)
    if res.get("is_error"):
        return _map_error(browser, res["content"][0]["text"])
    lines = [ln for ln in res["content"][0]["text"].split("\n") if ln.strip()]
    if not lines:
        return _ok("No tabs.")
    try:
        current = int(lines[0].strip())
    except ValueError:
        current = -1
    out = []
    for i, ln in enumerate(lines[1:], start=1):
        title, _, url = ln.partition("\t")
        mark = "* " if i == current else ""
        out.append(f"{i}. {mark}{title.strip()} — {url.strip()}")
    return _ok("\n".join(out) if out else "No tabs.")


@tool("browser_open", "Open an http(s) URL in the current browser (new tab by default)", {"url": str, "new_tab": bool})
@_guard
async def browser_open(args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    new_tab = bool(args.get("new_tab", True))
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return _err(str(e))
    u = _q(url)
    if browser == CHROME:
        script = (
            f'tell application "{CHROME}"\nactivate\n'
            + (f'tell front window to make new tab with properties {{URL:"{u}"}}\n' if new_tab
               else f'set URL of active tab of front window to "{u}"\n')
            + 'end tell'
        )
    else:
        script = (
            f'tell application "{SAFARI}"\nactivate\n'
            + (f'tell front window to set current tab to (make new tab with properties {{URL:"{u}"}})\n' if new_tab
               else f'set URL of current tab of front window to "{u}"\n')
            + 'end tell'
        )
    res = await _osascript(script)
    if res.get("is_error"):
        return _map_error(browser, res["content"][0]["text"])
    return _ok(f"Opened {url}")


@tool("browser_read", "Read the current tab: title, URL and visible text (capped)", {"max_chars": int})
@_guard
async def browser_read(args: dict) -> dict:
    try:
        max_chars = int(args.get("max_chars") or READ_DEFAULT)
    except (TypeError, ValueError):
        max_chars = READ_DEFAULT
    max_chars = max(READ_MIN, min(READ_MAX, max_chars))
    _, res = await _js(_JS_READ)
    if res.get("is_error"):
        return res
    data = _json(res) or {}
    text = " ".join(str(data.get("text", "")).split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…[truncated]"
    return _ok(f"{data.get('title', '')}\n{data.get('url', '')}\n{text}")


@tool("browser_find", "Find lines on the current page containing text (case-insensitive)", {"text": str})
@_guard
async def browser_find(args: dict) -> dict:
    needle = str(args.get("text", "")).strip()
    if not needle:
        return _err("text is required")
    _, res = await _js(_JS_FIND % (_JS_MATCH_HELPERS, json.dumps(needle), FIND_MAX_LINES))
    if res.get("is_error"):
        return res
    lines = (_json(res) or {}).get("lines") or []
    if not lines:
        return _ok("not found")
    return _ok("\n".join(f"{n}: {t}" for n, t in lines))


@tool("browser_click", "Click a link/button on the current page by its visible text or label", {"target": str})
@_guard
async def browser_click(args: dict) -> dict:
    target = str(args.get("target", "")).strip()
    if not target:
        return _err("target is required")
    _, res = await _js(_JS_CLICK % (_JS_MATCH_HELPERS, json.dumps(target)))
    if res.get("is_error"):
        return res
    clicked = (_json(res) or {}).get("clicked")
    if not clicked:
        return _err(f"no element matching '{target}'")
    return _ok(f"Clicked {clicked}")


@tool("browser_type", "Type text into a field on the current page (by placeholder/label/name), optionally pressing Enter", {"target": str, "text": str, "submit": bool})
@_guard
async def browser_type(args: dict) -> dict:
    target = str(args.get("target", "")).strip()
    text = str(args.get("text", ""))
    if not target:
        return _err("target is required")
    submit = "true" if bool(args.get("submit", False)) else "false"
    _, res = await _js(_JS_TYPE % (_JS_MATCH_HELPERS, json.dumps(target), json.dumps(text), submit))
    if res.get("is_error"):
        return res
    typed = (_json(res) or {}).get("typed")
    if not typed:
        return _err(f"no field matching '{target}'")
    return _ok(f"Typed into {typed}")


@tool("browser_scroll", "Scroll the current page: up, down, top or bottom", {"direction": str})
@_guard
async def browser_scroll(args: dict) -> dict:
    direction = str(args.get("direction", "down")).strip().lower()
    if direction not in _JS_SCROLL:
        return _err("direction must be up, down, top or bottom")
    _, res = await _js(_JS_SCROLL[direction])
    return res if res.get("is_error") else _ok(f"Scrolled {direction}")


@tool("browser_back", "Go back one page in the current tab", {})
@_guard
async def browser_back(args: dict) -> dict:
    _, res = await _js(_JS_BACK)
    return res if res.get("is_error") else _ok("Went back")


TOOLS = [browser_tabs, browser_open, browser_read, browser_find, browser_click, browser_type, browser_scroll, browser_back]
BROWSER_TOOL_NAMES = [t.name for t in TOOLS]
browser_server = create_sdk_mcp_server(name="browser", version="1.0.0", tools=TOOLS)
