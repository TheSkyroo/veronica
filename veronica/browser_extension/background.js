// Veronica browser bridge (Manifest V3 service worker, Chrome and Edge).
//
// Keeps one WebSocket open to the Veronica app on this PC
// (ws://127.0.0.1:<port>, default 8765) and answers its requests:
//   Veronica -> {id, op, ...args}
//   extension -> {id, ok: true, result} | {id, ok: false, error}
// The first message on every connection is {type: 'hello', token, browser,
// version}; Veronica closes the socket (code 4401) unless the token matches
// the pairing code it wrote to %USERPROFILE%\.veronica\browser_token.
// {type: 'focus'} tells Veronica this browser was just focused, so with
// both Chrome and Edge running it drives the one the user used last.
//
// Veronica can only ask for the fixed operations in OPS below; it never
// sends code. Page content goes back as data and is treated as untrusted
// on the Python side.

const DEFAULT_PORT = 8765;
const KEEPALIVE_MS = 20000;     // traffic every 20s keeps the MV3 worker alive (Chrome 116+)
const BACKOFF_MIN_MS = 1000;
const BACKOFF_MAX_MS = 15000;
const RAW_TEXT_MAX = 200000;    // hard cap on page text sent back; Veronica trims further
const ALARM = 'veronica-reconnect';

let ws = null;
let backoff = BACKOFF_MIN_MS;
let retryTimer = null;
let keepTimer = null;
let status = { state: 'disconnected', detail: '' };

function browserName() {
  return /\bEdg\//.test(navigator.userAgent) ? 'Edge' : 'Chrome';
}

function setStatus(state, detail = '') {
  status = { state, detail };
  chrome.storage.session?.set({ status }).catch(() => {});
}

async function settings() {
  const s = await chrome.storage.local.get({ token: '', port: DEFAULT_PORT });
  const port = parseInt(s.port, 10);
  return { token: String(s.token || '').trim(), port: port > 0 && port < 65536 ? port : DEFAULT_PORT };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function scheduleReconnect() {
  if (retryTimer) return;
  retryTimer = setTimeout(() => { retryTimer = null; connect(); }, backoff);
  backoff = Math.min(backoff * 2, BACKOFF_MAX_MS);
}

async function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  const { token, port } = await settings();
  if (!token) { setStatus('needs-token', 'Paste the pairing code from Veronica.'); return; }
  if (status.state === 'bad-token') return;   // wait for a new code (storage.onChanged resets this)
  let sock;
  try {
    sock = new WebSocket(`ws://127.0.0.1:${port}`);
  } catch (e) {
    setStatus('disconnected', String(e));
    scheduleReconnect();
    return;
  }
  ws = sock;
  setStatus('connecting', `127.0.0.1:${port}`);
  sock.onopen = () => {
    sock.send(JSON.stringify({ type: 'hello', token, browser: browserName(),
                               version: chrome.runtime.getManifest().version }));
    clearInterval(keepTimer);
    keepTimer = setInterval(() => send({ type: 'ping' }), KEEPALIVE_MS);
  };
  sock.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    onMessage(msg);
  };
  sock.onclose = (ev) => {
    if (ws === sock) ws = null;
    clearInterval(keepTimer);
    if (ev.code === 4401) {
      setStatus('bad-token', 'Veronica rejected the pairing code. Paste the current one.');
      return;
    }
    if (status.state !== 'bad-token') setStatus('disconnected', 'Is Veronica running?');
    scheduleReconnect();
  };
  sock.onerror = () => {};   // onclose follows and handles the retry
}

async function onMessage(msg) {
  if (msg.type === 'hello') {
    if (msg.ok) {
      backoff = BACKOFF_MIN_MS;
      setStatus('connected', browserName());
      reportFocus();
    }
    return;
  }
  if (msg.type === 'pong' || msg.id === undefined) return;
  try {
    const fn = OPS[msg.op];
    if (!fn) throw new Error(`unknown op '${msg.op}'`);
    send({ id: msg.id, ok: true, result: await fn(msg) });
  } catch (e) {
    send({ id: msg.id, ok: false, error: friendlyError(e) });
  }
}

function friendlyError(e) {
  const text = String((e && e.message) || e);
  if (/cannot access|cannot be scripted|chrome:\/\/|edge:\/\/|extensions gallery|webstore/i.test(text)) {
    return "Veronica can't access this page (browser settings, new-tab and extension store pages are off-limits).";
  }
  return text;
}

async function reportFocus() {
  try {
    const win = await chrome.windows.getLastFocused();
    if (win && win.focused) send({ type: 'focus' });
  } catch { /* no windows */ }
}

// -- operations -------------------------------------------------------------

async function lastNormalWindow() {
  try {
    return await chrome.windows.getLastFocused({ populate: true, windowTypes: ['normal'] });
  } catch {
    return null;
  }
}

async function activeTab() {
  const win = await lastNormalWindow();
  const tab = win && win.tabs && win.tabs.find((t) => t.active);
  if (!tab) throw new Error('No open browser tab.');
  return tab;
}

async function inPage(op, args) {
  const tab = await activeTab();
  const [res] = await chrome.scripting.executeScript({
    target: { tabId: tab.id }, func: pageOp, args: [op, args || {}, RAW_TEXT_MAX], world: 'ISOLATED',
  });
  if (!res) throw new Error('The page did not answer.');
  return res.result;
}

function isHttp(url) {
  try { return ['http:', 'https:'].includes(new URL(url).protocol); } catch { return false; }
}

const OPS = {
  async tabs() {
    const win = await lastNormalWindow();
    const tabs = (win && win.tabs) || [];
    return {
      browser: browserName(),
      tabs: tabs.map((t) => ({ title: t.title || '', url: t.url || t.pendingUrl || '', active: !!t.active })),
    };
  },
  async open({ url, new_tab }) {
    if (!isHttp(url)) throw new Error('only http(s) URLs are allowed');
    const win = await lastNormalWindow();
    if (!win) {
      await chrome.windows.create({ url, focused: true });
      return { opened: url };
    }
    if (new_tab !== false) {
      await chrome.tabs.create({ windowId: win.id, url, active: true });
    } else {
      const tab = win.tabs.find((t) => t.active);
      if (tab) await chrome.tabs.update(tab.id, { url });
      else await chrome.tabs.create({ windowId: win.id, url, active: true });
    }
    await chrome.windows.update(win.id, { focused: true });
    return { opened: url };
  },
  async ready_state() {
    const tab = await activeTab();
    return tab.status === 'complete' ? 'complete' : 'loading';
  },
  async back() {
    const tab = await activeTab();
    await chrome.tabs.goBack(tab.id);
    return 'ok';
  },
  read: () => inPage('read'),
  find: (m) => inPage('find', { text: m.text, max_lines: m.max_lines }),
  click: (m) => inPage('click', { target: m.target }),
  type: (m) => inPage('type', { target: m.target, text: m.text, submit: !!m.submit }),
  scroll: (m) => inPage('scroll', { direction: m.direction }),
};

// Runs inside the page (isolated world). Must be self-contained: Chrome
// serialises this function, so it can't see anything else in this file.
function pageOp(op, args, rawMax) {
  function norm(s) { return (s || '').replace(/\s+/g, ' ').trim().toLowerCase(); }
  function labelsOf(el) {
    const out = [el.innerText, el.getAttribute('aria-label'), el.value, el.title, el.alt,
                 el.placeholder, el.name, el.id];
    if (el.labels) { for (const l of el.labels) out.push(l.innerText); }
    return out.map(norm).filter(Boolean);
  }
  function visible(el) {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden';
  }
  // Prefer the innermost match: a <button> rather than the <div> around it.
  function innermost(cands) {
    const keep = cands.filter((a) => !cands.some((b) => a !== b && a.contains(b)));
    return keep.length ? keep[0] : null;
  }
  function findEl(sel, target) {
    const t = norm(target);
    const els = Array.from(document.querySelectorAll(sel)).filter(visible);
    const exact = els.filter((el) => labelsOf(el).indexOf(t) >= 0);
    if (exact.length) return innermost(exact);
    const partial = els.filter((el) => labelsOf(el).some((l) => l.indexOf(t) >= 0));
    return innermost(partial);
  }
  function describe(el) {
    const tag = el.tagName;
    let s;
    if (tag === 'INPUT' || tag === 'TEXTAREA') s = el.getAttribute('aria-label') || el.placeholder || el.name || el.value || '';
    else s = el.innerText || el.value || el.getAttribute('aria-label') || el.placeholder || '';
    return tag + ' ' + (s || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  }
  function bodyText() { return (document.body && document.body.innerText) || ''; }

  if (op === 'read') {
    const t = bodyText().replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').slice(0, rawMax);
    return { title: document.title, url: location.href, text: t };
  }
  if (op === 'find') {
    const q = norm(args.text), lines = bodyText().split('\n'), out = [];
    const max = Math.max(1, Math.min(50, args.max_lines || 10));
    for (let i = 0; i < lines.length && out.length < max; i++) {
      const l = lines[i].trim();
      if (l && norm(l).indexOf(q) >= 0) out.push([i + 1, l.slice(0, 160)]);
    }
    return { lines: out };
  }
  if (op === 'click') {
    const el = findEl('a,button,input[type=submit],input[type=button],[role=button],[role=link],[onclick],summary,label', args.target);
    if (!el) return { clicked: null };
    el.scrollIntoView({ block: 'center' });
    el.click();
    return { clicked: describe(el) };
  }
  if (op === 'type') {
    const el = findEl('input:not([type=hidden]):not([type=submit]):not([type=button]),textarea,[contenteditable]:not([contenteditable=false]),[role=textbox]', args.target);
    if (!el) return { typed: null };
    el.scrollIntoView({ block: 'center' });
    el.focus();
    const v = String(args.text || '');
    if (el.isContentEditable) {
      el.textContent = v;
    } else {
      const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value');
      if (d && d.set) d.set.call(el, v); else el.value = v;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    if (args.submit) {
      const opts = { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true };
      const ok = el.dispatchEvent(new KeyboardEvent('keydown', opts));
      el.dispatchEvent(new KeyboardEvent('keypress', opts));
      el.dispatchEvent(new KeyboardEvent('keyup', opts));
      // ok is false when a keydown handler called preventDefault, i.e. the
      // page handled Enter itself: don't submit twice.
      if (ok && el.form && document.activeElement === el) {
        if (el.form.requestSubmit) el.form.requestSubmit(); else el.form.submit();
      }
    }
    return { typed: describe(el) };
  }
  if (op === 'scroll') {
    const h = Math.round(window.innerHeight * 0.8);
    if (args.direction === 'down') window.scrollBy(0, h);
    else if (args.direction === 'up') window.scrollBy(0, -h);
    else if (args.direction === 'top') window.scrollTo(0, 0);
    else if (args.direction === 'bottom') window.scrollTo(0, document.body.scrollHeight);
    else throw new Error('direction must be up, down, top or bottom');
    return 'ok';
  }
  throw new Error(`unknown page op '${op}'`);
}

// -- lifecycle --------------------------------------------------------------

chrome.windows.onFocusChanged.addListener((id) => {
  if (id !== chrome.windows.WINDOW_ID_NONE) send({ type: 'focus' });
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local' || !(changes.token || changes.port)) return;
  setStatus('disconnected', '');
  backoff = BACKOFF_MIN_MS;
  if (ws) ws.close(); else connect();
});

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg && msg.type === 'status') { reply(status); return false; }
  if (msg && msg.type === 'reconnect') { connect(); reply(status); return false; }
  return false;
});

// The worker can still be stopped (e.g. while Veronica isn't running and
// nothing is connected); this alarm wakes it to try again.
chrome.alarms.create(ALARM, { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((a) => { if (a.name === ALARM) connect(); });

chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') chrome.runtime.openOptionsPage();
  connect();
});

connect();
