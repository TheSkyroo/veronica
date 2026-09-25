(() => {
  // Python pushes the whole state (`window.settings.state`) and answers each
  // posted command (`window.settings.reply(id, result)`). The page renders
  // one tab at a time from `model`; every control posts `{id, cmd, args}`.
  const TABS = [
    ['general', 'General'], ['voice', 'Voice'], ['listening', 'Listening'], ['briefings', 'Briefings'],
    ['brain', 'Brain'], ['history', 'History'], ['about', 'About'],
  ];
  const TAB_NAMES = new Set(TABS.map(t => t[0]));
  const LANGUAGES = [['en', 'English'], ['hi', 'Hindi'], ['auto', 'Auto (detect)']];
  const HUD_MODES = [['full', 'Full card'], ['mini', 'Mini orb']];
  const SPEED = {kind: 'float', label: 'Speaking speed', help: '', min: 0.7, max: 1.5, step: 0.05};
  const HISTORY_DEBOUNCE_MS = 200;
  const HISTORY_LIMIT = 200;
  const LATEST = "You're already on the latest.";

  let model = null;
  let tab = 'general';
  let restartRequired = false;
  let nextId = 1;
  const pending = new Map();      // id → {resolve}
  const sent = [];                // Playwright: messages that had no native handler
  const $ = sel => document.querySelector(sel);
  const tabsEl = $('#tabs');
  const paneEl = $('#pane');
  const bannerEl = $('#banner');

  // ---- transport -------------------------------------------------------------
  function native() {
    try {
      return window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.veronica || null;
    } catch (e) { return null; }
  }

  function post(cmd, args) {
    const id = nextId++;
    const msg = {id, cmd, args: args || {}};
    return new Promise(resolve => {
      pending.set(id, resolve);
      const h = native();
      if (h) {
        try { h.postMessage(msg); } catch (e) { pending.delete(id); resolve({ok: false, message: String(e)}); }
      } else {
        sent.push(msg);
      }
    });
  }

  window.settings = {
    state(json) {
      let next = json;
      if (typeof json === 'string') {
        try { next = JSON.parse(json); } catch (e) { return; }
      }
      if (!next || typeof next !== 'object') return;
      model = next;
      restartRequired = !!(model.meta && model.meta.restart_required);
      render();
    },
    reply(id, result) {
      const resolve = pending.get(id);
      pending.delete(id);
      if (resolve) resolve(result && typeof result === 'object' ? result : {});
    },
    select(name) {
      if (!TAB_NAMES.has(name)) name = 'general';
      tab = name;
      render();
      if (name === 'history') history.load();   // always fresh: turns accrue while the window is closed
    },
  };
  window.__settings = {sent, state: () => model, pending: () => pending.size, tab: () => tab};

  // ---- helpers -------------------------------------------------------------------
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const k in (attrs || {})) {
      const v = attrs[k];
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'disabled') node.disabled = !!v;
      else if (k === 'checked') node.checked = !!v;
      else if (k === 'value') node.value = v;
      else if (k.startsWith('on')) { if (typeof v === 'function') node.addEventListener(k.slice(2), v); }
      else node.setAttribute(k, v);
    }
    for (const c of (children || [])) if (c != null) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return node;
  }
  const fields = () => (model && model.meta && model.meta.fields) || {};
  const section = name => (model && model[name]) || {};

  function niceStep(span) {
    if (!(span > 0)) return 0.01;
    const raw = span / 50, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }
  function decimals(step) {
    const s = String(step);
    return s.includes('e-') ? parseInt(s.split('e-')[1], 10) : (s.split('.')[1] || '').length;
  }
  function fmt(value, spec) {
    if (spec.kind === 'int') return String(Math.round(Number(value)));
    const step = spec.step || niceStep(spec.max - spec.min);
    return Number(value).toFixed(decimals(step));
  }

  // ---- rows / controls ---------------------------------------------------------------
  // spec: {kind, label, help, choices|options, min, max, step, restart, disabled}
  function row(sec, key, spec, value) {
    const status = el('span', {class: 'status'});
    const r = el('div', {class: 'row' + (spec.disabled ? ' disabled' : ''), 'data-section': sec, 'data-key': key});
    const labelId = 'label-' + sec + '-' + key;
    const helpId = 'help-' + sec + '-' + key;
    const label = el('span', {class: 'label', id: labelId, text: spec.label});
    if (spec.restart) label.appendChild(el('span', {class: 'restart-tag', text: 'restart'}));
    r.appendChild(label);
    r.appendChild(el('span', {class: 'help', id: helpId, text: spec.help || ''}));
    const control = el('div', {class: 'control'});
    r.appendChild(control);
    r.appendChild(status);

    const setStatus = (text, error) => { status.textContent = text || ''; status.classList.toggle('error', !!error); };
    let revert = () => {};   // per control: put the model's value back after a refused change
    const commit = v => {
      setStatus('Applying…');
      post('set', {section: sec, key, value: v}).then(res => {
        if (res.restart_required) { restartRequired = true; updateBanner(); }
        if (res.ok === false) { setStatus(res.message || 'Failed', true); revert(); return; }
        setStatus(res.message || '');
      });
    };

    let input;
    if (spec.kind === 'bool') {
      input = el('input', {type: 'checkbox', class: 'toggle', 'data-key': key, checked: !!value, disabled: spec.disabled,
                           onchange: e => commit(!!e.target.checked)});
      revert = () => { input.checked = !!value; };
    } else if (spec.kind === 'choice') {
      input = el('select', {'data-key': key, disabled: spec.disabled, onchange: e => commit(e.target.value)});
      const options = spec.options || (spec.choices || []).map(c => [c, c]);
      for (const [id, name] of options) input.appendChild(el('option', {value: id, text: name}));
      input.value = value == null ? '' : String(value);
      revert = () => { input.value = value == null ? '' : String(value); };
    } else if (spec.kind === 'int' || spec.kind === 'float') {
      const step = spec.kind === 'int' ? 1 : (spec.step || niceStep(spec.max - spec.min));
      const valueEl = el('span', {class: 'value', text: fmt(value, spec)});
      input = el('input', {type: 'range', 'data-key': key, min: spec.min, max: spec.max, step, value: Number(value),
                           disabled: spec.disabled,
                           oninput: e => { valueEl.textContent = fmt(e.target.value, spec); },
                           onchange: e => commit(spec.kind === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))});
      revert = () => { input.value = Number(value); valueEl.textContent = fmt(value, spec); };
      labelled(input, labelId, spec.help ? helpId : null);
      control.appendChild(input);
      control.appendChild(valueEl);
      return r;
    } else {  // str / list
      const isList = spec.kind === 'list';
      const shown = isList ? (Array.isArray(value) ? value.join(', ') : String(value || '')) : (value == null ? '' : String(value));
      input = el('input', {type: 'text', class: spec.wide ? 'wide' : '', 'data-key': key, value: shown, disabled: spec.disabled,
                           placeholder: spec.placeholder || '', spellcheck: 'false', autocomplete: 'off'});
      input.dataset.committed = shown;
      revert = () => { input.value = shown; input.dataset.committed = shown; };
      const submit = () => {
        if (input.dataset.detaching) return;   // re-render pulled it out from under the caret
        const raw = input.value;
        if (raw === input.dataset.committed) return;
        input.dataset.committed = raw;
        commit(isList ? raw.split(',').map(s => s.trim()).filter(Boolean) : raw.trim());
      };
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
        else if (e.key === 'Escape') { input.value = input.dataset.committed; input.blur(); }
      });
      input.addEventListener('blur', submit);
    }
    control.appendChild(input);
    labelled(input, labelId, spec.help ? helpId : null);
    return r;
  }

  function labelled(input, labelId, helpId) {
    input.setAttribute('aria-labelledby', labelId);
    if (helpId) input.setAttribute('aria-describedby', helpId);
  }

  function settingRow(sec, key, extra) {
    const spec = Object.assign({}, fields()[key] || {kind: 'str', label: key}, extra || {});
    return row(sec, key, spec, section(sec)[key]);
  }

  function button(label, opts) {
    const o = opts || {};
    return el('button', Object.assign({type: 'button', class: o.class || 'btn', text: label, disabled: o.disabled}, o.attrs || {},
                                       {onclick: o.onclick}));
  }

  // ---- tabs ------------------------------------------------------------------------------
  const renderers = {
    general() {
      const g = section('general');
      const frag = [];
      frag.push(el('h2', {text: 'General'}));
      frag.push(el('p', {class: 'lead', text: 'How Veronica shows up and listens.'}));
      frag.push(row('general', 'language', {kind: 'choice', label: 'Language', help: 'Auto detects Hindi or English per turn.', options: LANGUAGES}, g.language));
      frag.push(row('general', 'start_at_login', {
        kind: 'bool', label: 'Start at login', disabled: !g.can_start_at_login,
        help: g.can_start_at_login ? 'Launch the app bundle when you log in.' : 'Build the app first (make app).',
      }, g.start_at_login));
      frag.push(row('general', 'hud_mode', {kind: 'choice', label: 'HUD style', help: 'Full card with the conversation, or a small orb.', options: HUD_MODES}, g.hud_mode));
      frag.push(settingRow('general', 'hud_hide_after_s'));
      frag.push(settingRow('general', 'hud_particles'));
      frag.push(settingRow('general', 'hud_intensity'));
      frag.push(settingRow('general', 'ptt_enabled'));
      frag.push(el('div', {class: 'actions'}, [
        button('Open Login Items…', {attrs: {'data-cmd': 'open_login_items'}, onclick: () => post('open_login_items')}),
      ]));
      return frag;
    },
    voice() {
      const v = section('voice');
      const voices = Array.isArray(v.voices) ? v.voices : [];
      const english = voices.filter(x => !x.hindi).map(x => [x.id, x.name]);
      const hindi = voices.filter(x => x.hindi).map(x => [x.id, x.name]);
      const test = lang => button(lang === 'hi' ? 'Test Hindi voice' : 'Test voice', {
        attrs: {'data-cmd': 'test_voice', 'data-lang': lang},
        onclick: () => post('test_voice', {lang}),
      });
      return [
        el('h2', {text: 'Voice'}),
        el('p', {class: 'lead', text: 'Pick how Veronica sounds. Changing a voice plays a short sample.'}),
        row('voice', 'voice', {kind: 'choice', label: 'English voice', options: english}, v.voice),
        row('voice', 'hindi_voice', {kind: 'choice', label: 'Hindi voice', options: hindi}, v.hindi_voice),
        row('voice', 'speed', SPEED, v.speed == null ? 1.0 : v.speed),
        el('div', {class: 'actions'}, [test('en'), test('hi')]),
      ];
    },
    listening() {
      const frag = [el('h2', {text: 'Listening'}), el('p', {class: 'lead', text: 'Wake word, turn-taking and how patient she is.'})];
      frag.push(el('h3', {text: 'Conversation'}));
      for (const k of ['followup_window_s', 'confirm_listen_s', 'vad_silence_ms', 'max_utterance_s']) frag.push(settingRow('listening', k));
      frag.push(el('h3', {text: 'Wake word'}));
      for (const k of ['wake_phrases', 'wake_min_rms', 'wake_window_s', 'wake_hop_s']) frag.push(settingRow('listening', k, k === 'wake_phrases' ? {wide: true} : null));
      frag.push(el('h3', {text: 'Microphone'}));
      frag.push(settingRow('listening', 'input_volume_floor'));
      return frag;
    },
    briefings() {
      const b = section('briefings');
      return [
        el('h2', {text: 'Briefings'}),
        el('p', {class: 'lead', text: 'A spoken morning briefing and gentle reminders.'}),
        row('briefings', 'briefing_enabled', {kind: 'bool', label: 'Morning briefing'}, b.briefing_enabled),
        row('briefings', 'briefing_time', {kind: 'str', label: 'Briefing time', help: '24-hour HH:MM.', placeholder: '08:00'}, b.briefing_time),
        row('briefings', 'nudges_enabled', {kind: 'bool', label: 'Reminder nudges'}, b.nudges_enabled),
        row('briefings', 'nudge_minutes', {kind: 'int', label: 'Nudge ahead (minutes)', help: 'How early before an event she speaks up.', min: 1, max: 60}, b.nudge_minutes == null ? 5 : b.nudge_minutes),
      ];
    },
    brain() {
      const b = section('brain');
      const frag = [
        el('h2', {text: 'Brain'}),
        el('p', {class: 'lead', text: 'Which assistant does the thinking, and how. Each uses its own login.'}),
      ];
      // What's answering right now: "Codex", or "Claude (for Codex)" while
      // the preferred brain is out (usage limit / not logged in).
      if (b.brain_label) frag.push(el('p', {class: 'lead', id: 'brain-label', text: 'Now on ' + b.brain_label + '.'}));
      const brains = ((fields().brain_backend || {}).choices || []).map(c => [c, c.charAt(0).toUpperCase() + c.slice(1)]);
      frag.push(settingRow('brain', 'brain_backend', {options: brains}));
      frag.push(settingRow('brain', 'brain_failover'));
      frag.push(settingRow('brain', 'brain_failover_order', {wide: true}));
      frag.push(settingRow('brain', 'brain_limit_cooldown_min'));
      frag.push(el('h3', {text: 'Their own tools'}));
      frag.push(settingRow('brain', 'codex_native_tools'));
      frag.push(settingRow('brain', 'antigravity_native_tools'));
      frag.push(settingRow('brain', 'copilot_native_tools'));
      frag.push(el('h3', {text: 'Thinking'}));
      frag.push(settingRow('brain', 'effort'));
      frag.push(settingRow('brain', 'memory_enabled'));
      frag.push(settingRow('brain', 'brain_cwd', {wide: true}));
      frag.push(settingRow('brain', 'computer_trust_s'));
      frag.push(settingRow('brain', 'preapprove_by_wording'));
      return frag;
    },
    history() {
      return [history.view()];
    },
    about() {
      const a = section('about');
      const upd = a.update || {};
      const statusEl = el('div', {class: 'status'});
      about.statusEl = statusEl;
      // A fresh update result / the updating flag flipping supersedes whatever
      // the last button reply said.
      const sig = (a.updating ? 'u' : '-') + '|' + (upd.detail || '');
      if (about.sig !== undefined && about.sig !== sig) { about.text = ''; about.cls = ''; }
      about.sig = sig;
      if (about.text) { statusEl.textContent = about.text; statusEl.className = 'status ' + (about.cls || ''); }
      else if (upd.detail) statusEl.textContent = upd.detail;
      const setStatus = (text, cls) => { about.text = text || ''; about.cls = cls || ''; statusEl.textContent = about.text; statusEl.className = 'status ' + about.cls; };
      const run = (cmd, busyText, onReply) => () => {
        setStatus(busyText, '');
        post(cmd).then(res => onReply(res || {}));
      };
      const buttons = el('div', {class: 'actions'}, [
        button('Check now', {attrs: {'data-cmd': 'check_update'}, onclick: run('check_update', 'Checking…', res => {
          if (res.ok === false) return setStatus(res.message || 'Check failed', 'error');
          setStatus(res.available ? (res.detail || 'An update is available.') : (res.detail || LATEST), res.available ? '' : 'good');
        })}),
        button('Update & restart', {class: 'primary', disabled: !!a.updating, attrs: {'data-cmd': 'update_now'}, onclick: run('update_now', 'Updating…', res => {
          setStatus(res.message || (res.ok === false ? 'Update failed' : 'Updating, back in a moment.'), res.ok === false ? 'error' : '');
        })}),
        button('Restart', {attrs: {'data-cmd': 'restart'}, onclick: run('restart', 'Restarting…', res => {
          setStatus(res.message || 'Restarting…', res.ok === false ? 'error' : '');
        })}),
        button('Open log', {attrs: {'data-cmd': 'open_logs'}, onclick: () => post('open_logs')}),
      ]);
      const dl = el('dl', {}, [
        el('dt', {text: 'Version'}), el('dd', {text: a.version || ''}),
        el('dt', {text: 'Build'}), el('dd', {}, [
          document.createTextNode(a.build || 'unknown'),
          a.dirty ? el('span', {class: 'dirty', text: '  (modified working tree)'}) : null,
        ]),
        el('dt', {text: 'Built'}), el('dd', {text: a.built_at || ''}),
        el('dt', {text: 'Update'}), el('dd', {text: a.updating ? 'Updating…' : (upd.available ? (upd.detail || 'Available') : 'None known')}),
        el('dt', {text: 'Log'}), el('dd', {class: 'log-path', text: a.log_path || ''}),
      ]);
      return [el('div', {class: 'about'}, [
        el('h2', {text: 'About'}),
        el('div', {class: 'version', text: a.describe || ('Veronica ' + (a.version || ''))}),
        dl, buttons, statusEl,
      ])];
    },
  };
  const about = {text: '', cls: '', statusEl: null, sig: undefined};

  // ---- history -----------------------------------------------------------------------------
  const history = {
    query: '', items: [], loading: false, message: '', confirming: false, timer: null, root: null,
    view() {
      const root = el('div', {class: 'history'});
      this.root = root;
      const search = el('input', {type: 'text', class: 'search', placeholder: 'Search what you said or what she answered', value: this.query,
                                  'aria-label': 'Search history', spellcheck: 'false', autocomplete: 'off'});
      search.addEventListener('input', () => {
        this.query = search.value;
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.load(), HISTORY_DEBOUNCE_MS);
      });
      search.addEventListener('keydown', e => { if (e.key === 'Escape') { search.value = ''; this.query = ''; this.load(); } });
      const clear = button('Clear all', {class: 'btn danger clear', onclick: () => { this.confirming = true; this.draw(); }});
      root.appendChild(el('div', {class: 'toolbar'}, [search, clear]));
      root.appendChild(el('div', {class: 'confirm hidden'}, [
        el('span', {class: 'msg', text: 'Forget every conversation? This cannot be undone.'}),
        button('Yes, clear', {class: 'btn danger yes', onclick: () => {
          this.confirming = false;
          this.query = '';
          search.value = '';
          post('clear_history').then(res => {
            this.message = res.ok === false ? (res.message || 'Could not clear history.') : '';
            this.sticky = !!this.message;
            this.load();
          });
          this.draw();
        }}),
        button('No', {class: 'btn no', onclick: () => { this.confirming = false; this.draw(); }}),
      ]));
      root.appendChild(el('div', {class: 'list'}));
      this.draw();
      return root;
    },
    load() {
      this.loading = true;
      this.draw();
      const q = this.query;
      post('history', {query: q, limit: HISTORY_LIMIT, offset: 0}).then(res => {
        if (q !== this.query) return;                  // a newer search is in flight
        this.loading = false;
        this.items = Array.isArray(res.items) ? res.items : [];
        if (res.ok === false) this.message = res.message || '';
        else if (!this.sticky) this.message = '';
        this.sticky = false;
        this.draw();
      });
    },
    draw() {
      const root = this.root;
      if (!root || !root.isConnected) return;
      root.querySelector('.confirm').classList.toggle('hidden', !this.confirming);
      const list = root.querySelector('.list');
      list.textContent = '';
      if (this.message) list.appendChild(el('div', {class: 'notice', role: 'alert', text: this.message}));
      if (this.loading && !this.items.length) list.appendChild(el('div', {class: 'loading', text: 'Loading…'}));
      else if (!this.items.length) list.appendChild(el('div', {class: 'empty', text: this.query ? 'No matches.' : 'Nothing yet.'}));
      for (const t of this.items) {
        list.appendChild(el('div', {class: 'turn', 'data-id': t.id}, [
          el('div', {class: 'when', text: when(t.ts)}),
          el('div', {class: 'you', text: 'You: ' + (t.heard || '')}),
          el('div', {class: 'her', text: 'Veronica: ' + (t.reply || '')}),
          button('Forget', {class: 'btn forget', onclick: () => post('forget_turn', {id: t.id}).then(res => {
            this.message = res.ok === false ? (res.message || 'Could not forget that.') : '';
            this.sticky = !!this.message;
            this.load();
          })}),
        ]));
      }
    },
  };

  function when(ts) {
    const s = String(ts || '');
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/);
    if (!m) return s;
    const date = new Date(+m[1], +m[2] - 1, +m[3]);
    let day;
    try { day = date.toLocaleDateString(undefined, {day: 'numeric', month: 'short', year: 'numeric'}); }
    catch (e) { day = m[3] + '/' + m[2] + '/' + m[1]; }
    return m[4] + ':' + m[5] + ' · ' + day;
  }

  // ---- render ---------------------------------------------------------------------------
  function updateBanner() {
    bannerEl.classList.toggle('hidden', !restartRequired);
  }

  function renderTabs() {
    if (tabsEl.childElementCount === 0) {
      for (const [name, label] of TABS) {
        tabsEl.appendChild(el('button', {type: 'button', role: 'tab', 'data-tab': name, text: label, onclick: () => window.settings.select(name)}));
      }
    }
    for (const b of tabsEl.children) {
      const on = b.dataset.tab === tab;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    }
  }

  function render() {
    renderTabs();
    updateBanner();
    // Remember an in-progress text edit / focus so a state push mid-typing
    // (they arrive after every applied change) doesn't eat the keystrokes.
    const active = document.activeElement;
    let keep = null;
    if (active && paneEl.contains(active) && active.dataset && active.dataset.key) {
      keep = {key: active.dataset.key, text: active.type === 'text' ? active.value : null,
              committed: active.dataset.committed, sel: active.type === 'text' ? active.selectionStart : null};
    }
    if (tab === 'history' && paneEl.dataset.tab === 'history' && paneEl.querySelector('.history')) {
      return;   // nothing on this tab renders from state; keep the list and its search box
    }
    paneEl.dataset.tab = tab;
    // Chromium fires `blur` on a focused input that is removed from the DOM;
    // flag it first so the text handler doesn't post the half-typed value.
    if (active && paneEl.contains(active) && active.dataset) active.dataset.detaching = '1';
    paneEl.textContent = '';
    if (!model) { paneEl.appendChild(el('p', {class: 'lead', text: 'Loading…'})); return; }
    const parts = (renderers[tab] || renderers.general)();
    for (const p of parts) if (p) paneEl.appendChild(p);
    if (keep) {
      const again = paneEl.querySelector('.control [data-key="' + keep.key + '"]');
      if (again) {
        if (keep.text != null && keep.text !== keep.committed && again.type === 'text') {
          again.value = keep.text;
          try { again.setSelectionRange(keep.sel, keep.sel); } catch (e) { /* not a text control */ }
        }
        again.focus({preventScroll: true});
      }
    }
    paneEl.scrollTop = 0;
  }

  bannerEl.querySelector('button').addEventListener('click', () => post('restart'));
  render();
})();
