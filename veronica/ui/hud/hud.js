(() => {
  const PALETTE = {
    idle:       {core:'#3a6df0', ring:'#4c7cff', glow:'rgba(76,124,255,.35)', speed:0.25},
    warming:    {core:'#6b7280', ring:'#9ca3af', glow:'rgba(156,163,175,.25)', speed:0.6},
    listening:  {core:'#22d3ee', ring:'#67e8f9', glow:'rgba(34,211,238,.45)', speed:0.6},
    thinking:   {core:'#f59e0b', ring:'#fbbf24', glow:'rgba(245,158,11,.40)', speed:1.6},
    speaking:   {core:'#8b5cf6', ring:'#a78bfa', glow:'rgba(139,92,246,.45)', speed:0.9},
    followup:   {core:'#22d3ee', ring:'#67e8f9', glow:'rgba(34,211,238,.30)', speed:0.4},
    confirming: {core:'#f97316', ring:'#fdba74', glow:'rgba(249,115,22,.45)', speed:0.8},
    error:      {core:'#ef4444', ring:'#f87171', glow:'rgba(239,68,68,.40)', speed:0.0},
  };
  const model = {state:'idle', heard:'', reply:'', tool:null, mic:0, ready:true,
                 voice:null, voiceStart:0, confirmStart:0};
  let micSmooth = 0, replyQueue = [], typing = false, replyGen = 0, pendingTimeout = null;

  const $ = id => document.getElementById(id);
  const heardEl = $('heard').querySelector('.msg');
  const replyEl = $('reply').querySelector('.msg');
  const toolEl = $('tool').querySelector('.msg');
  const badgeEl = $('tool').querySelector('.badge');

  function clearReply() {
    replyGen++;
    replyQueue = [];
    typing = false;
    if (pendingTimeout !== null) { clearTimeout(pendingTimeout); pendingTimeout = null; }
  }

  function typeNext() {
    if (typing || replyQueue.length === 0) return;
    typing = true;
    const gen = replyGen;
    const s = replyQueue.shift();
    const start = replyEl.textContent.length ? replyEl.textContent + ' ' : '';
    let i = 0;
    const step = () => {
      if (gen !== replyGen) { typing = false; return; }
      try {
        i = Math.min(s.length, i + 2);           // ~40 chars/s at 20 fps ticks
        replyEl.textContent = start + s.slice(0, i);
        if (i < s.length) { pendingTimeout = setTimeout(step, 50); return; }
      } finally {
        if (i >= s.length) { typing = false; pendingTimeout = null; }
      }
      typeNext();
    };
    step();
  }

  const hud = {
    push(ev) {
      const payload = ev && ev.payload;
      const kind = ev && ev.kind;
      try {
        switch (kind) {
          case 'state':
            if (payload === 'listening' && model.state !== 'followup') {
              model.heard = ''; model.reply = ''; clearReply();
              replyEl.textContent = ''; heardEl.textContent = '';
              model.tool = null; badgeEl.className = 'badge'; badgeEl.textContent = ''; toolEl.textContent = '';
            }
            if (payload === 'confirming') model.confirmStart = performance.now();
            model.state = payload; break;
          case 'heard': model.heard = payload || ''; heardEl.textContent = model.heard; break;
          case 'sentence': {
            const s = String(payload ?? '');
            if (!s) break;
            model.reply += (model.reply ? ' ' : '') + s; replyQueue.push(s); typeNext(); break;
          }
          case 'tool': {
            const t = payload && typeof payload === 'object' ? payload : {};
            const decision = t.decision || '';
            const summary = t.summary || '';
            model.tool = t; badgeEl.className = 'badge ' + decision;
            badgeEl.textContent = {auto:'⚡', ask:'?', allowed:'✓', declined:'✕'}[decision] || '';
            toolEl.textContent = summary.length > 60 ? summary.slice(0, 59) + '…' : summary;
            break;
          }
          case 'mic': model.mic = Math.max(0, Math.min(1, +payload || 0)); break;
          case 'voice': model.voice = payload; model.voiceStart = performance.now(); break;
          case 'warm': model.ready = !!(payload && payload.ready); break;
        }
      } catch (err) {
        console.error('hud.push failed', err);
      }
    },
    state() { return {state:model.state, heard:model.heard, reply:model.reply, tool:model.tool, mic:model.mic, ready:model.ready}; },
    setVisible(visible) {
      visible = !!visible;
      if (visible === rafActive) return;
      rafActive = visible;
      if (visible) { t0 = performance.now(); requestAnimationFrame(frame); }
    },
  };
  window.hud = hud;

  // ---- orb renderer --------------------------------------------------------
  const canvas = $('orb'), ctx = canvas.getContext('2d');
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = 150 * dpr; canvas.height = 150 * dpr; ctx.scale(dpr, dpr);
  const particles = Array.from({length: 28}, (_, i) => ({a: (i / 28) * Math.PI * 2, r: 52 + (i % 5) * 4, s: 0.2 + (i % 7) * 0.05}));
  let t0 = performance.now(), rot = 0;
  let rafActive = !(typeof document !== 'undefined' && document.visibilityState === 'hidden');

  if (typeof document !== 'undefined' && 'visibilityState' in document) {
    document.addEventListener('visibilitychange', () => {
      const visible = document.visibilityState !== 'hidden';
      if (visible === rafActive) return;
      rafActive = visible;
      if (visible) { t0 = performance.now(); requestAnimationFrame(frame); }
    });
  }

  function voiceLevel(now) {
    const v = model.voice; if (!v || !v.levels || !v.levels.length) return 0;
    const idx = Math.floor((now - model.voiceStart) / (v.step_ms || 50));
    return idx < v.levels.length ? v.levels[idx] : 0;
  }

  function frame(now) {
    const dt = (now - t0) / 1000; t0 = now;
    const p = PALETTE[model.state] || PALETTE.idle;
    rot += dt * p.speed;
    micSmooth += (model.mic - micSmooth) * 0.25;
    const cx = 75, cy = 75;
    ctx.clearRect(0, 0, 150, 150);

    let pulse = 0;
    if (model.state === 'listening' || model.state === 'followup') pulse = micSmooth;
    else if (model.state === 'speaking') pulse = voiceLevel(now);
    else if (model.state === 'thinking') pulse = 0.5 + 0.5 * Math.sin(now / 250);

    // glow
    const g = ctx.createRadialGradient(cx, cy, 10, cx, cy, 70);
    g.addColorStop(0, p.glow); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 150, 150);

    // core
    const coreR = 26 + pulse * 6;
    const cg = ctx.createRadialGradient(cx - 8, cy - 8, 4, cx, cy, coreR);
    cg.addColorStop(0, '#ffffff'); cg.addColorStop(0.25, p.core); cg.addColorStop(1, 'rgba(0,0,0,0.85)');
    ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.fillStyle = cg; ctx.fill();

    // rings
    ctx.lineWidth = 2 + pulse * 3; ctx.strokeStyle = p.ring;
    ctx.beginPath(); ctx.ellipse(cx, cy, 44 + pulse * 8, 30, rot, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.ellipse(cx, cy, 30, 44 + pulse * 8, -rot * 1.3, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;

    // particles / thinking dots
    ctx.fillStyle = p.ring;
    for (const q of particles) {
      const a = q.a + rot * q.s * (model.state === 'thinking' ? 4 : 1);
      const r = q.r + pulse * 6;
      ctx.globalAlpha = model.state === 'thinking' ? 0.9 : 0.35;
      ctx.beginPath(); ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r * 0.75, 1.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // confirming: "?" + countdown arc (5 s)
    if (model.state === 'confirming') {
      const frac = Math.max(0, 1 - (now - model.confirmStart) / 5000);
      ctx.beginPath(); ctx.arc(cx, cy, 58, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
      ctx.lineWidth = 3; ctx.strokeStyle = p.ring; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 26px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('?', cx, cy + 1);
    }
    // warming spinner
    if (model.state === 'warming') {
      ctx.beginPath(); ctx.arc(cx, cy, 58, rot * 2, rot * 2 + Math.PI * 0.6);
      ctx.lineWidth = 3; ctx.strokeStyle = p.ring; ctx.stroke();
    }
    if (rafActive) requestAnimationFrame(frame);
  }
  if (rafActive) requestAnimationFrame(frame);
})();
