// Options page: stores the pairing code (and port) in chrome.storage.local;
// the service worker reconnects whenever either changes.
const DEFAULT_PORT = 8765;
const $ = (id) => document.getElementById(id);

const LABELS = {
  connected: 'Connected to Veronica.',
  connecting: 'Connecting…',
  disconnected: 'Not connected. Is Veronica running?',
  'needs-token': 'Paste the pairing code to connect.',
  'bad-token': 'Veronica rejected this pairing code. Paste the current one from browser_token.',
};

async function showStatus() {
  try {
    const s = await chrome.runtime.sendMessage({ type: 'status' });
    $('status').textContent = (s && LABELS[s.state]) || '';
  } catch {
    $('status').textContent = '';
  }
}

async function load() {
  const s = await chrome.storage.local.get({ token: '', port: DEFAULT_PORT });
  $('token').value = s.token;
  $('port').value = s.port;
  showStatus();
}

$('save').addEventListener('click', async () => {
  const token = $('token').value.trim();
  const port = parseInt($('port').value, 10) || DEFAULT_PORT;
  await chrome.storage.local.set({ token, port });
  $('status').textContent = 'Saved. Connecting…';
  setTimeout(showStatus, 1500);
});

load();
setInterval(showStatus, 3000);
