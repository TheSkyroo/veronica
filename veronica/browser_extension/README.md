# Veronica browser extension

Lets Veronica read and act on the page you have open in **Google Chrome** or
**Microsoft Edge** on Windows: list tabs, open a URL, read or search the page,
click a link or button, type into a field, scroll and go back.

It connects only to the Veronica app on the same PC (`ws://127.0.0.1:8765`)
and only after you paste Veronica's pairing code. Veronica can ask for that
fixed list of actions; it cannot run arbitrary scripts in your pages.

## Install (one time)

Start Veronica once first, so it creates the pairing code file
`%USERPROFILE%\.veronica\browser_token`.

### Chrome

1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose this folder (`veronica\browser_extension`
   inside your Veronica install; Veronica's "no browser connected" message
   shows the full path).
4. The extension's options page opens. Open
   `%USERPROFILE%\.veronica\browser_token` in Notepad, copy the code, paste it
   into **Pairing code** and click **Save**. The page should say
   *Connected to Veronica.*

### Microsoft Edge

1. Open `edge://extensions`.
2. Turn on **Developer mode** (left sidebar).
3. Click **Load unpacked** and choose this folder.
4. Paste the pairing code into the options page as above. (To reopen it:
   `edge://extensions` -> Veronica -> **Details** -> **Extension options**.)

You can install it in both browsers. Veronica then uses the one whose window
you focused most recently, and in it the active tab of the front window.

## Good to know

- Chrome and Edge show a "developer mode extensions" notice at startup;
  that is expected for an unpacked extension.
- Browser-internal pages (`chrome://`, `edge://`, the new-tab page) and the
  extension stores can't be read or clicked; Veronica tells you so.
- If Veronica isn't running the extension retries quietly (at least every
  30 seconds), so it reconnects on its own when Veronica starts.
- To change the port, set `VERONICA_BROWSER_PORT` for Veronica and the same
  number in the extension's options.
- If you delete `browser_token`, Veronica makes a new code at its next start;
  paste the new one into the options page.
- The extension ID is fixed (`kbjjfiapkdanjhojndmgepdlnokoilhe`) by the key in
  `manifest.json`; Veronica only accepts connections from that ID. If you fork
  the extension with a different key, add its ID to
  `VERONICA_BROWSER_EXTENSION_ID` (comma-separated).
