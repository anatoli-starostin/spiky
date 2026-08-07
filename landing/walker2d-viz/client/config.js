// Runtime config for the Walker2d demo client — edit this, no build step needed.
//
// Point WALKER2D_WS at your deployed server's WebSocket URL. Rules:
//   ""  (empty)  -> connect to ws://<same-host-that-served-this-page>:8765
//                   (handy for local `python -m http.server`)
//   "wss://demo.example.com"  -> for a GitHub Pages (HTTPS) build talking to your TLS'd server.
//
// GitHub Pages is served over HTTPS, so it MUST use wss:// (a plain ws:// is blocked as mixed content).
//
// >>> DEPLOYER: replace YOUR_SERVER_HOST below with your server's TLS host before publishing to Pages —
//     the same value as your .env DOMAIN (e.g. demo.example.com, or a no-domain <dashed-ip>.sslip.io host).
//     The in-app server-URL field stays editable at runtime, so you can also just type a URL there to test.
window.WALKER2D_WS = "wss://YOUR_SERVER_HOST";
