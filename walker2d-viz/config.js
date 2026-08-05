// Runtime config for the Walker2d demo client — edit this, no build step needed.
//
// Point WALKER2D_WS at your deployed server's WebSocket URL. Rules:
//   ""  (empty)  -> connect to ws://<same-host-that-served-this-page>:8765
//                   (handy for local `python -m http.server`)
//   "wss://demo.example.com"  -> for a GitHub Pages (HTTPS) build talking to your TLS'd server.
//
// GitHub Pages is served over HTTPS, so it MUST use wss:// (a plain ws:// is blocked as mixed content).
window.WALKER2D_WS = "wss://89-169-96-79.sslip.io";
