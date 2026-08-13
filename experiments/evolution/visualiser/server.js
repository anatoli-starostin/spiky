// server.js — minimal, zero-dependency static file server for the
// hyperplane-LUT -> spiking-network visualiser.
//
//   node server.js            # serves on 0.0.0.0:8137
//   PORT=9000 node server.js  # override the port
//
// Binds 0.0.0.0 so the page is reachable over the tailnet (open
// http://nucstar:8137 from any machine on the tailnet).
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8137;
const HOST = '0.0.0.0';
const ROOT = path.join(__dirname, 'public');
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
};

http.createServer((req, res) => {
  let url = req.url.split('?')[0];
  if (url === '/') url = '/index.html';
  // basic path-traversal guard
  const safe = path.normalize(url).replace(/^(\.\.[/\\])+/, '');
  const file = path.join(ROOT, safe);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('404 not found'); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'text/plain' });
    res.end(data);
  });
}).listen(PORT, HOST, () => {
  console.log(`LUT->spiking visualiser running on http://${HOST}:${PORT}`);
  console.log(`Open it over the tailnet at  http://nucstar:${PORT}`);
});
