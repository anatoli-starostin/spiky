// evolved_assoc.js — viewer for the evolved Dale associative-memory winner. Topology canvas +
// store-then-recall before/after panels (dynamics precomputed from the real net). Dependency-free.
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');b.style.display='block';b.textContent='⚠ '+m;}
  window.addEventListener('error', function(e){showErr((e.error&&e.error.stack)||e.message);});
  var css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();};
  var DATA_URL='evolved_assoc_data.json';
  fetch(DATA_URL).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status+' '+DATA_URL);return r.json();})
    .then(init).catch(function(e){showErr('could not load '+DATA_URL+' — '+e.message);});

  var D, pos={};
  function roleColor(n){return n.role==='input'?css('--spike'):(n.exc?'#3fbf6f':'#e5534b');}

  function init(d){
    D=d;
    document.getElementById('chips').innerHTML=[
      'mean-gain <b>'+d.meta.mean_gain+'</b>',
      'generalization <b>'+Math.round(d.stats.gen_pos_frac*100)+'%</b> of fresh pairs',
      'neurons <b>'+d.nodes.length+'</b>',
      'excitatory <b>'+d.stats.exc_neurons+'</b> · inhibitory <b>'+d.stats.inh_neurons+'</b>',
      'synapses <b>'+d.stats.synapses+'</b>',
      'recurrent <b>'+d.stats.recurrent+'</b>',
      'exc delays <b>'+(d.stats.exc_delays.length?Math.min.apply(null,d.stats.exc_delays)+'–'+Math.max.apply(null,d.stats.exc_delays):'–')+'</b>'
    ].map(function(t){return '<span class="chip">'+t+'</span>';}).join('');
    setupCanvas(); layout(); draw();
    window.addEventListener('resize', function(){resize();layout();draw();});
    buildDemos();
    document.getElementById('side').innerHTML = sideText();
  }
  function sideText(){
    return '<h3>How to read it</h3><p class="small">This net was taught <b>2 arbitrary input→output-order mappings</b> per episode via spike-timing plasticity (STDP), then tested on the inputs alone. '
      +'Nodes: <span style="color:'+css('--spike')+'">cyan = input</span> (latency-coded), <span style="color:#3fbf6f">green = excitatory (RS)</span>, <span style="color:#e5534b">red = inhibitory (FS)</span> neurons; larger ringed = outputs. '
      +'Edges: green = excitatory (plastic, evolved delay), red = inhibitory (fixed weight, delay 1). Dashed = recurrent/lateral (where the memory lives). '
      +'Dale\'s law holds: sign is set by the presynaptic neuron type, and only excitatory synapses learn.</p>'
      +'<h3>Result</h3><p class="small">Mean storage-gain <b>'+D.meta.mean_gain+'</b> (recall_after − recall_before), and it <b>generalizes</b> — positive gain on '+Math.round(D.stats.gen_pos_frac*100)+'% of fresh, never-seen random-target pairs. Feedforward alone scores ≈ chance; the storage does the work.</p>';
  }

  var canv={};
  function setupCanvas(){var cv=document.getElementById('graph');canv={cv:cv,ctx:cv.getContext('2d'),w:0,h:+cv.getAttribute('height')};resize();}
  function resize(){var dpr=Math.min(2,window.devicePixelRatio||1);var w=Math.round(canv.cv.clientWidth||820);canv.cv.style.height=canv.h+'px';canv.cv.width=w*dpr;canv.cv.height=canv.h*dpr;canv.ctx.setTransform(dpr,0,0,dpr,0,0);canv.w=w;}
  function layout(){
    var W=canv.w,H=canv.h,pad=40;pos={};
    var ins=D.nodes.filter(function(n){return n.role==='input';});
    var outs=D.nodes.filter(function(n){return n.role==='output';});
    var hid=D.nodes.filter(function(n){return n.role==='hidden';});
    ins.forEach(function(n,i){pos[n.id]={x:pad+20,y:pad+i*(H-2*pad)/(ins.length-1||1)};});
    outs.forEach(function(n,i){pos[n.id]={x:W-pad-20,y:pad+30+i*(H-2*pad-60)/(outs.length-1||1)};});
    hid.forEach(function(n,i){pos[n.id]={x:W*0.5,y:H*0.5+(i-(hid.length-1)/2)*70};});
  }
  function ctrl(a,b,back){var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,dx=b.x-a.x,dy=b.y-a.y,l=Math.hypot(dx,dy)||1,bow=(back?-1:1)*Math.min(55,l*0.22);return{x:mx-dy/l*bow,y:my+dx/l*bow};}
  function draw(){
    var ctx=canv.ctx;ctx.clearRect(0,0,canv.w,canv.h);
    D.synapses.forEach(function(s){
      var a=pos[nid(s.src)],b=pos[nid(s.tgt)];if(!a||!b)return;
      var back=b.x<a.x-1;ctx.strokeStyle=s.exc?'#3fbf6f':'#e5534b';ctx.globalAlpha=s.recurrent?0.95:0.45;
      ctx.lineWidth=0.8+Math.min(4,s.w_mag*1.2);ctx.setLineDash(s.recurrent?[5,3]:[]);
      ctx.beginPath();
      if(s.src===s.tgt){ctx.moveTo(a.x,a.y-12);ctx.bezierCurveTo(a.x+44,a.y-48,a.x+44,a.y+48,a.x,a.y+12);}
      else{var cp=ctrl(a,b,back);ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(cp.x,cp.y,b.x,b.y);}
      ctx.stroke();
    });
    ctx.setLineDash([]);ctx.globalAlpha=1;
    D.nodes.forEach(function(n){
      var p=pos[n.id];if(!p)return;var r=n.role==='output'?15:11;
      if(n.role==='output'){ctx.strokeStyle=css('--out');ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,r+5,0,7);ctx.stroke();}
      ctx.fillStyle='#141b24';ctx.strokeStyle=roleColor(n);ctx.lineWidth=2.5;ctx.beginPath();
      if(n.role==='input')ctx.rect(p.x-r,p.y-r,2*r,2*r);else ctx.arc(p.x,p.y,r,0,7);
      ctx.fill();ctx.stroke();
      ctx.fillStyle=css('--ink');ctx.font='10px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(n.id, p.x, p.y);
    });
  }
  function nid(raw){for(var i=0;i<D.nodes.length;i++)if(D.nodes[i].raw===raw)return D.nodes[i].id;return raw;}

  function ord(a){return a.map(function(d){return 'o'+d;}).join(' › ');}
  function buildDemos(){
    var box=document.getElementById('demos');box.innerHTML='';
    D.demos.forEach(function(dm,i){
      var el=document.createElement('div');el.className='panel';
      var coldOK=dm.cold_fires, trOK=dm.trained_fires;
      el.innerHTML='<h2>Example '+(i+1)+' — teach an arbitrary mapping, then recall from the input alone</h2>'
        +'<p class="small">Input first-spike ticks: <span class="mono">'+Object.keys(dm.input_ticks).map(function(k){return k+'@'+dm.input_ticks[k];}).join('  ')+'</span> · taught target order <span class="mono">'+ord(dm.target_order)+'</span></p>'
        +'<div class="compare">'
        +'<div class="cmp"><h3>BEFORE storage (inputs only, no training)</h3>'
        +'<p class="small">outputs '+(coldOK?'fire':'<b style="color:var(--bad)">don\'t all fire</b>')+' — order <span class="mono">'+ord(dm.cold_order)+'</span></p>'
        +'<div class="pill '+(coldOK?'bad':'bad')+'">'+(coldOK?'no reliable order':'recall fails — untrained')+'</div></div>'
        +'<div class="cmp"><h3>AFTER 8 STDP storage epochs</h3>'
        +'<p class="small">outputs '+(trOK?'<b style="color:var(--ok)">fire</b>':'don\'t fire')+' — recalled order <span class="mono">'+ord(dm.trained_order)+'</span> vs target <span class="mono">'+ord(dm.target_order)+'</span></p>'
        +'<div class="pill '+(trOK?'ok':'bad')+'">'+(trOK?'storage enabled recall':'still silent')+'</div></div>'
        +'</div>';
      box.appendChild(el);
    });
  }
})();
