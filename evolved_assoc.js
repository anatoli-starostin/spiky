// evolved_assoc.js — interactive viewer for the evolved Dale associative-memory winner.
// Static topology canvas + an ANIMATION: pick a stored input, watch the real spike trains play
// back BEFORE (cold) vs AFTER (trained) STDP storage — neurons flash as they spike, signals travel
// along synapses respecting per-edge delay. All dynamics are the net's actual recorded spikes.
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');b.style.display='block';b.textContent='⚠ '+m;}
  window.addEventListener('error', function(e){showErr((e.error&&e.error.stack)||e.message);});
  var css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();};
  var DATA_URL='evolved_assoc_data.json';
  fetch(DATA_URL).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status+' '+DATA_URL);return r.json();})
    .then(init).catch(function(e){showErr('could not load '+DATA_URL+' — '+e.message);});

  var D;
  function roleColor(n){return n.role==='input'?css('--spike'):(n.exc?'#3fbf6f':'#e5534b');}
  function nid(raw){for(var i=0;i<D.nodes.length;i++)if(D.nodes[i].raw===raw)return D.nodes[i].id;return raw;}
  function nodeById(id){for(var i=0;i<D.nodes.length;i++)if(D.nodes[i].id===id)return D.nodes[i];return null;}

  function init(d){
    D=d;
    document.getElementById('chips').innerHTML=[
      'mean-gain <b>'+d.meta.mean_gain+'</b>',
      'generalization <b>'+Math.round(d.stats.gen_pos_frac*100)+'%</b> of fresh pairs',
      'neurons <b>'+d.nodes.length+'</b>',
      'excitatory <b>'+d.stats.exc_neurons+'</b> · inhibitory <b>'+d.stats.inh_neurons+'</b>',
      'synapses <b>'+d.stats.synapses+'</b>',
      'recurrent <b>'+d.stats.recurrent+'</b>'
    ].map(function(t){return '<span class="chip">'+t+'</span>';}).join('');
    setupStatic();
    document.getElementById('side').innerHTML=sideText();
    renderStdp();
    setupAnim();
    window.addEventListener('resize', function(){staticResize();layoutStatic();drawStatic();animResizeAll();});
  }
  function sideText(){
    return '<h3>How to read it</h3><p class="small">This net was taught <b>2 arbitrary input→output-order mappings</b> per episode via spike-timing plasticity (STDP), then tested on the inputs alone. '
      +'Nodes: <span style="color:'+css('--spike')+'">cyan = input</span> (latency-coded), <span style="color:#3fbf6f">green = excitatory (RS)</span>, <span style="color:#e5534b">red = inhibitory (FS)</span>; larger ringed = outputs. '
      +'Edges: green = excitatory (plastic, evolved delay), red = inhibitory (fixed, delay 1). Dashed = recurrent/lateral (where the memory lives).</p>'
      +'<h3>Result</h3><p class="small">Mean storage-gain <b>'+D.meta.mean_gain+'</b> (recall tau-b after − before), generalizing to '+Math.round(D.stats.gen_pos_frac*100)+'% of fresh random-target pairs. Before storage the outputs are <b>silent</b>; STDP makes them fire with timing that tracks the taught order (graded, tau-b ≈ 0.5–0.8).</p>';
  }

  function renderStdp(){
    var el=document.getElementById('stdp');if(!el)return;
    if(!D.stdp||!D.stdp.evolved){el.style.display='none';return;}
    var ev=D.stdp.evolved,sd=D.stdp.seed;
    var order=[['learning_rate','learning rate'],['ltp_max','LTP cap (ltp_max)'],['ltd_max','LTD cap (ltd_max)'],
               ['weight_scaling_cf','weight_scaling_cf'],['stdp_decay','stdp_decay'],['weight_decay','weight_decay'],['max_weight','max_weight']];
    var rows=order.filter(function(o){return ev[o[0]]!=null;}).map(function(o){
      var e=ev[o[0]],s=sd[o[0]],dir=e<s*0.98?'▼':(e>s*1.02?'▲':'≈');
      var gentle=(o[0]==='ltp_max'||o[0]==='ltd_max'||o[0]==='learning_rate');
      var col=gentle?(e<s?'var(--ok)':'var(--bad)'):'var(--ink)';
      return '<tr><td>'+o[1]+'</td><td class="mono">'+e+'</td><td class="mono" style="color:#66727f">'+s+'</td><td style="color:'+col+';text-align:center">'+dir+'</td></tr>';
    }).join('');
    el.innerHTML='<h2>The evolved learning rule</h2>'
      +'<p class="small">Unlike the fixed-STDP demos, <b>this net also evolved its own STDP rule</b> — the excitatory-plasticity genes were mutated and selected alongside the topology. The winner <b>beats the fixed-STDP net (0.708 vs 0.674)</b>, and it does so with <b>gentler plasticity</b>: smaller LTP/LTD caps and a lower learning rate than the hand-set seed.</p>'
      +'<table class="gtab"><tr><th>excitatory-STDP gene</th><th>evolved</th><th>seed</th><th>Δ</th></tr>'+rows+'</table>'
      +'<p class="small" style="margin-top:8px;">Standouts: <b>ltp_max '+ev.ltp_max+'</b> vs 1.0 and <b>ltd_max '+ev.ltd_max+'</b> vs 1.2 (potentiation/depression caps cut ~5×/2×), <b>learning_rate '+ev.learning_rate+'</b> vs 0.05. Across the whole top of the population learning_rate evolved <i>below</i> 0.05 — evolution favored small, well-tempered weight updates that store without destabilizing the net (aggressive STDP silences the outputs).</p>';
  }

  // ---------- shared geometry ----------
  function makePos(W,H){
    var pad=38,pos={};
    var ins=D.nodes.filter(function(n){return n.role==='input';});
    var outs=D.nodes.filter(function(n){return n.role==='output';});
    var hid=D.nodes.filter(function(n){return n.role==='hidden';});
    ins.forEach(function(n,i){pos[n.id]={x:pad+16,y:pad+i*(H-2*pad)/(ins.length-1||1)};});
    outs.forEach(function(n,i){pos[n.id]={x:W-pad-16,y:pad+24+i*(H-2*pad-48)/(outs.length-1||1)};});
    hid.forEach(function(n,i){pos[n.id]={x:W*0.5,y:H*0.5+(i-(hid.length-1)/2)*Math.min(70,(H-2*pad)/(hid.length||1))};});
    return pos;
  }
  function ctrl(a,b,back){var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,dx=b.x-a.x,dy=b.y-a.y,l=Math.hypot(dx,dy)||1,bow=(back?-1:1)*Math.min(48,l*0.22);return{x:mx-dy/l*bow,y:my+dx/l*bow};}
  function quad(a,cp,b,u){var m=1-u;return{x:m*m*a.x+2*m*u*cp.x+u*u*b.x,y:m*m*a.y+2*m*u*cp.y+u*u*b.y};}
  function edgePath(ctx,a,b,selfLoop){
    if(selfLoop){ctx.moveTo(a.x,a.y-11);ctx.bezierCurveTo(a.x+40,a.y-44,a.x+40,a.y+44,a.x,a.y+11);return null;}
    var back=b.x<a.x-1,cp=ctrl(a,b,back);ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(cp.x,cp.y,b.x,b.y);return cp;
  }

  // ---------- static topology canvas (#graph) ----------
  var sc={};
  function setupStatic(){var cv=document.getElementById('graph');sc={cv:cv,ctx:cv.getContext('2d'),h:+cv.getAttribute('height')};staticResize();layoutStatic();drawStatic();}
  function staticResize(){var dpr=Math.min(2,window.devicePixelRatio||1);var w=Math.round(sc.cv.clientWidth||820);sc.cv.style.height=sc.h+'px';sc.cv.width=w*dpr;sc.cv.height=sc.h*dpr;sc.ctx.setTransform(dpr,0,0,dpr,0,0);sc.w=w;}
  function layoutStatic(){sc.pos=makePos(sc.w,sc.h);}
  function drawStatic(){drawTopology(sc.ctx,sc.w,sc.h,sc.pos,null,1);}

  // ---------- topology + optional animation frame ----------
  // anim: {spikes:{id:[ticks]}, T, flash} or null for static
  function drawTopology(ctx,W,H,pos,anim,alphaScale){
    ctx.clearRect(0,0,W,H);
    // edges
    D.synapses.forEach(function(s){
      var a=pos[nid(s.src)],b=pos[nid(s.tgt)];if(!a||!b)return;
      ctx.strokeStyle=s.exc?'#3fbf6f':'#e5534b';ctx.globalAlpha=(s.recurrent?0.85:0.38)*alphaScale;
      ctx.lineWidth=0.8+Math.min(3.5,s.w_mag*1.1);ctx.setLineDash(s.recurrent?[5,3]:[]);
      ctx.beginPath();edgePath(ctx,a,b,s.src===s.tgt);ctx.stroke();
    });
    ctx.setLineDash([]);ctx.globalAlpha=1;
    // traveling signals
    if(anim){
      D.synapses.forEach(function(s){
        if(s.src===s.tgt)return; // self-loops: rely on node flash
        var a=pos[nid(s.src)],b=pos[nid(s.tgt)];if(!a||!b)return;
        var sp=anim.spikes[nid(s.src)];if(!sp)return;
        var back=b.x<a.x-1,cp=ctrl(a,b,back),dl=Math.max(1,s.delay);
        for(var i=0;i<sp.length;i++){
          var u=(anim.T-sp[i])/dl;
          if(u>=0&&u<=1){var p=quad(a,cp,b,u);
            ctx.globalAlpha=1;ctx.fillStyle=s.exc?'#8affb0':'#ff9a92';
            ctx.beginPath();ctx.arc(p.x,p.y,3.2,0,7);ctx.fill();
            ctx.globalAlpha=0.3;ctx.beginPath();ctx.arc(p.x,p.y,6,0,7);ctx.fill();}
        }
      });
      ctx.globalAlpha=1;
    }
    // nodes
    D.nodes.forEach(function(n){
      var p=pos[n.id];if(!p)return;var r=n.role==='output'?13:10;
      // spike glow
      if(anim){var sp=anim.spikes[n.id]||[],g=0;
        for(var i=0;i<sp.length;i++){var dt=anim.T-sp[i];if(dt>=0&&dt<anim.flash)g=Math.max(g,1-dt/anim.flash);}
        if(g>0){ctx.globalAlpha=g*0.9;ctx.fillStyle=roleColor(n);ctx.beginPath();ctx.arc(p.x,p.y,r+8*g+4,0,7);ctx.fill();ctx.globalAlpha=1;}
        var lit=g>0.02;
        ctx.fillStyle=lit?roleColor(n):'#141b24';
      } else ctx.fillStyle='#141b24';
      if(n.role==='output'){ctx.globalAlpha=1;ctx.strokeStyle=css('--out');ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,r+5,0,7);ctx.stroke();}
      ctx.strokeStyle=roleColor(n);ctx.lineWidth=2.3;ctx.beginPath();
      if(n.role==='input')ctx.rect(p.x-r,p.y-r,2*r,2*r);else ctx.arc(p.x,p.y,r,0,7);
      ctx.fill();ctx.stroke();
      ctx.fillStyle=css('--ink');ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(n.id,p.x,p.y);
    });
    ctx.globalAlpha=1;
  }

  // ---------- animation ----------
  var A={sel:0,T:0,playing:true,speed:12,last:0,end:60,conds:['cold','trained'],cv:{}};
  var FLASH=7;
  function ord(a){return a.map(function(d){return 'o'+d;}).join(' › ');}
  function setupAnim(){
    var inputs=D.anim.inputs;
    // selector
    var selWrap=document.getElementById('animSel');
    selWrap.innerHTML=inputs.map(function(inp,i){
      return '<button class="abtn" data-i="'+i+'">'+inp.label+'</button>';}).join('');
    selWrap.addEventListener('click',function(e){var b=e.target.closest('.abtn');if(!b)return;A.sel=+b.getAttribute('data-i');A.T=0;syncSel();});
    // controls
    document.getElementById('animPlay').addEventListener('click',function(){A.playing=!A.playing;this.textContent=A.playing?'❚❚ pause':'▶ play';});
    document.getElementById('animRestart').addEventListener('click',function(){A.T=0;A.playing=true;document.getElementById('animPlay').textContent='❚❚ pause';});
    var sp=document.getElementById('animSpeed');sp.addEventListener('input',function(){A.speed=+this.value;});
    // canvases
    A.cv.cold={cv:document.getElementById('animCold'),ctx:document.getElementById('animCold').getContext('2d')};
    A.cv.trained={cv:document.getElementById('animTrained'),ctx:document.getElementById('animTrained').getContext('2d')};
    animResizeAll();syncSel();
    A.last=0;requestAnimationFrame(loop);
  }
  function animResize(c){var dpr=Math.min(2,window.devicePixelRatio||1);var w=Math.round(c.cv.clientWidth||360);var h=300;c.cv.style.height=h+'px';c.cv.width=w*dpr;c.cv.height=h*dpr;c.ctx.setTransform(dpr,0,0,dpr,0,0);c.w=w;c.h=h;c.pos=makePos(w,h);}
  function animResizeAll(){animResize(A.cv.cold);animResize(A.cv.trained);}
  function syncSel(){
    var inputs=D.anim.inputs,inp=inputs[A.sel];
    Array.prototype.forEach.call(document.querySelectorAll('#animSel .abtn'),function(b,i){b.classList.toggle('on',i===A.sel);});
    // active window: last spike + max delay + tail
    var maxT=0,maxDl=1;D.synapses.forEach(function(s){maxDl=Math.max(maxDl,s.delay);});
    A.conds.forEach(function(cn){var sp=inp[cn].spikes;for(var k in sp)for(var j=0;j<sp[k].length;j++)maxT=Math.max(maxT,sp[k][j]);});
    A.end=Math.min(D.anim.n_ticks,maxT+maxDl+8);
    // input + target line
    var it=inp.input_ticks;
    document.getElementById('animInfo').innerHTML='<b>'+inp.label+'</b> · input latency <span class="mono">'+Object.keys(it).map(function(k){return k+'@'+it[k];}).join(' ')+'</span> · taught target order <span class="mono">'+ord(inp.target_order)+'</span>';
    setOutcome('cold',inp.cold);setOutcome('trained',inp.trained);
  }
  function setOutcome(cn,c){
    var el=document.getElementById('out_'+cn);
    if(c.fires) el.innerHTML='outputs <b style="color:var(--ok)">fire</b> — order <span class="mono">'+ord(c.order)+'</span> · recall tau-b <b style="color:'+(c.taub>0.02?'var(--ok)':'var(--bad)')+'">'+c.taub.toFixed(3)+'</b>';
    else el.innerHTML='outputs <b style="color:var(--bad)">don\'t all fire</b> — recall fails · tau-b <b>0.000</b>';
  }
  function loop(ts){
    var dt=A.last?(ts-A.last)/1000:0;A.last=ts;
    if(A.playing){A.T+=dt*A.speed;if(A.T>A.end+3)A.T=0;}
    document.getElementById('animClock').textContent='t = '+Math.min(A.end,Math.floor(A.T))+' / '+A.end+' ticks';
    var inp=D.anim.inputs[A.sel];
    drawTopology(A.cv.cold.ctx,A.cv.cold.w,A.cv.cold.h,A.cv.cold.pos,{spikes:inp.cold.spikes,T:A.T,flash:FLASH},1);
    drawTopology(A.cv.trained.ctx,A.cv.trained.w,A.cv.trained.h,A.cv.trained.pos,{spikes:inp.trained.spikes,T:A.T,flash:FLASH},1);
    requestAnimationFrame(loop);
  }
})();
