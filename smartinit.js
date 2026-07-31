// smartinit.js — viewer for the clean LUT smart-init winner (feed-forward mapping, NO STDP).
// Topology (Dale exc/inh, delays, latency-in -> detectors -> latency-out) + an interactive
// spike-playback: pick an input, watch the REAL recorded spikes propagate along synapses (real
// delays) until the 4 outputs fire in the taught-value order. Single condition (the evolved net).
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');b.style.display='block';b.textContent='⚠ '+m;}
  window.addEventListener('error', function(e){showErr((e.error&&e.error.stack)||e.message);});
  var css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();};
  var DATA_URL='smartinit_data.json';
  fetch(DATA_URL).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status+' '+DATA_URL);return r.json();})
    .then(init).catch(function(e){showErr('could not load '+DATA_URL+' — '+e.message);});

  var D;
  function roleColor(n){return n.role==='input'?css('--spike'):(n.exc?'#3fbf6f':'#e5534b');}
  function nodeById(id){for(var i=0;i<D.nodes.length;i++)if(D.nodes[i].id===id)return D.nodes[i];return null;}

  function init(d){
    D=d; var s=d.stats, m=d.meta;
    document.getElementById('chips').innerHTML=[
      'tau-b <b>'+m.taub_512+'</b> (512 real inputs)',
      'all-outputs-fire <b>'+Math.round(s.allfire_512*100)+'%</b>',
      'exact-order-match <b>'+Math.round(s.exact_512*100)+'%</b>',
      'neurons <b>'+s.neurons+'</b> ('+s.hidden+' hidden)',
      'synapses <b>'+s.synapses+'</b>',
      'excitatory <b>'+s.exc_neurons+'</b> · inhibitory <b>'+s.inh_neurons+'</b>'
    ].map(function(t){return '<span class="chip">'+t+'</span>';}).join('');
    document.getElementById('side').innerHTML=sideText();
    setupStatic();
    setupAnim();
    window.addEventListener('resize', function(){staticResize();layoutStatic();drawStatic();animResize();});
  }
  function sideText(){
    var s=D.stats;
    return '<h3>The smart-init story</h3><p class="small">The hand-built <i>exact</i> LUT→spiking circuit needed <b>'+s.construction_neurons+' neurons / '+s.construction_synapses+' synapses</b> of scaffolding — a clock, complement neurons, a one-hot row decoder. Here ALL of that is dropped. We seed a clean 3-layer net (latency-in → detectors → latency-out) directly from the trained LUT hyperplanes (tau-b ≈ '+D.meta.seed_taub+' at init), then run plain <b>weight+delay evolution</b> (no STDP, no training-at-inference).</p>'
      +'<h3>Result</h3><p class="small">It recovers to <b>tau-b '+D.meta.taub_512+'</b> on the 512 real-valued eval inputs (all 4 outputs fire on '+Math.round(s.allfire_512*100)+'% of them; exact output-order match '+Math.round(s.exact_512*100)+'%). And it does so while staying SMALL — <b>'+s.neurons+' neurons / '+s.synapses+' synapses</b>, even smaller than the '+s.seed_neurons+'/'+s.seed_synapses+' seed and far below the '+s.construction_neurons+'/'+s.construction_synapses+' scaffolded construction. It pruned rather than grew.</p>'
      +'<h3>How to read it</h3><p class="small">Nodes: <span style="color:'+css('--spike')+'">cyan = input</span> (6 real values, latency-coded — bigger value fires earlier), <span style="color:#3fbf6f">green = excitatory (RS)</span>, <span style="color:#e5534b">red = inhibitory (FS)</span>; ringed = the 4 outputs. Edges carry an evolved delay. tau-b ≈ 0.78 is strong rank-correlation of the output first-spike order with the LUT\'s stored-value order — NOT a perfect permutation (exact match ~55%).</p>';
  }

  // ---------- geometry (3 columns: inputs | hidden | outputs) ----------
  function makePos(W,H){
    var pad=36,pos={};
    var ins=D.nodes.filter(function(n){return n.role==='input';});
    var outs=D.nodes.filter(function(n){return n.role==='output';});
    var hid=D.nodes.filter(function(n){return n.role==='hidden';});
    ins.forEach(function(n,i){pos[n.id]={x:pad+14,y:pad+i*(H-2*pad)/(ins.length-1||1)};});
    outs.forEach(function(n,i){pos[n.id]={x:W-pad-14,y:pad+22+i*(H-2*pad-44)/(outs.length-1||1)};});
    hid.forEach(function(n,i){pos[n.id]={x:W*0.5,y:pad+i*(H-2*pad)/(hid.length-1||1)};});
    return pos;
  }
  function ctrl(a,b,back){var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,dx=b.x-a.x,dy=b.y-a.y,l=Math.hypot(dx,dy)||1,bow=(back?-1:1)*Math.min(46,l*0.2);return{x:mx-dy/l*bow,y:my+dx/l*bow};}
  function quad(a,cp,b,u){var mm=1-u;return{x:mm*mm*a.x+2*mm*u*cp.x+u*u*b.x,y:mm*mm*a.y+2*mm*u*cp.y+u*u*b.y};}

  function drawTopology(ctx,W,H,pos,anim){
    ctx.clearRect(0,0,W,H);
    D.synapses.forEach(function(s){
      var a=pos[s.src],b=pos[s.tgt];if(!a||!b)return;
      ctx.strokeStyle=s.exc?'#3fbf6f':'#e5534b';ctx.globalAlpha=(s.recurrent?0.85:0.4);
      ctx.lineWidth=0.8+Math.min(3.2,s.w_mag*1.1);ctx.setLineDash(s.recurrent?[5,3]:[]);
      ctx.beginPath();
      if(s.src===s.tgt){ctx.moveTo(a.x,a.y-10);ctx.bezierCurveTo(a.x+38,a.y-40,a.x+38,a.y+40,a.x,a.y+10);}
      else{var back=b.x<a.x-1,cp=ctrl(a,b,back);ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(cp.x,cp.y,b.x,b.y);}
      ctx.stroke();
    });
    ctx.setLineDash([]);ctx.globalAlpha=1;
    if(anim){
      D.synapses.forEach(function(s){
        if(s.src===s.tgt)return;var a=pos[s.src],b=pos[s.tgt];if(!a||!b)return;
        var sp=anim.spikes[s.src];if(!sp)return;
        var back=b.x<a.x-1,cp=ctrl(a,b,back),dl=Math.max(1,s.delay);
        for(var i=0;i<sp.length;i++){var u=(anim.T-sp[i])/dl;
          if(u>=0&&u<=1){var p=quad(a,cp,b,u);
            ctx.globalAlpha=1;ctx.fillStyle=s.exc?'#8affb0':'#ff9a92';ctx.beginPath();ctx.arc(p.x,p.y,3.2,0,7);ctx.fill();
            ctx.globalAlpha=0.3;ctx.beginPath();ctx.arc(p.x,p.y,6,0,7);ctx.fill();}}
      });
      ctx.globalAlpha=1;
    }
    D.nodes.forEach(function(n){
      var p=pos[n.id];if(!p)return;var r=n.role==='output'?13:10;
      if(anim){var sp=anim.spikes[n.id]||[],g=0;
        for(var i=0;i<sp.length;i++){var dt=anim.T-sp[i];if(dt>=0&&dt<anim.flash)g=Math.max(g,1-dt/anim.flash);}
        if(g>0){ctx.globalAlpha=g*0.9;ctx.fillStyle=roleColor(n);ctx.beginPath();ctx.arc(p.x,p.y,r+8*g+4,0,7);ctx.fill();ctx.globalAlpha=1;}
        ctx.fillStyle=g>0.02?roleColor(n):'#141b24';
      } else ctx.fillStyle='#141b24';
      if(n.role==='output'){ctx.globalAlpha=1;ctx.strokeStyle=css('--out');ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,r+5,0,7);ctx.stroke();}
      ctx.strokeStyle=roleColor(n);ctx.lineWidth=2.3;ctx.beginPath();
      if(n.role==='input')ctx.rect(p.x-r,p.y-r,2*r,2*r);else ctx.arc(p.x,p.y,r,0,7);
      ctx.fill();ctx.stroke();
      ctx.fillStyle=css('--ink');ctx.font='8px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(n.id,p.x,p.y);
    });
    ctx.globalAlpha=1;
  }

  // ---------- static ----------
  var sc={};
  function setupStatic(){var cv=document.getElementById('graph');sc={cv:cv,ctx:cv.getContext('2d'),h:+cv.getAttribute('height')};staticResize();layoutStatic();drawStatic();}
  function staticResize(){var dpr=Math.min(2,window.devicePixelRatio||1);var w=Math.round(sc.cv.clientWidth||820);sc.cv.style.height=sc.h+'px';sc.cv.width=w*dpr;sc.cv.height=sc.h*dpr;sc.ctx.setTransform(dpr,0,0,dpr,0,0);sc.w=w;}
  function layoutStatic(){sc.pos=makePos(sc.w,sc.h);}
  function drawStatic(){drawTopology(sc.ctx,sc.w,sc.h,sc.pos,null);}

  // ---------- animation ----------
  var A={sel:0,T:0,playing:true,speed:10,last:0,end:50,cv:{}}, FLASH=7;
  function ord(a){return a.map(function(d){return 'o'+d;}).join(' › ');}
  function setupAnim(){
    var inputs=D.anim.inputs;
    document.getElementById('animSel').innerHTML=inputs.map(function(inp,i){
      return '<button class="abtn" data-i="'+i+'">'+inp.label+' · τ '+inp.taub.toFixed(2)+'</button>';}).join('');
    document.getElementById('animSel').addEventListener('click',function(e){var b=e.target.closest('.abtn');if(!b)return;A.sel=+b.getAttribute('data-i');A.T=0;syncSel();});
    document.getElementById('animPlay').addEventListener('click',function(){A.playing=!A.playing;this.textContent=A.playing?'❚❚ pause':'▶ play';});
    document.getElementById('animRestart').addEventListener('click',function(){A.T=0;A.playing=true;document.getElementById('animPlay').textContent='❚❚ pause';});
    document.getElementById('animSpeed').addEventListener('input',function(){A.speed=+this.value;});
    A.cv={cv:document.getElementById('anim'),ctx:document.getElementById('anim').getContext('2d')};
    animResize();syncSel();A.last=0;requestAnimationFrame(loop);
  }
  function animResize(){var c=A.cv,dpr=Math.min(2,window.devicePixelRatio||1);var w=Math.round(c.cv.clientWidth||700);var h=340;c.cv.style.height=h+'px';c.cv.width=w*dpr;c.cv.height=h*dpr;c.ctx.setTransform(dpr,0,0,dpr,0,0);c.w=w;c.h=h;c.pos=makePos(w,h);}
  function syncSel(){
    var inp=D.anim.inputs[A.sel];
    Array.prototype.forEach.call(document.querySelectorAll('#animSel .abtn'),function(b,i){b.classList.toggle('on',i===A.sel);});
    var maxT=0,maxDl=1;D.synapses.forEach(function(s){maxDl=Math.max(maxDl,s.delay);});
    for(var k in inp.spikes)for(var j=0;j<inp.spikes[k].length;j++)maxT=Math.max(maxT,inp.spikes[k][j]);
    A.end=Math.min(D.anim.n_ticks,maxT+maxDl+6);
    var match=JSON.stringify(inp.out_order)===JSON.stringify(inp.target_order);
    document.getElementById('animInfo').innerHTML='Input (6 real values): <span class="mono">['+inp.x.join(', ')+']</span> → latency <span class="mono">'
      +Object.keys(inp.input_ticks).map(function(k){return k+'@'+inp.input_ticks[k];}).join(' ')+'</span>';
    document.getElementById('animOut').innerHTML='LUT target order <span class="mono">'+ord(inp.target_order)+'</span> · net output order <span class="mono">'+ord(inp.out_order)+'</span> · recall tau-b <b style="color:'+(inp.taub>0.6?'var(--ok)':inp.taub>0.2?'#e5c07b':'var(--bad)')+'">'+inp.taub.toFixed(3)+'</b> '
      +(match?'<span class="pill ok">exact order match</span>':'<span class="pill" style="background:#3a3320;color:#e5c07b">partial — rank-correlated</span>');
  }
  function loop(ts){
    var dt=A.last?(ts-A.last)/1000:0;A.last=ts;
    if(A.playing){A.T+=dt*A.speed;if(A.T>A.end+3)A.T=0;}
    document.getElementById('animClock').textContent='t = '+Math.min(A.end,Math.floor(A.T))+' / '+A.end;
    var inp=D.anim.inputs[A.sel];
    drawTopology(A.cv.ctx,A.cv.w,A.cv.h,A.cv.pos,{spikes:inp.spikes,T:A.T,flash:FLASH});
    requestAnimationFrame(loop);
  }
})();
