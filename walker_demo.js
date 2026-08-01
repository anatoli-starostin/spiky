// walker_demo.js — Walker2d LUT→spiking actor, live in the browser.
// int4 integer weights + integer input latencies => the ramp comparison Σ w·t < θ is exact integer;
// we run the whole pipeline (17 latency inputs → 32 tables' 6-bit addresses → sum → 6 latency-coded
// action spikes) in JS and animate it on a 256-tick window, then decode the output spikes back and
// compare to the LUT oracle. Self-contained: model embedded in walker_model.json.
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');b.style.display='block';b.textContent='⚠ '+m;}
  window.addEventListener('error', function(e){showErr((e.error&&e.error.stack)||e.message);});
  var css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();};
  fetch('walker_model.json').then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
    .then(init).catch(function(e){showErr('could not load walker_model.json — '+e.message);});

  var M, sel=0, T=0, playing=true, speed=1, last=0, TT, DT, WIN;
  var pow2=[32,16,8,4,2,1];

  function compute(obs){
    var wq=M.wq,bq=M.bq,wtq=M.wtq,ws=M.ws,bs=M.bs,wsc=M.wsc,c=M.meta.c,al=M.meta.alpha,win=M.meta.win;
    var n=M.meta.n,k=M.meta.k,NT=M.meta.tables;
    var it=[]; for(var j=0;j<n;j++){var t=Math.round(c-al*obs[j]); it.push(Math.max(1,Math.min(win-1,t)));}
    var act=new Array(6).fill(0), actGT=new Array(6).fill(0), addrs=[], addrsGT=[];
    for(var t2=0;t2<NT;t2++){
      var bits=0, bitsGT=0;
      for(var i=0;i<k;i++){
        var proj=0, sw=0, projf=0;
        for(var j2=0;j2<n;j2++){ proj+=wq[t2][i][j2]*it[j2]; sw+=wq[t2][i][j2]; projf+=(wq[t2][i][j2]*ws[t2])*obs[j2]; }
        var theta=c*sw + al*(bs[t2]/ws[t2])*bq[t2][i];
        bits=(bits<<1)|((proj<theta)?1:0);
        bitsGT=(bitsGT<<1)|((projf + bq[t2][i]*bs[t2] > 0)?1:0);
      }
      addrs.push(bits); addrsGT.push(bitsGT);
      for(var ch=0;ch<6;ch++){ act[ch]+=wtq[t2][bits][ch]*wsc[t2]; actGT[ch]+=wtq[t2][bitsGT][ch]*wsc[t2]; }
    }
    // output latency: bigger action -> earlier spike. tick relative to decode; absolute = DT + tick.
    var oc=M.meta.out_c, oa=M.meta.out_alpha, tOut=[], dec=[];
    for(var ch2=0;ch2<6;ch2++){ var to=Math.round(oc-oa*act[ch2]); to=Math.max(1,Math.min(win-1,to)); tOut.push(to); dec.push((oc-to)/oa); }
    var addrMatch=0; for(var q=0;q<NT;q++) if(addrs[q]===addrsGT[q]) addrMatch++;
    return {it:it,addrs:addrs,addrMatch:addrMatch/NT,act:act,actGT:actGT,tOut:tOut,dec:dec};
  }

  function init(d){
    M=d; WIN=M.meta.win; DT=M.meta.decode_tick; TT=DT+WIN;
    document.getElementById('chips').innerHTML=[
      'model <b>int4</b> Walker2d LUT actor','tables <b>'+M.meta.tables+'</b>','k <b>'+M.meta.k+'</b> · n <b>'+M.meta.n+'</b>',
      'window <b>'+WIN+'</b> ticks','output <b>6</b> action means (latency-coded)'
    ].map(function(t){return '<span class="chip">'+t+'</span>';}).join('');
    var s=document.getElementById('preset');
    s.innerHTML=M.presets.map(function(p,i){return '<option value="'+i+'">'+p.name+'</option>';}).join('');
    s.addEventListener('change',function(){sel=+this.value;T=0;syncSel();});
    document.getElementById('play').addEventListener('click',function(){playing=!playing;this.textContent=playing?'❚❚ pause':'▶ play';});
    document.getElementById('restart').addEventListener('click',function(){T=0;playing=true;document.getElementById('play').textContent='❚❚ pause';});
    document.getElementById('speed').addEventListener('change',function(){speed=+this.value;});
    cv=document.getElementById('raster'); ctx=cv.getContext('2d'); resize();
    window.addEventListener('resize',function(){resize();});
    syncSel(); last=0; requestAnimationFrame(loop);
  }

  var cv,ctx,W,H;
  function resize(){var dpr=Math.min(2,window.devicePixelRatio||1);W=Math.round(cv.clientWidth||820);H=+cv.getAttribute('height');cv.style.height=H+'px';cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}

  var R;
  function syncSel(){
    var p=M.presets[sel]; R=compute(p.obs);
    document.getElementById('obs').innerHTML='<b>obs (17-dim):</b> <span class="mono">['+p.obs.map(function(v){return v.toFixed(2);}).join(', ')+']</span>';
    // compare table
    var rows='<tr><th>action dim</th><th>LUT oracle</th><th>spiking (decoded spike time)</th><th>Δ</th><th></th></tr>';
    var allok=true;
    for(var ch=0;ch<6;ch++){
      var gt=R.actGT[ch], sp=R.dec[ch], err=Math.abs(gt-sp), ok=err<0.15; if(!ok)allok=false;
      rows+='<tr><td>a'+ch+'</td><td class="mono">'+gt.toFixed(3)+'</td><td class="mono">'+sp.toFixed(3)+'</td><td class="mono">'+err.toFixed(3)+'</td><td>'+(ok?'<span style="color:var(--ok)">✓</span>':'<span style="color:var(--bad)">✕</span>')+'</td></tr>';
    }
    document.getElementById('cmp').innerHTML='<table class="gtab">'+rows+'</table>'
      +'<p class="small" style="margin-top:6px;">Per-table address-match vs oracle: <b>'+Math.round(R.addrMatch*100)+'%</b>. Overall action match: '
      +(allok?'<b style="color:var(--ok)">MATCH ✓</b> (all 6 dims within 0.15)':'<b style="color:#e5c07b">close</b> (some dim off — rare address flip / output quant)')+'</p>';
  }

  function xt(tick){return 46+(W-60)*tick/TT;}
  function loop(ts){
    var dt=last?(ts-last)/1000:0; last=ts;
    if(playing){T+=dt*speed*90; if(T>TT+40)T=0;}
    draw();
    requestAnimationFrame(loop);
  }
  function flash(tick){var d=T-tick; return (d>=0&&d<14)?1-d/14:0;}
  function draw(){
    ctx.clearRect(0,0,W,H);
    var nIn=M.meta.n, padTop=24, rowH=(H-70)/(nIn+6+2);
    // section labels
    ctx.fillStyle=css('--mut')||'#8b98a5'; ctx.font='11px ui-monospace,monospace'; ctx.textAlign='left';
    ctx.fillText('17 inputs → latency-coded spikes',46,14);
    // clock line (address resolved)
    ctx.strokeStyle='#3a4550'; ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(xt(WIN),padTop-4); ctx.lineTo(xt(WIN),H-20); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#66727f'; ctx.fillText('addresses resolved (t='+WIN+')',xt(WIN)+4,padTop+2);
    // input rows
    for(var j=0;j<nIn;j++){
      var y=padTop+j*rowH+8; ctx.fillStyle='#243040'; ctx.fillRect(46,y,W-60,0.5);
      ctx.fillStyle='#4a5560'; ctx.font='9px ui-monospace'; ctx.textAlign='right'; ctx.fillText('i'+j,42,y+3);
      var tk=R.it[j], g=flash(tk);
      ctx.globalAlpha=1; ctx.fillStyle=g>0?css('--spike')||'#4fd1e0':'#2b6f78';
      if(g>0){ctx.globalAlpha=g*0.7;ctx.beginPath();ctx.arc(xt(tk),y,6+4*g,0,7);ctx.fill();ctx.globalAlpha=1;ctx.fillStyle=css('--spike')||'#4fd1e0';}
      ctx.beginPath(); ctx.arc(xt(tk),y,3.2,0,7); ctx.fill();
    }
    // output rows
    var oy0=padTop+(nIn+1.2)*rowH+8;
    ctx.fillStyle=css('--mut')||'#8b98a5'; ctx.textAlign='left'; ctx.fillText('6 action means → latency-coded output spikes',46,oy0-8);
    for(var ch=0;ch<6;ch++){
      var y2=oy0+ch*rowH+8, at=DT+R.tOut[ch], g2=flash(at);
      ctx.fillStyle='#243040'; ctx.fillRect(46,y2,W-60,0.5);
      ctx.fillStyle='#4a5560'; ctx.font='9px ui-monospace'; ctx.textAlign='right'; ctx.fillText('a'+ch,42,y2+3);
      ctx.fillStyle=g2>0?'#c678dd':'#6a4a78';
      if(g2>0){ctx.globalAlpha=g2*0.8;ctx.beginPath();ctx.arc(xt(at),y2,7+4*g2,0,7);ctx.fill();ctx.globalAlpha=1;}
      ctx.fillStyle='#c678dd'; ctx.beginPath(); ctx.arc(xt(at),y2,3.6,0,7); ctx.fill();
      if(T>=at){ctx.fillStyle=css('--ink')||'#dfe6ee';ctx.font='9px ui-monospace';ctx.textAlign='left';ctx.fillText(R.dec[ch].toFixed(2),xt(at)+8,y2+3);}
    }
    // playhead
    ctx.strokeStyle=css('--spike')||'#4fd1e0'; ctx.globalAlpha=0.6; ctx.beginPath(); ctx.moveTo(xt(T),padTop-6); ctx.lineTo(xt(T),H-16); ctx.stroke(); ctx.globalAlpha=1;
    document.getElementById('tick').textContent='t = '+Math.min(TT,Math.floor(T))+' / '+TT+' ticks';
  }
})();
