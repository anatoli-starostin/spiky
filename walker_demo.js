// walker_demo.js — Walker2d LUT→spiking actor, replaying a REAL spnet spike trace.
// Every node lights EXACTLY at its recorded spike tick; comparators that never cross stay dark
// (bit 0); edges carry a pulse only when a real spike propagates. Three synced views: an aggregate
// graph (32 tables), a per-table INSPECT view (6 comparator bits → 1-of-64 address by coincidence),
// and a timeline raster. Model + per-preset traces embedded in walker_model.json.
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');if(b){b.style.display='block';b.textContent='⚠ '+m;}}
  window.addEventListener('error', function(e){showErr((e.error&&e.error.stack)||e.message);});
  fetch('walker_model.json').then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
    .then(init).catch(function(e){showErr('could not load walker_model.json — '+e.message);});

  var M, sel=0, tsel=0, T=0, playing=true, speed=1, last=0, started=false, TT, DT, WIN, NT=32, K=6, errShown=false;
  var pow2=[32,16,8,4,2,1];

  function oracleAddr(obs){                 // int4-dequant LUT addresses on the raw obs (for match stat)
    var wq=M.wq,bq=M.bq,ws=M.ws,bs=M.bs, a=[];
    for(var t=0;t<NT;t++){var bits=0;
      for(var i=0;i<K;i++){var pf=0;for(var j=0;j<17;j++)pf+=(wq[t][i][j]*ws[t])*obs[j];
        bits=(bits<<1)|((pf+bq[t][i]*bs[t]>0)?1:0);}
      a.push(bits);}
    return a;
  }

  function init(d){
    M=d; WIN=M.meta.win; DT=M.meta.decode_tick; TT=DT+WIN;
    document.getElementById('chips').innerHTML=[
      'model <b>int4</b> Walker2d actor','tables <b>32</b>','k <b>6</b> · n <b>17</b>',
      'window <b>'+WIN+'</b> ticks','<b>REAL</b> spnet spike trace'
    ].map(function(t){return '<span class="chip">'+t+'</span>';}).join('');
    var s=document.getElementById('preset');
    s.innerHTML=M.presets.map(function(p,i){return '<option value="'+i+'">'+p.name+'</option>';}).join('');
    s.addEventListener('change',function(){sel=+this.value;T=0;syncSel();});
    var ts=document.getElementById('tablesel');
    if(ts){var o='';for(var t=0;t<NT;t++)o+='<option value="'+t+'">table '+t+'</option>';ts.innerHTML=o;
      ts.addEventListener('change',function(){tsel=+this.value;});}
    document.getElementById('play').addEventListener('click',function(){playing=!playing;this.textContent=playing?'❚❚ pause':'▶ play';});
    document.getElementById('restart').addEventListener('click',function(){T=0;playing=true;document.getElementById('play').textContent='❚❚ pause';});
    document.getElementById('speed').addEventListener('change',function(){speed=+this.value;});
    gcv=document.getElementById('graph'); gctx=gcv?gcv.getContext('2d'):null;
    icv=document.getElementById('inspect'); ictx=icv?icv.getContext('2d'):null;
    rcv=document.getElementById('raster'); rctx=rcv?rcv.getContext('2d'):null;
    if(!gctx&&!rctx){showErr('no canvas context');return;}
    resize(); window.addEventListener('resize',function(){resize();drawAll();});
    syncSel(); drawAll(); last=0; started=false; requestAnimationFrame(loop);
  }

  var gcv,gctx,gW,gH,icv,ictx,iW,iH,rcv,rctx,rW,rH,R,GT,ORA;
  function sizeC(el,c){var dpr=Math.min(2,window.devicePixelRatio||1);var r=el.getBoundingClientRect();
    var w=Math.max(320,Math.round(r.width||el.clientWidth||640)),h=+el.getAttribute('height')||360;
    el.style.height=h+'px';el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);c.setTransform(dpr,0,0,dpr,0,0);return [w,h];}
  function resize(){
    if(gctx){var a=sizeC(gcv,gctx);gW=a[0];gH=a[1];}
    if(ictx){var b=sizeC(icv,ictx);iW=b[0];iH=b[1];}
    if(rctx){var e=sizeC(rcv,rctx);rW=e[0];rH=e[1];}
  }

  function syncSel(){
    var p=M.presets[sel]; R=p.trace; GT=p.gt; ORA=oracleAddr(p.obs);
    document.getElementById('obs').innerHTML='<b>obs (17-dim):</b> <span class="mono">['+p.obs.map(function(v){return v.toFixed(2);}).join(', ')+']</span>';
    var oc=M.meta.out_c, oa=M.meta.out_alpha, rows='<tr><th>action</th><th>LUT oracle</th><th>spiking (from output spike time)</th><th>Δ</th><th></th></tr>', allok=true;
    for(var ch=0;ch<6;ch++){var gt=GT[ch], dec=(oc-R.out_tick[ch])/oa, err=Math.abs(gt-dec), ok=err<0.15; if(!ok)allok=false;
      rows+='<tr><td>a'+ch+'</td><td class="mono">'+gt.toFixed(3)+'</td><td class="mono">'+dec.toFixed(3)+'</td><td class="mono">'+err.toFixed(3)+'</td><td>'+(ok?'<span style="color:var(--ok)">✓</span>':'<span style="color:var(--bad)">✕</span>')+'</td></tr>';}
    var am=0;for(var t=0;t<NT;t++)if(R.addr[t]===ORA[t])am++;
    document.getElementById('cmp').innerHTML='<table class="gtab">'+rows+'</table>'
      +'<p class="small" style="margin-top:6px;">Per-table address-match vs oracle: <b>'+Math.round(am/NT*100)+'%</b>. Overall action match: '
      +(allok?'<b style="color:var(--ok)">MATCH ✓</b>':'<b style="color:#e5c07b">close</b> (a rare boundary flip)')+'</p>';
  }

  function lerp(a,b,f){return {x:a.x+(b.x-a.x)*f,y:a.y+(b.y-a.y)*f};}
  function flash(tick){if(tick<0)return 0;var d=T-tick;return (d>=0&&d<16)?1-d/16:0;}
  function node(c,p,color,g,r){
    if(g>0){c.globalAlpha=g*0.85;c.fillStyle=color;c.beginPath();c.arc(p.x,p.y,r+7*g,0,7);c.fill();c.globalAlpha=1;}
    c.globalAlpha=1;c.fillStyle=g>0?'#ffffff':color;c.beginPath();c.arc(p.x,p.y,r,0,7);c.fill();
  }

  // ---------- aggregate graph: 32 table clusters, real comparator firing, explicit comp->addr edges ----------
  function drawGraph(){
    if(!gctx||!R)return; var c=gctx; c.fillStyle='#0b0f16'; c.fillRect(0,0,gW,gH);
    var padY=22, ph=gH-2*padY-14;
    var ins=[]; for(var j=0;j<17;j++) ins.push({x:gW*0.05,y:padY+j*ph/16});
    var outs=[]; for(var o=0;o<6;o++) outs.push({x:gW*0.965,y:padY+ph*0.14+o*ph*0.72/5});
    var cols=4,rows=8, gx0=gW*0.14, gx1=gW*0.9, gy0=padY, cw=(gx1-gx0)/cols, chh=ph/rows;
    function cl(t){var cx=t%cols,cy=(t/cols)|0; return {x:gx0+cx*cw, y:gy0+cy*chh};}
    function comp(t,i){var b=cl(t); return {x:b.x+cw*0.16, y:b.y+chh*0.18+i*chh*0.6/5};}
    function addrN(t){var b=cl(t); return {x:b.x+cw*0.66, y:b.y+chh*0.42};}
    // faint input->cluster + addr->output bundles
    c.lineWidth=0.5; c.strokeStyle='rgba(63,169,184,0.06)';
    for(j=0;j<17;j++){c.beginPath();c.moveTo(ins[j].x,ins[j].y);c.lineTo(gx0-6,gH/2);c.stroke();}
    c.strokeStyle='rgba(168,111,192,0.09)';
    for(o=0;o<6;o++){c.beginPath();c.moveTo(gx1+2,gH/2);c.lineTo(outs[o].x,outs[o].y);c.stroke();}
    // per-cluster comparator->address EXPLICIT edges + pulses (layer2->layer3, visible)
    for(var t=0;t<NT;t++){var an=addrN(t);
      for(var i=0;i<K;i++){var cp=comp(t,i), ct=R.comp_tick[t*K+i], fired=ct>=0;
        c.lineWidth=0.6; c.strokeStyle=fired?'rgba(120,200,215,0.28)':'rgba(90,90,110,0.14)';
        c.beginPath();c.moveTo(cp.x,cp.y);c.lineTo(an.x,an.y);c.stroke();
        if(fired){var t1=R.addr_tick[t]; if(T>=ct&&T<=t1+2){var f=t1>ct?(T-ct)/(t1-ct):1; var pp=lerp(cp,an,Math.min(1,f)); c.fillStyle='#8ff2ff';c.beginPath();c.arc(pp.x,pp.y,2.2,0,7);c.fill();}}
      }
    }
    // pulses addr->output
    for(o=0;o<6;o++){var s0=R.addr_tick[0],s1=DT+R.out_tick[o];/*from cluster region*/ var src={x:gx1+2,y:gH/2}; if(T>=WIN&&T<=s1){var ff=(T-WIN)/(s1-WIN);var q=lerp(src,outs[o],Math.min(1,ff));c.fillStyle='#f0c0ff';c.beginPath();c.arc(q.x,q.y,3,0,7);c.fill();}}
    // nodes: comparators glow at REAL ticks (dark if never), addresses at addr_tick
    for(t=0;t<NT;t++){for(i=0;i<K;i++){var ct2=R.comp_tick[t*K+i];node(c,comp(t,i),'#4fd1e0',flash(ct2),2.2);}
      node(c,addrN(t),'#9a7fc0',flash(R.addr_tick[t]),3.2);}
    for(j=0;j<17;j++) node(c,ins[j],'#3fa9b8',flash(R.in_tick[j]),3.6);
    for(o=0;o<6;o++) node(c,outs[o],'#c678dd',flash(DT+R.out_tick[o]),5.4);
    c.fillStyle='#66727f';c.font='10px ui-monospace,monospace';c.textAlign='center';
    c.fillText('17 obs',gW*0.05,gH-4);c.fillText('32 tables: 6 comparators → address (real staggered spikes; dark = bit 0)',gW*0.5,gH-4);c.fillText('6 actions',gW*0.955,gH-4);
    c.textAlign='left';c.fillStyle='#8b98a5';c.fillText(T<WIN?'comparators crossing threshold at their real times…':(T<DT?'addresses resolved':'emitting action spikes'),8,13);
  }

  // ---------- per-table INSPECT: 6 comparator bits -> 1-of-64 address via coincidence ----------
  function drawInspect(){
    if(!ictx||!R)return; var c=ictx; c.fillStyle='#0b0f16'; c.fillRect(0,0,iW,iH);
    var t=tsel, addr=R.addr[t], padY=30, ph=iH-2*padY;
    var comps=[]; for(var i=0;i<K;i++) comps.push({x:iW*0.16,y:padY+i*ph/5});
    var an={x:iW*0.66,y:padY+ph*0.5}, out={x:iW*0.9,y:padY+ph*0.5};
    // edges comp->address (explicit)
    for(i=0;i<K;i++){var ct=R.comp_tick[t*K+i], fired=ct>=0;
      c.lineWidth=fired?1.6:1; c.strokeStyle=fired?'rgba(120,220,235,0.5)':'rgba(90,90,110,0.25)'; c.setLineDash(fired?[]:[4,3]);
      c.beginPath();c.moveTo(comps[i].x,comps[i].y);c.lineTo(an.x,an.y);c.stroke(); c.setLineDash([]);
      if(fired){var t1=R.addr_tick[t]; if(T>=ct&&T<=t1+3){var f=t1>ct?(T-ct)/(t1-ct):1;var pp=lerp(comps[i],an,Math.min(1,f));c.fillStyle='#8ff2ff';c.beginPath();c.arc(pp.x,pp.y,3.6,0,7);c.fill();}}
    }
    c.lineWidth=1.4;c.strokeStyle='rgba(198,120,221,0.5)';c.beginPath();c.moveTo(an.x,an.y);c.lineTo(out.x,out.y);c.stroke();
    // comparator nodes + labels (bit, fire tick or DARK)
    for(i=0;i<K;i++){var ct3=R.comp_tick[t*K+i], bit=ct3>=0?1:0;
      node(c,comps[i],'#4fd1e0',flash(ct3),7);
      c.fillStyle=bit?'#e8eef5':'#7a8593';c.font='11px ui-monospace';c.textAlign='right';
      c.fillText('h'+i+' = '+bit,comps[i].x-14,comps[i].y-6);
      c.fillStyle='#66727f';c.font='9px ui-monospace';c.fillText(bit?('fires t='+ct3):'never (bit 0)',comps[i].x-14,comps[i].y+8);}
    // address node fires (coincidence) at addr_tick
    node(c,an,'#c678dd',flash(R.addr_tick[t]),11);
    c.fillStyle='#e8eef5';c.font='12px ui-monospace';c.textAlign='center';c.fillText('addr '+addr,an.x,an.y-18);
    c.fillStyle='#8b98a5';c.font='10px ui-monospace';c.fillText('coincidence',an.x,an.y+22);
    node(c,out,'#c678dd',flash(DT+R.out_tick[Math.min(5,t%6)]),6);
    // caption: 6-bit pattern -> address
    var bstr=''; for(i=0;i<K;i++) bstr+=(R.comp_tick[t*K+i]>=0?'1':'0');
    c.fillStyle='#cdd6df';c.font='12px ui-monospace';c.textAlign='left';
    c.fillText('table '+t+':  bits [b5..b0] = '+bstr+'  →  1-of-64 address = '+addr,12,16);
    c.fillStyle='#8b98a5';c.font='10px ui-monospace';
    c.fillText('each comparator fires (=1) when its ramp crosses threshold, or stays dark (=0); the address neuron fires when its 6 required bits coincide.',12,iH-8);
  }

  // ---------- timeline raster: real input, comparator-crossing, output spikes ----------
  function drawRaster(){
    if(!rctx||!R)return; var c=rctx; c.fillStyle='#0b0f16'; c.fillRect(0,0,rW,rH);
    function xt(tk){return 44+(rW-56)*tk/TT;}
    var padTop=26, band1=padTop, inH=70, compY=padTop+82, compH=120, outY=padTop+220, outH=60;
    c.fillStyle='#66727f';c.font='10px ui-monospace';c.textAlign='left';
    c.fillText('17 inputs',8,band1-6);c.fillText('192 comparators (crossing ticks; gaps = bit 0)',8,compY-6);c.fillText('6 action outputs',8,outY-6);
    // clock line
    c.strokeStyle='#3a4550';c.setLineDash([4,3]);c.beginPath();c.moveTo(xt(WIN),padTop-4);c.lineTo(xt(WIN),outY+outH);c.stroke();c.setLineDash([]);
    c.fillStyle='#66727f';c.fillText('t='+WIN,xt(WIN)+3,padTop-4);
    // inputs
    for(var j=0;j<17;j++){var y=band1+8+j*(inH-8)/16, tk=R.in_tick[j],g=flash(tk);
      if(g>0){c.globalAlpha=g*0.8;c.fillStyle='#5fe3f2';c.beginPath();c.arc(xt(tk),y,5+4*g,0,7);c.fill();c.globalAlpha=1;}
      c.fillStyle=g>0?'#8ff2ff':'#3fa9b8';c.beginPath();c.arc(xt(tk),y,2.6,0,7);c.fill();}
    // comparators: 192 rows compressed into the band
    for(var ci=0;ci<192;ci++){var ct=R.comp_tick[ci]; if(ct<0)continue; var y2=compY+ (ci/192)*compH, g2=flash(ct);
      c.fillStyle=g2>0?'#8ff2ff':'rgba(79,209,224,0.5)';c.beginPath();c.arc(xt(ct),y2,g2>0?3:1.6,0,7);c.fill();}
    // outputs
    for(var o=0;o<6;o++){var y3=outY+8+o*(outH-8)/5, at=DT+R.out_tick[o],g3=flash(at);
      if(g3>0){c.globalAlpha=g3*0.85;c.fillStyle='#d98fee';c.beginPath();c.arc(xt(at),y3,7+4*g3,0,7);c.fill();c.globalAlpha=1;}
      c.fillStyle=g3>0?'#f0c0ff':'#a86fc0';c.beginPath();c.arc(xt(at),y3,3.4,0,7);c.fill();
      if(T>=at){c.fillStyle='#e8eef5';c.font='9px ui-monospace';c.textAlign='left';c.fillText(((M.meta.out_c-R.out_tick[o])/M.meta.out_alpha).toFixed(2),xt(at)+8,y3+3);}}
    // playhead
    var px=xt(T);c.globalAlpha=0.25;c.strokeStyle='#5fe3f2';c.lineWidth=6;c.beginPath();c.moveTo(px,padTop-6);c.lineTo(px,outY+outH);c.stroke();
    c.globalAlpha=0.95;c.lineWidth=1.4;c.beginPath();c.moveTo(px,padTop-6);c.lineTo(px,outY+outH);c.stroke();c.globalAlpha=1;c.lineWidth=1;
    var el=document.getElementById('tick'); if(el)el.textContent='t = '+Math.min(TT,Math.floor(T))+' / '+TT;
  }

  function drawAll(){ drawGraph(); drawInspect(); drawRaster(); }
  function loop(ts){
    requestAnimationFrame(loop);
    try{
      if(!started){started=true; resize();}
      var dt=last?Math.min(0.05,(ts-last)/1000):0; last=ts;
      if(playing){T+=dt*speed*130; if(T>TT+50)T=0;}
      drawAll();
    }catch(e){ if(!errShown){errShown=true; showErr('animation error: '+(e.message||e));} }
  }
})();
