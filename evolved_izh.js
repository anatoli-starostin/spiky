// evolved_izh.js — dependency-free viewer for the evolved Izhikevich RS/FS network.
// Same idiom as inspector.js: linked canvas graph + spike raster, #side details, live playhead.
// Difference: the Izhikevich simulation is run IN-BROWSER (faithful port) rather than loaded.
(function () {
  'use strict';
  function showErr(m){var b=document.getElementById('errbar');b.style.display='block';b.textContent='⚠ demo error — '+m;if(window.console)console.error(m);}
  window.addEventListener('error',function(e){showErr((e.error&&e.error.stack)||e.message);});
  var css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();};

  var D=null,NT=44,P=null;
  var INPUTS=[],OUTPUTS=[],HIDDEN=[],byId={},outSyn={};
  var sim=null,curIdx=0,simT=0,Tmax=44,playing=false,speed=1,lastTs=null,dirty=true;
  var sel=null,hover=null,pos={},wmax=1;

  var DATA_URL='evolved_izh_data.json';
  fetch(DATA_URL).then(function(r){
    if(!r.ok)throw new Error('HTTP '+r.status+' fetching '+DATA_URL);
    return r.text().then(function(t){ try{return JSON.parse(t);}catch(_){throw new Error('non-JSON body from '+DATA_URL+' (starts with "'+t.slice(0,30).replace(/\s+/g,' ')+'")');} });
  }).then(function(d){D=d;init();}).catch(function(e){showErr('could not load data — '+e.message);});

  // ---------------- faithful Izhikevich sim (port of the reference simulator) ----------------
  function simulate(inputTicks){
    var V={},U={},first={},spikes={},vtrace={},os={};
    D.nodes.forEach(function(n){var p=P[n.type];V[n.id]=p.c;U[n.id]=p.b*p.c;first[n.id]=null;spikes[n.id]=[];vtrace[n.id]=[];});
    D.synapses.forEach(function(s){(os[s.src]=os[s.src]||[]).push(s);});
    var arrivals={},events=[],forced={},inSet={};
    Object.keys(inputTicks).forEach(function(nid){var t=inputTicks[nid];(forced[t]=forced[t]||[]).push(nid);inSet[nid]=1;});
    function emit(nid,t){
      if(first[nid]===null)first[nid]=t; spikes[nid].push(t);
      (os[nid]||[]).forEach(function(s){var at=t+s.delay;(arrivals[at]=arrivals[at]||[]).push({tgt:s.tgt,w:s.w});
        events.push({src:s.src,tgt:s.tgt,emit:t,arrive:at,exc:s.exc});});
    }
    for(var t=0;t<=NT;t++){
      (forced[t]||[]).forEach(function(nid){if(first[nid]===null)emit(nid,t);});
      var inj={}; (arrivals[t]||[]).forEach(function(a){inj[a.tgt]=(inj[a.tgt]||0)+a.w;});
      D.nodes.forEach(function(n){
        if(inSet[n.id]){vtrace[n.id].push(first[n.id]!==null&&t>=first[n.id]?30:-65);return;}
        var p=P[n.type],v=V[n.id],u=U[n.id];
        if(v>p.vth){v=p.c;u+=p.d;}
        var I=inj[n.id]||0;
        for(var k=0;k<2;k++){v=v+0.5*((p.cf2*v+p.cf1)*v+p.cf0-u+I);}
        u=u+p.a*(p.b*v-u); V[n.id]=v;U[n.id]=u; vtrace[n.id].push(v);
        if(v>=p.vth)emit(n.id,t);
      });
    }
    return {first:first,spikes:spikes,vtrace:vtrace,events:events};
  }
  function taub(fo,tv){var v=fo.map(function(f){return -f;}),u=tv,C=0,Dd=0,Tu=0,Tv=0;
    for(var a=0;a<4;a++)for(var b=a+1;b<4;b++){var su=Math.sign(u[a]-u[b]),sv=Math.sign(v[a]-v[b]),p=su*sv;
      if(p>0)C++;else if(p<0)Dd++;else if(su===0&&sv!==0)Tu++;else if(sv===0&&su!==0)Tv++;}
    var den=Math.sqrt((C+Dd+Tu)*(C+Dd+Tv));return ((den>0?(C-Dd)/den:0)+1)/2;}
  var ordStr=function(a){return a.map(function(d){return 'o'+d;}).join(' > ');};

  // ---------------- init ----------------
  function init(){
    NT=D.meta.N_TICKS; P=D.neuron_params; Tmax=NT;
    INPUTS=D.nodes.filter(function(n){return n.role==='input';}).map(function(n){return n.id;});
    OUTPUTS=D.nodes.filter(function(n){return n.role==='output';}).map(function(n){return n.id;});
    HIDDEN=D.nodes.filter(function(n){return n.role==='hidden';}).map(function(n){return n.id;});
    D.nodes.forEach(function(n){byId[n.id]=n;});
    D.synapses.forEach(function(s){(outSyn[s.src]=outSyn[s.src]||[]).push(s);});
    wmax=Math.max.apply(null,D.synapses.map(function(s){return Math.abs(s.w_raw);}).concat([1e-6]));

    buildInputUI();
    document.getElementById('glegend').innerHTML =
      '<span style="color:'+css('--spike')+'">■ input</span> · '
      +'<span style="color:'+css('--lut')+'">● RS (excitatory)</span> · '
      +'<span style="color:'+css('--warn')+'">● FS (inhibitory)</span> · '
      +'<span style="color:'+css('--out')+'">◉ output</span> · '
      +'edges: <span style="color:#5b9dff">■ −w (inh)</span>/<span style="color:#ff6b6b">■ +w (exc)</span>, dashed = delay>1, moving dot = a spike in flight';
    document.querySelector('h1').textContent='evolved spiking net — Izhikevich RS/FS (tau-b '+D.meta.stored_taub+')';
    setupCanvas(); bindControls(); loadInput(0);
    requestAnimationFrame(loop);
  }

  function buildInputUI(){
    var sel=document.getElementById('inputSel');
    D.demo_inputs.forEach(function(d,i){var o=document.createElement('option');o.value=i;o.textContent=(i+1)+') input '+(i+1);sel.appendChild(o);});
    sel.addEventListener('change',function(){loadInput(+sel.value);});
  }
  function loadInput(idx){
    curIdx=idx; var d=D.demo_inputs[idx];
    sim=simulate(d.input_ticks); simT=0; playing=false; setPlay(false); dirty=true;
    renderInfo(); showPanel();
  }
  function netOrder(){ // current predicted order by first-spike (unfired last)
    return OUTPUTS.map(function(o,i){return i;}).sort(function(a,b){
      var fa=sim.first[OUTPUTS[a]],fb=sim.first[OUTPUTS[b]];
      fa=fa===null?1e9:fa; fb=fb===null?1e9:fb; return fa-fb||a-b;});
  }
  function renderInfo(){
    var d=D.demo_inputs[curIdx];
    var fo=OUTPUTS.map(function(o){return sim.first[o]!==null?sim.first[o]:NT+1;});
    var allf=fo.every(function(f){return f<=NT;});
    var tb=taub(fo,d.tvals);
    var no=netOrder(), match=JSON.stringify(no)===JSON.stringify(d.true_order);
    document.getElementById('variantInfo').innerHTML =
      '<b>input x</b> = <span class="mono">['+d.x.join(', ')+']</span>'
      +' <span class="small" style="color:var(--muted)">→ first-spike ticks '+INPUTS.map(function(id){return d.input_ticks[id];}).join(', ')+'</span><br>'
      +'<b>oracle target order</b>: <span class="mono">'+ordStr(d.true_order)+'</span> '
      +'<span class="small" style="color:var(--muted)">(values ['+d.tvals.join(', ')+'])</span><br>'
      +'<b>evolved net</b>: output first-spike order <span class="mono">'+ordStr(no)+'</span> — '
      +(allf?(match?'<span style="color:var(--ok)">✓ exact match</span>':'<span style="color:var(--warn)">partial</span>'):'<span style="color:var(--warn)">outputs still firing…</span>')
      +' &nbsp; <b>tau-b</b> = <span class="mono" style="color:var(--spike)">'+tb.toFixed(3)+'</span>'
      +' <span class="small" style="color:var(--muted)">(reference '+d.ref_taub+')</span>';
  }

  // ---------------- canvas ----------------
  var canv={};
  function setupCanvas(){
    ['graph','raster'].forEach(function(id){var cv=document.getElementById(id);canv[id]={cv:cv,ctx:cv.getContext('2d'),w:0,h:0,hCSS:+cv.getAttribute('height')};});
    resize(); window.addEventListener('resize',function(){resize();dirty=true;});
    var g=canv.graph.cv;
    g.addEventListener('mousemove',function(e){hover=pick(e);dirty=true;g.style.cursor=hover?'pointer':'default';});
    g.addEventListener('click',function(e){sel=pick(e);showPanel();dirty=true;});
    var r=canv.raster.cv,drag=false;
    var scrub=function(e){var t=rasterTick(e);if(t!=null){simT=t;playing=false;setPlay(false);dirty=true;}};
    r.addEventListener('mousedown',function(e){drag=true;scrub(e);});
    r.addEventListener('mousemove',function(e){if(drag)scrub(e);});
    window.addEventListener('mouseup',function(){drag=false;});
  }
  function resize(){
    var dpr=Math.min(2,window.devicePixelRatio||1);
    for(var id in canv){var c=canv[id],w=Math.round(c.cv.clientWidth||800),h=c.hCSS;
      if(w===c.w)continue; c.cv.style.height=h+'px';c.cv.width=Math.max(1,w*dpr);c.cv.height=h*dpr;
      c.ctx.setTransform(dpr,0,0,dpr,0,0);c.w=w;c.h=h;}
    layout();
  }
  function layout(){
    var c=canv.graph; if(!c)return; pos={};
    var padT=30,padB=24,H=c.h-padT-padB;
    function place(ids,x){ids.forEach(function(id,i){pos[id]={x:x,y:padT+(ids.length===1?H/2:i*H/(ids.length-1))};});}
    place(INPUTS, 46);
    place(OUTPUTS, c.w-46);
    // hidden: two columns in the middle band
    HIDDEN.forEach(function(id,i){var col=i%2, rows=Math.ceil(HIDDEN.length/2);
      pos[id]={x:c.w*0.40+col*c.w*0.20, y:padT+(rows===1?H/2:Math.floor(i/2)*H/(rows-1))};});
  }
  function ctrl(a,b){ // perpendicular-bowed control point for a quadratic edge
    var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy)||1;
    var back=b.x<a.x-1, bow=(back?-1:1)*Math.min(60,len*0.22);
    return {x:mx-dy/len*bow, y:my+dx/len*bow};
  }
  function bez(a,c,b,t){var mt=1-t;return {x:mt*mt*a.x+2*mt*t*c.x+t*t*b.x, y:mt*mt*a.y+2*mt*t*c.y+t*t*b.y};}

  function vFill(id,t){var tr=sim.vtrace[id];if(!tr)return -65;return tr[Math.max(0,Math.min(tr.length-1,Math.floor(t)))];}
  function firedNear(id,t){var f=sim.spikes[id]||[];for(var i=0;i<f.length;i++)if(Math.abs(t-f[i])<0.5)return true;return false;}
  function vColor(v){var s=Math.max(0,Math.min(1,(v+65)/95));
    var st=[[22,30,44],[36,90,130],[0,180,216],[255,184,77],[255,240,200]];
    var seg=Math.min(st.length-2,Math.floor(s*(st.length-1))),f=s*(st.length-1)-seg,a=st[seg],b=st[seg+1];
    return 'rgb('+a.map(function(c,i){return Math.round(c+(b[i]-c)*f);}).join(',')+')';}
  function roleColor(n){return n.role==='input'?css('--spike'):n.role==='output'?css('--out'):(n.type==='FS'?css('--warn'):css('--lut'));}

  function drawGraph(){
    var c=canv.graph,ctx=c.ctx; ctx.clearRect(0,0,c.w,c.h);
    // edges
    D.synapses.forEach(function(s){
      var a=pos[s.src],b=pos[s.tgt]; if(!a||!b)return;
      var mag=Math.min(1,Math.abs(s.w_raw)/wmax), hot=sel&&sel.kind==='edge'&&sel.o===s;
      ctx.strokeStyle=s.exc?'#ff6b6b':'#5b9dff'; ctx.globalAlpha=hot?0.95:0.16+0.5*mag; ctx.lineWidth=(hot?2:0)+0.6+2.2*mag;
      ctx.setLineDash(s.delay>1?[4,3]:[]);
      ctx.beginPath();
      if(s.src===s.tgt){ctx.moveTo(a.x,a.y-12);ctx.bezierCurveTo(a.x+42,a.y-46,a.x+42,a.y+46,a.x,a.y+12);}
      else{var cp=ctrl(a,b);ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(cp.x,cp.y,b.x,b.y);}
      ctx.stroke();
    });
    ctx.setLineDash([]); ctx.globalAlpha=1;
    // in-flight spikes
    D.synapses&&sim.events.forEach(function(ev){
      if(simT>ev.emit&&simT<=ev.arrive){var a=pos[ev.src],b=pos[ev.tgt];if(!a||!b)return;
        var f=(simT-ev.emit)/(ev.arrive-ev.emit),cp=(ev.src===ev.tgt)?{x:a.x+42,y:a.y}:ctrl(a,b),pt=bez(a,cp,b,f);
        ctx.fillStyle=ev.exc?'#ff9a9a':'#8fc0ff';ctx.beginPath();ctx.arc(pt.x,pt.y,3.4,0,7);ctx.fill();}
    });
    // nodes
    D.nodes.forEach(function(n){
      var p=pos[n.id];if(!p)return; var flash=firedNear(n.id,simT), r=n.role==='output'?13:n.role==='input'?11:11;
      ctx.globalAlpha=1;
      if(n.role==='output'){ctx.strokeStyle=css('--out');ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,r+5,0,7);ctx.stroke();}
      ctx.fillStyle=flash?'#fff2a8':vColor(vFill(n.id,simT));
      ctx.strokeStyle=roleColor(n); ctx.lineWidth=(sel&&sel.kind==='node'&&sel.o===n)?3.5:2.2;
      ctx.beginPath();
      if(n.role==='input')ctx.rect(p.x-r,p.y-r,2*r,2*r); else ctx.arc(p.x,p.y,flash?r*1.28:r,0,7);
      ctx.fill();ctx.stroke();
      ctx.fillStyle=css('--ink');ctx.font='10px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(n.role==='hidden'?n.type:n.id, p.x, p.y+(n.role==='hidden'?0:0));
    });
  }
  function drawRaster(){
    var c=canv.raster,ctx=c.ctx; ctx.clearRect(0,0,c.w,c.h);
    var order=INPUTS.concat(HIDDEN).concat(OUTPUTS), padL=54,padR=14,padT=12,padB=18;
    var rowH=(c.h-padT-padB)/order.length, X=function(t){return padL+t/NT*(c.w-padL-padR);};
    order.forEach(function(id,i){
      var y=padT+i*rowH+rowH/2, n=byId[id];
      ctx.fillStyle=css('--muted');ctx.font='10px ui-monospace,monospace';ctx.textAlign='right';ctx.textBaseline='middle';
      ctx.fillText(n.role==='hidden'?id.slice(0,4)+'·'+n.type:id, padL-6, y);
      ctx.strokeStyle=css('--edge');ctx.lineWidth=1;ctx.globalAlpha=.5;ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(c.w-padR,y);ctx.stroke();ctx.globalAlpha=1;
      (sim.spikes[id]||[]).forEach(function(t){
        ctx.fillStyle=roleColor(n);ctx.beginPath();ctx.arc(X(t),y,t<=simT?3.4:2,0,7);ctx.globalAlpha=t<=simT?1:.35;ctx.fill();ctx.globalAlpha=1;});
    });
    // playhead
    ctx.strokeStyle=css('--spike');ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(X(simT),padT-4);ctx.lineTo(X(simT),c.h-padB+4);ctx.stroke();
    ctx.fillStyle=css('--muted');ctx.font='10px ui-monospace,monospace';ctx.textAlign='left';
    ctx.fillText('tick '+Math.floor(simT)+' / '+NT, padL, c.h-6);
  }
  function rasterTick(e){var c=canv.raster,rc=c.cv.getBoundingClientRect(),x=(e.clientX-rc.left),padL=54,padR=14;
    var t=(x-padL)/(c.w-padL-padR)*NT;return Math.max(0,Math.min(NT,t));}
  function pick(e){var c=canv.graph,rc=c.cv.getBoundingClientRect(),mx=e.clientX-rc.left,my=e.clientY-rc.top,best=null,bd=1e9;
    D.nodes.forEach(function(n){var p=pos[n.id];if(!p)return;var d=Math.hypot(mx-p.x,my-p.y);if(d<16&&d<bd){bd=d;best={kind:'node',o:n};}});
    if(best)return best;
    D.synapses.forEach(function(s){var a=pos[s.src],b=pos[s.tgt];if(!a||!b)return;var cp=(s.src===s.tgt)?{x:a.x+42,y:a.y}:ctrl(a,b);
      for(var t=0;t<=1;t+=0.1){var pt=bez(a,cp,b,t),d=Math.hypot(mx-pt.x,my-pt.y);if(d<6&&d<bd){bd=d;best={kind:'edge',o:s};}}});
    return best;}
  function showPanel(){
    var s=document.getElementById('side');
    if(!sel){s.innerHTML='<h3>Click a node or edge</h3><p class="small">Nodes: inputs (cyan squares, latency-forced), hidden & output circles colored by RS/FS type; fill = membrane potential, gold flash = spike. Edges: red +w (exc) / blue −w (inh), width ∝ |weight|, dashed = delay>1.</p>';return;}
    if(sel.kind==='node'){var n=sel.o,p=P[n.type],ff=sim.first[n.id];
      s.innerHTML='<h3>'+n.id+' — '+(n.role==='hidden'?'hidden ':'')+n.role+'</h3><table>'
        +row('type',n.type+(n.type==='FS'?' (inhibitory)':' (excitatory)'))+row('a',p.a)+row('b',p.b)+row('c (reset v)',p.c)+row('d (reset u)',p.d)
        +row('threshold',p.vth+' mV')+row('first spike',ff===null?'— (silent)':'tick '+ff)+row('# spikes',(sim.spikes[n.id]||[]).length)+'</table>';
    } else {var s2=sel.o;
      s.innerHTML='<h3>'+s2.src+' → '+s2.tgt+'</h3><table>'
        +row('weight (raw)',(s2.w_raw>=0?'+':'')+s2.w_raw)+row('weight (×64)',s2.w.toFixed(1))+row('sign',s2.exc?'excitatory':'inhibitory')+row('delay',s2.delay+' ticks')+'</table>';}
  }
  function row(k,v){return '<tr><td>'+k+'</td><td class="mono">'+v+'</td></tr>';}

  // ---------------- loop / controls ----------------
  function setPlay(p){playing=p;var b=document.getElementById('play');if(b)b.textContent=p?'⏸ Pause':'▶ Play';}
  function loop(ts){
    if(playing){if(lastTs==null)lastTs=ts;var dt=(ts-lastTs)/1000;lastTs=ts;simT+=dt*speed*11;
      if(simT>=NT){simT=NT;setPlay(false);}dirty=true;} else lastTs=null;
    if(dirty&&sim){drawGraph();drawRaster();renderInfo();if(!sel)showPanel();dirty=false;}
    requestAnimationFrame(loop);
  }
  function bindControls(){
    document.getElementById('play').addEventListener('click',function(){if(simT>=NT)simT=0;setPlay(!playing);});
    document.getElementById('step').addEventListener('click',function(){playing=false;setPlay(false);simT=Math.min(NT,Math.floor(simT)+1);dirty=true;});
    document.getElementById('restart').addEventListener('click',function(){simT=0;playing=false;setPlay(false);dirty=true;});
    document.getElementById('speed').addEventListener('change',function(e){speed=+e.target.value;});
  }
})();
