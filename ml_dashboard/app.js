// ============================================================
// app.js ? AI Fraud Detection Dashboard Logic
// ============================================================

Chart.register(ChartDataLabels);

// ?? State ??
let trained = false;
let allCharts = {};
let miniCMs   = {};

// ?? Helpers ??
function hex2rgba(hex, a) {
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}
function destroyChart(id) {
  if(allCharts[id]) { allCharts[id].destroy(); delete allCharts[id]; }
}
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function pct(v,d=2){ return `${(v*100).toFixed(d)}%`; }
function fmt(v,d=4){ return typeof v==='number' ? v.toFixed(d) : v; }

// ?? Section Navigation ??
document.querySelectorAll('.nav-tab').forEach(tab=>{
  tab.addEventListener('click',()=>{
    document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s=>s.style.display='none');
    tab.classList.add('active');
    document.getElementById(`section-${tab.dataset.section}`).style.display='flex';
    document.getElementById(`section-${tab.dataset.section}`).style.flexDirection='column';
    document.getElementById(`section-${tab.dataset.section}`).style.gap='1.5rem';
  });
});

// ?? Model Sidebar List ??
function buildModelList() {
  const el = document.getElementById('modelList');
  el.innerHTML = '';
  Object.keys(MODELS).forEach((name,i)=>{
    const color = MODEL_COLORS[name]||'#64748b';
    const div = document.createElement('div');
    div.className='model-item checked';
    div.innerHTML=`<input type="checkbox" class="model-cb" id="cb${i}" checked/>
      <label for="cb${i}" class="model-name">${name}</label>
      <div class="model-dot" style="background:${color}"></div>`;
    div.addEventListener('click',e=>{
      if(e.target.tagName!=='INPUT'){
        const cb=div.querySelector('input'); cb.checked=!cb.checked;
      }
      div.classList.toggle('checked',div.querySelector('input').checked);
    });
    el.appendChild(div);
  });
}

// ?? Train Button ??
document.getElementById('trainBtn').addEventListener('click', async()=>{
  const btn=document.getElementById('trainBtn');
  btn.disabled=true; btn.innerHTML=`<span class="spinner"></span> Training...`;
  const wrap=document.getElementById('progressWrap');
  const fill=document.getElementById('progressFill');
  const lbl=document.getElementById('progressLabel');
  wrap.style.display='block';

  const names=Object.keys(MODELS);
  for(let i=0;i<names.length;i++){
    const pct2=Math.round(((i+1)/names.length)*100);
    fill.style.width=pct2+'%';
    lbl.textContent=`?? ${names[i]} ? ${pct2}%`;
    await sleep(350);
  }
  fill.style.width='100%'; lbl.textContent='? All models trained!';
  btn.disabled=false; btn.innerHTML='?? Retrain All Models';
  trained=true;
  renderModelResults();
  renderComparison();
  renderRiskScorecard();
  setTimeout(()=>wrap.style.display='none',2000);
});

// ============================================================
// EDA CHARTS
// ============================================================
function renderEDA() {
  renderClassDist();
  renderAmountDist();
  renderTimeDist();
  renderFraudCorr();
  renderStatsTable();
  renderHeatmap();
  renderBoxPlot();
}

function renderClassDist() {
  destroyChart('classDist');
  const ctx=document.getElementById('classDistChart').getContext('2d');
  allCharts['classDist']=new Chart(ctx,{
    type:'doughnut',
    data:{
      labels:['Legit (Class 0)','Fraud (Class 1)'],
      datasets:[{
        data:[DS.legitCount,DS.fraudCount],
        backgroundColor:['rgba(16,185,129,0.75)','rgba(239,68,68,0.8)'],
        borderColor:['#10b981','#ef4444'],
        borderWidth:2, hoverOffset:12,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,cutout:'72%',
      plugins:{
        legend:{labels:{color:'#94a3b8',font:{size:12}},position:'bottom'},
        datalabels:{
          formatter:(v,ctx2)=>{
            const t=ctx2.dataset.data.reduce((a,b)=>a+b,0);
            return `${((v/t)*100).toFixed(3)}%`;
          },
          color:'#fff',font:{weight:'bold',size:13},
        },
        tooltip:{callbacks:{label:c=>`${c.label}: ${c.raw.toLocaleString()}`}}
      }
    }
  });
}

function renderAmountDist() {
  destroyChart('amountDist');
  const ctx=document.getElementById('amountDistChart').getContext('2d');
  allCharts['amountDist']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:AMOUNT_BINS,
      datasets:[
        {label:'Legitimate',data:LEGIT_AMOUNTS,backgroundColor:'rgba(16,185,129,0.6)',borderColor:'#10b981',borderWidth:1.5,borderRadius:5,yAxisID:'y'},
        {label:'Fraud',     data:FRAUD_AMOUNTS, backgroundColor:'rgba(239,68,68,0.8)',  borderColor:'#ef4444',borderWidth:1.5,borderRadius:5,yAxisID:'y1'},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8'}},datalabels:{display:false}},
      scales:{
        x:{ticks:{color:'#64748b'},grid:{display:false}},
        y:{type:'linear',position:'left',ticks:{color:'#10b981',callback:v=>v>=1000?`${(v/1000).toFixed(0)}k`:v},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Legit Count',color:'#10b981',font:{size:10}}},
        y1:{type:'linear',position:'right',ticks:{color:'#ef4444'},grid:{drawOnChartArea:false},title:{display:true,text:'Fraud Count',color:'#ef4444',font:{size:10}}}
      }
    }
  });
}

function renderTimeDist() {
  destroyChart('timeDist');
  const ctx=document.getElementById('timeDistChart').getContext('2d');
  allCharts['timeDist']=new Chart(ctx,{
    type:'line',
    data:{
      labels:TIME_HOURS,
      datasets:[
        {label:'Legitimate',data:LEGIT_TIME,borderColor:'#10b981',backgroundColor:'rgba(16,185,129,0.1)',fill:true,tension:0.4,borderWidth:2.5,pointRadius:4,pointBackgroundColor:'#10b981',yAxisID:'y'},
        {label:'Fraud',     data:FRAUD_TIME, borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.15)', fill:true,tension:0.4,borderWidth:2.5,pointRadius:4,pointBackgroundColor:'#ef4444',yAxisID:'y1'},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8'}},datalabels:{display:false}},
      scales:{
        x:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{color:'#10b981',callback:v=>v>=1000?`${(v/1000).toFixed(0)}k`:v},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Legit',color:'#10b981',font:{size:10}}},
        y1:{position:'right',ticks:{color:'#ef4444'},grid:{drawOnChartArea:false},title:{display:true,text:'Fraud',color:'#ef4444',font:{size:10}}}
      }
    }
  });
}

function renderFraudCorr() {
  destroyChart('corrFraud');
  const ctx=document.getElementById('corrFraudChart').getContext('2d');
  const sorted=[...FEATURE_FRAUD_CORR].sort((a,b)=>a.r-b.r);
  allCharts['corrFraud']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:sorted.map(d=>d.feat),
      datasets:[{
        label:'Pearson r with Class',
        data:sorted.map(d=>d.r),
        backgroundColor:sorted.map(d=>d.r<0?'rgba(239,68,68,0.7)':'rgba(59,130,246,0.7)'),
        borderColor:sorted.map(d=>d.r<0?'#ef4444':'#3b82f6'),
        borderWidth:1.5,borderRadius:5,
      }]
    },
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{anchor:'end',align:'end',formatter:v=>v.toFixed(3),color:'#94a3b8',font:{size:9}},
        tooltip:{callbacks:{label:c=>`r = ${c.raw.toFixed(4)}`}}
      },
      scales:{
        x:{ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.06)'},title:{display:true,text:'Pearson Correlation with Fraud Class',color:'#64748b',font:{size:10}}},
        y:{ticks:{color:'#e2e8f0',font:{size:10}},grid:{display:false}}
      }
    }
  });
}

function renderStatsTable() {
  let html=`<table class="ml-table"><thead><tr>
    <th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th>
  </tr></thead><tbody>`;
  FEATURE_STATS.forEach(s=>{
    const isFraud=s.feat==='Class';
    html+=`<tr${isFraud?' class="fraud-row"':''}><td class="model-cell">${s.feat}</td>
    <td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.max}</td></tr>`;
  });
  html+=`</tbody></table>`;
  document.getElementById('statsTableWrap').innerHTML=html;
}

function renderHeatmap() {
  destroyChart('heatmap');
  const ctx=document.getElementById('heatmapChart').getContext('2d');
  const n=CORR_FEATS.length;
  const pts=[];
  CORR_MATRIX.forEach((row,i)=>row.forEach((v,j)=>pts.push({x:j,y:n-1-i,v})));

  function corrColor(v){
    v=Math.max(-1,Math.min(1,v));
    if(v>=0){ const t=v; return `rgba(${Math.round(59+t*196)},${Math.round(130*(1-t)+68*t)},${Math.round(246*(1-t)+68*t)},0.85)`; }
    else{ const t=-v; return `rgba(${Math.round(59*(1-t)+239*t)},${Math.round(130*(1-t)+68*t)},${Math.round(246*(1-t)+68*t)},0.85)`; }
  }

  allCharts['heatmap']=new Chart(ctx,{
    type:'scatter',
    data:{datasets:[{data:pts,pointRadius:0}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{
          display:true,
          formatter:(v,c)=>c.dataset.data[c.dataIndex].v.toFixed(1),
          color:c=>(Math.abs(c.dataset.data[c.dataIndex].v)>0.4?'#fff':'#94a3b8'),
          font:{size:7,weight:'bold'}
        },
        tooltip:{callbacks:{label:c=>{const d=c.raw;return `${CORR_FEATS[d.x]} ? ${CORR_FEATS[n-1-d.y]}: ${d.v.toFixed(3)}`;}}}
      },
      scales:{
        x:{min:-0.5,max:n-0.5,ticks:{stepSize:1,callback:v=>CORR_FEATS[Math.round(v)]||'',color:'#64748b',font:{size:8},maxRotation:45},grid:{display:false}},
        y:{min:-0.5,max:n-0.5,ticks:{stepSize:1,callback:v=>CORR_FEATS[n-1-Math.round(v)]||'',color:'#64748b',font:{size:8}},grid:{display:false}}
      }
    },
    plugins:[{id:'hmCells',beforeDraw(chart){
      const{ctx:c,scales:{x,y}}=chart;
      const cw=x.getPixelForValue(1)-x.getPixelForValue(0);
      const ch=y.getPixelForValue(0)-y.getPixelForValue(1);
      pts.forEach(p=>{
        c.fillStyle=corrColor(p.v);
        c.fillRect(x.getPixelForValue(p.x)-cw/2, y.getPixelForValue(p.y)-ch/2, cw, ch);
      });
    }}]
  });
}

function renderBoxPlot() {
  destroyChart('boxPlot');
  const ctx=document.getElementById('boxPlotChart').getContext('2d');
  // Approximate quartile stats for Amount: Legit vs Fraud
  const bins2=['$0','$0-22','$22-77','$77-208','$208-2k','>$2k'];
  allCharts['boxPlot']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:bins2,
      datasets:[
        {label:'Legit Amount',  data:[11200,52300,68100,52000,48200,2615], backgroundColor:'rgba(16,185,129,0.6)',borderColor:'#10b981',borderWidth:1.5,borderRadius:5},
        {label:'Fraud Amount',  data:[75,95,82,68,92,80],                  backgroundColor:'rgba(239,68,68,0.7)',  borderColor:'#ef4444',borderWidth:1.5,borderRadius:5},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8'}},datalabels:{display:false}},
      scales:{
        x:{ticks:{color:'#64748b',font:{size:9}},grid:{display:false}},
        y:{ticks:{color:'#64748b',callback:v=>v>=1000?`${(v/1000).toFixed(0)}k`:v},grid:{color:'rgba(255,255,255,0.05)'}}
      }
    }
  });
}

// ============================================================
// TRAIN/TEST SPLIT SECTION
// ============================================================
function renderSplitSection() {
  buildTrainMatrix();
  buildTestMatrix();
  renderSplitBalance();
  renderFeatureRange();
}

function buildMatrix(rows, wrapperId) {
  const cols=['Time','V1','V2','V3','V4','V5','Amount','Class'];
  let h=`<table class="ml-table"><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(r=>{
    const isFraud=r.cls===1;
    h+=`<tr${isFraud?' class="fraud-row"':''}>`+
      `<td>${r.time}</td><td>${r.v1.toFixed(3)}</td><td>${r.v2.toFixed(3)}</td>`+
      `<td>${r.v3.toFixed(3)}</td><td>${r.v4.toFixed(3)}</td><td>${r.v5.toFixed(3)}</td>`+
      `<td>${r.amount.toFixed(2)}</td>`+
      `<td><span class="${isFraud?'fraud-tag':'legit-tag'}">${isFraud?'FRAUD':'LEGIT'}</span></td>`+
      `</tr>`;
  });
  h+=`</tbody></table>`;
  document.getElementById(wrapperId).innerHTML=h;
}

function buildTrainMatrix(){ buildMatrix(TRAIN_ROWS,'trainMatrixWrap'); }
function buildTestMatrix(){ buildMatrix(TEST_ROWS,'testMatrixWrap'); }

function renderSplitBalance() {
  destroyChart('splitBal');
  const ctx=document.getElementById('splitBalanceChart').getContext('2d');
  allCharts['splitBal']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:['Train Set','Test Set'],
      datasets:[
        {label:'Legit',data:[227452,56863],backgroundColor:'rgba(16,185,129,0.7)',borderColor:'#10b981',borderWidth:1.5,borderRadius:6},
        {label:'Fraud',data:[394,98],        backgroundColor:'rgba(239,68,68,0.8)',  borderColor:'#ef4444',borderWidth:1.5,borderRadius:6},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{labels:{color:'#94a3b8'}},
        datalabels:{display:true,formatter:v=>v.toLocaleString(),color:'#fff',font:{size:10,weight:'bold'},anchor:'center',align:'center'}
      },
      scales:{
        x:{ticks:{color:'#94a3b8'},grid:{display:false}},
        y:{ticks:{color:'#64748b',callback:v=>v>=1000?`${(v/1000).toFixed(0)}k`:v},grid:{color:'rgba(255,255,255,0.05)'}}
      }
    }
  });
}

function renderFeatureRange() {
  destroyChart('featRange');
  const ctx=document.getElementById('featureRangeChart').getContext('2d');
  const feats=['V1','V4','V10','V14','V17','Amount'];
  const means=[-0.001,0.0001,0.0002,0.0002,0.0002,88.35];
  const stds =[1.959,1.416,1.088,0.958,0.995,250.12];
  allCharts['featRange']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:feats,
      datasets:[
        {label:'Mean?Std',data:stds.map((s,i)=>s),backgroundColor:feats.map((_,i)=>`hsl(${200+i*35},70%,55%)`).map(c=>c.replace('hsl','hsla').replace(')',',0.65)')),borderRadius:5,borderWidth:0},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},datalabels:{display:true,formatter:(v,c)=>feats[c.dataIndex],color:'#fff',font:{size:9,weight:'bold'}}},
      scales:{
        x:{ticks:{color:'#64748b'},grid:{display:false}},
        y:{ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Standard Deviation',color:'#64748b',font:{size:10}}}
      }
    }
  });
}

// ============================================================
// MODEL RESULTS SECTION
// ============================================================
function renderModelResults() {
  document.getElementById('untrainedMsg').style.display='none';
  document.getElementById('modelResultsContent').style.display='block';
  document.getElementById('modelResultsBadge').textContent='Credit Card Fraud ? All 7 Models Trained';

  // KPIs
  const names=Object.keys(MODELS);
  const bestF1=names.reduce((a,b)=>MODELS[a].f1>MODELS[b].f1?a:b);
  const bestAUC=names.reduce((a,b)=>MODELS[a].roc_auc>MODELS[b].roc_auc?a:b);
  const bestRec=names.reduce((a,b)=>MODELS[a].recall>MODELS[b].recall?a:b);
  const bestPrc=names.reduce((a,b)=>MODELS[a].precision>MODELS[b].precision?a:b);

  const kpis=[
    {icon:'??',label:'Best F1-Score',     val:pct(MODELS[bestF1].f1,2),      sub:bestF1,            color:'#10b981'},
    {icon:'??',label:'Best ROC-AUC',      val:MODELS[bestAUC].roc_auc.toFixed(4), sub:bestAUC,      color:'#3b82f6'},
    {icon:'??',label:'Best Fraud Recall', val:pct(MODELS[bestRec].recall,2), sub:bestRec,           color:'#f59e0b'},
    {icon:'??',label:'Best Precision',    val:pct(MODELS[bestPrc].precision,2),sub:bestPrc,         color:'#8b5cf6'},
  ];
  const kpiEl=document.getElementById('modelKpis');
  kpiEl.innerHTML=kpis.map(k=>`
    <div class="kpi-card" style="--accent:${k.color}">
      <div class="kpi-icon">${k.icon}</div>
      <div class="kpi-val">${k.val}</div>
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-sub">${k.sub}</div>
    </div>`).join('');

  // Model cards
  const grid=document.getElementById('modelCardsGrid');
  grid.innerHTML='';
  names.forEach((name,idx)=>{
    const m=MODELS[name];
    const color=MODEL_COLORS[name]||'#64748b';
    const cm=m.cm;
    const tn=cm[0][0],fp=cm[0][1],fn=cm[1][0],tp=cm[1][1];

    const card=document.createElement('div');
    card.className='model-card fade-up';
    card.style.animationDelay=`${idx*0.08}s`;
    card.innerHTML=`
      <div class="model-card-head">
        <div class="model-card-icon" style="background:${hex2rgba(color,0.15)};color:${color}">${MODEL_ICONS[name]||'??'}</div>
        <div>
          <div class="model-card-name">${name}</div>
          <div class="model-card-desc">${MODEL_DESC[name]||''}</div>
        </div>
      </div>
      <div class="model-metrics-grid">
        <div class="model-metric"><div class="model-metric-lbl">Accuracy</div><div class="model-metric-val" style="color:#3b82f6">${pct(m.accuracy,4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">Precision</div><div class="model-metric-val" style="color:#10b981">${pct(m.precision,4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">Recall</div><div class="model-metric-val" style="color:#f59e0b">${pct(m.recall,4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">F1-Score</div><div class="model-metric-val" style="color:#8b5cf6">${pct(m.f1,4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">ROC-AUC</div><div class="model-metric-val" style="color:#ef4444">${m.roc_auc.toFixed(4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">PR-AUC</div><div class="model-metric-val" style="color:#ec4899">${m.pr_auc.toFixed(4)}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">True Pos (TP)</div><div class="model-metric-val" style="color:#10b981">${tp}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">False Neg (FN)</div><div class="model-metric-val" style="color:#ef4444">${fn}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">False Pos (FP)</div><div class="model-metric-val" style="color:#f59e0b">${fp}</div></div>
        <div class="model-metric"><div class="model-metric-lbl">True Neg (TN)</div><div class="model-metric-val" style="color:#3b82f6">${tn.toLocaleString()}</div></div>
      </div>
      <div class="model-chart-area"><canvas id="mdlRadar${idx}"></canvas></div>
    `;
    grid.appendChild(card);
    requestAnimationFrame(()=>{
      const rc=document.getElementById(`mdlRadar${idx}`);
      if(!rc) return;
      new Chart(rc.getContext('2d'),{
        type:'radar',
        data:{
          labels:['Accuracy?100','Precision','Recall','F1-Score','ROC-AUC'],
          datasets:[{
            label:name,
            data:[m.accuracy*100, m.precision*100, m.recall*100, m.f1*100, m.roc_auc*100],
            backgroundColor:hex2rgba(color,0.18),borderColor:color,borderWidth:2,
            pointBackgroundColor:color,pointRadius:3,
          }]
        },
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false},datalabels:{display:false}},
          scales:{r:{min:0,max:100,ticks:{color:'#64748b',font:{size:7},backdropColor:'transparent',stepSize:25},
          grid:{color:'rgba(255,255,255,0.07)'},pointLabels:{color:'#94a3b8',font:{size:9}}}}}
      });
    });
  });
}

// ============================================================
// COMPARISON SECTION
// ============================================================
function renderComparison() {
  document.getElementById('untrainedMsg2').style.display='none';
  document.getElementById('compareContent').style.display='block';

  const names=Object.keys(MODELS);
  renderAccComp(names);
  renderPRF1Comp(names);
  renderROCComp(names);
  renderPRCurveComp(names);
  renderAllCMs(names);
  renderMetricsTable(names);
}

function renderAccComp(names) {
  destroyChart('accComp');
  const ctx=document.getElementById('accCompChart').getContext('2d');
  allCharts['accComp']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:names,
      datasets:[{
        label:'Accuracy (%)',
        data:names.map(n=>MODELS[n].accuracy*100),
        backgroundColor:names.map(n=>hex2rgba(MODEL_COLORS[n]||'#64748b',0.75)),
        borderColor:names.map(n=>MODEL_COLORS[n]||'#64748b'),
        borderWidth:2,borderRadius:8,
      }]
    },
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{anchor:'end',align:'end',formatter:v=>`${v.toFixed(3)}%`,color:'#94a3b8',font:{size:9,weight:'bold'}}
      },
      scales:{
        x:{min:97,max:100.5,ticks:{color:'#64748b',callback:v=>`${v}%`},grid:{color:'rgba(255,255,255,0.05)'}},
        y:{ticks:{color:'#e2e8f0',font:{size:11}},grid:{display:false}}
      }
    }
  });
}

function renderPRF1Comp(names) {
  destroyChart('prf1Comp');
  const ctx=document.getElementById('prf1CompChart').getContext('2d');
  allCharts['prf1Comp']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:names,
      datasets:[
        {label:'Precision',data:names.map(n=>MODELS[n].precision*100),backgroundColor:'rgba(59,130,246,0.7)',borderColor:'#3b82f6',borderWidth:1,borderRadius:4},
        {label:'Recall',   data:names.map(n=>MODELS[n].recall*100),   backgroundColor:'rgba(245,158,11,0.7)',borderColor:'#f59e0b',borderWidth:1,borderRadius:4},
        {label:'F1-Score', data:names.map(n=>MODELS[n].f1*100),       backgroundColor:'rgba(139,92,246,0.7)',borderColor:'#8b5cf6',borderWidth:1,borderRadius:4},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8'}},datalabels:{display:false}},
      scales:{
        x:{ticks:{color:'#64748b',font:{size:9.5},maxRotation:20},grid:{display:false}},
        y:{min:0,max:110,ticks:{color:'#64748b',callback:v=>`${v}%`},grid:{color:'rgba(255,255,255,0.05)'}}
      }
    }
  });
}

function renderROCComp(names) {
  destroyChart('rocComp');
  const ctx=document.getElementById('rocCompChart').getContext('2d');
  const datasets=names.map(name=>{
    const auc=MODELS[name].roc_auc;
    const color=MODEL_COLORS[name]||'#64748b';
    const pts=[];
    for(let x=0;x<=1;x+=0.04){
      const exp=(1-auc)/Math.max(auc,0.01);
      pts.push({x:parseFloat(x.toFixed(2)),y:parseFloat(Math.min(1,Math.pow(x,exp)).toFixed(4))});
    }
    return{label:`${name} (${auc.toFixed(4)})`,data:pts,borderColor:color,backgroundColor:'transparent',borderWidth:2,pointRadius:0,tension:0.3,fill:false};
  });
  datasets.push({label:'Random (0.5)',data:[{x:0,y:0},{x:1,y:1}],borderColor:'rgba(255,255,255,0.15)',borderDash:[5,4],borderWidth:1.5,pointRadius:0,fill:false});
  allCharts['rocComp']=new Chart(ctx,{
    type:'line',data:{datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:12},position:'bottom'},datalabels:{display:false}},
      scales:{
        x:{type:'linear',min:0,max:1,title:{display:true,text:'False Positive Rate',color:'#64748b',font:{size:10}},ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{type:'linear',min:0,max:1,title:{display:true,text:'True Positive Rate',color:'#64748b',font:{size:10}},ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}

function renderPRCurveComp(names) {
  destroyChart('prCurve');
  const ctx=document.getElementById('prCurveChart').getContext('2d');
  // Simulate Precision-Recall curve from pr_auc
  const datasets=names.map(name=>{
    const pr_auc=MODELS[name].pr_auc;
    const color=MODEL_COLORS[name]||'#64748b';
    const pts=[];
    for(let r=0;r<=1;r+=0.05){
      const p=Math.max(0.01, Math.min(0.99, pr_auc + (pr_auc-0.5)*(1-r)*0.4 + (1-r)*pr_auc*0.3));
      pts.push({x:parseFloat(r.toFixed(2)),y:parseFloat(p.toFixed(4))});
    }
    return{label:`${name} (PR-AUC=${pr_auc.toFixed(4)})`,data:pts,borderColor:color,backgroundColor:'transparent',borderWidth:2,pointRadius:0,tension:0.3,fill:false};
  });
  // Baseline (random)
  datasets.push({label:`Baseline (${(DS.fraudRate).toFixed(4)})`,data:[{x:0,y:DS.fraudRate},{x:1,y:DS.fraudRate}],borderColor:'rgba(255,255,255,0.15)',borderDash:[5,4],borderWidth:1.5,pointRadius:0,fill:false});
  allCharts['prCurve']=new Chart(ctx,{
    type:'line',data:{datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:12},position:'bottom'},datalabels:{display:false}},
      scales:{
        x:{type:'linear',min:0,max:1,title:{display:true,text:'Recall',color:'#64748b',font:{size:10}},ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{type:'linear',min:0,max:1,title:{display:true,text:'Precision',color:'#64748b',font:{size:10}},ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}

function renderCMCanvas(canvas, cm, modelName) {
  const n=cm.length;
  const labels=['Legit','Fraud'].slice(0,n);
  const flat=[];
  const colors=[];
  const total=cm.flat().reduce((a,b)=>a+b,0);
  cm.forEach((row,i)=>row.forEach((v,j)=>{
    flat.push({x:j,y:n-1-i,v,pct:(v/total*100).toFixed(2)});
    colors.push(i===j?(i===0?'rgba(16,185,129,0.65)':'rgba(16,185,129,0.65)'):( v>0?'rgba(239,68,68,0.65)':'rgba(255,255,255,0.03)'));
  }));
  // diagonal: TN=green, TP=green; off-diagonal: red
  cm.forEach((row,i)=>row.forEach((v,j)=>{
    const idx=i*n+j;
    if(i===j) colors[idx]='rgba(16,185,129,0.65)';
    else if(v>0) colors[idx]='rgba(239,68,68,0.6)';
    else colors[idx]='rgba(255,255,255,0.03)';
  }));

  const ex=Chart.getChart(canvas); if(ex) ex.destroy();
  return new Chart(canvas.getContext('2d'),{
    type:'scatter',
    data:{datasets:[{data:flat,pointRadius:0}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        title:{display:!!modelName,text:modelName,color:'#94a3b8',font:{size:9}},
        datalabels:{display:true,formatter:(v,c)=>{const d=c.dataset.data[c.dataIndex];return `${d.v}\n(${d.pct}%)`;},color:'#fff',font:{size:n>2?8:11,weight:'bold'},align:'center',anchor:'center'},
        tooltip:{callbacks:{label:c=>{const d=c.raw;return `Pred:${labels[d.x]} True:${labels[n-1-d.y]} ? ${d.v}`;}}},
      },
      scales:{
        x:{min:-0.5,max:n-0.5,title:{display:true,text:'Predicted',color:'#64748b',font:{size:9}},ticks:{stepSize:1,callback:v=>labels[Math.round(v)]||'',color:'#94a3b8',font:{size:9}},grid:{display:false}},
        y:{min:-0.5,max:n-0.5,title:{display:true,text:'Actual',color:'#64748b',font:{size:9}},ticks:{stepSize:1,callback:v=>labels[n-1-Math.round(v)]||'',color:'#94a3b8',font:{size:9}},grid:{display:false}}
      }
    },
    plugins:[{id:'cmCells',beforeDraw(chart){
      const{ctx:c,scales:{x,y}}=chart;
      const cw=x.getPixelForValue(1)-x.getPixelForValue(0);
      const ch=y.getPixelForValue(0)-y.getPixelForValue(1);
      flat.forEach((p,i)=>{
        c.fillStyle=colors[i];
        c.fillRect(x.getPixelForValue(p.x)-cw/2,y.getPixelForValue(p.y)-ch/2,cw,ch);
        c.strokeStyle='rgba(255,255,255,0.06)';
        c.strokeRect(x.getPixelForValue(p.x)-cw/2,y.getPixelForValue(p.y)-ch/2,cw,ch);
      });
    }}]
  });
}

function renderAllCMs(names) {
  const grid=document.getElementById('cmGridAll');
  grid.innerHTML='';
  Object.keys(miniCMs).forEach(k=>{miniCMs[k].destroy();delete miniCMs[k];});
  names.forEach(name=>{
    const cm=MODELS[name].cm;
    const color=MODEL_COLORS[name]||'#64748b';
    const wrap=document.createElement('div'); wrap.className='cm-mini';
    const title=document.createElement('div'); title.className='cm-mini-title';
    title.style.color=color; title.textContent=name;
    const canvas=document.createElement('canvas'); canvas.height=150;
    wrap.appendChild(title); wrap.appendChild(canvas);
    grid.appendChild(wrap);
    const ch=renderCMCanvas(canvas,cm,'');
    if(ch) miniCMs[name]=ch;
  });
}

function renderMetricsTable(names) {
  const bestAcc=Math.max(...names.map(n=>MODELS[n].accuracy));
  const bestPrc=Math.max(...names.map(n=>MODELS[n].precision));
  const bestRec=Math.max(...names.map(n=>MODELS[n].recall));
  const bestF1 =Math.max(...names.map(n=>MODELS[n].f1));
  const bestAUC=Math.max(...names.map(n=>MODELS[n].roc_auc));
  const bestPRAUC=Math.max(...names.map(n=>MODELS[n].pr_auc));

  let html=`<table class="ml-table"><thead><tr>
    <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
    <th>F1-Score</th><th>ROC-AUC</th><th>PR-AUC</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th>
  </tr></thead><tbody>`;
  names.forEach(n=>{
    const m=MODELS[n];
    const [tn,fp,fn,tp]=[m.cm[0][0],m.cm[0][1],m.cm[1][0],m.cm[1][1]];
    html+=`<tr>
      <td class="model-cell">${MODEL_ICONS[n]||''} ${n}</td>
      <td class="${m.accuracy===bestAcc?'best':''}">${pct(m.accuracy,4)}</td>
      <td class="${m.precision===bestPrc?'best':''}">${pct(m.precision,4)}</td>
      <td class="${m.recall===bestRec?'best':''}">${pct(m.recall,4)}</td>
      <td class="${m.f1===bestF1?'best':''}">${pct(m.f1,4)}</td>
      <td class="${m.roc_auc===bestAUC?'best':''}">${m.roc_auc.toFixed(4)}</td>
      <td class="${m.pr_auc===bestPRAUC?'best':''}">${m.pr_auc.toFixed(4)}</td>
      <td class="best">${tp}</td>
      <td class="worst">${fp}</td>
      <td class="worst">${fn}</td>
      <td>${tn.toLocaleString()}</td>
    </tr>`;
  });
  html+=`</tbody></table>`;
  document.getElementById('metricsTableWrap').innerHTML=html;
}

// ============================================================
// RISK SCORECARD SECTION
// ============================================================
function renderRiskScorecard() {
  document.getElementById('untrainedMsg3').style.display='none';
  document.getElementById('riskContent').style.display='block';

  renderRankChart();
  renderFeatImportance();
  renderFNChart();
  renderImpactChart();
  renderThreshChart();
  renderRiskTable();
}

function renderRankChart() {
  destroyChart('rankChart');
  const ctx=document.getElementById('rankChart').getContext('2d');
  const names=Object.keys(MODELS);
  // Score = 0.4*F1 + 0.35*PRAUC + 0.25*Recall  (fraud-focused composite)
  const scores=names.map(n=>0.4*MODELS[n].f1+0.35*MODELS[n].pr_auc+0.25*MODELS[n].recall);
  const sorted=[...names.map((n,i)=>({n,s:scores[i]}))].sort((a,b)=>b.s-a.s);
  allCharts['rankChart']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:sorted.map(d=>d.n),
      datasets:[{
        label:'Fraud-Focused Score',
        data:sorted.map(d=>d.s*100),
        backgroundColor:sorted.map(d=>hex2rgba(MODEL_COLORS[d.n]||'#64748b',0.75)),
        borderColor:sorted.map(d=>MODEL_COLORS[d.n]||'#64748b'),
        borderWidth:2,borderRadius:8,
      }]
    },
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{anchor:'end',align:'end',formatter:v=>`${v.toFixed(1)}`,color:'#94a3b8',font:{size:10,weight:'bold'}},
        tooltip:{callbacks:{label:c=>`Score: ${c.raw.toFixed(2)}/100 (40%?F1 + 35%?PR-AUC + 25%?Recall)`}}
      },
      scales:{
        x:{min:0,max:100,ticks:{color:'#64748b',callback:v=>`${v}`},grid:{color:'rgba(255,255,255,0.05)'}},
        y:{ticks:{color:'#e2e8f0',font:{size:10.5}},grid:{display:false}}
      }
    }
  });
}

function renderFeatImportance() {
  destroyChart('featImp');
  const ctx=document.getElementById('featImpChart').getContext('2d');
  const sorted=[...FEAT_IMPORTANCE].sort((a,b)=>b.imp-a.imp);
  allCharts['featImp']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:sorted.map(d=>d.feat),
      datasets:[{
        label:'Feature Importance',
        data:sorted.map(d=>d.imp*100),
        backgroundColor:sorted.map((_,i)=>`hsl(${0+i*12},75%,55%)`).map(c=>c.replace('hsl','hsla').replace(')',',0.75)')),
        borderRadius:5,borderWidth:0,
      }]
    },
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},datalabels:{anchor:'end',align:'end',formatter:v=>`${v.toFixed(1)}%`,color:'#94a3b8',font:{size:9}}},
      scales:{
        x:{ticks:{color:'#64748b',callback:v=>`${v}%`},grid:{color:'rgba(255,255,255,0.05)'}},
        y:{ticks:{color:'#e2e8f0',font:{size:10}},grid:{display:false}}
      }
    }
  });
}

function renderFNChart() {
  destroyChart('fnChart');
  const ctx=document.getElementById('fnChart').getContext('2d');
  const names=Object.keys(MODELS);
  const fns=names.map(n=>MODELS[n].cm[1][0]);
  allCharts['fnChart']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:names,
      datasets:[{
        label:'False Negatives (Missed Fraud)',
        data:fns,
        backgroundColor:names.map(n=>hex2rgba(MODEL_COLORS[n]||'#64748b',0.75)),
        borderColor:names.map(n=>MODEL_COLORS[n]||'#64748b'),
        borderWidth:1.5,borderRadius:6,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{display:true,anchor:'end',align:'end',formatter:v=>`${v} missed`,color:'#fca5a5',font:{size:9,weight:'bold'}},
        tooltip:{callbacks:{label:c=>`${c.raw} fraud cases MISSED (out of 98)`}}
      },
      scales:{
        x:{ticks:{color:'#64748b',font:{size:9},maxRotation:25},grid:{display:false}},
        y:{ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Missed Frauds (FN)',color:'#64748b',font:{size:10}}}
      }
    }
  });
}

function renderImpactChart() {
  destroyChart('impactChart');
  const ctx=document.getElementById('impactChart').getContext('2d');
  const names=Object.keys(MODELS);
  const avgFraudAmt=122.21;
  // Estimated missed fraud cost = FN * avg fraud amount
  const costs=names.map(n=>MODELS[n].cm[1][0]*avgFraudAmt);
  allCharts['impactChart']=new Chart(ctx,{
    type:'bar',
    data:{
      labels:names,
      datasets:[{
        label:'Est. Missed Fraud Cost ($)',
        data:costs,
        backgroundColor:names.map(n=>hex2rgba(MODEL_COLORS[n]||'#64748b',0.72)),
        borderColor:names.map(n=>MODEL_COLORS[n]||'#64748b'),
        borderWidth:1.5,borderRadius:6,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        datalabels:{display:true,anchor:'end',align:'end',formatter:v=>`$${v.toFixed(0)}`,color:'#fca5a5',font:{size:9,weight:'bold'}},
      },
      scales:{
        x:{ticks:{color:'#64748b',font:{size:9},maxRotation:25},grid:{display:false}},
        y:{ticks:{color:'#64748b',callback:v=>`$${v}`},grid:{color:'rgba(255,255,255,0.05)'},title:{display:true,text:'Financial Impact ($)',color:'#64748b',font:{size:10}}}
      }
    }
  });
}

function renderThreshChart() {
  destroyChart('threshChart');
  const ctx=document.getElementById('threshChart').getContext('2d');
  allCharts['threshChart']=new Chart(ctx,{
    type:'line',
    data:{
      labels:THRESHOLDS.map(t=>`${(t*100).toFixed(0)}%`),
      datasets:[
        {label:'Precision',data:THRESH_PREC.map(v=>v*100),borderColor:'#3b82f6',borderWidth:2,tension:0.4,pointRadius:4,fill:false},
        {label:'Recall',   data:THRESH_REC.map(v=>v*100), borderColor:'#f59e0b',borderWidth:2,tension:0.4,pointRadius:4,fill:false},
        {label:'F1-Score', data:THRESH_F1.map(v=>v*100),  borderColor:'#8b5cf6',borderWidth:2.5,tension:0.4,pointRadius:4,fill:false},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}},datalabels:{display:false}},
      scales:{
        x:{ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'},title:{display:true,text:'Decision Threshold',color:'#64748b',font:{size:10}}},
        y:{min:0,max:110,ticks:{color:'#64748b',callback:v=>`${v}%`},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}

function renderRiskTable() {
  const names=Object.keys(MODELS);
  const avgAmt=122.21;
  let html=`<table class="ml-table"><thead><tr>
    <th>Model</th><th>F1 (Fraud)</th><th>ROC-AUC</th><th>PR-AUC</th>
    <th>TP (Caught)</th><th>FN (Missed)</th><th>FP (False Alarm)</th>
    <th>Est. Cost Saved</th><th>Risk Level</th>
  </tr></thead><tbody>`;
  names.forEach(n=>{
    const m=MODELS[n];
    const [tn,fp,fn,tp]=[m.cm[0][0],m.cm[0][1],m.cm[1][0],m.cm[1][1]];
    const saved=((tp*avgAmt)).toFixed(0);
    const missed=((fn*avgAmt)).toFixed(0);
    const riskLvl=fn<=20?'?? Low':fn<=35?'?? Medium':'?? High';
    const score=0.4*m.f1+0.35*m.pr_auc+0.25*m.recall;
    html+=`<tr>
      <td class="model-cell">${MODEL_ICONS[n]||''} ${n}</td>
      <td class="${m.f1===Math.max(...names.map(x=>MODELS[x].f1))?'best':''}">${pct(m.f1,4)}</td>
      <td>${m.roc_auc.toFixed(4)}</td>
      <td>${m.pr_auc.toFixed(4)}</td>
      <td class="best">${tp} / 98</td>
      <td class="${fn<=20?'best':'worst'}">${fn}</td>
      <td>${fp}</td>
      <td class="best">$${parseInt(saved).toLocaleString()}</td>
      <td>${riskLvl}</td>
    </tr>`;
  });
  html+=`</tbody></table>`;
  document.getElementById('riskTableWrap').innerHTML=html;
}

// ============================================================
// INIT
// ============================================================
function init() {
  buildModelList();
  renderEDA();
  renderSplitSection();
}

document.addEventListener('DOMContentLoaded', init);
