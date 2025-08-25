async function openAIChat(apiKey, model, messages, temperature = 0.7, top_p = 1.0, extra = {}) {
  const useCustom = document.getElementById('useCustomApiBase');
  const base = (useCustom && useCustom.checked ? (document.getElementById('apiBase')?.value?.trim()) : '') || 'https://api.openai.com/v1';
  const url = base.replace(/\/$/, '') + '/chat/completions';
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {})
    },
    body: JSON.stringify({ model, messages, temperature, top_p, ...extra })
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`OpenAI error ${res.status}: ${errText}`);
  }
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content || '';
  return text.trim();
}

function escapeHtml(str){
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

async function generateOnce(apiKey, model, prompt, params) {
  return openAIChat(apiKey, model, [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: prompt }
  ], params.temperature, params.top_p, params.extra);
}

async function judgeOnce(apiKey, model, output, params, judgeId) {
  const evalPrompt = `You are Judge #${judgeId}. Evaluate the OUTPUT below which was generated at temperature ${params.temperature} and top_p ${params.top_p}.
Return a STRICT minified JSON object with numeric fields only (no text outside JSON):
{"relevance": float0to100, "clarity": float0to100, "utility": float0to100, "creativity": float0to100, "coherence": float0to100, "safety": float0to100, "overall": float0to100}
Output between triple dashes:
---
${output}
---`;
  const systemPrompt = (document.getElementById('judgeSystemPrompt')?.value || '').trim() || 'Return only the JSON.';
  const raw = await openAIChat(apiKey, model, [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: evalPrompt }
  ], 0.2, 1.0);
  try {
    const jsonText = (raw.match(/\{[\s\S]*\}/) || [raw])[0];
    const obj = JSON.parse(jsonText);
    return {
      relevance: +obj.relevance || 0,
      clarity: +obj.clarity || 0,
      utility: +obj.utility || 0,
      creativity: +obj.creativity || 0,
      coherence: +obj.coherence || 0,
      safety: +obj.safety || 0,
      overall: +obj.overall || 0,
    };
  } catch (e) {
    const num = (raw.match(/\d+(?:\.\d+)?/) || [0])[0];
    return { relevance: 0, clarity: 0, utility: 0, creativity: 0, coherence: 0, safety: 0, overall: +num };
  }
}

function mean(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0) / arr.length : 0; }

function aggregateScores(scores) {
  const keys = ['relevance','clarity','utility','creativity','coherence','safety','overall'];
  const out = {};
  for (const k of keys) out[k] = +mean(scores.map(s=>s[k]||0)).toFixed(2);
  return out;
}

async function standardMode(apiKey, model, prompt, arms, judges) {
  const outputs = {};
  const details = {};
  const overalls = {};

  await Promise.all(arms.map(async (arm) => {
    const key = JSON.stringify(arm);
    const text = await generateOnce(apiKey, model, prompt, arm);
    outputs[key] = text;
    const judgeResults = await Promise.all(Array.from({length: judges}).map((_,i)=>
      judgeOnce(apiKey, model, text, arm, i+1)
    ));
    const agg = aggregateScores(judgeResults);
    details[key] = agg;
    overalls[key] = agg.overall;
  }));

  const ranked = Object.entries(overalls).sort((a,b)=>b[1]-a[1]);
  return { outputs, details, ranked };
}

async function advancedModeUCB(apiKey, model, prompt, arms, judges, rounds, c) {
  const keys = arms.map(a=>JSON.stringify(a));
  const pulls = Object.fromEntries(keys.map(k=>[k,0]));
  const sums = Object.fromEntries(keys.map(k=>[k,0]));
  const best = Object.fromEntries(keys.map(k=>[k,{overall:-1,text:'',detail:{}}]));
  let total = 0;

  // init
  for (const arm of arms) {
    const k = JSON.stringify(arm);
    const text = await generateOnce(apiKey, model, prompt, arm);
    const judgeResults = await Promise.all(Array.from({length: judges}).map((_,i)=>
      judgeOnce(apiKey, model, text, arm, i+1)
    ));
    const agg = aggregateScores(judgeResults);
    pulls[k] += 1; sums[k] += agg.overall; total += 1;
    if (agg.overall > best[k].overall) best[k] = {overall: agg.overall, text, detail: agg};
  }

  for (let r = 0; r < rounds - 1; r++) {
    const ucb = {};
    for (const k of keys) {
      const m = pulls[k] ? (sums[k]/pulls[k]) : Infinity;
      const bonus = pulls[k] ? c * Math.sqrt(Math.log(Math.max(1,total)) / pulls[k]) : Infinity;
      ucb[k] = m + bonus;
    }
    const nextK = keys.sort((a,b)=>ucb[b]-ucb[a])[0];
    const arm = JSON.parse(nextK);
    const text = await generateOnce(apiKey, model, prompt, arm);
    const judgeResults = await Promise.all(Array.from({length: judges}).map((_,i)=>
      judgeOnce(apiKey, model, text, arm, i+1)
    ));
    const agg = aggregateScores(judgeResults);
    pulls[nextK] += 1; sums[nextK] += agg.overall; total += 1;
    if (agg.overall > best[nextK].overall) best[nextK] = {overall: agg.overall, text, detail: agg};
  }

  const means = Object.fromEntries(keys.map(k=>[k, pulls[k] ? (sums[k]/pulls[k]) : 0]));
  const rankedKeys = keys.slice().sort((a,b)=>means[b]-means[a]);
  const bestK = rankedKeys[0];
  return { bestK, best: best[bestK], means, pulls };
}

function getEl(id){ return document.getElementById(id); }
function setText(id, txt){ getEl(id).textContent = txt; }
function appendLog(msg){ const el=getEl('runLog'); if(!el) return; el.textContent += `\n${msg}`; el.scrollTop = el.scrollHeight; }

function renderArmsTable(arms){
  const tbody = getEl('armsTable').querySelector('tbody');
  tbody.innerHTML = '';
  for (const arm of arms){
    const k = JSON.stringify(arm);
    const tr = document.createElement('tr');
    tr.id = `arm-${btoa(k).replace(/=/g,'')}`;
    tr.innerHTML = `
      <td class="status status-wait">waiting</td>
      <td class="pulls">0</td>
      <td class="mean">-</td>
      <td class="best">-</td>
      <td><details><summary>view</summary><div class="arm-detail"></div></details></td>
    `;
    tbody.appendChild(tr);
  }
}

function updateArmRow(arm, data){
  const k = JSON.stringify(arm);
  const id = `arm-${btoa(k).replace(/=/g,'')}`;
  const tr = getEl(id);
  if (!tr) return;
  if (data.status) { const s = tr.querySelector('.status'); s.textContent = data.status; s.className = `status ${data.statusClass||''}`; }
  if (data.pulls !== undefined) tr.querySelector('.pulls').textContent = String(data.pulls);
  if (data.mean !== undefined) tr.querySelector('.mean').textContent = (data.mean===null?'-':Number(data.mean).toFixed(2));
  if (data.best !== undefined) tr.querySelector('.best').textContent = (data.best===null?'-':Number(data.best).toFixed(2));
  if (data.detail) tr.querySelector('.arm-detail').innerHTML = data.detail;
}

document.addEventListener('DOMContentLoaded', () => {
  // Chart setup
  let chart;
  function ensureChart(){
    const ctx = getEl('scoreChart');
    if (!ctx) return null;
    if (chart) return chart;
    chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: [{ label: 'temp vs mean score', data: [], borderColor:'#00ff9c', backgroundColor:'rgba(0,255,156,0.3)' }]},
      options: {
        scales: {
          x: { title: { display:true, text:'temperature' }, grid: { color:'#0b442f' }, ticks:{ color:'#b5f5d2' } },
          y: { title: { display:true, text:'mean judge score' }, suggestedMin:0, suggestedMax:100, grid: { color:'#0b442f' }, ticks:{ color:'#b5f5d2' } }
        },
        plugins: { legend: { labels: { color:'#b5f5d2' } } }
      }
    });
    return chart;
  }
  function addChartPoint(temp, mean){
    const c = ensureChart(); if (!c) return;
    c.data.datasets[0].data.push({ x: temp, y: mean });
    c.update('none');
  }
  // Custom API base toggle
  const useCustom = getEl('useCustomApiBase');
  const apiBaseField = getEl('apiBaseField');
  if (useCustom && apiBaseField){
    const savedUse = localStorage.getItem('autotemp_use_custom_api') === '1';
    useCustom.checked = savedUse;
    apiBaseField.style.display = savedUse ? '' : 'none';
    useCustom.addEventListener('change', ()=>{
      apiBaseField.style.display = useCustom.checked ? '' : 'none';
      localStorage.setItem('autotemp_use_custom_api', useCustom.checked ? '1' : '0');
    });
  }
  // Run state and controls
  let running = false;
  let cancelled = false;
  let runResults = { arms: [], records: [] };
  const runBtn = getEl('runBtn');
  const stopBtn = getEl('stopBtn');
  const downloadBtn = getEl('downloadBtn');
  function enableRunButtons(isRunning){
    running = !!isRunning;
    if (runBtn) runBtn.disabled = running;
    if (stopBtn) stopBtn.disabled = !running;
    if (downloadBtn) downloadBtn.disabled = running || !runResults.records.length;
  }
  function recordResult(arm, output, judges){
    try { runResults.records.push({ timestamp: Date.now(), arm, output, judges }); } catch(e) {}
    if (downloadBtn) downloadBtn.disabled = running ? true : !runResults.records.length;
  }
  function downloadJSON(){
    const blob = new Blob([JSON.stringify(runResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'autotemp_results.json'; a.click();
    URL.revokeObjectURL(url);
  }
  if (downloadBtn) downloadBtn.addEventListener('click', downloadJSON);
  if (stopBtn) stopBtn.addEventListener('click', () => { if (running) { cancelled = true; appendLog('Stop requested: finishing current step then halting.'); } });
  // Persist judge system prompt
  const judgeSystemPromptEl = getEl('judgeSystemPrompt');
  if (judgeSystemPromptEl){
    const saved = localStorage.getItem('autotemp_judge_system_prompt');
    if (saved) judgeSystemPromptEl.value = saved;
    else judgeSystemPromptEl.value = 'You are a strict evaluator. Return only a minified JSON with the numeric fields: {"relevance","clarity","utility","creativity","coherence","safety","overall"}. Do not include text outside JSON.';
    judgeSystemPromptEl.addEventListener('input', ()=> localStorage.setItem('autotemp_judge_system_prompt', judgeSystemPromptEl.value));
  }
  const judges = getEl('judges');
  const rounds = getEl('rounds');
  const explorationC = getEl('explorationC');
  judges.addEventListener('input', ()=> setText('judgesVal', judges.value));
  rounds.addEventListener('input', ()=> setText('roundsVal', rounds.value));
  explorationC.addEventListener('input', ()=> setText('cVal', (+explorationC.value).toFixed(2)));

  // Dotbar helpers to visualize/select discrete values
  function initDotbar(containerId, min, max, step, parseFn, inputId){
    const bar = getEl(containerId); const inp = getEl(inputId);
    if (!bar || !inp) return;
    bar.innerHTML = '<div class="range"></div><div class="labels"><span>'+min+'</span><span>'+max+'</span></div>';
    const values = (inp.value||'').split(',').map(s=>parseFn(s.trim())).filter(v=>!Number.isNaN(v));
    const active = new Set(values.map(v=>String(v)));
    const count = Math.floor((max-min)/step)+1;
    for (let i=0;i<count;i++){
      const v = +(min + i*step).toFixed( (step<1 && step>0) ? 2 : 0 );
      const dot = document.createElement('div'); dot.className='dot '+(active.has(String(v))?'':'inactive');
      dot.style.left = `${(i/(count-1))*100}%`;
      dot.title = String(v);
      dot.addEventListener('click', ()=>{
        const key = String(v);
        if (active.has(key)) active.delete(key); else active.add(key);
        dot.classList.toggle('inactive');
        const list = Array.from(active).map(x=>parseFn(x));
        list.sort((a,b)=>a-b);
        inp.value = list.join(',');
      });
      bar.appendChild(dot);
    }
  }

  initDotbar('tempDots', 0.0, 1.5, 0.1, parseFloat, 'temperatures');
  initDotbar('topDots', 0.0, 1.0, 0.1, parseFloat, 'tops');
  // Reduce noise: ~30% fewer dots
  initDotbar('maxTokDots', 128, 8192, 512, x=>parseInt(x,10), 'maxTokens');
  initDotbar('freqDots', 0.0, 2.0, 0.15, parseFloat, 'freqPen');
  initDotbar('presDots', 0.0, 2.0, 0.15, parseFloat, 'presPen');

  getEl('runBtn').addEventListener('click', async () => {
    if (running) { appendLog('Run already in progress. Please wait or press Stop.'); return; }
    const apiKey = getEl('apiKey').value.trim();
    const remember = getEl('rememberKey').checked;
    if (!apiKey) { alert('Please enter an API key.'); return; }
    if (remember) localStorage.setItem('autotemp_api_key', apiKey); else localStorage.removeItem('autotemp_api_key');

    const model = getEl('model').value.trim() || 'gpt-4o-mini';
    const temps = getEl('temperatures').value.split(',').map(s=>parseFloat(s.trim())).filter(n=>!Number.isNaN(n));
    const tops = getEl('tops').value.split(',').map(s=>parseFloat(s.trim())).filter(n=>!Number.isNaN(n));
    const maxTokens = getEl('maxTokens').value.split(',').map(s=>parseInt(s.trim(),10)).filter(n=>!Number.isNaN(n));
    const freqPen = getEl('freqPen').value.split(',').map(s=>parseFloat(s.trim())).filter(n=>!Number.isNaN(n));
    const presPen = getEl('presPen').value.split(',').map(s=>parseFloat(s.trim())).filter(n=>!Number.isNaN(n));
    const stopRaw = getEl('stopSeqs').value.trim();
    const stopTokens = stopRaw ? stopRaw.split(',').map(s=>s.replace(/\\n/g,'\n')) : undefined;
    const j = parseInt(getEl('judges').value, 10) || 3;
    const auto = getEl('autoSelect').checked;
    const adv = getEl('advancedMode').checked;
    const r = parseInt(getEl('rounds').value, 10) || 5;
    const c = parseFloat(getEl('explorationC').value) || 1.0;
    const prompt = getEl('userPrompt').value.trim();
    if (!prompt) { alert('Enter a prompt.'); return; }

    // build arms (Cartesian product)
    function cartesian(arrs){ return arrs.reduce((a,b)=> a.flatMap(x=> b.map(y=>[...x,y])), [[]]); }
    const lists = [temps, tops, maxTokens, freqPen, presPen];
    const combos = cartesian(lists);
    const arms = combos.map(([temperature, top_p, max_tokens, frequency_penalty, presence_penalty]) => ({
      temperature, top_p,
      extra: {
        max_tokens,
        frequency_penalty,
        presence_penalty,
        ...(stopTokens ? { stop: stopTokens } : {})
      }
    }));

    const status = getEl('status');
    const results = getEl('results');
    results.textContent = '';
    status.textContent = 'Running...';
    appendLog(`Initialized ${arms.length} arms. Judges=${j}. Advanced=${adv ? 'UCB' : 'Standard'}.`);
    renderArmsTable(arms);
    // initialize run state
    cancelled = false;
    runResults = { arms, records: [] };
    enableRunButtons(true);
    try {
      const c = ensureChart(); if (c){ c.data.datasets[0].data = []; c.update('none'); }
      if (!adv) {
        const outputs = {}; const details = {}; const overalls = {};
        for (const arm of arms){
          if (cancelled) break;
          updateArmRow(arm, { status:'running', statusClass:'status-running' });
          appendLog(`Generating for arm ${JSON.stringify(arm)}...`);
          const text = await generateOnce(apiKey, model, prompt, arm);
          outputs[JSON.stringify(arm)] = text;
          appendLog(`Judging arm ${JSON.stringify(arm)}...`);
          const judgeResults = await Promise.all(Array.from({length: j}).map((_,i)=> judgeOnce(apiKey, model, text, arm, i+1)));
          const agg = aggregateScores(judgeResults);
          recordResult(arm, text, agg);
          details[JSON.stringify(arm)] = agg; overalls[JSON.stringify(arm)] = agg.overall;
          const paramHtml = `<div class="arm-params"><span class="label">Params</span><pre>${escapeHtml(JSON.stringify(arm, null, 2))}</pre></div>`;
          const outputHtml = `<div class="arm-output-box"><pre>${escapeHtml(text)}</pre></div>`;
          const scoresHtml = `<div class="arm-scores">Scores: <code>${escapeHtml(JSON.stringify(agg))}</code></div>`;
          updateArmRow(arm, { status:'done', statusClass:'status-done', pulls:1, mean:agg.overall, best:agg.overall, detail: paramHtml + outputHtml + scoresHtml });
          if (typeof arm.temperature === 'number') addChartPoint(arm.temperature, agg.overall);
        }
        const ranked = Object.entries(overalls).sort((a,b)=>b[1]-a[1]);
        if (auto) {
          const [bestK, bestScore] = ranked[0];
          const arm = JSON.parse(bestK);
          results.textContent = `Best Arm ${bestK} | Overall ${bestScore}\n` + outputs[bestK] + "\n\n" + `Judges: ${JSON.stringify(details[bestK])}`;
        } else {
          results.textContent = ranked.map(([t, s])=>
            `Arm ${t} | Overall ${s} | Detail ${JSON.stringify(details[t])}\n${outputs[t]}`
          ).join('\n\n');
        }
      } else {
        // Transparent UCB loop with UI updates
        const keys = arms.map(a=>JSON.stringify(a));
        const pulls = Object.fromEntries(keys.map(k=>[k,0]));
        const sums = Object.fromEntries(keys.map(k=>[k,0]));
        const best = Object.fromEntries(keys.map(k=>[k,{overall:-1,text:'',detail:{}}]));
        let total = 0;
        for (const arm of arms){ updateArmRow(arm, { status:'running', statusClass:'status-running' }); }
        // init pull each arm
        for (const arm of arms){
          if (cancelled) break;
          appendLog(`Init pull -> ${JSON.stringify(arm)}`);
          const k = JSON.stringify(arm);
          const text = await generateOnce(apiKey, model, prompt, arm);
          const judgeResults = await Promise.all(Array.from({length: j}).map((_,i)=> judgeOnce(apiKey, model, text, arm, i+1)));
          const agg = aggregateScores(judgeResults);
          recordResult(arm, text, agg);
          pulls[k] += 1; sums[k] += agg.overall; total += 1;
          if (agg.overall > best[k].overall) best[k] = {overall: agg.overall, text, detail: agg};
          const paramHtml = `<div class="arm-params"><span class="label">Params</span><pre>${escapeHtml(JSON.stringify(arm, null, 2))}</pre></div>`;
          const outputHtml = `<div class="arm-output-box"><pre>${escapeHtml(text)}</pre></div>`;
          const scoresHtml = `<div class="arm-scores">Scores: <code>${escapeHtml(JSON.stringify(agg))}</code></div>`;
          updateArmRow(arm, { pulls:pulls[k], mean:(sums[k]/pulls[k]), best:best[k].overall, detail: paramHtml + outputHtml + scoresHtml });
          if (typeof arm.temperature === 'number') addChartPoint(arm.temperature, agg.overall);
        }
        for (let i=0;i<r-1;i++){
          if (cancelled) break;
          // compute UCB
          const ucb = {};
          for (const arm of arms){
            const k = JSON.stringify(arm);
            const m = pulls[k] ? (sums[k]/pulls[k]) : Infinity;
            const bonus = pulls[k] ? c * Math.sqrt(Math.log(Math.max(1,total)) / pulls[k]) : Infinity;
            ucb[k] = m + bonus;
          }
          const nextK = keys.slice().sort((a,b)=>ucb[b]-ucb[a])[0];
          const arm = JSON.parse(nextK);
          appendLog(`Round ${i+1}: selecting arm ${nextK} (UCB=${ucb[nextK].toFixed(3)})`);
          const text = await generateOnce(apiKey, model, prompt, arm);
          const judgeResults = await Promise.all(Array.from({length: j}).map((_,i)=> judgeOnce(apiKey, model, text, arm, i+1)));
          const agg = aggregateScores(judgeResults);
          recordResult(arm, text, agg);
          pulls[nextK] += 1; sums[nextK] += agg.overall; total += 1;
          if (agg.overall > best[nextK].overall) best[nextK] = {overall: agg.overall, text, detail: agg};
          const paramHtml = `<div class=\"arm-params\"><span class=\"label\">Params</span><pre>${escapeHtml(JSON.stringify(arm, null, 2))}</pre></div>`;
          const outputHtml = `<div class=\"arm-output-box\"><pre>${escapeHtml(text)}</pre></div>`;
          const scoresHtml = `<div class=\"arm-scores\">Scores: <code>${escapeHtml(JSON.stringify(agg))}</code></div>`;
          updateArmRow(arm, { pulls:pulls[nextK], mean:(sums[nextK]/pulls[nextK]), best:best[nextK].overall, detail: paramHtml + outputHtml + scoresHtml });
          if (typeof arm.temperature === 'number') addChartPoint(arm.temperature, agg.overall);
        }
        for (const arm of arms){ updateArmRow(arm, { status:'done', statusClass:'status-done' }); }
        const means = Object.fromEntries(keys.map(k=>[k, pulls[k] ? (sums[k]/pulls[k]) : 0]));
        const ranked = keys.slice().sort((a,b)=>means[b]-means[a]);
        const bestK = ranked[0];
        const bestArm = JSON.parse(bestK);
        appendLog(`Complete. Best ${bestK} mean=${means[bestK].toFixed(2)} best_overall=${best[bestK].overall.toFixed(2)}`);
        if (auto){
          results.textContent = `Advanced (UCB) — Best Arm ${bestK} | Mean ${means[bestK].toFixed(2)} | Best Overall ${best[bestK].overall.toFixed(2)}\n` + best[bestK].text + "\n\n" + `Detail: ${JSON.stringify(best[bestK].detail)}`;
        } else {
          const lines = [`Advanced (UCB) — Best ${bestK}`, best[bestK].text, '', `Detail: ${JSON.stringify(best[bestK].detail)}`, ''];
          for (const k of ranked){ lines.push(`Arm ${k}: pulls=${pulls[k]}, mean_overall=${means[k].toFixed(2)}, best_overall=${best[k].overall.toFixed(2)}`); }
          results.textContent = lines.join('\n');
        }
      }
      status.textContent = cancelled ? 'Stopped.' : 'Done.';
      enableRunButtons(false);
    } catch (e) {
      status.textContent = 'Error';
      results.textContent = String(e?.message || e);
      enableRunButtons(false);
    }
  });
});


