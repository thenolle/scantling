const STYLE = `
:root{--background:0 0% 100%;--foreground:240 10% 3.9%;--card:0 0% 100%;--muted:240 4.8% 95.9%;--muted-foreground:240 3.8% 46.1%;--border:240 5.9% 90%;--chart-1:221 83% 53%;--chart-2:160 60% 40%;--chart-3:25 90% 55%;--chart-4:280 65% 55%;--ok:142 71% 35%;--warn:38 92% 50%;--bad:0 72% 51%;--radius:.5rem}
@media(prefers-color-scheme:dark){:root{--background:240 10% 3.9%;--foreground:0 0% 98%;--card:240 10% 6%;--muted:240 3.7% 15.9%;--muted-foreground:240 5% 64.9%;--border:240 3.7% 15.9%;--chart-1:217 91% 65%;--chart-2:160 60% 50%;--chart-3:30 90% 60%;--chart-4:280 70% 68%}}
*{box-sizing:border-box}
body{margin:0;background:hsl(var(--background));color:hsl(var(--foreground));font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:20px;font-weight:600;letter-spacing:-.02em;margin:0}
.sub{color:hsl(var(--muted-foreground));font-size:13px}
.grid{display:grid;gap:12px;margin-bottom:12px}
.tiles{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}
.charts{grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
.card{background:hsl(var(--card));border:1px solid hsl(var(--border));border-radius:var(--radius);padding:14px 16px;min-width:0}
.card h2{font-size:13px;font-weight:500;color:hsl(var(--muted-foreground));margin:0 0 6px}
.value{font-size:24px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.small{font-size:12px;color:hsl(var(--muted-foreground));font-variant-numeric:tabular-nums}
.badge{display:inline-flex;align-items:center;gap:6px;border:1px solid hsl(var(--border));border-radius:999px;padding:2px 10px;font-size:12px;font-weight:500}
.dot{width:8px;height:8px;border-radius:50%;background:hsl(var(--muted-foreground))}
.running .dot{background:hsl(var(--ok));box-shadow:0 0 0 3px hsl(var(--ok)/.2)}
.stalled .dot{background:hsl(var(--warn))}.finished .dot{background:hsl(var(--chart-1))}.stopped .dot{background:hsl(var(--bad))}
.bar{height:8px;border-radius:999px;background:hsl(var(--muted));overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:hsl(var(--foreground));transition:width .4s}
canvas{width:100%;height:220px;display:block}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}
.legend span{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:hsl(var(--muted-foreground))}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:12px;margin:0}
.kv dt{color:hsl(var(--muted-foreground))}.kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.log{font:12px/1.5 ui-monospace,Consolas,monospace;max-height:280px;overflow:auto;white-space:pre-wrap}
.log .warn{color:hsl(var(--warn))}.log .error{color:hsl(var(--bad))}.log time{color:hsl(var(--muted-foreground))}
[hidden]{display:none!important}
`

const SCRIPT = `
const $ = (id) => document.getElementById(id)
const state = { plan: null, status: 'idle', steps: [], evals: [], checkpoints: [], lines: [], summary: null, dataset: null, past: [], system: [], now: Date.now(), lastEntryAt: 0 }
const css = (name) => 'hsl(' + getComputedStyle(document.documentElement).getPropertyValue(name).trim() + ')'
const fmt = (n, d) => Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: d || 0, minimumFractionDigits: d || 0 }) : '–'
const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
const span = (minutes) => minutes >= 60 ? Math.floor(minutes / 60) + ' h ' + Math.round(minutes % 60) + ' min' : Math.round(minutes) + ' min'
const ema = (values, a) => { let m; return values.map((v) => (m = m === undefined ? v : m * (1 - a) + v * a)) }
const last = (list) => list[list.length - 1]

function chart(canvas, series, o) {
	const dpr = window.devicePixelRatio || 1
	const w = canvas.clientWidth, h = canvas.clientHeight
	if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr) }
	const c = canvas.getContext('2d')
	c.setTransform(dpr, 0, 0, dpr, 0, 0)
	c.clearRect(0, 0, w, h)
	c.font = '11px ui-sans-serif,system-ui'
	const all = series.flatMap((s) => s.points)
	if (all.length < 2) { c.fillStyle = css('--muted-foreground'); c.fillText('waiting for data…', 12, 24); return }
	const xs = all.map((p) => p[0])
	const ys = all.map((p) => p[1]).sort((a, b) => a - b)
	const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs)
	const y0 = o.min !== undefined ? o.min : ys[0]
	let y1 = o.max !== undefined ? o.max : o.clip ? ys[Math.floor((ys.length - 1) * o.clip)] : ys[ys.length - 1]
	if (y1 <= y0) y1 = y0 + 1
	const pad = { l: 48, r: 10, t: 8, b: 20 }
	const X = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r)
	const Y = (y) => pad.t + (1 - (Math.min(Math.max(y, y0), y1) - y0) / (y1 - y0)) * (h - pad.t - pad.b)
	c.lineWidth = 1
	for (let i = 0; i <= 4; i++) {
		const y = y0 + ((y1 - y0) * i) / 4
		c.strokeStyle = css('--border'); c.beginPath(); c.moveTo(pad.l, Y(y)); c.lineTo(w - pad.r, Y(y)); c.stroke()
		c.fillStyle = css('--muted-foreground'); c.textAlign = 'left'; c.fillText((o.fmt || fmt)(y), 4, Y(y) + 3)
	}
	c.fillStyle = css('--muted-foreground'); c.textAlign = 'left'; c.fillText(fmt(x0), pad.l, h - 5); c.textAlign = 'right'; c.fillText(fmt(x1), w - pad.r, h - 5)
	for (const s of series) {
		c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = s.width || 1.5; c.setLineDash(s.dash || []); c.globalAlpha = s.alpha || 1
		if (s.dots) { for (const p of s.points) { c.beginPath(); c.arc(X(p[0]), Y(p[1]), 3, 0, 6.3); c.fill() } }
		else { c.beginPath(); s.points.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1])))); c.stroke() }
	}
	c.setLineDash([]); c.globalAlpha = 1
}

function legend(id, items) {
	const box = $(id); box.replaceChildren()
	for (const item of items) { const s = document.createElement('span'); const i = document.createElement('i'); i.style.background = item[1]; s.append(i, item[0]); box.append(s) }
}

function tile(title, value, sub, bar) {
	const card = document.createElement('div'); card.className = 'card'
	const h = document.createElement('h2'); h.textContent = title
	const v = document.createElement('div'); v.className = 'value'; v.textContent = value
	const s = document.createElement('div'); s.className = 'small'; s.textContent = sub || ''
	card.append(h, v, s)
	if (bar !== undefined) { const b = document.createElement('div'); b.className = 'bar'; const f = document.createElement('div'); f.style.width = Math.min(100, bar) + '%'; b.append(f); card.append(b) }
	return card
}

function kv(id, rows) {
	const dl = $(id); dl.replaceChildren()
	for (const row of rows) { const dt = document.createElement('dt'); dt.textContent = row[0]; const dd = document.createElement('dd'); dd.textContent = row[1]; dl.append(dt, dd) }
}

function brief(text) {
	try { const o = JSON.parse(text); return Object.keys(o).map((k) => k + '=' + o[k]).join(' ') } catch (e) { return text }
}

function draw() {
	const steps = state.steps, tip = last(steps), plan = state.plan, f = plan ? plan.fields : {}
	const stalled = state.status === 'running' && state.now - state.lastEntryAt > 90000
	const label = stalled ? 'stalled' : state.status
	$('status').className = 'badge ' + label
	$('statusText').textContent = label
	$('subtitle').textContent = plan ? (f.name ? f.name + ' · ' : '') + (f.preset ? f.preset + ' · ' : '') + 'started ' + new Date(plan.startedAt).toLocaleString() : 'No training run in the log yet'
	const total = tip ? tip.total : f.steps || 0
	const percent = total && tip ? (tip.step / total) * 100 : 0
	const ev = last(state.evals), sys = last(state.system), gpu = sys && sys.gpu
	const eta = tip ? tip.etaMinutes : 0
	const finishes = tip && state.status === 'running' ? clock(Date.parse(tip.time) + eta * 60000) : ''
	const tiles = $('tiles'); tiles.replaceChildren()
	tiles.append(
		tile('Progress', tip ? fmt(tip.step) + ' / ' + fmt(total) : '–', fmt(percent, 1) + '%' + (f.epochs ? ' · ' + fmt((f.epochs * percent) / 100, 2) + ' of ' + f.epochs + ' epochs' : ''), percent),
		tile('Throughput', tip ? fmt(tip.tokensPerSecond) + ' tok/s' : '–', tip && tip.gpuMs !== undefined ? 'step ' + fmt(tip.gpuMs + tip.optimizerMs + tip.transferMs) + ' ms' : ''),
		tile('Time left', state.status === 'running' && tip ? span(eta) : state.status === 'finished' ? 'done' : '–', finishes ? 'finishes about ' + finishes : ''),
		tile('Tokens seen', tip && f.tokensPerStep ? fmt((tip.step * f.tokensPerStep) / 1e6, 1) + ' M' : '–', f.trainTokens ? 'of ' + fmt(f.trainTokens / 1e6, 1) + ' M per epoch' : ''),
		tile('Train loss', tip ? fmt(last(ema(steps.slice(-30).map((p) => p.loss), 0.2)), 3) : '–', tip ? 'grad norm ' + fmt(tip.gradNorm, 2) + ' · lr ' + tip.lr.toExponential(2) : ''),
		tile('Validation', ev ? fmt(ev.loss, 4) : '–', ev ? 'ppl ' + fmt(ev.perplexity, 2) + ' · ' + fmt(ev.bitsPerByte, 3) + ' bpb · step ' + fmt(ev.step) : ''),
		tile('GPU', gpu ? fmt(gpu.utilization) + '%' : '–', gpu ? fmt(gpu.memoryUsedMiB) + ' / ' + fmt(gpu.memoryTotalMiB) + ' MiB · ' + gpu.temperatureC + ' °C' + (gpu.powerW !== undefined ? ' · ' + fmt(gpu.powerW) + ' W' : '') : 'waiting for nvidia-smi'),
		tile('CPU · RAM', sys ? fmt(sys.cpuPercent) + '%' : '–', sys ? fmt(sys.memoryUsedBytes / 1073741824, 1) + ' / ' + fmt(sys.memoryTotalBytes / 1073741824, 1) + ' GiB' + (sys.memoryUsedBytes / sys.memoryTotalBytes > 0.9 ? ' · low memory' : '') : '')
	)
	const loss = steps.map((p) => [p.step, p.loss])
	const lossSeries = [
		{ points: loss, color: css('--chart-1'), alpha: 0.25 },
		{ points: ema(loss.map((p) => p[1]), 0.1).map((v, i) => [loss[i][0], v]), color: css('--chart-1'), width: 2 },
		{ points: state.evals.map((e) => [e.step, e.loss]), color: css('--chart-3'), dots: true }
	]
	if (f.unigramBaseline && tip) lossSeries.push({ points: [[steps[0].step, f.unigramBaseline], [tip.step, f.unigramBaseline]], color: css('--muted-foreground'), dash: [4, 4], width: 1 })
	chart($('lossChart'), lossSeries, { clip: 0.97, fmt: (v) => fmt(v, 2) })
	legend('lossLegend', [['train loss (smoothed)', css('--chart-1')], ['validation', css('--chart-3')], ['unigram baseline', css('--muted-foreground')]])
	chart($('speedChart'), [{ points: steps.map((p) => [p.step, p.tokensPerSecond]), color: css('--chart-2'), width: 2 }], { min: 0 })
	const timed = steps.filter((p) => p.gpuMs !== undefined)
	chart($('phaseChart'), [
		{ points: timed.map((p) => [p.step, p.gpuMs]), color: css('--chart-1') },
		{ points: timed.map((p) => [p.step, p.optimizerMs]), color: css('--chart-3') },
		{ points: timed.map((p) => [p.step, p.transferMs]), color: css('--chart-4') }
	], { min: 0 })
	legend('phaseLegend', [['GPU compute (ms)', css('--chart-1')], ['optimizer (ms)', css('--chart-3')], ['transfers (ms)', css('--chart-4')]])
	chart($('lrChart'), [{ points: steps.map((p) => [p.step, p.lr]), color: css('--chart-4'), width: 2 }], { min: 0, fmt: (v) => v.toExponential(1) })
	chart($('sysChart'), [
		{ points: state.system.map((s, i) => [i, s.gpu ? s.gpu.utilization : 0]), color: css('--chart-2'), width: 2 },
		{ points: state.system.map((s, i) => [i, s.cpuPercent]), color: css('--chart-3'), width: 2 }
	], { min: 0, max: 100 })
	legend('sysLegend', [['GPU %', css('--chart-2')], ['CPU %', css('--chart-3')]])
	const rows = []
	for (const key of ['name', 'dataset', 'parameters', 'trainTokens', 'tokensPerStep', 'epochs', 'microBatch', 'accumulation', 'gpuEstimate', 'unigramBaseline', 'deviceNote']) if (f[key] !== undefined) rows.push([key, typeof f[key] === 'number' ? fmt(f[key], key === 'epochs' || key === 'unigramBaseline' ? 2 : 0) : String(f[key])])
	kv('planList', rows)
	kv('checkpointList', state.checkpoints.slice(-8).reverse().map((c) => [c.id, fmt(c.evalLoss, 4) + (c.promoted ? ' ✓' : '')]))
	const d = state.dataset
	$('datasetCard').hidden = !d
	if (d) kv('datasetList', [['phase', d.phase + (d.finished ? ' (finished)' : '')], ['documents', fmt(d.done) + (d.total ? ' / ' + fmt(d.total) : '')], ['from cache', fmt(d.cached)], ['updated', new Date(d.updatedAt).toLocaleTimeString()]])
	const log = $('log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8
	log.replaceChildren()
	for (const line of state.lines.slice(-80)) {
		const row = document.createElement('div'); row.className = line.level
		const t = document.createElement('time'); t.textContent = new Date(line.time).toLocaleTimeString() + ' '
		row.append(t, line.message + (line.text ? ' ' + brief(line.text) : '')); log.append(row)
	}
	if (atEnd) log.scrollTop = log.scrollHeight
}

let queued = false
const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw() }) } }

function apply(d) {
	if (d.type === 'plan') { state.plan = d.plan; state.steps = []; state.evals = []; state.checkpoints = []; state.status = 'running'; state.summary = null }
	else if (d.type === 'step') { state.steps.push(d.point); state.lastEntryAt = Date.parse(d.point.time) }
	else if (d.type === 'eval') state.evals.push(d.point)
	else if (d.type === 'checkpoint') state.checkpoints.push(d.point)
	else if (d.type === 'status') { state.status = d.status; state.summary = d.summary }
	else if (d.type === 'dataset') state.dataset = d.dataset
	else if (d.type === 'log') { state.lines.push(d.line); if (state.lines.length > 200) state.lines.shift(); state.lastEntryAt = Date.parse(d.line.time) }
}

const source = new EventSource('/events')
source.onopen = () => { $('connection').textContent = 'live' }
source.onerror = () => { $('connection').textContent = 'reconnecting…' }
source.addEventListener('snapshot', (e) => { Object.assign(state, JSON.parse(e.data)); schedule() })
source.addEventListener('delta', (e) => { apply(JSON.parse(e.data)); schedule() })
source.addEventListener('system', (e) => { state.system.push(JSON.parse(e.data)); if (state.system.length > 150) state.system.shift(); schedule() })
source.addEventListener('tick', (e) => { const t = JSON.parse(e.data); state.now = t.now; state.lastEntryAt = t.lastEntryAt || state.lastEntryAt; schedule() })
window.addEventListener('resize', schedule)
schedule()
`

const chartCard = (title: string, canvas: string, legend: string): string => `<div class="card"><h2>${title}</h2><canvas id="${canvas}"></canvas><div class="legend" id="${legend}"></div></div>`

/** The monitor page. Everything is drawn from event data with `textContent` and canvas, never as HTML. */
export function monitorPage(): string {
	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scantling · live monitor</title>
<style>${STYLE}</style>
</head>
<body>
<main>
<header>
<div><h1>Live monitor</h1><div class="sub" id="subtitle"></div></div>
<div><span class="small" id="connection">connecting…</span> <span id="status" class="badge"><span class="dot"></span><span id="statusText">idle</span></span></div>
</header>
<div class="grid tiles" id="tiles"></div>
<div class="grid charts">
${chartCard('Loss', 'lossChart', 'lossLegend')}
${chartCard('Throughput (tokens/s)', 'speedChart', 'speedLegend')}
${chartCard('Where each step goes', 'phaseChart', 'phaseLegend')}
${chartCard('Learning rate', 'lrChart', 'lrLegend')}
${chartCard('Machine load', 'sysChart', 'sysLegend')}
<div class="card"><h2>Run</h2><dl class="kv" id="planList"></dl></div>
<div class="card"><h2>Checkpoints</h2><dl class="kv" id="checkpointList"></dl></div>
<div class="card" id="datasetCard" hidden><h2>Dataset build</h2><dl class="kv" id="datasetList"></dl></div>
</div>
<div class="card"><h2>Log</h2><div class="log" id="log"></div></div>
</main>
<script>${SCRIPT}</script>
</body>
</html>`
}
