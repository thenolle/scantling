const SCRIPT = `
const $ = (id) => document.getElementById(id)
let controller = null
async function showModel() {
	const response = await fetch('/api/model')
	const model = await response.json()
	$('model').textContent = model.name + ' · ' + model.parameters.toLocaleString() + ' parameters · context ' + model.context + ' tokens' + (model.step > 0 ? ' · step ' + model.step : '')
}
async function run() {
	controller = new AbortController()
	$('go').disabled = true
	$('stop').disabled = false
	$('out').textContent = ''
	$('status').textContent = 'generating…'
	const body = { prompt: $('prompt').value, maxTokens: Number($('max').value), temperature: Number($('temp').value), topK: Number($('topk').value), topP: Number($('topp').value), repetitionPenalty: Number($('rep').value), raw: $('raw').checked }
	try {
		const response = await fetch('/api/complete', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal })
		if (!response.ok || !response.body) {
			$('status').textContent = (await response.text()) || 'error ' + response.status
			return
		}
		const reader = response.body.getReader()
		const decoder = new TextDecoder()
		let buffer = ''
		for (;;) {
			const { value, done } = await reader.read()
			if (done) break
			buffer += decoder.decode(value, { stream: true })
			let at
			while ((at = buffer.indexOf('\\n\\n')) !== -1) {
				const line = buffer.slice(0, at)
				buffer = buffer.slice(at + 2)
				if (!line.startsWith('data: ')) continue
				const item = JSON.parse(line.slice(6))
				if (item.text) $('out').textContent += item.text
				if (item.done) $('status').textContent = item.done + ' · ' + item.tokens + ' tokens'
				if (item.error) $('status').textContent = item.error
			}
		}
	} catch (error) {
		if (error.name !== 'AbortError') $('status').textContent = String(error)
		else $('status').textContent = 'stopped'
	} finally {
		$('go').disabled = false
		$('stop').disabled = true
	}
}
$('go').addEventListener('click', run)
$('stop').addEventListener('click', () => controller && controller.abort())
$('append').addEventListener('click', () => { $('prompt').value += $('out').textContent; $('out').textContent = '' })
showModel()
`

const STYLE = `
:root { color-scheme: light dark; --gap: 0.75rem; font-family: system-ui, sans-serif; }
body { margin: 0 auto; max-width: 62rem; padding: 1.5rem; }
h1 { font-size: 1.25rem; margin: 0 0 0.25rem; }
#model, #status { opacity: 0.7; font-size: 0.85rem; }
textarea, pre { width: 100%; box-sizing: border-box; font: 0.9rem/1.45 ui-monospace, Consolas, monospace; padding: 0.75rem; border: 1px solid #8884; border-radius: 0.5rem; background: transparent; }
textarea { min-height: 12rem; resize: vertical; }
pre { min-height: 12rem; white-space: pre-wrap; margin: 0; }
.row { display: flex; flex-wrap: wrap; gap: var(--gap); align-items: end; margin: var(--gap) 0; }
label { display: grid; gap: 0.2rem; font-size: 0.8rem; }
input[type=number] { width: 6rem; padding: 0.3rem; }
button { padding: 0.45rem 0.9rem; border-radius: 0.4rem; border: 1px solid #8886; background: #8882; cursor: pointer; }
`

/** The page of the completion playground. It sets text only through `textContent`, never as HTML. */
export function chatPage(): string {
	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scantling · completion playground</title>
<style>${STYLE}</style>
</head>
<body>
<h1>Completion playground</h1>
<div id="model">loading model…</div>
<p>This is a base model trained on code. Write the start of a file and it continues it; it does not follow instructions yet.</p>
<textarea id="prompt" spellcheck="false" placeholder="export function"></textarea>
<div class="row">
<label>Max tokens<input id="max" type="number" min="1" max="512" value="128"></label>
<label>Temperature<input id="temp" type="number" min="0" max="2" step="0.1" value="0.7"></label>
<label>Top-k<input id="topk" type="number" min="0" max="1000" value="40"></label>
<label>Top-p<input id="topp" type="number" min="0.01" max="1" step="0.05" value="0.95"></label>
<label>Repetition penalty<input id="rep" type="number" min="1" max="2" step="0.05" value="1.1"></label>
<label><span>Raw prompt</span><input id="raw" type="checkbox"></label>
<button id="go">Generate</button>
<button id="stop" disabled>Stop</button>
<button id="append">Append to prompt</button>
<span id="status"></span>
</div>
<pre id="out" aria-live="polite"></pre>
<script>${SCRIPT}</script>
</body>
</html>`
}
