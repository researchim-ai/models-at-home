// Smoke an unpacked Linux desktop app on DISPLAY with its real preload and Python backend.
// Usage: node scripts/smoke-desktop.cjs /path/to/models-at-home-studio
const { spawn } = require('child_process')
const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = require('fs')
const { tmpdir } = require('os')
const path = require('path')
const net = require('net')
const assert = require('assert/strict')

const delay = ms => new Promise(resolve => setTimeout(resolve, ms))

async function main() {
  assert(process.argv[2], 'Provide the unpacked application executable')
  const root = mkdtempSync(path.join(tmpdir(), 'mah-desktop-smoke-'))
  const config = path.join(root, 'config')
  for (const name of ['Models at Home Studio', 'models-at-home-studio']) {
    const folder = path.join(config, name)
    mkdirSync(folder, { recursive: true })
    writeFileSync(path.join(folder, 'config.json'), JSON.stringify({ backendMode: 'native' }))
  }
  const server = net.createServer()
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  await new Promise(resolve => server.close(resolve))
  const child = spawn(path.resolve(process.argv[2]), [
    '--no-sandbox', '--disable-gpu', `--remote-debugging-port=${port}`,
  ], {
    detached: true,
    env: { ...process.env, XDG_CONFIG_HOME: config, XDG_CACHE_HOME: path.join(root, 'cache'),
      MODELS_AT_HOME_ROOT: path.join(root, 'workspace') },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let logs = ''
  child.stdout.on('data', chunk => { logs += chunk.toString() })
  child.stderr.on('data', chunk => { logs += chunk.toString() })
  let socket
  const pending = new Map()
  let sequence = 0
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)) }, 10000)
    pending.set(id, value => { clearTimeout(timer); value.error ? reject(new Error(value.error.message)) : resolve(value.result) })
    socket.send(JSON.stringify({ id, method, params }))
  })
  const evaluate = async expression => {
    const result = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text)
    return result.result.value
  }
  try {
    let target
    let lastTargets = []
    for (let attempt = 0; attempt < 120; attempt++) {
      if (child.exitCode !== null) throw new Error(`Desktop exited: ${logs.slice(-3000)}`)
      try {
        const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then(r => r.json())
        lastTargets = targets.map(t => ({ type: t.type, url: t.url }))
        target = targets.find(t => t.type === 'page' && !t.url.startsWith('devtools:'))
        if (target) break
      } catch { /* Chromium is starting. */ }
      await delay(250)
    }
    assert(target, `No renderer target (exit=${child.exitCode}, signal=${child.signalCode}, targets=${JSON.stringify(lastTargets)}): ${logs.slice(-3000)}`)
    socket = new WebSocket(target.webSocketDebuggerUrl)
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject })
    socket.onmessage = event => {
      const value = JSON.parse(event.data)
      if (value.id && pending.has(value.id)) { pending.get(value.id)(value); pending.delete(value.id) }
    }
    const waitFor = async text => {
      for (let attempt = 0; attempt < 160; attempt++) {
        if (await evaluate(`Boolean(document.body?.innerText.toLowerCase().includes(${JSON.stringify(text.toLowerCase())}))`)) return
        await delay(250)
      }
      throw new Error(`Renderer missing ${text}: ${await evaluate('document.body.innerText')}`)
    }
    const screenshot = async name => {
      if (!process.env.MAH_DESKTOP_SCREENSHOT_DIR) return
      mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR, { recursive: true })
      await delay(350)
      const shot = await command('Page.captureScreenshot', { format: 'png' })
      writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR, `${name}.png`), Buffer.from(shot.data, 'base64'))
    }
    await waitFor('Readiness')
    assert.equal(await evaluate('window.electronAPI.backend.status().then(s => s.running)'), true)
    for (const [route, text] of [
      ['/', 'Readiness'], ['/vlm', 'VLM Stage'], ['/vla', 'VLA Configuration'], ['/world-models', 'World Models Studio'], ['/builder', 'Saved Blueprints'],
      ['/study', 'Select a document from the sidebar'], ['/notebooks', 'JupyterLab Connection'],
      ['/agent', 'llama-server'], ['/settings', 'Configure your studio preferences'],
    ]) {
      await evaluate(`location.hash = ${JSON.stringify(route)}`)
      await waitFor(text)
      if (route === '/builder') await waitFor('token_embedding')
      await screenshot(route === '/' ? 'training-studio' : route.slice(1))
      // Both themes and the minimum supported desktop width must keep the shell usable.
      await evaluate(`document.querySelector('button[aria-label="Switch to light theme"]').click()`)
      assert.equal(await evaluate(`document.documentElement.classList.contains('dark')`), false)
      assert.equal(await evaluate(`getComputedStyle(document.body).backgroundColor`), 'rgb(255, 255, 255)', 'Streamlit light canvas must be white')
      assert.equal(await evaluate(`getComputedStyle(document.querySelector('.studio-sidebar')).backgroundColor`), 'rgb(240, 242, 246)', 'Streamlit sidebar must be grey')
      if (route === '/') await screenshot('training-studio-light')
      await command('Emulation.setDeviceMetricsOverride', { width: 1024, height: 768, deviceScaleFactor: 1, mobile: false })
      assert.equal(await evaluate(`document.documentElement.scrollWidth <= innerWidth`), true, `Shell overflow: ${route}`)
      const overflows = await evaluate(`Array.from(document.querySelectorAll('.studio-content, .studio-content>div, .studio-content>div>main, .studio-content>div>div.flex-1')).filter(el => el.scrollWidth > el.clientWidth + 1).map(el => ({className: el.className, width: el.clientWidth, scroll: el.scrollWidth}))`)
      assert.deepEqual(overflows, [], `Content overflow: ${route}`)
      await screenshot(`${route === '/' ? 'training-studio' : route.slice(1)}-1024`)
      await command('Emulation.clearDeviceMetricsOverride')
      await evaluate(`document.querySelector('button[aria-label="Switch to dark theme"]').click()`)
      assert.equal(await evaluate(`document.documentElement.classList.contains('dark')`), true)
      if (route === '/vla') {
        for (const [kind,text] of [['openvla','Action-token cross entropy'],['smolvla','Train action expert only'],['pi0','LeRobot/OpenPI'],['act','ACT transformer width']]) {
          await evaluate(`(() => {const s=document.querySelector('[aria-label="VLA architecture"]');s.value=${JSON.stringify(kind)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`)
          await waitFor(text)
          assert.equal(await evaluate(`document.querySelector('[aria-label="VLA action type"]').disabled`),true)
          assert.equal(await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Начать VLA обучение')).disabled`),true)
        }
        await screenshot('vla-standalone-act')
        if (process.env.MAH_SMOKE_ACT_TRAIN==='1') {
          for(let attempt=0;attempt<100&&await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Синтетическое демо')).disabled`);attempt++)await delay(100)
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Синтетическое демо')).click()`)
          for(const [id,value] of [['vla-max_steps','2'],['vla-save_every','1'],['vla-eval_every','1'],['vla-batch_size','1'],['vla-image_size','32'],['vla-act_dim_model','32'],['vla-act_encoder_layers','1']]) {
            await evaluate(`(() => {const f=document.getElementById(${JSON.stringify(id)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(f,${JSON.stringify(value)});f.dispatchEvent(new Event('input',{bubbles:true}));})()`)
          }
          await evaluate(`(() => {const s=document.querySelector('[aria-label="VLA device"]');s.value='cpu';s.dispatchEvent(new Event('change',{bubbles:true}));})()`)
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Проверить VLA')).click()`)
          await waitFor('VLA проверка пройдена')
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Начать VLA обучение')).click()`)
          await waitFor('Policy export:')
          await waitFor('Action MAE')
          await evaluate(`[...document.querySelectorAll('p')].find(p=>p.textContent.startsWith('Policy export:')).scrollIntoView({block:'center'})`)
          await screenshot('vla-act-training')
          console.log('PASS packaged actual ACT demo preflight, own-objective training, validation, metrics and export')
        }
        await evaluate(`(() => {const s=document.querySelector('[aria-label="VLA architecture"]');s.value='native_vla';s.dispatchEvent(new Event('change',{bubbles:true}));})()`)
        console.log('PASS packaged OpenVLA, SmolVLA, ACT and pi0 selectors, contracts and training gate')
      }
      if (route === '/vlm') {
        assert.equal(await evaluate(`[...document.querySelectorAll('button')].find(button => button.textContent.includes('Start Training'))?.disabled`), true)
        await waitFor('Проверить эксперимент')
        assert.equal(await evaluate(`Boolean(document.querySelector('option[value="Qwen/Qwen3.5-0.8B"]'))`), true)
        assert.equal(await evaluate(`Boolean(document.querySelector('option[value="Qwen/Qwen3-VL-2B-Instruct"]'))`), true)
        assert.equal(await evaluate(`document.querySelector('option[value="vlm_grpo"]')?.disabled`), false)
        assert.equal(await evaluate(`Boolean(document.querySelector('select[aria-label="VLM backend"] option[value="unsloth"]'))`), true)
        await evaluate(`(() => { const select = document.querySelector('option[value="vlm_grpo"]').parentElement;
          select.value = 'vlm_grpo'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
        await waitFor('Policy objective')
        assert.equal(await evaluate(`document.querySelector('select[aria-label="VLM GRPO algorithm"] option[value="sdpo"]')?.disabled`), false)
        assert.equal(await evaluate(`Boolean(document.querySelector('select[aria-label="VLM GRPO algorithm"] option[value="gspo"]'))`), false)
        console.log('PASS packaged modern VLM presets, native GRPO/SDPO, Unsloth and preflight gate')
      }
      if (route === '/vla') {
        assert.equal(await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Начать VLA обучение'))?.disabled`), true)
        assert.equal(await evaluate(`Boolean(document.querySelector('select[aria-label="VLA domain"] option[value="desktop"]'))`), true)
        if (process.env.MAH_SMOKE_VLA_TRAIN === '1') {
          await evaluate(`(() => { const select = document.querySelector('select[aria-label="VLA domain"]'); select.value = 'desktop'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
          for (let attempt = 0; attempt < 80; attempt++) {
            if (await evaluate(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('Синтетическое демо') && !b.disabled)`)) break
            await delay(250)
          }
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Синтетическое демо')).click()`)
          for (const [id, value] of [['vla-max_steps', '2'], ['vla-save_every', '1'], ['vla-eval_every', '1'], ['vla-batch_size', '2']]) {
            await evaluate(`(() => { const field = document.getElementById(${JSON.stringify(id)}); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, ${JSON.stringify(value)}); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
          }
          await evaluate(`(() => { const select = document.querySelector('select[aria-label="VLA device"]'); select.value = 'cpu'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Проверить VLA')).click()`)
          await waitFor('VLA проверка пройдена')
          assert.equal(await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Начать VLA обучение')).disabled`), false)
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Начать VLA обучение')).click()`)
          await waitFor('Policy export:')
          await waitFor('Discrete accuracy:')
          assert.equal(await evaluate(`getComputedStyle(document.querySelector('.recharts-line path')).stroke`), 'rgb(233, 69, 96)', 'VLA loss must use the red training curve')
          assert.equal(await evaluate(`document.body.innerText.includes('Discrete accuracy:')`), true, 'VLA hybrid validation accuracy missing')
          if (process.env.MAH_DESKTOP_SCREENSHOT_DIR) {
            mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR, { recursive: true })
            const shot = await command('Page.captureScreenshot', { format: 'png' })
            writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR, 'vla-training.png'), Buffer.from(shot.data, 'base64'))
          }
          await evaluate(`[...document.querySelectorAll('[role="tab"]')].find(b => b.textContent === 'Action preview').click()`)
          const document = await command('DOM.getDocument')
          const input = await command('DOM.querySelector', { nodeId: document.root.nodeId, selector: '#vla-image' })
          await command('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [path.resolve(path.dirname(process.argv[2]), 'resources/examples/vla/desktop/frames/0.png')] })
          for (let attempt = 0; attempt < 80; attempt++) {
            if (await evaluate(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('Предсказать действия') && !b.disabled)`)) break
            await delay(250)
          }
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Предсказать действия')).click()`)
          await waitFor('"continuous"')
          await waitFor('"discrete"')
          console.log('PASS packaged VLA synthetic desktop preflight, actual train, metrics/chart, export and action prediction')
        }
      }
      if (route === '/world-models') {
        await waitFor('DIAMOND · CS:GO')
        assert.equal(await evaluate(`Boolean(document.querySelector('option[value="genie4"]'))`), false)
        assert.equal(await evaluate(`Boolean(document.querySelector('option[value="diamond_atari"]'))`), true)
        console.log('PASS packaged embedded DIAMOND/local Cosmos catalog')
        if (process.env.MAH_SMOKE_LOCAL_WORLD === '1') {
          const weights = process.env.MAH_SMOKE_DIAMOND_PATH
          assert(weights, 'MAH_SMOKE_DIAMOND_PATH is required')
          for (const [id,value] of [['world-play-path',weights],['world-play-denoising','1']]) {
            await evaluate(`(() => { const field=document.getElementById(${JSON.stringify(id)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(field,${JSON.stringify(value)});field.dispatchEvent(new Event('input',{bubbles:true})); })()`)
          }
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Загрузить модель').click()`)
          await waitFor('Шаг 0')
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Один шаг').click()`)
          await waitFor('Шаг 1')
          assert.equal(await evaluate(`Boolean(document.querySelector('img[alt="Local world frame"]'))`),true)
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Играть').click();document.querySelector('[aria-label="Local world viewport"]').focus()`)
          await command('Input.dispatchKeyEvent',{type:'keyDown',key:'w',code:'KeyW',windowsVirtualKeyCode:87})
          for (let i=0;i<80;i++) {
            if (await evaluate(`Number(document.body.innerText.match(/Шаг (\\d+) ·/)?.[1]??0)>=4`)) break
            await delay(100)
          }
          await command('Input.dispatchKeyEvent',{type:'keyUp',key:'w',code:'KeyW',windowsVirtualKeyCode:87})
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Пауза').click()`)
          assert.equal(await evaluate(`Number(document.body.innerText.match(/Шаг (\\d+) ·/)?.[1]??0)>=4`),true,'Autoplay did not advance model frames')

          if (process.env.MAH_DESKTOP_SCREENSHOT_DIR) {
            mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR,{recursive:true})
            const shot=await command('Page.captureScreenshot',{format:'png'})
            writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR,'world-local-play.png'),Buffer.from(shot.data,'base64'))
          }
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Сбросить сцену').click()`)
          await waitFor('Шаг 0')
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Освободить модель').click()`)
          for(let i=0;i<80;i++){if(await evaluate(`[...document.querySelectorAll('button')].some(b=>b.textContent==='Загрузить модель'&&!b.disabled)`))break;await delay(100)}
          await evaluate(`(() => {const select=document.querySelector('select[aria-label="Foundation model"]');select.value='diamond_atari';select.dispatchEvent(new Event('change',{bubbles:true}));})()`)
          await waitFor('Screenshot Atari')
          await evaluate(`(() => {const field=document.getElementById('world-play-path');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(field,${JSON.stringify(weights)});field.dispatchEvent(new Event('input',{bubbles:true}));const device=document.querySelector('select[aria-label="World play device"]');device.value='cpu';device.dispatchEvent(new Event('change',{bubbles:true}));})()`)
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Загрузить модель').click()`)
          await waitFor('Шаг 0')
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Один шаг').click()`)
          await waitFor('Шаг 1')
          assert.equal(await evaluate(`/Шаг 1 · cpu/.test(document.body.innerText)`),true)
          await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Освободить модель').click()`)
          console.log('PASS real embedded DIAMOND GPU controls/autoplay/reset/release and CPU Breakout packaged example in Electron')
        }
        if (process.env.MAH_SMOKE_FOUNDATION_PREFLIGHT === '1') {
          await evaluate(`(() => { const select=document.querySelector('select[aria-label="Foundation model"]');select.value='cosmos3_edge';select.dispatchEvent(new Event('change',{bubbles:true})); })()`)
          await waitFor('Cosmos · локальная генерация')
          await evaluate(`(() => { const field = document.querySelector('textarea[aria-label="World prompt"]'); Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(field, 'A quiet mountain lake'); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Проверить Cosmos')).click()`)
          await waitFor('Cosmos preflight:')
          assert.equal(await evaluate(`/ModuleNotFoundError|No module named|Traceback/.test(document.body.innerText)`), false, 'Foundation worker dependency/interpreter failure')
          assert.equal(await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Остановить').disabled`), true)
          if (process.env.MAH_DESKTOP_SCREENSHOT_DIR) {
            mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR, { recursive: true })
            const shot = await command('Page.captureScreenshot', { format: 'png' })
            writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR, 'world-foundation.png'), Buffer.from(shot.data, 'base64'))
          }
          console.log('PASS packaged real foundation preflight without starting generation')
        }
      }
      if (route === '/world-models' && process.env.MAH_SMOKE_WORLD_TRAIN === '1') {
        await evaluate(`[...document.querySelectorAll('[role="tab"]')].find(b => b.textContent === 'Обучение').click()`)
        for (let attempt = 0; attempt < 80; attempt++) {
          if (await evaluate(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('Синтетический пример') && !b.disabled)`)) break
          await delay(250)
        }
        await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Синтетический пример')).click()`)
        for (const [id, value] of [['world-max_steps', '2'], ['world-save_every', '1'], ['world-eval_every', '1'], ['world-batch_size', '2']]) {
          await evaluate(`(() => { const field = document.getElementById(${JSON.stringify(id)}); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, ${JSON.stringify(value)}); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
        }
        await evaluate(`(() => { const select = document.querySelector('select[aria-label="World device"]'); select.value = 'cpu'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
        await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Проверить данные')).click()`)
        await waitFor('Preflight: готово')
        if (process.env.MAH_SMOKE_WORLD_RESOURCES === '1') {
          await waitFor('Измерить 5 шагов')
          await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Измерить 5 шагов').click()`)
          await waitFor('Примерное время на 2 шагов:')
          await waitFor('Измерено на этой машине + запас')
          assert(await evaluate(`document.body.innerText.includes('RAM свободно:')`))
          if (process.env.MAH_DESKTOP_SCREENSHOT_DIR) {
            mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR, { recursive: true })
            await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Измерить 5 шагов').scrollIntoView({block:'center'})`)
            const shot = await command('Page.captureScreenshot', { format: 'png' })
            writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR, 'world-resources.png'), Buffer.from(shot.data, 'base64'))
          }
          console.log('PASS packaged hardware hints, actual training benchmark, measured memory and ETA')
        }
        await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Обучать').click()`)
        await waitFor('Экспорт:')
        if (process.env.MAH_DESKTOP_SCREENSHOT_DIR) {
          mkdirSync(process.env.MAH_DESKTOP_SCREENSHOT_DIR, { recursive: true })
          const shot = await command('Page.captureScreenshot', { format: 'png' })
          writeFileSync(path.join(process.env.MAH_DESKTOP_SCREENSHOT_DIR, 'world-training.png'), Buffer.from(shot.data, 'base64'))
        }
        await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Открыть Rollout')).click()`)
        const document = await command('DOM.getDocument')
        const input = await command('DOM.querySelector', { nodeId: document.root.nodeId, selector: 'input[type="file"]' })
        await command('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [path.resolve(path.dirname(process.argv[2]), 'resources/examples/world-models/frames/validation-0-0.png')] })
        for (let attempt = 0; attempt < 80; attempt++) {
          if (await evaluate(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('Предсказать будущее') && !b.disabled)`)) break
          await delay(250)
        }
        await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Предсказать будущее')).click()`)
        await waitFor('t+4')
        assert.equal(await evaluate(`document.querySelectorAll('figure img').length`), 4)
        console.log('PASS packaged World Models demo preflight, actual train, metrics, export and free-running rollout')
      }
      console.log(`PASS file renderer ${route}`)
    }
    await evaluate('location.hash = "/builder"')
    await waitFor('token_embedding')
    assert.equal(await evaluate(`(() => {
      const button = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'token_embedding');
      button?.click(); return Boolean(button);
    })()`), true)
    await waitFor('1 blocks')
    assert.equal(await evaluate(`(() => {
      const button = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Train this model'));
      const ready = Boolean(button && !button.disabled); button?.click(); return ready;
    })()`), true)
    await waitFor('Readiness')
    assert.equal(await evaluate(`[...document.querySelectorAll('select')].some(select => select.value === 'blueprint')`), true)
    console.log('PASS packaged Builder save and handoff to Training Studio')
    if (process.env.MAH_SMOKE_BUILDER === '1') {
      await evaluate('location.hash = "/builder"')
      await waitFor('Start from a recipe')
      await waitFor('token_embedding')
      for (const [id, value] of [['builder-hidden', '32'], ['builder-vocab', '100'], ['builder-positions', '64']]) {
        await evaluate(`(() => { const field = document.getElementById(${JSON.stringify(id)}); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, ${JSON.stringify(value)}); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
      }
      for(let attempt=0;attempt<100&&await evaluate(`document.getElementById('builder-recipe').disabled`);attempt++)await delay(100)
      await evaluate(`(() => { const select = document.getElementById('builder-recipe'); select.value = 'looped'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
      await waitFor('5 blocks')
      await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Analyze architecture').click()`)
      await waitFor('Architecture validated')
      const parameters = await evaluate(`document.body.innerText.match(/([0-9,]+) unique parameters/)[1]`)
      await evaluate(`document.querySelector('[aria-label="loop block 3"]').click()`)
      await waitFor('Block inspector')
      for (const suffix of ['iterations', 'eval_iterations']) {
        await evaluate(`(() => { const field = document.querySelector('.builder-saved input[id$="-${suffix}"]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, '6'); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
      }
      await evaluate(`(() => { const section = [...document.querySelectorAll('.builder-saved details')].find(d => d.querySelector('summary')?.textContent.includes('Depth sampling')); section.open = true; const label = [...section.querySelectorAll('label')].find(l => l.textContent === 'checkpoint'); label.querySelector('input').click(); const field = section.querySelector('input[id$="-gradient_steps"]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, '2'); field.dispatchEvent(new Event('input', { bubbles: true })); })()`)
      await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Analyze architecture').click()`)
      await waitFor('shared body × 6')
      await waitFor('Architecture validated')
      assert.equal(await evaluate(`document.body.innerText.match(/([0-9,]+) unique parameters/)[1]`), parameters, 'Loop depth must not multiply physical parameters')
      await evaluate(`(() => { const select = document.querySelector('.builder-saved select[aria-label="Add body block"]'); select.value = 'dropout'; select.dispatchEvent(new Event('change', { bubbles: true })); })()`)
      await waitFor('Container body · 3')
      await evaluate(`document.querySelector('button[aria-label="Undo architecture edit"]').click()`)
      await waitFor('Container body · 2')
      await evaluate(`document.querySelector('[aria-label="llama_block block 4"] button[aria-label="Duplicate block"]').click()`)
      await waitFor('6 blocks')
      await evaluate(`document.querySelector('button[aria-label="Undo architecture edit"]').click()`)
      await waitFor('5 blocks')
      await evaluate(`document.querySelector('button[aria-label="Switch to light theme"]').click()`)
      await evaluate(`document.querySelectorAll('[data-sonner-toast] button[data-close-button]').forEach(button => button.click())`)
      await delay(4500)
      await screenshot('model-builder-looped')
      await command('Emulation.setDeviceMetricsOverride', { width: 1024, height: 768, deviceScaleFactor: 1, mobile: false })
      assert.equal(await evaluate(`document.querySelector('.studio-content').scrollWidth <= document.querySelector('.studio-content').clientWidth`), true)
      await screenshot('model-builder-looped-1024')
      await command('Emulation.clearDeviceMetricsOverride')
      await evaluate(`document.querySelector('button[aria-label="Switch to dark theme"]').click()`)
      await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Save Blueprint')).click()`)
      await waitFor('сохранён')
      await command('Page.reload')
      await waitFor('shared body × 6')
      await waitFor('5 blocks')
      assert.equal(await evaluate(`JSON.parse(localStorage.getItem('mah-builder-draft')).blocks[2].params.gradient_steps`), 2)
      assert.equal(await evaluate(`JSON.parse(localStorage.getItem('mah-builder-draft')).blocks[2].params.checkpoint`), true)
      await evaluate(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('Train this model')).click()`)
      await waitFor('Readiness')
      console.log('PASS editable loop body, parameter sharing, independent copies, undo, meta-analysis, responsive layout, persisted draft and training handoff')
    }
    if (process.env.MAH_SMOKE_RESEARCH === '1') {
      await evaluate('location.hash = "/builder"')
      await waitFor('Start from a recipe')
      for(let attempt=0;attempt<100&&await evaluate(`document.getElementById('builder-recipe').disabled`);attempt++)await delay(100)
      await evaluate(`(() => { const s=document.getElementById('builder-recipe');s.value='lab_tiny';s.dispatchEvent(new Event('change',{bubbles:true})); })()`)
      await waitFor('5 blocks')
      await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Analyze architecture').click()`)
      await waitFor('Architecture validated')
      await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Run synthetic probe').click()`)
      await waitFor('Synthetic loss:')
      await waitFor('Causality: passed')
      assert(await evaluate(`!document.body.innerText.includes('update norm 0 ·')`))
      await screenshot('research-lab-signals')
      await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Pin baseline').click()`)
      await evaluate(`[...document.querySelectorAll('[role="tab"]')].find(b=>b.textContent==='Graph').click()`)
      await screenshot('research-lab-graph')
      await evaluate(`document.querySelector('[aria-label="loop block 3"]').click()`)
      await evaluate(`[...document.querySelectorAll('summary')].find(b=>b.textContent.includes('Understand this block')).click()`)
      await waitFor('Universal Transformers')
      assert(await evaluate(`[...document.querySelectorAll('a')].some(a=>a.href.includes('1807.03819'))`))
      await screenshot('research-lab-knowledge')
      await evaluate(`document.querySelector('button[aria-label="Switch to light theme"]').click()`)
      await screenshot('research-lab-light')
      await command('Emulation.setDeviceMetricsOverride',{width:1024,height:768,deviceScaleFactor:1,mobile:false})
      assert.equal(await evaluate(`document.querySelector('.studio-content').scrollWidth<=document.querySelector('.studio-content').clientWidth`),true)
      await screenshot('research-lab-1024')
      await command('Emulation.clearDeviceMetricsOverride')
      await evaluate(`document.querySelector('button[aria-label="Switch to dark theme"]').click()`)
      await evaluate(`[...document.querySelectorAll('[role="tab"]')].find(b=>b.textContent==='Overview').click()`)
      for (const task of ['vlm','vla']) {
        await evaluate('location.hash = "/builder"')
        await waitFor('Research Lab')
        await evaluate(`(() => {const s=document.getElementById('lab-task');s.value=${JSON.stringify(task)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`)
        await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent==='Analyze architecture').click()`)
        await waitFor('Architecture validated')
        await evaluate(`[...document.querySelectorAll('button')].find(b=>b.textContent.includes(${JSON.stringify('Train in '+task.toUpperCase()+' Studio')})).click()`)
        await waitFor(task==='vlm'?'VLM Stage · Research blueprint':'VLA Configuration')
        if(task==='vla')assert.equal(await evaluate(`document.querySelector('[aria-label="VLA architecture"]').value`),'blueprint_vla')
        await screenshot('research-handoff-'+task)
      }
      console.log('PASS Research Lab real CPU probe, causality, graph, paper references, baseline, responsive themes and VLM/VLA handoffs')
    }
    await evaluate('location.hash = "/settings"')
    await waitFor('Configure your studio preferences')
    await evaluate(`document.querySelector('button[aria-label="Switch to light theme"]').click()`)
    // Exercise packaged hash routing on a full reload, not just in-app navigation.
    await command('Page.reload')
    await waitFor('Configure your studio preferences')
    assert.equal(await evaluate(`document.documentElement.classList.contains('dark')`), false, 'Theme must survive reload')
    await evaluate(`document.querySelector('button[aria-label="Switch to dark theme"]').click()`)
    await evaluate(`document.querySelector('button[aria-label="Переключить на русский"]').click()`)
    assert.equal(await evaluate(`document.querySelector('button[aria-label="Switch to English"]') !== null`), true)
    console.log('PASS packaged reload, theme persistence, language switch and native backend health')
    await evaluate('window.close()').catch(() => {})
    for (let attempt = 0; attempt < 60 && child.exitCode === null && child.signalCode === null; attempt++) await delay(250)
    assert(child.exitCode !== null || child.signalCode !== null, 'Desktop did not shut down')
  } catch (error) {
    console.error('Renderer at failure:', await evaluate('document.body.innerText').catch(() => 'unavailable'))
    throw error
  } finally {
    if (child.exitCode === null && child.signalCode === null && socket?.readyState === WebSocket.OPEN) {
      await evaluate('window.close()').catch(() => {})
      for (let attempt = 0; attempt < 24 && child.exitCode === null && child.signalCode === null; attempt++) await delay(250)
    }
    socket?.close()
    if (child.exitCode === null && child.signalCode === null) {
      // This PID is the process group created above, never an unrelated application.
      try { process.kill(-child.pid, 'SIGKILL') } catch { /* exited */ }
    }
    rmSync(root, { recursive: true, force: true })
  }
}
main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1 })
