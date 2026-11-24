import assert from 'node:assert/strict'
import { readFile,stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { content,commands,repo } from '../content.js'
const root=path.resolve(fileURLToPath(new URL('..',import.meta.url)))
function shape(a,b,where='content'){
  assert.equal(typeof a,typeof b,`Translation type mismatch: ${where}`)
  if(Array.isArray(a)){assert.equal(a.length,b.length,`Translation array mismatch: ${where}`);a.forEach((v,i)=>shape(v,b[i],`${where}[${i}]`))}
  else if(a&&typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),`Translation keys mismatch: ${where}`);for(const k of Object.keys(a))shape(a[k],b[k],`${where}.${k}`)}
  else if(typeof a==='string')assert(b.trim(),`Empty translation: ${where}`)
}
shape(content.en,content.ru)
let localReferences=0
for(const [lang,file] of [['en','index.html'],['en','en/index.html'],['ru','ru/index.html']]){
  const filename=path.join(root,'dist',file),html=await readFile(filename,'utf8')
  assert(html.includes(`<html lang="${lang}">`))
  assert.equal((html.match(/<h1>/g)||[]).length,1)
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1])
  assert.equal(new Set(ids).size,ids.length,`Duplicate IDs in ${file}`)
  for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){
    const link=m[1]
    if(link.startsWith('#')){assert(ids.includes(link.slice(1)),`Missing anchor ${link}`);continue}
    if(link.startsWith(repo+'/blob/HEAD/')){await stat(path.join(root,'..',link.split('/blob/HEAD/')[1]));continue}
    if(/^[a-z]+:/.test(link))continue
    const target=path.resolve(path.dirname(filename),link)
    const info=await stat(target);if(info.isDirectory())await stat(path.join(target,'index.html'));localReferences++
  }
  for(const workspace of content[lang].workspaces){await stat(path.join(root,'assets','screenshots',workspace.image+'.webp'));assert(workspace.alt.trim())}
}
for(const command of Object.values(commands))assert(command.includes('npm run dev')&&command.includes('MAH_PYTHON'))
const child=spawn(process.execPath,[path.join(root,'scripts','serve.mjs'),'--production','--port','0'],{stdio:['ignore','pipe','pipe']})
let output=''
try{
 const url=await new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>reject(new Error('Preview server did not start')),10000)
  child.stdout.on('data',data=>{output+=data;const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timer);resolve(match[0])}})
  child.once('error',reject);child.once('exit',code=>{clearTimeout(timer);reject(new Error(`Preview exited ${code}`))})
 })
 for(const resource of ['/','/en/','/ru/','/style.css','/site.js','/page.js','/content.js','/assets/mark.svg','/assets/fonts/manrope-variable.ttf']){
  const res=await fetch(url+resource);assert.equal(res.status,200,resource)
  if(resource.endsWith('.css'))assert(res.headers.get('content-type')?.startsWith('text/css'))
  if(resource.endsWith('.js'))assert(res.headers.get('content-type')?.includes('javascript'))
  if(resource.endsWith('.ttf'))assert.equal(res.headers.get('content-type'),'font/ttf')
  const data=await res.arrayBuffer();assert(data.byteLength>0,resource)
 }
 assert.equal((await fetch(url+'/missing-file')).status,404)
 assert.equal((await fetch(url+'/ru',{redirect:'manual'})).status,308)
 console.log(`PASS EN/RU translation parity, HTML structure, ${localReferences} local references, all screenshot assets, production routes and MIME responses`)
}finally{child.kill('SIGTERM')}
