import { mkdir,rm,writeFile,cp } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { renderPage } from '../page.js'
const root=fileURLToPath(new URL('..',import.meta.url))
const output=path.join(root,'dist')
await rm(output,{recursive:true,force:true})
await mkdir(output,{recursive:true})
for(const file of ['style.css','site.js','page.js','content.js'])await cp(path.join(root,file),path.join(output,file))
await cp(path.join(root,'assets'),path.join(output,'assets'),{recursive:true})
await writeFile(path.join(output,'index.html'),renderPage('en','./'))
for(const lang of ['en','ru']){await mkdir(path.join(output,lang),{recursive:true});await writeFile(path.join(output,lang,'index.html'),renderPage(lang,'../'))}
await writeFile(path.join(output,'404.html'),renderPage('en','./'))
console.log(`Built static EN/RU website: ${output}`)
