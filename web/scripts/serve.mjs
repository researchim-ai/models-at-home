import http from 'node:http'
import { readFile,stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { renderPage } from '../page.js'
const root=path.resolve(fileURLToPath(new URL('..',import.meta.url)))
const production=process.argv.includes('--production')
const directory=production?path.join(root,'dist'):root
const arg=process.argv.indexOf('--port')
const port=Number(arg>=0?process.argv[arg+1]:process.env.PORT||4173)
const host=process.env.HOST||'127.0.0.1'
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.webp':'image/webp','.ttf':'font/ttf','.txt':'text/plain; charset=utf-8'}
const server=http.createServer(async(req,res)=>{
  try{
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'});res.end();return}
    const url=new URL(req.url,'http://local')
    const pathname=decodeURIComponent(url.pathname)
    if(!production&&['/','/en/','/ru/'].includes(pathname)){
      res.writeHead(200,{'Content-Type':types['.html'],'Cache-Control':'no-store'});res.end(req.method==='HEAD'?'':renderPage(pathname==='/ru/'?'ru':'en',pathname==='/'?'./':'../'));return
    }
    const file=path.resolve(directory,'.'+pathname)
    if(file!==directory&&!file.startsWith(directory+path.sep)){res.writeHead(403);res.end('Forbidden');return}
    const info=await stat(file)
    if(info.isDirectory()&&!pathname.endsWith('/')){res.writeHead(308,{'Location':url.pathname+'/'+url.search});res.end();return}
    const target=info.isDirectory()?path.join(file,'index.html'):file
    const data=await readFile(target)
    res.writeHead(200,{'Content-Type':types[path.extname(target)]||'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
    res.end(req.method==='HEAD'?'':data)
  }catch{res.writeHead(404,{'Content-Type':'text/plain; charset=utf-8'});res.end('Not found')}
})
server.listen(port,host,()=>console.log(`Models at Home website: http://${host}:${server.address().port}${production?' (static production build)':''}`))
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>server.close(()=>process.exit(0)))
