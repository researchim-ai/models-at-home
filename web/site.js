import { content, commands } from './content.js'
import { renderWorkspace, renderBlock } from './page.js'

const lang=document.documentElement.lang==='ru'?'ru':'en'
const c=content[lang]
const base=document.body.dataset.base || './'
const nav=document.getElementById('site-nav')
const menu=document.querySelector('.menu-toggle')
function closeMenu(){nav.classList.remove('is-open');menu.setAttribute('aria-expanded','false')}
menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));nav.classList.toggle('is-open',open)})
nav.addEventListener('click',event=>{if(event.target.closest('a'))closeMenu()})
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeMenu()})
document.addEventListener('click',event=>{if(!nav.contains(event.target)&&!menu.contains(event.target))closeMenu()})
window.matchMedia('(min-width: 961px)').addEventListener('change',event=>{if(event.matches)closeMenu()})
for(const link of document.querySelectorAll('[data-locale]'))link.addEventListener('click',()=>{if(location.hash)link.hash=location.hash})

function tabs(selector,callback){
  const buttons=[...document.querySelectorAll(selector)]
  const select=button=>{for(const b of buttons){const active=b===button;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1}callback(button)}
  for(const [index,button] of buttons.entries()){
    button.addEventListener('click',()=>select(button))
    button.addEventListener('keydown',event=>{
      let next
      if(['ArrowRight','ArrowDown'].includes(event.key))next=(index+1)%buttons.length
      if(['ArrowLeft','ArrowUp'].includes(event.key))next=(index-1+buttons.length)%buttons.length
      if(event.key==='Home')next=0
      if(event.key==='End')next=buttons.length-1
      if(next!==undefined){event.preventDefault();select(buttons[next]);buttons[next].focus();buttons[next].scrollIntoView({block:'nearest',inline:'nearest'})}
    })
  }
}
let activeWorkspace=0
tabs('[data-workspace]',button=>{
  activeWorkspace=Number(button.dataset.workspace)
  const panel=document.getElementById('workspace-panel')
  panel.innerHTML=renderWorkspace(c,activeWorkspace,base)
  panel.setAttribute('aria-labelledby',button.id)
})
tabs('[data-block]',button=>{const panel=document.getElementById('block-panel');panel.innerHTML=renderBlock(c,Number(button.dataset.block));panel.setAttribute('aria-labelledby',button.id)})
let selectedOS='unix'
tabs('[data-os]',button=>{selectedOS=button.dataset.os;document.getElementById('install-command').textContent=commands[selectedOS];document.getElementById('command-panel').setAttribute('aria-labelledby',button.id);document.getElementById('copy-status').textContent=''})
const copyButton=document.getElementById('copy-commands')
let copyTimer
copyButton.addEventListener('click',async()=>{
  const command=commands[selectedOS]
  const status=document.getElementById('copy-status')
  try{
    await navigator.clipboard.writeText(command)
    copyButton.querySelector('span').textContent=c.setup.copied;copyButton.setAttribute('aria-label',c.setup.copied);status.textContent=c.setup.copied
  }catch{
    const range=document.createRange();range.selectNodeContents(document.getElementById('install-command'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent=c.setup.copyFailed
  }
  clearTimeout(copyTimer);copyTimer=setTimeout(()=>{copyButton.querySelector('span').textContent=c.setup.copy;copyButton.setAttribute('aria-label',c.setup.copy)},2400)
})

const dialog=document.getElementById('screenshot-dialog')
let galleryIndex=0
let galleryTrigger=null
function displayImage(index){
  galleryIndex=(index+c.workspaces.length)%c.workspaces.length
  const w=c.workspaces[galleryIndex]
  const image=document.getElementById('lightbox-image')
  image.src=`${base}assets/screenshots/${w.image}.webp`;image.alt=w.alt
  document.getElementById('lightbox-caption').textContent=w.label
  document.getElementById('lightbox-count').textContent=`${galleryIndex+1} / ${c.workspaces.length}`
}
document.addEventListener('click',event=>{
  const target=event.target.closest('[data-zoom]')
  if(target){galleryTrigger=target;displayImage(Number(target.dataset.index));dialog.showModal();document.body.classList.add('modal-open')}
})
document.getElementById('close-lightbox').addEventListener('click',()=>dialog.close())
document.getElementById('prev-screenshot').addEventListener('click',()=>displayImage(galleryIndex-1))
document.getElementById('next-screenshot').addEventListener('click',()=>displayImage(galleryIndex+1))
dialog.addEventListener('click',event=>{if(event.target===dialog){const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)dialog.close()}})
dialog.addEventListener('keydown',event=>{if(event.key==='ArrowRight'){event.preventDefault();displayImage(galleryIndex+1)}if(event.key==='ArrowLeft'){event.preventDefault();displayImage(galleryIndex-1)}})
dialog.addEventListener('close',()=>{document.body.classList.remove('modal-open');galleryTrigger?.focus()})

if('IntersectionObserver' in window&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){
  const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('is-visible');observer.unobserve(entry.target)}},{threshold:.08})
  for(const item of document.querySelectorAll('.section-heading,.manifesto-main,.principles,.research-top,.workflow-grid,.start-copy,.faq-heading')){
    if(item.getBoundingClientRect().top>window.innerHeight){item.classList.add('reveal-ready');observer.observe(item)}
  }
}
