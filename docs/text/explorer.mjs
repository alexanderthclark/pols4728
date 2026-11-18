import {inspectTerm,termEquations} from './visuals.mjs?v=20261108-peloton-1';
import {explorerTerms as vocabulary} from './model.mjs?v=20261108-peloton-1';

/** Keep the table stable while a cell's document and term drive the calculation. */
export function initializeExplorer(root){
  root.innerHTML=inspectTerm();
  const cells=[...root.querySelectorAll('.matrix-cell')];
  const rows=[...root.querySelectorAll('tbody tr')];
  const headings=[...root.querySelectorAll('thead th')].slice(1);
  const result=root.querySelector('.explorer-result');
  const scroller=root.querySelector('.explorer-scroll');
  let selected=cells.find(cell=>cell.getAttribute('aria-pressed')==='true');

  function select(cell){
    if(!cell||cell===selected)return;
    selected=cell;
    const doc=Number(cell.dataset.doc),col=Number(cell.dataset.col);
    for(const button of cells){
      const active=button===cell;
      button.tabIndex=active?0:-1;
      button.setAttribute('aria-pressed',String(active));
      button.closest('td').classList.toggle('focus',Number(button.dataset.col)===col);
    }
    rows.forEach((row,i)=>row.classList.toggle('active',i===doc));
    headings.forEach((heading,i)=>heading.classList.toggle('term-focus',i===col));
    result.innerHTML=termEquations(vocabulary[col],doc);
  }
  const targetCell=event=>event.target.closest('.matrix-cell');
  const onSelect=event=>select(targetCell(event));
  function onKeydown(event){
    const cell=targetCell(event);
    if(!cell||event.altKey||event.metaKey)return;
    let doc=Number(cell.dataset.doc),col=Number(cell.dataset.col);
    switch(event.key){
      case 'ArrowLeft': col--;break;
      case 'ArrowRight': col++;break;
      case 'ArrowUp': doc--;break;
      case 'ArrowDown': doc++;break;
      case 'Home': col=0;if(event.ctrlKey)doc=0;break;
      case 'End': col=vocabulary.length-1;if(event.ctrlKey)doc=rows.length-1;break;
      default:return;
    }
    event.preventDefault();
    doc=Math.max(0,Math.min(rows.length-1,doc));
    col=Math.max(0,Math.min(vocabulary.length-1,col));
    const next=cells[doc*vocabulary.length+col];
    select(next);
    next.focus({preventScroll:true});
    // Reveal the focused column within the table, without moving the page.
    if(scroller){
      const cellBounds=next.getBoundingClientRect(),bounds=scroller.getBoundingClientRect();
      const labelWidth=rows[doc].querySelector('th').getBoundingClientRect().width;
      if(cellBounds.left<bounds.left+labelWidth)scroller.scrollLeft-=bounds.left+labelWidth-cellBounds.left;
      else if(cellBounds.right>bounds.right)scroller.scrollLeft+=cellBounds.right-bounds.right;
    }
  }
  const handlers={pointerover:onSelect,focusin:onSelect,click:onSelect,keydown:onKeydown};
  for(const [name,handler] of Object.entries(handlers))root.addEventListener(name,handler);
  return ()=>{for(const [name,handler] of Object.entries(handlers))root.removeEventListener(name,handler);};
}
