/* Static architecture overview with labels attached to its components. */
(() => {
  const blue = '#234E70', teal = '#007F73', rust = '#A54F32', purple = '#7B4F9D';
  const rect = (x,y,w,h,fill,stroke='none',radius=3,opacity=1) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="1.6" opacity="${opacity}"/>`;
  const label = (x,y,text,color='#242424',size=18,anchor='middle') => `<text x="${x}" y="${y}" fill="${color}" font-family="var(--serif)" font-size="${size}" text-anchor="${anchor}">${text}</text>`;
  const matrix = (x,y,rows,cols,size,gap,color,causal=false) => {
    let result='';
    for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
      const blocked=causal && col>row;
      result+=rect(x+col*(size+gap),y+row*(size+gap),size,size,blocked?'white':color,blocked?'#d4d8da':'none',1,blocked?1:.35+((row*3+col*5)%7)*.09);
    }
    return result;
  };
  const join = (x,y,color) => `<circle cx="${x}" cy="${y}" r="10" fill="white" stroke="${color}" stroke-width="1.8"/><path d="M${x-5} ${y} H${x+5} M${x} ${y-5} V${y+5}" fill="none" stroke="${color}" stroke-width="1.6"/>`;

  function draw(kind){
    const id=`overview-${kind}`;
    const arrow=(d,color=blue,end=true,width=2) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"${end?` marker-end="url(#${id}-${color===teal?'teal':color===rust?'rust':'blue'})"`:''}/>`;
    let svg='<defs>';
    for(const [name,color] of [['blue',blue],['teal',teal],['rust',rust]])svg+=`<marker id="${id}-${name}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6Z" fill="${color}"/></marker>`;
    svg+='</defs>';

    // Residual stream and its two additive branches.
    svg+=arrow('M324 650 V486',blue,true,2.8)+arrow('M324 464 V291',blue,false,2.8)+arrow('M324 269 V205',blue,true,2.8);
    svg+=arrow('M324 600 H85 V570')+arrow('M324 600 H216 V570');
    svg+=arrow('M85 500 V475 H313',teal)+arrow('M216 500 V475 H313',teal);
    svg+=arrow('M324 450 H145 V436',rust)+arrow('M145 300 V280 H313',rust);
    svg+=join(324,475,teal)+join(324,280,rust);
    svg+=label(339,350,'Residual',blue,16,'start')+label(339,369,'stream',blue,16,'start');

    // Embeddings plus positions; the initial four-row residual matrix.
    svg+=rect(245,650,158,56,'white','#d4d8da');
    svg+=matrix(256,657,4,6,5,2,blue)+join(324,672,blue)+matrix(350,657,4,6,5,2,purple);
    svg+=label(324,700,'Embeddings + positions',blue,14);
    for(const y of [609,405,229]){
      svg+=rect(297,y-4,54,38,'white',blue,3);
      svg+=matrix(301,y,4,6,6,2,blue);
    }

    // Two attention heads: full connectivity versus a causal triangle.
    for(const [head,x] of [34,165].entries()){
      svg+=rect(x,500,102,70,'#e1f0ed',teal);
      svg+=label(x+51,519,`Head ${head+1}`,teal,16);
      svg+=matrix(x+34,528,4,4,7,2,teal,kind==='decoder');
    }
    svg+=label(150,625,kind==='encoder'?'Attention layer':'Causal attention layer',teal,18);

    // One MLP, showing four separate rows through the same feature maps.
    svg+=rect(52,300,186,136,'#f4eae6',rust);
    svg+=label(145,321,'MLP',rust,21)+label(145,339,'shared across tokens',rust,14);
    for(let row=0;row<4;row++){
      const y=352+row*22;
      svg+=arrow(`M94 ${y+3} H112`,rust,true,1.2)+arrow(`M163 ${y+3} H184`,rust,true,1.2);
      svg+=matrix(64,y,1,6,4,1,blue)+matrix(119,y-2,1,10,3,1,rust)+matrix(190,y,1,6,4,1,rust);
    }

    // Output structures: pool to one scalar, or retain all token rows.
    svg+=rect(200,78,200,125,'white','#d4d8da');
    if(kind==='encoder'){
      svg+=label(300,98,'Mean pooling',blue,18);
      for(let row=0;row<4;row++){
        const y=121+row*21;
        svg+=matrix(214,y,1,6,5,2,blue);
        svg+=arrow(`M255 ${y+2.5} C280 ${y+2.5}, 285 151, 309 151`,blue,false,1.4);
      }
      svg+=arrow('M309 151 H320',blue,true,1.4)+matrix(328,148,1,6,5,2,blue);
      svg+=arrow('M324 78 V42')+`<circle cx="324" cy="30" r="8" fill="${blue}"/>`;
      svg+=label(305,35,'Scalar prediction',blue,18,'end')+label(310,64,'Linear head',blue,15,'end');
    }else{
      svg+=label(300,98,'Unembedding + softmax',blue,16);
      for(let row=0;row<4;row++){
        const y=121+row*21;
        svg+=matrix(214,y,1,6,5,2,blue)+arrow(`M258 ${y+3} H292`,blue,true,1.4);
        for(let col=0;col<8;col++){
          const height=3+((row*5+col*3)%7)*1.6;
          svg+=rect(302+col*10,y+9-height,7,height,blue,'none',.8,.4+((row+col)%4)*.15);
        }
      }
      svg+=arrow('M324 78 V61')+matrix(290,14,4,8,6,3,blue);
      svg+=label(276,25,'Next-token',blue,18,'end')+label(276,45,'distributions',blue,18,'end');
    }
    return svg;
  }

  for(const kind of ['encoder','decoder'])document.getElementById(`overview-${kind}`).innerHTML=draw(kind);
  // Keep the overview free of visible navigation text without shifting layout.
  new IntersectionObserver(([entry])=>{
    document.body.classList.toggle('overview-visible',entry.intersectionRatio>=.6);
  },{threshold:[0,.6]}).observe(document.getElementById('overview'));
})();
