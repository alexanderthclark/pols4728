import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {startLesson} from '../docs/text/bootstrap.mjs';
import {chapters} from '../docs/text/lesson.mjs';

const source = fs.readFileSync(new URL('../docs/text/index.html', import.meta.url), 'utf8');

test('without JavaScript the complete lesson precedes its conclusion and chapter links work', () => {
  const transcriptStart = source.indexOf('<section class="transcript"');
  const conclusionStart = source.indexOf('<section class="ending"');
  assert.ok(transcriptStart > source.indexOf('<div id="chapters"></div>'));
  assert.ok(transcriptStart < conclusionStart);
  const transcript = source.slice(transcriptStart, conclusionStart);
  for (const chapter of chapters) {
    assert.ok(transcript.includes(`<details open id="${chapter.id}" data-chapter-id="${chapter.id}">`));
    for (const stage of chapter.stages) assert.ok(transcript.includes(`<h3>${stage.title}</h3>`));
  }
  assert.equal((transcript.match(/<h3>/g) ?? []).length, 17);
  assert.match(source, /id="explore" hidden/);
});

// Small DOM doubles track the observable loading contract, including node order.
function pageFixture() {
  const readingChapters = chapters.map(chapter => ({id:chapter.id, dataset:{chapterId:chapter.id}, open:true}));
  const transcript = {querySelectorAll:() => readingChapters};
  const root = {tracks:[], querySelectorAll(){return this.tracks;}, replaceChildren(){this.tracks=[];}};
  const explorer = {hidden:true};
  const conclusion = {before(node){move(node, main.children.indexOf(conclusion));}};
  const main = {children:[root,transcript,conclusion,explorer], append(node){move(node,this.children.length);}};
  function move(node,index){
    const existing=main.children.indexOf(node);
    if(existing!==-1){main.children.splice(existing,1);if(existing<index)index--;}
    main.children.splice(index,0,node);
  }
  const page = {querySelector:selector=>({'#chapters':root,'#transcript':transcript,'#explore':explorer,'#compare':conclusion,main}[selector])};
  const initializeStory = () => {root.tracks=chapters.map(chapter=>({id:'',dataset:{chapterId:chapter.id}}));};
  return {page,root,transcript,readingChapters,explorer,conclusion,main,initializeStory};
}

test('a failed module request preserves the full, expanded reading route', async () => {
  const f=pageFixture(), errors=[];
  const ready=await startLesson({page:f.page,loadStory:async()=>{throw new Error('Module request failed');},reportError:error=>errors.push(error)});
  assert.equal(ready,false);
  assert.equal(errors.length,1);
  assert.equal(f.root.tracks.length,0);
  assert.ok(f.main.children.indexOf(f.transcript)<f.main.children.indexOf(f.conclusion));
  assert.deepEqual(f.readingChapters.map(c=>[c.id,c.open]),chapters.map(c=>[c.id,true]));
  assert.equal(f.explorer.hidden,true);
});

test('an initialization error removes partial scenes without removing the reading version', async () => {
  const f=pageFixture();
  const ready=await startLesson({page:f.page,loadStory:async()=>({initializeStory(){f.initializeStory();throw new Error('Diagram failed');}}),reportError:()=>{}});
  assert.equal(ready,false);
  assert.equal(f.root.tracks.length,0);
  assert.deepEqual(f.readingChapters.map(c=>[c.id,c.open]),chapters.map(c=>[c.id,true]));
  assert.equal(f.main.children.filter(node=>node===f.transcript).length,1);
});

test('successful initialization transfers anchors and moves one collapsed reading version to the end', async () => {
  const f=pageFixture();
  const ready=await startLesson({page:f.page,loadStory:async()=>({initializeStory:f.initializeStory}),reportError:assert.fail});
  assert.equal(ready,true);
  assert.deepEqual(f.root.tracks.map(c=>c.id),chapters.map(c=>c.id));
  assert.deepEqual(f.readingChapters.map(c=>[c.id,c.open]),chapters.map(c=>[`reading-${c.id}`,false]));
  assert.equal(f.main.children.at(-1),f.transcript);
  assert.equal(f.main.children.filter(node=>node===f.transcript).length,1);
  assert.equal(f.explorer.hidden,false);
});

test('a failure after initialization disposes listeners and restores original anchors', async () => {
  const f=pageFixture();
  let disposed=false;
  f.main.append=()=>{throw new Error('Finalization failed');};
  const ready=await startLesson({page:f.page,loadStory:async()=>({initializeStory(){f.initializeStory();return()=>{disposed=true;};}}),reportError:()=>{}});
  assert.equal(ready,false);
  assert.equal(disposed,true);
  assert.equal(f.root.tracks.length,0);
  assert.deepEqual(f.readingChapters.map(c=>[c.id,c.open]),chapters.map(c=>[c.id,true]));
  assert.ok(f.main.children.indexOf(f.transcript)<f.main.children.indexOf(f.conclusion));
});
