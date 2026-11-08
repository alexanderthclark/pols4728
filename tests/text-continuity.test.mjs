import assert from 'node:assert/strict';
import test from 'node:test';
import {storyGroups} from '../docs/text/story.mjs';
import {morphNode} from '../docs/text/visual-update.mjs';
import {chapters} from '../docs/text/lesson.mjs';

// Minimal tree objects test node identity, rather than just equal markup.
class Node {
  constructor(name, attrs={}, children=[]){
    this.nodeType=name==='#text'?3:1;this.localName=name;this.namespaceURI='test';
    this.attrs=new Map(Object.entries(attrs));this.childNodes=[];this.parentNode=null;
    this.nodeValue=this.nodeType===3?attrs.value:null;
    children.forEach(child=>this.insertBefore(child,null));
  }
  get attributes(){return [...this.attrs].map(([name,value])=>({name,value}));}
  get lastChild(){return this.childNodes.at(-1);}
  getAttribute(name){return this.attrs.get(name)??null;}
  hasAttribute(name){return this.attrs.has(name);}
  setAttribute(name,value){this.attrs.set(name,value);}
  removeAttribute(name){this.attrs.delete(name);}
  insertBefore(child,before){child.remove();const index=before?this.childNodes.indexOf(before):this.childNodes.length;this.childNodes.splice(index,0,child);child.parentNode=this;}
  remove(){if(this.parentNode){this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this),1);this.parentNode=null;}}
  cloneNode(deep){return new Node(this.localName,Object.fromEntries(this.attrs),deep?this.childNodes.map(child=>child.cloneNode(true)):[]);}
}
const cell=(key,value)=>new Node('td',{'data-key':key},[new Node('#text',{value})]);

test('bag of words and TF–IDF share a single pinned sequence before the applications',()=>{
  const groups=storyGroups();
  assert.deepEqual(groups.map(group=>group.chapters.map(chapter=>chapter.id)),[['ai-writing'],['bag-of-words','tf-idf'],['authorship'],['bertopic']]);
  assert.deepEqual(groups.map(group=>group.application),[false,false,true,true]);
  const main=groups[1],bag=chapters.find(chapter=>chapter.id==='bag-of-words');
  assert.equal(main.steps[bag.stages.length-1].scene.visual,'matrix');
  assert.equal(main.steps[bag.stages.length].scene.visual,'tf');
  assert.equal(main.steps[bag.stages.length].localStage,0);
  assert.equal(main.steps[bag.stages.length].offset,bag.stages.length);
  assert.equal(main.steps.at(-1).scene.visual,'cosine');
});

test('scroll updates preserve matrix cells and text nodes while changing weights and highlights',()=>{
  const political=cell('political','1'),citizen=cell('citizen','1');
  const number=citizen.childNodes[0],row=new Node('tr',{class:'counts'},[political,citizen]);
  const next=new Node('tr',{class:'weighted','data-focus':'citizen'},[cell('political','0'),cell('citizen','0.69')]);
  morphNode(row,next);
  assert.equal(row.childNodes[0],political);assert.equal(row.childNodes[1],citizen);
  assert.equal(citizen.childNodes[0],number);assert.equal(number.nodeValue,'0.69');
  assert.equal(row.getAttribute('class'),'weighted');
  morphNode(row,new Node('tr',{class:'counts'},[cell('political','1'),cell('citizen','1')]));
  assert.equal(citizen.childNodes[0],number);assert.equal(number.nodeValue,'1');
  assert.equal(row.hasAttribute('data-focus'),false);
});

test('keyed visual objects retain identity when their order changes or an annotation is added',()=>{
  const one=cell('one','1'),two=cell('two','2'),root=new Node('g',{},[one,two]);
  morphNode(root,new Node('g',{},[cell('two','3'),cell('one','4'),new Node('text',{},[new Node('#text',{value:'note'})])]));
  assert.equal(root.childNodes[0],two);assert.equal(root.childNodes[1],one);
  assert.equal(two.childNodes[0].nodeValue,'3');assert.equal(root.childNodes.length,3);
  morphNode(root,new Node('g',{},[cell('one','1')]));
  assert.deepEqual(root.childNodes,[one]);assert.equal(one.childNodes[0].nodeValue,'1');
});
