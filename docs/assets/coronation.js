(() => {
  const root = document.getElementById('course-coronation');
  if (!root) return;

  const image = root.querySelector('img.course-logo');
  const svg = root.querySelector('.character-layer');
  if (!image || !svg) return;

  const layer = svg.querySelector('.animated-eyes');
  const defs = svg.querySelector('.eye-clips');
  const crown = svg.querySelector('.descending-crown');
  const crownContact = svg.querySelector('.crown-contact-shadow');
  const rays = svg.querySelector('.crown-rays-mask');
  const mouth = svg.querySelector('.animated-mouth');
  const oMouth = svg.querySelector('.o-mouth');
  const smile = svg.querySelector('.pleased-smile');
  if (![layer, defs, crown, crownContact, rays, mouth, oMouth, smile].every(Boolean)) return;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const NS = 'http://www.w3.org/2000/svg';
  const idleDelay = 5000;
  const duration = 4900;
  const landingAt = 3.62;
  const eyes = [
    {name:'m-left',cx:322,cy:451,rx:27,ry:25,px:337,py:453,r:11,delay:.58,path:'M321 426 C337 425 350 436 350 451 C351 465 338 477 322 476 C307 476 296 465 296 451 C296 437 306 427 321 426 Z'},
    {name:'m-right',cx:375,cy:448,rx:27,ry:26,px:390,py:451,r:11,delay:.58,path:'M374 422 C389 421 401 433 401 448 C402 463 391 474 376 474 C361 476 350 464 350 449 C349 434 359 423 374 422 Z'},
    {name:'l-left',cx:630,cy:423,rx:26,ry:27,px:643,py:431,r:11,delay:.70,path:'M629 396 C644 396 656 408 656 424 C657 439 646 449 631 449 C615 450 604 438 604 424 C604 408 615 397 629 396 Z'},
    {name:'l-right',cx:680,cy:420,rx:24,ry:27,px:693,py:429,r:11,delay:.70,path:'M679 393 C695 392 704 404 704 420 C704 435 694 446 680 446 C665 447 655 435 656 421 C655 406 666 394 679 393 Z'},
    {name:'amp-left',cx:902,cy:483,rx:15,ry:16,px:912,py:479,r:7,delay:.84,path:'M902 468 C911 468 918 475 918 484 C918 493 912 501 903 500 C893 500 887 493 887 484 C887 475 894 468 902 468 Z'},
    {name:'amp-right',cx:934,cy:487,rx:16,ry:17,px:940,py:484,r:7,delay:.84,path:'M933 472 C943 472 950 479 949 489 C949 499 942 505 933 505 C923 504 916 497 917 487 C917 478 923 472 933 472 Z'},
    {name:'a-left',cx:1142,cy:357,rx:25,ry:25,px:1129,py:345,r:11,delay:.46,path:'M1141 332 C1157 331 1167 343 1167 358 C1167 372 1156 383 1142 382 C1126 383 1117 371 1118 356 C1118 342 1127 333 1141 332 Z'},
    {name:'a-right',cx:1190,cy:365,rx:25,ry:25,px:1177,py:354,r:11,delay:.46,path:'M1190 340 C1205 339 1215 351 1214 366 C1215 380 1205 390 1191 389 C1175 390 1165 378 1166 364 C1166 350 1177 341 1190 340 Z'},
    {name:'i-left',cx:1442,cy:352,rx:23,ry:27,px:1434,py:335,r:10.5,delay:.36,path:'M1441 325 C1457 323 1466 337 1465 353 C1465 368 1456 379 1441 378 C1426 379 1419 367 1420 352 C1420 337 1427 327 1441 325 Z'},
    {name:'i-right',cx:1489,cy:357,rx:23,ry:28,px:1478,py:341,r:10.5,delay:.36,path:'M1488 328 C1503 327 1511 341 1511 357 C1511 373 1502 383 1488 382 C1473 383 1465 371 1466 355 C1466 340 1474 330 1488 328 Z'}
  ];
  function element(tag, attrs) {
    const e = document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([key,value]) => e.setAttribute(key,value));
    return e;
  }
  eyes.forEach(eye => {
    const clip = element('clipPath',{id:'coronation-'+eye.name});
    clip.append(element('path',{d:eye.path}));
    defs.append(clip);
    const group = element('g',{'data-character':eye.name.split('-')[0]});
    group.append(element('ellipse',{cx:eye.cx,cy:eye.cy,rx:eye.rx+4,ry:eye.ry+4,fill:'#fff'}));
    const inner = element('g',{});
    inner.append(element('path',{d:eye.path,fill:'#fff',stroke:'#161616','stroke-width':eye.r<8?4.5:5.8,'stroke-linejoin':'round'}));
    const clipped = element('g',{'clip-path':'url(#coronation-'+eye.name+')'});
    const pupil = element('circle',{cx:eye.px,cy:eye.py,r:eye.r,fill:'#161616'});
    clipped.append(pupil);
    inner.append(clipped);
    group.append(inner);
    layer.append(group);
    eye.inner=inner;
    eye.pupil=pupil;
  });
  const ease = x => { const t=Math.min(1,Math.max(0,x)); return t*t*(3-2*t); };
  const lerp = (a,b,t) => a+(b-a)*t;
  function blink(t, at, length) {
    const u=(t-at)/length;
    return u<=0||u>=1 ? 1 : 1-.96*Math.sin(Math.PI*u);
  }
  function paint(t) {
    layer.setAttribute('opacity',1);
    // Convert the page's top edge to logo coordinates. Include the rotated
    // crown and its shadow so it enters from fully above the page. Measuring
    // during active frames also keeps the path aligned when the layout changes.
    const bounds=image.getBoundingClientRect();
    const scale=bounds.width/1692 || 1;
    const startY=-(bounds.top+window.scrollY)/scale-180;
    const descent=ease((t-.42)/3.2);
    const crownY=lerp(startY,121.5,descent);
    crown.setAttribute('opacity',t>=.42?1:0);
    crown.setAttribute('transform','translate(1379.6 '+crownY+')');
    crownContact.setAttribute('opacity',ease((t-(landingAt-.46))/.46));
    rays.setAttribute('opacity',ease((t-.15)/1.2));
    // Keep the original mouth covered while the quiet O becomes a smile.
    const smileProgress=ease((t-(landingAt+.21))/.65);
    mouth.setAttribute('opacity',ease((t-.36)/.45));
    oMouth.setAttribute('opacity',1-smileProgress);
    smile.setAttribute('opacity',smileProgress);
    eyes.forEach(eye => {
      const isI=eye.name.startsWith('i-');
      const gaze=ease((t-eye.delay)/.85);
      const dx=1472-eye.cx,dy=crownY+86-eye.cy, distance=Math.hypot(dx,dy);
      const reach=eye.r<8?7:12;
      let x=lerp(eye.px,eye.cx+dx/distance*reach,gaze);
      let y=lerp(eye.py,eye.cy+dy/distance*reach,gaze);
      let openness=1;
      if(isI) {
        const settle=ease((t-(landingAt+.16))/.7);
        x=lerp(x,eye.cx,settle);
        y=lerp(y,eye.cy-1,settle);
        openness=blink(t,.08,.30)*blink(t,landingAt+.21,.34);
      }
      eye.pupil.setAttribute('cx',x.toFixed(2));
      eye.pupil.setAttribute('cy',y.toFixed(2));
      eye.inner.setAttribute('transform','translate('+eye.cx+' '+eye.cy+') scale(1 '+openness.toFixed(3)+') translate('+(-eye.cx)+' '+(-eye.cy)+')');
    });
  }

  let state = 'waiting';
  let remaining = idleDelay;
  let idleStarted = null;
  let timer = 0;
  let frame = 0;
  let elapsed = 0;
  let animationStarted = null;
  let inView = false;
  let imageReady = image.complete && image.naturalWidth > 0;
  let disposed = false;
  let observer = null;
  const activityEvents = ['scroll', 'pointerdown', 'keydown', 'touchstart'];
  svg.dataset.coronationState = state;

  function setState(next) {
    state = next;
    svg.dataset.coronationState = next;
  }

  function canCount() {
    return imageReady && inView && !document.hidden && root.isConnected;
  }

  function pauseCountdown(now = performance.now()) {
    clearTimeout(timer);
    timer = 0;
    if (idleStarted !== null) {
      remaining = Math.max(0, remaining - (now - idleStarted));
      idleStarted = null;
    }
  }

  function resumeCountdown() {
    if (disposed || state !== 'waiting' || !canCount() || idleStarted !== null) return;
    idleStarted = performance.now();
    timer = setTimeout(() => {
      pauseCountdown();
      if (!root.isConnected) {
        cleanup();
      } else if (canCount()) {
        if (remaining === 0) begin();
        else resumeCountdown();
      }
    }, remaining);
  }

  function cleanup() {
    if (disposed) return;
    disposed = true;
    clearTimeout(timer);
    cancelAnimationFrame(frame);
    timer = 0;
    frame = 0;
    observer?.disconnect();
    activityEvents.forEach(type => document.removeEventListener(type, meaningfulActivity, true));
    document.removeEventListener('visibilitychange', visibilityChanged);
    window.removeEventListener('resize', updateInView);
    image.removeEventListener('load', imageLoaded);
    image.removeEventListener('error', imageFailed);
    if (reducedMotion.removeEventListener) {
      reducedMotion.removeEventListener('change', motionPreferenceChanged);
    } else if (reducedMotion.removeListener) {
      reducedMotion.removeListener(motionPreferenceChanged);
    }
  }

  function complete() {
    paint(duration / 1000);
    setState('crowned');
    cleanup();
  }

  function animate(now) {
    frame = 0;
    if (disposed || state !== 'animating') return;
    if (!root.isConnected) {
      cleanup();
      return;
    }
    if (document.hidden || !inView) {
      pauseAnimation(now);
      return;
    }
    const progress = Math.min(duration, elapsed + now - animationStarted);
    paint(progress / 1000);
    if (progress >= duration) complete();
    else frame = requestAnimationFrame(animate);
  }

  function pauseAnimation(now = performance.now()) {
    cancelAnimationFrame(frame);
    frame = 0;
    if (animationStarted !== null) {
      elapsed = Math.min(duration, elapsed + now - animationStarted);
      animationStarted = null;
    }
  }

  function resumeAnimation() {
    if (disposed || state !== 'animating' || document.hidden || !inView || animationStarted !== null) return;
    if (reducedMotion.matches || elapsed >= duration) {
      complete();
      return;
    }
    animationStarted = performance.now();
    frame = requestAnimationFrame(animate);
  }

  function begin() {
    if (disposed || state !== 'waiting' || !canCount()) return;
    pauseCountdown();
    setState('animating');
    if (reducedMotion.matches) {
      complete();
    } else {
      paint(0);
      resumeAnimation();
    }
  }

  function meaningfulActivity() {
    if (disposed) return;
    if (!root.isConnected) {
      cleanup();
      return;
    }
    if (!observer) updateInView();
    if (state !== 'waiting') return;
    pauseCountdown();
    remaining = idleDelay;
    resumeCountdown();
  }

  function visibilityChanged() {
    if (disposed) return;
    if (!root.isConnected) {
      cleanup();
      return;
    }
    if (document.hidden) {
      pauseCountdown();
      if (state === 'animating') pauseAnimation();
    } else if (state === 'waiting') {
      // Start a fresh clock baseline so hidden time never counts as idle.
      if (!observer) updateInView();
      resumeCountdown();
    } else if (state === 'animating') {
      resumeAnimation();
    }
  }

  function setInView(visible) {
    if (disposed || visible === inView) return;
    if (state === 'waiting') pauseCountdown();
    else if (!visible) pauseAnimation();
    inView = visible;
    if (state === 'waiting') resumeCountdown();
    else if (visible) resumeAnimation();
  }

  function updateInView() {
    if (disposed) return;
    const bounds = image.getBoundingClientRect();
    const width = Math.max(0, Math.min(bounds.right, window.innerWidth) - Math.max(bounds.left, 0));
    const height = Math.max(0, Math.min(bounds.bottom, window.innerHeight) - Math.max(bounds.top, 0));
    const area = bounds.width * bounds.height;
    setInView(area > 0 && width * height / area >= 0.1);
  }

  function imageLoaded() {
    imageReady = image.naturalWidth > 0;
    pauseCountdown();
    remaining = idleDelay;
    if (!observer) updateInView();
    resumeCountdown();
  }

  function imageFailed() {
    cleanup();
  }

  function motionPreferenceChanged() {
    if (state === 'animating' && reducedMotion.matches) complete();
  }

  activityEvents.forEach(type => document.addEventListener(type, meaningfulActivity, {capture: true, passive: true}));
  document.addEventListener('visibilitychange', visibilityChanged);
  image.addEventListener('load', imageLoaded);
  image.addEventListener('error', imageFailed);
  if (reducedMotion.addEventListener) {
    reducedMotion.addEventListener('change', motionPreferenceChanged);
  } else if (reducedMotion.addListener) {
    reducedMotion.addListener(motionPreferenceChanged);
  }

  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver(entries => {
      if (!root.isConnected) {
        cleanup();
        return;
      }
      const entry = entries[0];
      setInView(entry.isIntersecting && entry.intersectionRatio >= 0.1);
    }, {threshold: 0.1});
    observer.observe(image);
  } else {
    window.addEventListener('resize', updateInView, {passive: true});
    updateInView();
  }
  resumeCountdown();
})();
