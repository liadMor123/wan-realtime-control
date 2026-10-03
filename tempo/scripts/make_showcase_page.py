#!/usr/bin/env python3
"""Build the showcase web page from results/showcase/manifest.json (scripts/build_showcase_manifest.py).

  make_showcase_page.py <showcase dir>   -> <showcase dir>/index.html (media/ referenced relatively)
"""
import html
import json
import os
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEMPO, "results", "showcase")
man = json.load(open(os.path.join(D, "manifest.json")))

PAGE = r"""<title>Tempo Showcase</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Schibsted+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F3F4F1; --surface:#FFFFFF; --ink:#16191B; --muted:#5D666B; --line:#D8DCD6;
  --on:#0E7C66; --on-soft:#CFE9E1; --off:#E4E7E2; --b0:#6B7378; --chip:#EEF1EC; --head:#16191B;
  --display:"Bricolage Grotesque",ui-sans-serif,system-ui,sans-serif;
  --body:"Schibsted Grotesk",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --ground:#0F1213; --surface:#161A1C; --ink:#E6EAE7; --muted:#97A19E; --line:#2A3133;
  --on:#3CC2A0; --on-soft:#173A32; --off:#22282A; --b0:#9AA3A8; --chip:#1D2325; --head:#E6EAE7; color-scheme:dark;}}
:root[data-theme="dark"]{
  --ground:#0F1213; --surface:#161A1C; --ink:#E6EAE7; --muted:#97A19E; --line:#2A3133;
  --on:#3CC2A0; --on-soft:#173A32; --off:#22282A; --b0:#9AA3A8; --chip:#1D2325; --head:#E6EAE7; color-scheme:dark;}
body{background:var(--ground);color:var(--ink);font:400 15px/1.55 var(--body);}
.wrap{max-width:1120px;margin:0 auto;padding-inline:20px;padding-block:40px 64px;}
header{display:grid;gap:14px;max-width:760px;margin-bottom:28px}
h1{font:700 clamp(30px,5vw,46px)/1.05 var(--display);letter-spacing:-.01em;margin:0;color:var(--head);text-wrap:balance}
.lede{color:var(--muted);margin:0;max-width:65ch}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:22px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:6px}
nav{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--ground);border-bottom:1px solid var(--line);
  display:flex;gap:4px;flex-wrap:wrap;padding-block:10px;margin-bottom:8px}
nav a{font:500 13px/1 var(--body);color:var(--muted);text-decoration:none;padding:8px 12px;border-radius:999px}
nav a:hover,nav a:focus-visible{color:var(--ink);background:var(--chip);outline:none}
section{padding-top:28px}
h2{font:600 22px/1.2 var(--display);margin:0 0 4px}
.sec-note{color:var(--muted);margin:0 0 18px;font-size:14px;max-width:70ch}
.ex{border-top:1px solid var(--line);padding-block:22px;display:grid;gap:12px}
.ex-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 16px;align-items:baseline}
.ex-label{font:600 15px/1.3 var(--body)}
.src{font:400 12px/1.3 var(--mono);color:var(--muted)}
.prompt{margin:0;font-size:14px;color:var(--ink);max-width:90ch}
.prompt b{font-weight:600;color:var(--on)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:640px){.pair{grid-template-columns:1fr}}
figure{margin:0;display:grid;gap:6px}
video{width:100%;max-width:100%;aspect-ratio:832/480;background:#000;border-radius:6px;display:block}
figcaption{display:flex;justify-content:space-between;gap:8px;font-size:13px;color:var(--muted)}
figcaption .arm{font-weight:600;color:var(--ink)}
figcaption .arm.l{color:var(--on)}
.acc{font:500 12px/1 var(--mono);background:var(--chip);border-radius:4px;padding:4px 6px;color:var(--ink);white-space:nowrap}
.strip{display:grid;gap:4px}
.track{position:relative;display:grid;grid-template-columns:repeat(81,1fr);gap:1px;height:14px;cursor:pointer}
.track span{background:var(--off);border-radius:1px}
.track span.on{background:var(--on)}
.track .ph{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);left:0;pointer-events:none}
.track-lbl{display:flex;justify-content:space-between;font:400 11px/1 var(--mono);color:var(--muted)}
.ctl{display:flex;align-items:center;gap:10px}
button{font:500 13px/1 var(--body);color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:6px;padding:8px 12px;cursor:pointer}
button:focus-visible{outline:2px solid var(--on);outline-offset:2px}
.clock{font:400 12px/1 var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
footer{margin-top:40px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px;display:grid;gap:8px;max-width:80ch}
footer a,.lede a{color:var(--on)}
@media (prefers-reduced-motion: reduce){video{animation:none}}
</style>
<div class="wrap">
<header>
  <h1>When the object appears</h1>
  <p class="lede">Each row runs the <em>same prompt with the same seed</em> twice: plain Wan2.1 (or Self-Forcing), and the same
  model with our forward-only per-frame attention bias <b>L</b> (β = γ = 2). Nothing else differs. The prompts are the
  ones on <a href="https://shira-schiber.github.io/TempoControl/" target="_blank" rel="noopener">TempoControl's project page</a>.</p>
  <div class="legend"><span><i style="background:var(--on)"></i>frames where the object should be visible</span>
  <span><i style="background:var(--off)"></i>frames where it should not</span>
  <span>Accuracy = official TempoControl metric for that video (share of sampled frames correct)</span></div>
</header>
<nav aria-label="Sections">__NAV__</nav>
__SECTIONS__
<footer>
  <p><b>What this is.</b> Seed 42 for every video. B0 is the unmodified model; L adds +2 to the object's text tokens in
  cross-attention on frames where it should be visible and −2 where it should not, in the conditional branch only. For two objects,
  the static object gets +2 on every frame. No extra steps, no gradients, +0.9 % time.</p>
  <p><b>Provenance.</b> Single-object prompts dog (2nd), umbrella, skateboard and apple are verbatim benchmark prompts; their videos are the
  study's step-4 (Wan) and step-3b (Self-Forcing) runs. "Dog, last second" is not in the benchmark and the two-object prompts use the
  website's wording (the benchmark file words them differently), so those were generated for this page with the same protocol.
  Six prompts and one seed are illustrative only; the benchmark results (e.g. +15.5 points on all 80 single-object prompts,
  +9.6 on all 82 pairs) are in the study's findings.</p>
  <p><b>Not shown.</b> TempoControl's own output videos are not downloadable from their page, so there is no third column, and
  their movement and audio examples need a method this study did not build.</p>
</footer>
</div>
<script>
(function(){
  const FPS=16, N=81;
  document.querySelectorAll('.ex').forEach(ex=>{
    const vids=[...ex.querySelectorAll('video')], master=vids[0];
    const ph=ex.querySelectorAll('.ph'), clock=ex.querySelector('.clock'), btn=ex.querySelector('button');
    const tick=()=>{
      const t=master.currentTime||0, d=master.duration||N/FPS, f=Math.min(N-1,Math.floor(t*FPS));
      ph.forEach(p=>p.style.left=(100*Math.min(t/d,1))+'%');
      clock.textContent='frame '+String(f).padStart(2,'0')+' / '+(N-1)+'  ·  '+t.toFixed(2)+' s';
      for(let i=1;i<vids.length;i++){ if(Math.abs(vids[i].currentTime-t)>0.06) vids[i].currentTime=t; }
      if(!master.paused) requestAnimationFrame(tick);
    };
    const play=()=>{vids.forEach(v=>{const p=v.play(); if(p) p.catch(()=>{});}); btn.textContent='Pause'; requestAnimationFrame(tick);};
    const pause=()=>{vids.forEach(v=>v.pause()); btn.textContent='Play'; tick();};
    btn.addEventListener('click',()=> master.paused?play():pause());
    master.addEventListener('loadedmetadata',tick);
    ex.querySelectorAll('.track').forEach(tr=>tr.addEventListener('click',e=>{
      const r=tr.getBoundingClientRect(), x=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
      vids.forEach(v=>v.currentTime=x*(master.duration||N/FPS)); tick();
    }));
    if('IntersectionObserver' in window){
      new IntersectionObserver(es=>es.forEach(en=>{ if(en.isIntersecting) play(); else pause(); }),{threshold:0.35}).observe(ex);
    }
  });
})();
</script>
"""


def frames_on(mask):
    return [int(mask[(k + 3) // 4]) == 1 for k in range(81)]


def track(mask, label):
    cells = "".join('<span class="on"></span>' if o else "<span></span>" for o in frames_on(mask))
    first = next((k for k, o in enumerate(frames_on(mask)) if o), None)
    lbl = f"{html.escape(label)}: visible from frame {first} ({first / 16:.2f} s)" if first not in (None, 0) else \
          f"{html.escape(label)}: visible throughout"
    return (f'<div class="strip"><div class="track" title="click to seek">{cells}<div class="ph"></div></div>'
            f'<div class="track-lbl"><span>{lbl}</span><span>5.06 s</span></div></div>')


def highlight(prompt, words):
    out = html.escape(prompt)
    for w in sorted({w for w in words if w}, key=len, reverse=True):
        out = out.replace(html.escape(w), f"<b>{html.escape(w)}</b>")
    return out


nav, secs = [], []
model_b0 = {"Wan2.1-T2V-1.3B": "Wan 2.1, no control", "Self-Forcing": "Self-Forcing, no control"}
for s in man["sections"]:
    nav.append(f'<a href="#{s["id"]}">{html.escape(s["title"])}</a>')
    note = {"wan-one": "Single object: the object should appear only in the stated second.",
            "sf-one": "The same four prompts in Self-Forcing, Wan2.1 distilled to stream chunk by chunk in real time; "
                      "L is applied per chunk. Gradient-based methods cannot run here.",
            "wan-two": "Two objects: the first object should be there from the start, the second only in the second half."}[s["id"]]
    rows = []
    for e in s["entries"]:
        b0, l_ = e["arms"]["b0"], e["arms"]["l"]
        tracks = track(e["mask"], e["temp_object"])
        if e.get("static_object"):
            tracks = track([1] * 21, e["static_object"]) + tracks
        rows.append(f"""<article class="ex">
  <div class="ex-head"><span class="ex-label">{html.escape(e['label'])}</span><span class="src">{html.escape(e['source'])}</span></div>
  <p class="prompt">“{highlight(e['prompt'], [e['temp_object'], e.get('static_object')])}”</p>
  <div class="pair">
    <figure><video src="{b0['file']}" muted loop playsinline preload="metadata"></video>
      <figcaption><span class="arm">{model_b0[s['model']]}</span><span class="acc">accuracy {b0['accuracy']:.2f}</span></figcaption></figure>
    <figure><video src="{l_['file']}" muted loop playsinline preload="metadata"></video>
      <figcaption><span class="arm l">with L (ours)</span><span class="acc">accuracy {l_['accuracy']:.2f}</span></figcaption></figure>
  </div>
  {tracks}
  <div class="ctl"><button type="button">Play</button><span class="clock">frame 00 / 80</span></div>
</article>""")
    secs.append(f'<section id="{s["id"]}"><h2>{html.escape(s["title"])}</h2><p class="sec-note">{note}</p>{"".join(rows)}</section>')

open(os.path.join(D, "index.html"), "w").write(PAGE.replace("__NAV__", "".join(nav)).replace("__SECTIONS__", "".join(secs)))
print("wrote", os.path.join(D, "index.html"))
