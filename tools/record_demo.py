"""Record the demo video: drive the app, burn captions into the frames, no voice.

Run against a live server:

    python -m uvicorn collate.api:app --port 8080     # in one terminal
    python tools/record_demo.py                        # in another

Produces `demo/collate-demo.webm`, and `collate-demo.mp4` beside it when a
transcoder is available.

Why a script rather than a screen capture: the timings are the script, so a
caption that reads badly is a one-line edit and a re-run, and the take is
identical every time. The captions are injected into the page rather than
composited afterwards, which keeps everything in one file and means the text
scales with the recording.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo"
BASE = "http://127.0.0.1:8080"
W, H = 1600, 900

# The caption bar and a focus ring, injected before any page script runs so
# they survive navigation. Styled to match the application rather than sitting
# on top of it looking like a different product.
OVERLAY = """
window.__collate_overlay = () => {
  if (document.getElementById('cap-bar')) return;
  const css = document.createElement('style');
  css.textContent = `
    #cap-bar {
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 99999;
      background: rgba(28,26,22,0.94); color: #f7f4ee;
      padding: 18px 42px 20px; border-top: 2px solid #f7dd8a;
      font-family: "Iowan Old Style","Palatino Linotype",Georgia,serif;
      transition: opacity .35s ease; opacity: 0;
    }
    #cap-bar.on { opacity: 1; }
    #cap-main { font-size: 25px; line-height: 1.3; }
    #cap-sub  { font-family: "Cascadia Mono",Consolas,monospace; font-size: 13px;
                color: #b9b0a0; margin-top: 7px; letter-spacing: .02em; }
    #cap-main b { color: #f7dd8a; font-weight: 600; }
    .cap-focus {
      outline: 3px solid #f7dd8a !important;
      outline-offset: 2px !important;
      transition: outline-color .3s ease;
    }
    #cap-title {
      position: fixed; inset: 0; z-index: 99998; background: #f7f4ee;
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; text-align: center;
      font-family: "Iowan Old Style","Palatino Linotype",Georgia,serif;
      transition: opacity .5s ease;
    }
    #cap-title h1 { font-size: 62px; margin: 0; color: #1c1a16; font-weight: 600; }
    #cap-title p  { font-size: 22px; color: #6b6355; font-style: italic; margin: 10px 0 0; }
    #cap-title .n {
      font-family: "Cascadia Mono",Consolas,monospace; font-size: 14px;
      color: #857c6c; margin-top: 26px; letter-spacing: .12em;
    }
  `;
  document.head.appendChild(css);
  const bar = document.createElement('div');
  bar.id = 'cap-bar';
  bar.innerHTML = '<div id="cap-main"></div><div id="cap-sub"></div>';
  document.body.appendChild(bar);
};
window.__cap = (main, sub) => {
  window.__collate_overlay();
  const bar = document.getElementById('cap-bar');
  document.getElementById('cap-main').innerHTML = main || '';
  document.getElementById('cap-sub').textContent = sub || '';
  bar.classList.toggle('on', !!main);
};
window.__focus = (sel) => {
  document.querySelectorAll('.cap-focus').forEach(e => e.classList.remove('cap-focus'));
  if (!sel) return;
  const el = document.querySelector(sel);
  if (el) { el.classList.add('cap-focus'); el.scrollIntoView({block:'center', behavior:'smooth'}); }
};
window.__title = (h, p, n) => {
  window.__collate_overlay();
  let t = document.getElementById('cap-title');
  if (!t) { t = document.createElement('div'); t.id = 'cap-title'; document.body.appendChild(t); }
  t.innerHTML = `<h1>${h}</h1><p>${p}</p><div class="n">${n || ''}</div>`;
  t.style.opacity = '1';
};
window.__untitle = () => {
  const t = document.getElementById('cap-title');
  if (t) { t.style.opacity = '0'; setTimeout(() => t.remove(), 600); }
};
"""


class Take:
    """One recording, with a running clock so pacing problems are visible."""

    def __init__(self, page):
        self.page = page
        self.t0 = time.monotonic()

    def mark(self, label: str) -> None:
        print(f"  [{time.monotonic() - self.t0:5.1f}s] {label}")

    def cap(self, main: str, sub: str = "", hold: float = 0.0) -> None:
        self.page.evaluate("([m,s]) => window.__cap(m,s)", [main, sub])
        if hold:
            self.page.wait_for_timeout(int(hold * 1000))

    def focus(self, selector: str | None) -> None:
        self.page.evaluate("s => window.__focus(s)", selector)

    def wait(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    _EASE = """([sel, y, d]) => {
      const el = sel ? document.querySelector(sel) : document.scrollingElement;
      if (!el) return;
      const start = el.scrollTop, delta = y - start, t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / (d * 1000));
        const e = k < 0.5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;   // easeInOutQuad
        el.scrollTop = start + delta * e;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }"""

    def scroll_to(self, selector: str | None, y: int, seconds: float = 0.9) -> None:
        """Ease a specific scroller to an absolute position.

        Absolute rather than relative, and JS rather than mouse.wheel, because
        the wheel scrolls whatever happens to be under the cursor - which during
        the first recording meant a caption pointed at a footnote that was still
        six hundred pixels below the fold.
        """
        self.page.evaluate(self._EASE, [selector, y, seconds])
        self.page.wait_for_timeout(int(seconds * 1000) + 160)

    def pane(self, y: int, seconds: float = 0.9) -> None:
        self.scroll_to("#evidence", y, seconds)

    def page_scroll(self, y: int, seconds: float = 0.9) -> None:
        self.scroll_to(None, y, seconds)


def run() -> Path:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.webm"):
        old.unlink()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(OUT),
            record_video_size={"width": W, "height": H},
        )
        ctx.add_init_script(OVERLAY)
        page = ctx.new_page()
        t = Take(page)

        page.goto(BASE + "/", wait_until="networkidle")
        page.evaluate("window.__collate_overlay()")

        # -- title ---------------------------------------------------------
        page.evaluate(
            "() => window.__title('Collate',"
            "'a critical apparatus for documents',"
            "'6 DOCUMENTS &nbsp;·&nbsp; 511 PAGES &nbsp;·&nbsp; 1,967 CLAIMS &nbsp;·&nbsp; 704 FINDINGS')"
        )
        t.wait(4.2)
        page.evaluate("window.__untitle()")
        t.wait(0.8)
        t.mark("title done")

        # -- premise -------------------------------------------------------
        t.cap(
            "Six documents disagree about the same numbers. Which disagreements are <b>real</b>?",
            "Delhivery filings and three institutional reports on the Indian economy",
            5.0,
        )
        t.cap(
            "A fact is not just a value. It carries a <b>period</b>, a <b>basis</b> and <b>units</b> — "
            "and when two facts differ, the thing that differs <b>is the explanation</b>.",
            "frame = entity + measure + period + basis + unit",
            6.5,
        )
        t.focus(".highlights")
        t.cap(
            "Four findings, one of each kind.",
            "each card is a live query, not a fixture",
            3.0,
        )
        t.focus(None)
        t.mark("premise done")

        # -- case 1: corroboration ----------------------------------------
        page.click(".hl >> nth=0")
        page.wait_for_timeout(900)
        t.cap(
            "<b>Agreement.</b> The annual report prints ₹85,942.34 million. "
            "The earnings deck prints 8,594 under ₹ Cr.",
            "case 1 — corroborated across documents, expressed differently",
            6.0,
        )
        t.cap(
            "Those two strings share almost no characters. <b>They are the same number.</b>",
            "UNIT_SCALE — both normalise to 85,942,340,000 INR",
            5.0,
        )
        t.pane(560)
        t.cap(
            "Every claim is anchored to a rectangle on a real page. "
            "If the quote cannot be found again, <b>it is thrown away</b>.",
            "the highlighted box is the sentence the claim came from",
            6.0,
        )
        t.pane(1150)
        t.wait(2.2)
        t.mark("case 1 done")

        # -- case 3: explained by basis -----------------------------------
        t.pane(0, 0.4)
        page.click(".hl >> nth=1")
        page.wait_for_timeout(900)
        t.cap(
            "<b>Explained.</b> 8,123 against 10,077 — twenty-four percent apart, "
            "same company, same year, same units.",
            "case 3 — an apparent contradiction, resolved",
            6.0,
        )
        t.focus("table.frame-diff tr.differs")
        t.cap(
            "One is <b>standalone</b>, the other <b>consolidated</b>. "
            "The highlighted row is the reason.",
            "BASIS_MISMATCH — different figures by construction, not rival measurements",
            6.5,
        )
        t.focus(None)
        t.mark("case 3 done")

        # -- case 2: real contradiction ------------------------------------
        t.pane(0, 0.4)
        page.click(".hl >> nth=3")
        page.wait_for_timeout(900)
        t.cap(
            "<b>A real conflict.</b> The Economic Survey says net FDI was $10.1bn. "
            "The RBI says $26.8bn. Same year, same units.",
            "case 2 — nothing in either frame accounts for the gap",
            6.5,
        )
        t.pane(430)
        t.cap(
            "Rules decide the verdict. The model is asked only about conflicts, "
            "and its note sits <b>beside</b> the verdict — never replacing it.",
            "here it agrees: the quotes do not reconcile",
            6.0,
        )
        t.pane(1130, 1.4)
        t.cap(
            "But look at the page. <b>Footnote 95 defines net FDI as inward minus outward.</b> "
            "The answer was in the document, just outside the span we anchored.",
            "the sharpest statement of where this system currently stops",
            7.5,
        )
        t.wait(1.6)
        t.mark("case 2 done")

        # -- case 4: failures ----------------------------------------------
        page.goto(BASE + "/quarantine", wait_until="networkidle")
        page.evaluate("window.__collate_overlay()")
        t.wait(0.6)
        t.cap(
            "<b>What it threw away.</b> 391 claims rejected — every quote must be "
            "found again on its page, character for character.",
            "case 4 — failures, measured rather than claimed",
            6.0,
        )
        t.page_scroll(430)
        t.cap(
            "The RBI report first anchored at <b>39%</b>. The rejected quotes read as "
            "perfect prose that was not on the page.",
            "the cause was two-column layout — the columns were being read interleaved",
            6.5,
        )
        t.cap(
            "Fixing the reading order took it to <b>71%</b>. "
            "The gate had been catching all of it.",
            "24 of the earnings deck's 27 pages hold their numbers inside charts — counted, not guessed",
            6.0,
        )
        t.mark("case 4 done")

        # -- generality -----------------------------------------------------
        page.goto(BASE + "/", wait_until="networkidle")
        page.evaluate("window.__collate_overlay()")
        t.wait(0.6)
        page.click("text=Real GDP growth")
        page.wait_for_timeout(1100)
        t.cap(
            "One subject, twenty readings, three institutions — "
            "the Survey, the RBI and the IMF, side by side.",
            "grouped by subject, ordered by period",
            6.0,
        )
        t.pane(620)
        t.cap(
            "Nothing is hardcoded. The vocabulary starts <b>empty</b> — "
            "852 subjects, all learned from the documents.",
            "pointed at this assignment's own PDF it returned: doc_type 'hiring assignment', publisher 'Superjoin'",
            6.5,
        )
        t.pane(0, 0.4)
        t.focus("#dropzone")
        t.cap(
            "Drop any PDF here and it joins the apparatus.",
            "ingest runs in the background and reports each stage",
            4.5,
        )
        t.focus(None)
        t.mark("generality done")

        # -- close -----------------------------------------------------------
        page.evaluate(
            "() => window.__title('Collate',"
            "'rules decide · the model explains · every claim anchored',"
            "'github.com/Dexter722005/collate')"
        )
        t.wait(4.5)
        t.mark("closing card")

        page.wait_for_timeout(400)
        video = page.video
        ctx.close()
        browser.close()
        path = Path(video.path())

    final = OUT / "collate-demo.webm"
    if path != final:
        shutil.move(str(path), final)
    return final


def transcode(webm: Path) -> Path | None:
    """WebM plays everywhere that matters, but MP4 is the safer thing to hand in."""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    mp4 = webm.with_suffix(".mp4")
    cmd = [exe, "-y", "-i", str(webm), "-c:v", "libx264", "-preset", "medium",
           "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  transcode failed:", (r.stderr or "")[-400:])
        return None
    return mp4


if __name__ == "__main__":
    print("recording (the app is driven live; this takes about as long as the video)...")
    started = time.monotonic()
    webm = run()
    print(f"\nwebm: {webm}  ({webm.stat().st_size / 1e6:.1f} MB, {time.monotonic() - started:.0f}s)")
    mp4 = transcode(webm)
    if mp4:
        print(f"mp4 : {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
    else:
        print("mp4 : not produced (no transcoder) - the webm uploads fine to YouTube and Drive")
    sys.exit(0)
