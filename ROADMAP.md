# StromaSense Web — Build Roadmap (COMPRESSED — portal closes Sep 6, 11:59 PM)

**Real deadline: the submission portal (repo + slides + write-up) closes
Sep 6, 11:59 PM.** Sep 10 is the Grand Finale presentation date, not the
submission deadline — don't confuse the two. As of writing this, the build
had not started yet. That means everything below is checkpoint-based, not
day-based: one gate, then a branch, not a leisurely six-day plan.

**Frozen, not deleted:** the Model Card generator, latency display,
tooltips, hospital CTA, and the whole 3D-theming conversation. All of it
either already exists in this repo or was explicitly decided against for
this deadline. None of it gets touched again until the checkpoint below
passes. If you're reading this roadmap and thinking about visual polish
before the checkpoint is green, stop and go do the checkpoint.

The app *code* already exists — every page, model loading, Grad-CAM,
attention rollout, caching. This is not a from-scratch build. What's
unverified is whether your actual weights run through that code at all,
because the source notebook cell this was extracted from never had a saved
execution. That's the one real unknown standing between you and a working
submission, and it gets resolved in the next few hours, not gradually
across a week.

---

## Step 1 (do this now) — the one gate that decides everything else

- [ ] Pull the four patient-split weights off Drive: YOLO `.pt`, ResNet50
      classifier, ResNet50 feature-extractor, ViT feature-extractor,
      XGBoost.
- [ ] `pip install -r requirements.txt` into a fresh venv.
- [ ] Fill the `TODO`s in `src/config.py` with your real filenames (local
      paths are fine for now — you don't need HF Hub set up to test this).
- [ ] Run `python scripts/test_pipeline_standalone.py <one_known_image>`.

**Timebox this to 3 hours from when you start.** If it's producing a sane
prediction — even an ugly one — by then, go to Step 2. If it's still
broken and the bug isn't obviously a quick fix, stop debugging and jump
straight to the **Fallback path** below. Do not let this eat the whole
night; there is no time buffer left to spend on a single stubborn bug.

---

## Step 2 — if the pipeline works: minimal live app

- [ ] 6-10 real images per gallery (BreaKHis + BACH) into
      `assets/sample_images/`, manifests filled in (~45 min).
- [ ] `streamlit run app.py` locally, click every page, fix anything
      broken. Try It is the only page that must be perfect — the others
      just need to not crash (1-2 hrs).
- [ ] Push to GitHub, weights excluded per `.gitignore` (30 min).
- [ ] Attempt an HF Space deploy. **Hard budget: 1-2 hours.** If the build
      log is still fighting you past that, stop — screen-record the local
      app working instead and submit that recording alongside the repo. A
      video of something real beats a public URL still broken at 11 PM.
- [ ] Write-up: lift directly from `README.md` — the numbers, the
      limitations, the patient-split story are already written (30-60 min).
- [ ] Slides: minimal, reuse the metrics and the mockup screenshots from
      this conversation, don't design from scratch (1-2 hrs).
- [ ] Submit with real buffer before 11:59 PM — portal upload issues are
      not a good place to discover you had no slack left.

---

## Fallback path — if Step 1 doesn't pass in time, this is not a consolation prize

- [ ] Go back into the original Colab notebook — the environment this code
      is *proven* to run in — and re-execute the relevant cells to
      generate a handful of real predictions with real Grad-CAM and
      attention-rollout overlays. Screenshot or export them.
- [ ] Submission becomes: the cleaned-up notebooks, those exported result
      images, `README.md`, and a write-up that says plainly — given the
      timeline, this presents validated results from the working research
      pipeline rather than a live demo, with full code available for
      review.
- [ ] This is a legitimate, defensible entry. A live app still broken when
      the portal closes is a worse submission than this. Don't burn your
      last hours trying to force the live version through once the
      checkpoint has already told you it's not going to make it cleanly.

---

## Explicitly out of scope for this deadline

- Anything 3D — CSS tilt effects, WebGL, all of it, regardless of how it
  looked in the mockup comparison
- Open, unrestricted image upload as the primary interaction mode
- A hosted, always-on production service beyond the hackathon
- The three-way fine-tuning ablation
- Any language stronger than "research prototype" / "decision-support" /
  "not for clinical use"
- Any further scope discussion before Step 1's checkpoint is green
