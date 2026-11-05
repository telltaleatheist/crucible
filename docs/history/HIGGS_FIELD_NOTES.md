# Higgs Field Notes

The working reference for fine-tuning, serving and evaluating Higgs Audio v3 TTS voices (Mistborn, deathstalker,
Owen Morgan, Third Reich, sigma): corpus building, training, checkpoint picking, ladders, deployment, and the
BookForge / clipforge tools. Same rule as the Orpheus `TESTING_BIBLE.md`: measure, then narrate.

**Pruned 2026-09-25.** Superseded advice, stale state snapshots and retired-stack instructions were removed. The full
pre-prune text (§1-§8, §4n.1-§4n.95 with every experiment and retraction) is in git at commit **3f82feb** - use
`git show 3f82feb:docs/HIGGS_FIELD_NOTES.md` for history. Old § numbers are kept in parentheses on the headings that
absorbed them, and the map at the end resolves every old reference.

Companions: `HIGGS_MODEL_LEDGER.md` (per-run table), `docs/TEMPER.md` (post-corpus record + report server),
`docs/ASR_CALLSITES.md` (Whisper -> Qwen migration inventory). Campaign evidence: `E:\training\_campaigns\YYYY-MM-DD-<slug>\`.
Legacy campaign root (scripts still resolve through it): `E:\training\_campaigns\2026-09-01-cod-full-rebuild\higgs\`
(`night2/`, `night3/`, `night4/`, `v3_ft/`).

Confidence tags where they matter: **[HIGH]** measured with a control/reproduction; **[MED]** measured once or mechanism
inferred; **[LOW]** reported by others or one observation with an alternative explanation (Owen, 2026-09-04).

---

## 0. Standing rules (process)

- **One stage at a time on the GPU** (Owen: "one thing at a time so we dont thrash"). Every GPU consumer takes the lock,
  including Ollama (§6.2).
- **Nothing deploys to production, the Mac or HuggingFace without Owen's explicit go for that run.** Owen 2026-09-09:
  *"dont deploy to huggingface until i give an explicit green light - huggingface is our backup ... the old models are
  safe to erase from the mac/pc once we deploy."*
- **WSL is staging, never storage.** Owen: *"wsl as a staging area, never a storage area. corpuses are supposed to be
  removed after theyre trained"*; *"erase the model merges, keep the adapters"*. A merge is disposable only while its
  adapter survives (8 GB merge vs 252 MB adapter; HF repos are squashed, so the adapter is the real backup) (was §4n.81).
- **Campaign dirs are dated**: `E:\training\_campaigns\YYYY-MM-DD-<slug>\`, never `nightN` (Owen 2026-09-10: *"this
  project will be going for months"*). Legacy `night2/3/4` are not renamed because scripts resolve through them;
  clipforge reads the stage dir from `CLIPFORGE_CAMPAIGN_STAGE` (default `night4`). Run artifacts go in
  `higgs/runs/<YYYY-MM-DD_HHMM>_<voice>_<stage>/` with `run.json`; datasets and merged checkpoints deliberately did not
  move (was §4n.49, §4n.64.6).
- **Measure the thing the model learned from**: training clips or the master they were cut from - never upload parts,
  raw sources or pre-enhancer files (material discarded downstream produced several wrong conclusions, §4n.61.4).
- **A short run showing no defect is not evidence the defect is absent** (the tail runaway developed with training;
  a 135-step quick train "cleared" it and cost two full trains, §4n.53).
- **Assert on an enumeration, not the absence of a complaint**: count wavs/rows/files; an exit code is not a result
  (rsync exit 0 put 8 GB in a bogus dir; a ladder with 0 renders printed DONE; normalize_text exit 0 truncated at 64 KiB).
- **Cleanup manifests live outside the tree being deleted**; before deleting, grep `night4/` and chain scripts for the
  path (the 09-24 cleanup deleted `build/dn/*_dn_44k.flac` that `night4/speaker_refs/mb.json` still referenced, §4n.94).
- **Parse-check every edited script before commit/launch**: `ast.parse`, `node --check`, and `bash -n` run with WSL's
  own bash (§6.1).
- **Every file created outside a git repo is recorded in the file ledger** (Owen, 2026-10-02, all agents). The CLI is
  `C:/Users/tellt/AppData/Local/Programs/Python/Python311/python.exe C:/Users/tellt/.claude/file-ledger/ledger.py`.
  - Classify on creation: `set <path> --why "..." --expires 7d|2d|60d` or `--keep`.
  - Renders handed to Owen go to `E:\claude-files\outbox\<date>\` (7d); scratch goes to `E:\claude-files\scratch\`;
    backups go via `ledger backup` to the Mac's `/Volumes/Callisto/backups`.
  - Never leave files loose in ~, WSL ~, Projects, Desktop or Downloads.
  - Open with Owen: whether campaign dirs under `E:\training\_campaigns` stay the evidence home or move to
    `E:\claude-files\keep\`. For now they stay, ledgered with expiries.
  - The Mac's `/Volumes/Callisto/training` staging (Sep 7-8, 50 GB) was deleted 2026-10-02. Its titles experiment is
    at `/Volumes/Callisto/backups/2026-10-02-titles-headline27b-transcript-evidence`.

---

## 1. The engine (was §1, §4, §4c)

- **[HIGH] Model** `bosonai/higgs-audio-v3-tts-4b`: Qwen3 ~4B decoder (36 layers, hidden 2560, GQA 32/8), context
  **8,192 tokens**, Boson Higgs Audio v3 Research and Non-Commercial License (Owen: BookForge is personal use).
  Higgs Audio v2 was dropped 2026-09-04 (Owen: *"basically just Orpheus and we know Orpheus better"*).
- **[HIGH] Audio tokenizer**: 8 codebooks x 1024 (+ stream bos/eos 1024/1025), **25 frames/s**, delay pattern,
  **24 kHz mono**. Codec API trap: `higgs_v3_codec.encode()` returns `[T, 8]`, `decode()` expects `[8, T]` - transpose
  between them.
- **No voice presets**: zero-shot clone from a reference, or a fine-tune. v3 has NO `<|scene_desc_*|>` and no neutral
  emotion token; an unknown control token is read as words and wrecks the render (coverage 0.000).
- **Trainer is ours** (v3 ships none): `v3_ft/higgs_v3_model.py` + `train_lora.py`, from-scratch PyTorch LoRA over the
  Qwen3 backbone + Boson audio head; rows in v3 chat framing, text masked, loss on the 8 codebook heads + stream eos,
  delay pattern as at inference.
- **Sequence math**: 100 s of audio = 2,500 frames; `--max-seq-len 4096` holds ~2,100 chars and truncates nothing on
  22-100 s clips. (Orpheus trap: SNAC runs ~86 tok/s, so the same clip is ~8,600 tokens there.)
- **[HIGH] Serving needs MERGED weights** - no runtime LoRA for this model class. `v3_ft/merge_for_serving.py --lora
  <ckpt> --out <dir>` streams base shards and applies `W + (alpha/r) B A` to touched matrices only; CPU-only, measured
  **~50 s and 8.0 GB** per checkpoint (09-09; ~30 s on 09-19), so it runs beside a GPU ladder. It aborts if the merged
  matrix count differs from the adapter's A/B pairs (a silent name miss would serve the base model). `train_lora.py`
  writes `generation_config.override.json` into every run dir at creation (the merge requires it).
- **Loss does not grade behaviour.** A 0.04-nat holdout gap does not resolve a behavioural difference; the lowest-loss
  checkpoint was the pausy/looping one on mb_v7, ds_v7 and the T1 quicktrain (§3.8). Holdout loss is not comparable
  across corpora.

---

## 2. Sampling (was §4n.30, §4n.31, §4n.43, §4n.61.1)

- **Ladder/screen instrument: temperature 0.8 / top_p 0.95 / top_k 50**, one source of truth
  `pipeline/higgs_ladder.py HIGGS_SAMPLING` (Owen 2026-09-06: *"across the board, for everything everywhere. we
  shouldn't deviate from the default unless we have a very good reason"* - Boson's model-card example).
  `render_meta.json` records the sampling actually sent.
- **App (BookForge engine-level): 0.7** - Owen 2026-09-09 on re-rendered known-gibberish chunks: *"gibberish gets cut
  in half at 0.7. set it to be global for higgs renders. streaming and rendering both."* (bookforge 8e64648f; a
  measurement outranks a default). **On Crucible (checked 2026-09-25, `crucible/voices/mistborn.toml`)** both arms
  serve `sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }` as take 0, and 0.7 is the FIRST RETAKE rung
  (`[[voice.takes]] temperature = 0.7`: a chunk that failed at 0.8 is retried at a different setting, Owen's ruling
  2026-09-14). So served = 0.8 = what the ladder renders. **UNRESOLVED - ask Owen:** whether the app-level 0.7 above
  still applies anywhere now that renders go through Crucible.
- **Every request must send all three values**: SGLang-Omni applies NO top_k/top_p unless the request carries them
  (without them a chunk ran to the cap with 80 s of silence). The merged dir's `generation_config.json` stays at
  1.0/0.95/50/rep 1.0 as a server-side fallback; per-request sampling overrides it.
- **[HIGH] What temperature moves** (sweep 0.6-1.2, 98 renders/point, 2 voices; plus arms at 0.2/0.6/0.7): 0.6-0.9 is
  flat for early stops and run-ons; **1.0 is where babble and dropped spans begin**, so 0.9 is the highest safe value;
  0.2 is significantly worse (p=0.001; Owen: *"i dont want to go below 0.6"*). Temperature moves the BABBLE class, not
  the EARLY-STOP class (early stop is the duration prior, not sampling). Pace rises ~0.15 chars/s per 0.1 T, so pace
  must be measured at the served temperature.
- **repetition_penalty is refuted** and the per-voice `sampling` block was removed from `higgs-models.json`: mlx-audio's
  higgs_audio_v3 has no repetition penalty at all, yet the Mac reproduced mistborn's pause profile identically.
- Select test material for the failure being tested: a prose bank with zero babble cannot answer a babble question.

---

## 3. The recipe, end to end

Order (Owen 2026-09-25; updated 2026-10-09 by deathstalker v16, the deployed recipe of record
`recipes/deathstalker_v15sep/`): **source audio (Adobe website, located, brightened, THEN vocal-stem separated) ->
`clipforge book-text` (generate-sentences on Crucible -> cues corrected to the reader -> AI cleanup -> exclusions) ->
slice (rows <= 1,000 chars) -> mix (no bed, no pause cap) -> train -> pause screen (stall guard on) -> ladder -> band +
pace -> deploy (HF + both Crucible arms).** The Qwen row gate was dropped 2026-09-29; the bed is replaced by the stall
guard (§3.5, §3.5.2).

### Pipeline decisions (Owen, 2026-09-25) - the three rules every corpus follows

1. **Text comes from the EPUB, checked and corrected by Qwen ASR - never Whisper.**
   - ASR = **qwen3-asr-1.7b**, alignment = **qwen3-aligner**, both as Crucible jobs (*"abandon whisper ... for
     everything"*; *"all gpu use in bookforge should run through crucible"*).
   - **Never send silence to the ASR** (*"we should never send silence to the ASR tool"*): digital silence is cut out
     before the upload and word times mapped back (Qwen has no VAD and invents speech over silence).
   - The EPUB supplies the words; the ASR LOCATES and CHECKS them. **When the reader diverges, the reader wins**
     (*"if it comes down to picking between ASR and the book, we should pick the ASR"* - meaning dropped clauses,
     paraphrase, additions, edition differences: *"correct the vtt so it reflects the real audio so we arent losing
     training data"*). **When it is only a spelling the ASR got wrong, the EPUB wins** - names and near-miss words
     ("Elend" not "Ellen", "shattering" not "shuddering"; Owen: *"the epub should win on shattering/shuddering"*).
   - Every run writes `discrepancies.json` (inserts, omissions, paraphrase, pace outliers, non-speech, music under
     speech, not-in-recording, the corrections made, barely-matched cues). This is standard book prep, not a
     training-only step (*"a normal part of the process of prepping a book"*).
2. **Training text = render text: the AI cleanup runs before training, always.** (*"make ai cleanup a part of the
   process before we train, so what goes in matches what will come out"*.) The same triaged Foundry cleanup the render
   applies (triage on qwen3.5-9b, then the cleaner on what it flags), over the corrected VTT, before slicing. A corpus
   whose text did not go through it is not training-ready.
3. **The corpus is cut for the band we serve: 400-1,000 characters, nothing longer than ~1,000.** (*"setting our
   longest chunk to 1000 characters rather than 1600 ... focus on the band we'll be using, which will be 400 - 1000 ...
   above 1000, it starts losing prosody anyway"*.) `slice_vtt --long-max-chars 1000 --run-median-s 45`; the short
   tier (single sentences) still covers below 400. The ladder may still MEASURE to 1,600 - that is its range, not the
   corpus length.

One verb does rules 1-2 and the exclusions: `clipforge book-text` (§4).

**Correcting to the reader, as ruled by ear (Owen's two spot checks, 2026-09-26).** The ASR stays the default for
what the reader said: *"we should stick to the ASR when possible because of book version differences and
paraphrasing"*. A one-word ASR choice among near-homophones ("in" / "an" / "and", "is" / "as") is "a fully defensible
choice" even where Owen would pick the book's word. What the correction must NOT do, each fixed in bookforge:
- replace a NAME with a mishearing ("Chantal" -> "Gentile"): book names (capitalised mid-sentence, never lower-case) are
  never replaced (4f580564);
- write digits for spoken numbers ("nine thousand" -> "9000"): numbers compare by value (4f580564);
- drop a sentence's first or last 1-2 words the ASR missed ("If neither..."): not applied; longer dropped clauses still
  follow the reader (4f580564);
- leave a period after an expanded abbreviation ("Saint. Stephen's", "Lieutenant. Robin"): the SPOKEN word with no
  period - "Matt." -> "Matthew", "St." -> "Saint" (45f1518e);
- cut off or orphan the reader's trailing additions ("Ephesians 5 verse 21"): up to 4 trailing words belong to the
  sentence (4f580564) - but never the NEXT sentence's opening words ("...1994. He", "Hurry. I") (45f1518e).
**The correction finds PARAPHRASE; it does not supersede the book** (Owen 2026-09-27: "the books proper nouns and
unusual words should be trusted. So this will really be about finding paraphrasing, not superseding the book").
Book names and the book's unusual words (lower-case, >= 6 letters, used <= 2 times in the whole book) are never
replaced. A SECOND OPINION from a different ASR family (whisper-large-v3-turbo, bookforge 91b8fac4 + the region vote)
re-hears every cue Qwen still changes, and votes per REGION (a run of Qwen's changes): if Whisper's words over that
stretch are at least as similar to Qwen's as to the book's, Qwen's version stands (a real departure; ties to Qwen, whom
Owen trusts more); if closer to the book, Qwen misheard and the book keeps it. Owen's worked case: book "Check table 21",
Qwen "the back of the book", Whisper "the back of a book" -> Qwen's version whole (not a splice). Every vote, with its
similarities and margin, lands in discrepancies.json; the close calls feed the review page. Live test + scoring against
Owen's 30 judged clips (`2026-09-26-ds-v14/spotcheck/answer_key.json`) when the GPU is free.
Known gap: the AI cleanup occasionally leaves a year as digits ("replaced in 1994") - a Foundry cleanup issue, minor.
Spot-check sampler: `2026-09-26-ds-v14/spotcheck.py` - 10 random corrections + 5 kept deletions from TRAINED cues only,
cut from the training master with book vs trains-on text; Owen listens before a corpus is sliced.

**Dialogue is per voice** (Owen 2026-09-26): nonfiction keeps its dialogue - Celebration of Discipline, McKinley,
Leadership Pipeline, every Third Reich book (`book-text --dialogue keep`); fiction narrators exclude it (Mistborn
prose-only, Marked Man never, 3000 Degrees) - the default `--dialogue exclude`. **Deathstalker (Owen 2026-10-09): no
narrator fiction dialogue in ANY book** - *"He does Australian character voices ... They sound like different people
entirely. It was tested and makes the model measurably worse."* Never propose dialogue data to fix deathstalker's
emphasis.

### 3.1 Source audio (was §4n.93, §4n.95, §4n.35.2-4, §4n.22, §4n.19, §4n.57, §4n.92)

- **Enhancer = the Adobe Podcast WEBSITE only.** Owen 2026-09-24: *"premiere files are unnecessary now"*, *"keep the
  adobe web files when we find them"*. Settings never change between batches: **Enhance Speech v2 70 %, background 0,
  music 0** (return names end `-esv2-70p-bg-m-music-m.wav`); a settings change brings back the timbre step. Budget
  ~4 h/day (4 x 59-min parts).
- **Consistent source is the rule.** Owen 2026-09-08 after mb_v3 (Adobe + denoised mixed, 27 h): *"prosody is good but
  the voices/clips we used definitely werent normalized via adobe. i can hear quality differences between them."* More
  data does not fix a mixed-source corpus. And: *"Higgs will normalize the background audio and it'll normalize the
  speaker audio to a point, but if they're too different between corpus sources, it won't matter and it'll bleed
  through. Adobe normalization is still the preferred method for dramatically different audio qualities."*
- **Quality over hours.** Owen: *"14 - 18 hours is fine. having high quality audio is far superior to having lots of
  poor quality audio."* Evidence: deathstalker v13 (one consistent enhanced sound, clean cuts, no invented sounds, no
  music) - *"the prosody is better than any other so far"* - after corpus size and RVC had been ruled out as the lever
  (v13 changed several things at once, so which fix mattered most is not isolated).
- A normal clean recording needs no pre-cleanup (Third Reich tr_v3 was trained on raw decodes - no denoise, no Adobe, no
  bed); egregiously bad audio still gets cleaned.
- **Upload regions start and end at a pause centre** (`build_adobe_upload.py`, 111f340): edges come from the slicer's
  silence rule (SIL_DB 10, CUT_DB 6, MIN_PAUSE_S 0.2, floor = max(p5, p95 - 55) over a WIDE 10 s context - a narrow
  window counts speech as floor); an edge cue with no pause is trimmed off; long regions split at their longest pause;
  `--lead-s` is retired. Audit with `pipeline/enhance/region_edges.py` (HoA region starts on sound 95.2 % -> 0.1 %).
  FE's 9-11 % "on sound" flags are its noisy raw pauses: 0 of 3,046 FE edges within 20 dB of speech.
- **One parts folder per voice** (Owen 2026-10-09: *"consolidate all parts into a single folder, same as deathstalker"*):
  - deathstalker: `2026-10-02-ds-adobe-full/parts/`.
  - Mistborn: `2026-09-25-mb-adobe-full/parts/`, holding all 64 sent parts and 51 returns from five batches. Old
    folder names are mapped in `parts/PATHS.md`.
  - Any return at Enhance Speech v2 70 % / background 0 / music 0 is usable whatever its age (Owen: *"they're safe to
    use, even if they're older"*).
  - Mistborn still to send (10-09): WoA 09-12, HoA 01-07, AoL 01-02 = 13 parts, 12.5 h.
  - **Live span files were repointed to `parts/`** (12 files; every wav reference checked to exist). The first pass
    missed backslash paths: the Bash tool's heredoc turned `\` into `\` in the regex (§6.1 trap). The fix was
    written with the Write tool.
- **Mistborn v18 (started 2026-10-09, Owen: "go ahaed. gpu is done"):** the deathstalker v16 formula on all Mistborn
  returns. Campaign `2026-10-09-mb-v18`.
  - **Step 1, FE (complete on Adobe):** `separate_mb_masters.py fe` (tmux mbsep). It is the ds script with Mistborn
    paths, run locally rather than through Crucible's denoise job: same checkpoint, and 1,619 spans would mean 1,619
    upload-and-job round trips.
  - `after_fe.sh` (detached) measures the floor, then re-runs FE book-text on the current BookForge into `text/fe`.
  - WoA/HoA/AoL wait for Owen's 13 parts. Their 09-28 VTTs do not cover post-09-28 audio, so book-text is re-run after
    the masters are rebuilt.
  - **FE done 2026-10-09 17:42; ready for prep.**
    - Separated 15:14-16:48. The paired-frame floor dropped 11.7 dB.
    - book-text on bookforge edae284d: 9,452 cues, 226 corrected, 1,349 excluded (dialogue 160, barely matched 1,177,
      pace 12).
    - The excl_v15 rule keeps 9.27 h inside website audio (v17 text: 9.28 h).
    - Text: `2026-10-09-mb-v18/text/fe/fe_clean.vtt` + `fe_exclude_ids.txt`. Not trained (Owen: wait for tonight's
      parts).
  - **2026-10-11, the rest (Owen: "prep the new parts ... tonight at midnight ill probably pull those last two from
    adobe web and we can train overnight").** All returns are in except hoa part 03 and aol part 01. Scripts in the campaign:
    - `prep_book.sh <bk>`: build_web_master (old spans + every map batch of the book) -> brighten.py <bk> ->
      separate_mb_masters.py <bk> (clears `sep_work/<bk>` first: span files are index-named, so a rebuilt master must
      never reuse them) -> floor -> book-text on the un-brightened located master. Old masters go to `pre_v18/`.
      WoA (complete, 12/12) launched 13:30.
    - `prep_v18.sh slice <bks>` / `merge`: the ds v16 prep (excl_v15, v16 slice rule, cut-audit) -> mb18_l / mb18_s in
      `E:/training/mistborn/build19_v18`. FE + WoA slice after WoA's text.
    - `tonight.sh` (detached) waits for the two returns (parts/ or Downloads, full length within 50 ms, not growing),
      runs HoA + AoL, slices, merges -> PREP.DONE.
    - `mb_v18_chain.sh` (tmux mbv18) = ds_v16_chain.sh by sed: no bed, no pause cap, r32/a64, screen bank_mb_len, best
      pauser + latest clean read Mutineers' Moon ch. 8. GO.marker set on Owen's "train overnight".
    - Paused 10-11 ~13:40 (GPU taken). `prep_book.sh <bk> cpu` added: stops after locate + brighten (CPU only) and
      sets `<BK>.CPU_DONE.marker`; a plain rerun resumes at separation. WoA CPU done 14:02 (2,385 located, 0 dropped,
      16.67 h website audio; brightness -14.50 -> -8.88 dB).
    - 10-12 00:11, all 64 returns in: `overnight.sh` replaces tonight.sh. It runs the HoA + AoL CPU steps, waits for 01:30
      and an idle card (util < 20 %, < 6 GiB, 3 checks), then separation + book-text for woa/hoa/aol, slicing for all
      four books and the merge. Chain relaunched in tmux mbv18 with GO. `ladder_v18.sh` (tmux mbv18l, ds ladder_v16 by sed,
      band 300-800) waits for the chain's ALL_DONE: screen -> ladder top 2 -> pick (Owen 00:15: "prosody gains").
    - CPU steps 00:40: HoA 2,682 located, 17 dropped (weak match, peak 0.40-0.50; nothing misplaced), 17.56 h website
      audio, -17.97 -> -9.36 dB; AoL 960 located, 1 dropped, 5.98 h, -15.44 -> -9.34 dB.
    - 01:12 Owen "GPU is free. Go ahead and start": overnight.sh stopped at its 01:30 wait, `overnight_gpu.sh` runs its
      steps 3-4 now. **separate_mb_masters.py loaded the whole master + a same-size output buffer as float32** (~45 GB
      for WoA/HoA's 28-29 h timelines; FE's 24.6 h only squeezed through on swap). WoA sat in D state at 12/12 GB WSL.
      Fixed: spans are read with seek, the output FLAC is written sequentially (zeros between spans, the source's gaps
      still peak-checked), memory flat. Smoke-tested on a synthetic master (exact length, aligned). Old copy `.pre_stream`.
    - PREP.DONE 07:51 (slice: FE 9.3 h, WoA 11.9, HoA 15.1, AoL 4.3; cut-audit <= 5 flagged per book; merge 2770 + 2487
      rows). **The mix exited 3 at 07:52 on `--served-cap-chars 700 --served-cap-strict`**: v18 rows p75 = 692 (v16 ~709),
      24.1 % of rows at/above 700. The check drops no rows and Mistborn serves 800 (ladder-measured), so a value copied
      from v16 stopped the run; the GPU sat idle ~2 h. Owen 09:50: "Just wait until tonight" - train held, chains exited.
    - **10-13 run** (relaunch_train.sh 00:06, served-cap check removed): mix 3,242 rows / 34.0 h, leakage PASS. Train
      00:21, 3.53 s/step, EARLY STOP at 3610 (best holdout 3.5416 @ 3040). Screen (bank_mb_len, 19 ckpts, all screened):
      **0 clean** - defects 0-2 but pausing scores 14-63 (< 70): pauses long, median 1.0-1.9 s vs the corpus 0.76.
      Same shape as the 10-01 no-bed A/B (3-37); mb v16 (bed) scored 60-85, ds v16 (no bed) 81-85. Picks: ckpt-1900.
      Ladder (300-800, 168 renders each, stall guard on): 1520 score 55.6, 5 trunc (3.0 %); 1900 50.0, 9 trunc (5.4 %);
      inside noise -> pick ckpt-1900. vs the deployed v16 3375 ladder (09-26): score 59.2, 19 trunc (9.0 %), median
      0.82 s -> v18 truncates less, pauses ~0.2 s longer. Sample samples/mb_v18_1900_mutineers_moon_ch8.wav.
    - **Ear (10-13):** Owen, ch. 8 side by side with v16 (guard on): "prosody sounds about the same". v18 went high
      (a Vin-like character register) on ONE piece, the dialogue opener MM01 (f0 p95 122 Hz vs v16 90); the rest
      matched. A second, dialogue-dense passage (`mm_dialogue_bank.json` MD01-08, `render_dialogue.sh`) showed no
      spike (v18 p95 85-103 Hz vs v16 87-108). Quoted speech in v18 rows is ~0.3 % (9 rows, letters/epigraphs), same
      as v16 - not the leak. Italic text (unquoted thoughts) rose: HoA italic share 7.5 -> 11.7 %, rows with >= 40
      italic letters 21 -> 28 % (WoA, HoA). Suspect: Vin's italic thoughts voiced in character in the new parts.
      Open fix (not run): f0-scan the training clips, flag high ones new since v16, filter italics/pitch, retrain.
    - **DEPLOYED (10-13)** on Owen's "v18 sounds fine. lets use that as the deployed model" (HF + PC + Mac).
      Card facts from 1900's own ladder: pace 13.06 (median of 61 clean 300-600 renders, p05 12.09 / p95 15.34),
      band 300-600 (3/128, Wilson hi 6.7 %; narrower than v16's 300-800), cap 1000 = longest training row.
      Scripts `E:/training/_campaigns/2026-10-13-mb-v18-deploy`. Fallback: v16 adapter 2026-09-26-mb-v16/runs/mb_v16/ckpt-3375.
- **Don't re-buy returned audio**: `--skip-spans <file>` (0.5 s margin), `--max-parts N`. `build/adobe_spans_ledger.json`
  MISSES the second 09-08 batch; build the skip list from ledger + `adobe_v2/<bk>_ad2_spans.json` + every later
  campaign's `adobe_map.json` (all on `<bk>_master_48k.flac` time). Upload masters must be on the library-m4b timeline
  (FE: `raw/fe_raw_44k.flac`; the "mistborn FULL merged.flac" master drifts 0 ms at 17 min, -89 ms at 11 h, **-216 ms
  at 23.6 h** and is never used for cutting).
- **LOCATE every return; never place by recorded time** (`build_web_master.py`, c29696b; `--new-map` /
  `--new-returns-glob` repeatable, 079eaea): each span's first 30 s is cross-correlated against the m4b at 16 kHz
  (+-1.5 s); peak < 0.5 = left out and reported; spans > 90 s are also located at their END and must agree within 5 ms.
  Output `<bk>_web_48k.flac` (website audio on the m4b timeline, digital silence elsewhere) + `<bk>_web_spans.json`.
  A return PART holding many spans from different chapters is not a "clip" - use whole-book mode on the located master.
- **Brighten and match every piece** (Owen 2026-09-25: *"lets brighten everything and match all the pieces so they all
  sound the same"*; he had heard *"some mistborn generated sentences sound clean, some sound like mumbles"*). Darkness
  is per BOOK: brightness 10log10(E 2-8 kHz / E 0.1-2 kHz) on speech frames FE -11.6, WoA -14.5, HoA -18.0, AoL
  -15.5 dB (HoA dark at source; WoA's source is bright and Adobe darkened it ~5 dB). `2026-09-25-mb-brighten/brighten.py`:
  target = the narrator's own tone from 48 x 30 s of the ORIGINAL FE + WoA m4bs (-9.5 dB); per-SPAN `eq_match`
  (spans < 30 s take their return file's mean); sample-exact. Result FE -8.87 / WoA -8.85 / HoA -9.35 / AoL -9.29 dB
  (6.4 dB spread -> 0.5). Outputs `masters_eq/<bk>_web_eq_48k.flac`. Use `oaconvolve` (fftconvolve OOMed on a 40-min span).
- **[MED] Then SEPARATE the vocal stem, AFTER the EQ** (2026-10-06/09; Owen: *"we have to re-separate AFTER eq,
  because we found that EQing amps up the background noise"*).
  - **Why:** Enhance Speech at 70 % mixes the original back in. The website returns keep a scaled copy of the
    reader's raw room tone about 14 dB down, the same shape in every band.
  - **Effect:** brightening lifts that copy 2-3 dB above 2.5 kHz, the model learns it, and it renders it louder.
  - **Fix:** RoFormer denoise dry stem per website span (`tone/separate_eq_masters.py`, or Crucible's `denoise` job,
    the same checkpoint). It runs about 10x realtime and is refit to the span's exact sample count.
  - **Evidence:** a one-variable A/B. ds_v15 (noise in) vs ds_v15sep (separated), same everything else. Owen heard
    better prosody after separation (*"the best prosody ive heard from deathstalker so far"*). One A/B, so likely
    rather than proven.
  - **Never re-brighten a separated master:** brighten reads `masters/`, so it puts the noise back.
  - **[HIGH] Measure the drop on the SAME pause frames before and after** (`2026-10-09-mb-v18/paired_floor.py`).
    The first check (`floor_check.py`, `masters_book.sh`) took each file's quietest 3 % of NON-ZERO frames.
    Separation pushes many pause frames to ~digital silence, which that rule excludes, so the "after" floor was read
    from breaths and word tails. FE read a 1.2 dB drop while 11 % of its frames had gone below -100 dB.
  - **Paired results** (before = un-brightened master, so the hiss band is understated by the EQ's 2-3 dB):

    | book | total | <500 Hz | 500 Hz-2 kHz | 4-10 kHz |
    |---|---|---|---|---|
    | d3k | 12.6 | 9.8 | 3.3 | 9.6 |
    | cd | 8.1 | 1.9 | -0.5 | 3.2 |
    | lp | 11.0 (old check: 5.3) | 8.6 | 3.0 | 9.3 |
    | mm | 11.3 | 11.8 | 2.7 | 11.6 |
    | mck | 8.0 (old check: 6.9) | 5.1 | 0.6 | 7.2 |
    | Mistborn FE | 11.7 | 6.7 | 0.4 | 7.8 |

  - **What stays:** the 500 Hz-2 kHz band moves 0-3 dB in every book. The earlier "lp voice-band residual" was
    mostly the biased check.
- **Concatenated masters defeat the whole-book aligner; use the span map** (§4n.35.3, .7): per-book Adobe masters are
  gapless concatenations of spans with `source_start/source_end`, so `source_time = span.source_start + (master_time -
  cumulative_span_start)`. Membership is TIME containment (`cue.start >= span.source_start and cue.end <=
  span.source_end`, ~0.05 s tail tolerance), never the `run_ci0/run_ci1` index range (its first cue can start 237-994 s
  before the span audio).
- Never assume a separator/denoiser/enhancer keeps the input sample rate (§6.5).

### 3.2 Text (was §4n.91, §4n.95, §4n.93, §4n.26, §4n.34.8-9, §4n.35.6, §4n.4, §4n.9)

- **ASR = Qwen for everything.** Owen 2026-09-25: *"i definitely want to abandon whisper and use qwen asr and qwen align
  moving forward. for everything. it's far superior"*. **qwen3-asr-1.7b (never 0.6b)** and **qwen3-aligner**, through
  Crucible's `asr` and `align` verbs; the WSL `qwen-align` env (`qwen_asr` 0.0.6) only where Crucible cannot be used yet.
  Whisper-scored and Qwen-scored numbers are different instruments - re-baseline before comparing.
- **The book text is the truth.** Owen: *"we still want to align to the actual book text because the book text contains
  proper nouns and stuff. generate-sentences is supposed to be exact."* ASR only LOCATES and CHECKS; the aligner is fed
  the BOOK's text. No ASR text ever becomes corpus text.
- **generate-sentences on Crucible** (bookforge 47ffd266, 0d3214e9, fdeef6f3, f340a7c4; Owen: *"all gpu use in bookforge
  should run through crucible ... we dont ever directly call the qwen model"*): ASR the book -> diff heard words vs EPUB
  (unique 4/3/2-gram anchors + LIS, edit-distance DP; near-miss = 40 % of letters, so "Ellen" matches "Elend") -> a
  sentence is PLACED only when every token matches in one contiguous run; the rest are DISPUTED and re-timed by the
  aligner on the book's text in windows that hug the heard words (2 s margin, <= 180 s) -> cue edges moved to pause
  centres (search 0.3 s, reach 1.0 s, flagged when no pause) -> VTT + `align-report.json` + `discrepancies.json`.
  First live run (AoL web master, 4.88 h of returns on a 10.8 h timeline): asr **~130x realtime**, 15 GB on the card per
  job, whole run 548 s, re-runs from the cached transcript ~2 min; agreement with the library VTT 99.4 %; new edges sit
  ~0.4 s earlier than the library's at both ends.
  - **Partial-audio fix (f340a7c4)**: unspoken sentences had been smeared over silence between passages (one "Miles
    snapped." over 151 s). Now a sentence is not placed if two of its words are > 3 s apart or it spans > 1 s/word + 2 s;
    windows hug the densest cluster of heard words and never send a no-heard-word run into a gap > 20 s. Longest cue
    165 -> 15.8 s; silence inside cues 58 -> 6.9 min.
  - **NEVER SEND SILENCE TO THE ASR** (Owen 2026-09-25; bookforge 5c803b1b, 9f0b3992). Qwen3-ASR has no VAD (Crucible
    refuses `vad_filter` for it) and transcribes silence: WoA's website master (6.6 h of returns on a 29 h timeline) gave
    83 min of an invented sentence ("The first thing that you need to do is to get a good quality of light") on repeat.
    The run now measures the level envelope FIRST, cuts every stretch <= -80 dBFS for >= 2 s out of the audio (pieces
    padded 0.3 s, 1 s gaps, streamed to `<transcript cache>.compact.wav`, rebuilt only when the pieces change), sends
    only that to the ASR and maps every word back (`shared/sentence-align/silence-compact.ts`). A whole audiobook (room
    tone) passes through unchanged. Heard words over -80 dBFS silence are still dropped as a second guard. AoL: 10.80 h
    -> 4.97 h sent, silence inside cues 6.9 -> 4.4 min, cues with > 2 s silence 68 -> 31, library agreement 99.4 %
    unchanged; WoA: 28.95 h -> 6.76 h sent. discrepancies.json gained `not_in_recording` (an unplaced run whose gap is
    mostly digital silence - absent from this partial recording, not an omission), so `text_not_in_audio` is only what
    the reader left out (AoL: 1,162 entries / 357 min -> 677 / 28 min).
  - Lists that can be long are never spread into Math.min/max (a partial recording's gap is hours of frames: WoA died
    "Maximum call stack size exceeded", bookforge 1af9c10e).
- **The cue says what the reader said - a standard step of book prep** (Owen 2026-09-25: *"correct the vtt so it
  reflects the real audio so we arent losing training data"*, *"a normal part of the process of prepping a book. running
  ASR on the text, checking discrepancies, building a full, reliable VTT file from the ASR and the epub"*; bookforge
  `correct-to-heard.ts`). After placement, each cue's book text is aligned word by word with the words heard in its
  span: the BOOK's word wherever the reader said it (exact, near-miss spelling - so proper nouns stay the book's - and
  compounds either way), the READER's word where it genuinely differs (book punctuation kept), insertions kept,
  omissions removed. A cue agreeing on < 30 % is left as the book has it (misplaced, not reworded) and is a candidate
  for exclusion. Every change is listed under `corrections` in discrepancies.json. **Such a cue is misplaced and is
  not written to the VTT at all** (bookforge 2026-09-25): the forced aligner places whatever text it is given, and on
  WoA the EPUB's front matter (review blurbs, "Contents", "Acknowledgments") had been laid over the spoken title -
  707 of 5,631 cues, present unnoticed in every VTT before the agreement check existed. So paraphrases, edition differences
  and self-corrections become training data that matches its audio instead of rows to drop.
- **discrepancies.json** (Owen: *"we need some way to tell when audio contains content that isnt found in the text, and
  cases where text doesnt exist in the audio"*): `audio_not_in_text` (inserts), `text_not_in_audio` (omissions),
  `paraphrase` (normalised in-order word agreement < 85 %), `pace_outlier` (chars/s vs rolling median of 40 neighbours,
  +-35 %, sentences >= 40 chars and 2 s), `non_speech_audio` (loud with no heard word), `music_under_speech` (5+
  consecutive sentence pauses never within 15 dB of the floor). On AoL: ~3 % flagged (133 paraphrases, 125 inserts, 36
  pace outliers), incl. a real misread-and-correction and an edition difference. Exclude high-severity ones.
- **Training text = render text** (Owen 2026-09-25: *"the text should look just like it will when it's rendered. all of
  it."*). BookForge's cleanup is book-level, BEFORE splitting/alignment (Foundry clean-text: punctuation spec -> number
  rules -> the model, optional triage pre-pass). The corpus builder had only ever run the first two stages; on v15 ~60
  of 2,913 rows (2 %) differed (footnote markers glued to words "do,1 especially"; chapter numbers + small-caps openers
  glued to prose). Now: `clipforge cleanup` over the VTT (§4). Parity items that already bit: numbers read as words (a
  model trained on "1945" served "nineteen forty-five" is a spec mix), em-dashes kept as U+2014 (an old cutter rewrote
  them to " - "; the corpus had zero and every model skipped em-dash spans), all-caps folded to Title Case exactly as
  BookForge does at render (`pipeline/normalization/caps_fold.py`, Owen ruled FOLD), footnote markers stripped.
- **Dialogue goes out by the WHOLE SENTENCE** (Owen 2026-09-24: *"we exclude the whole sentence containing dialogue, so
  we don't get stray 'he said' in the corpus without context"*). `dialogue_cues_strict.py` (ead234e) drops any cue with a
  quote mark or inside an open quote, the attribution cue after it (speech verb in its first 6 words, or <= 4 words), and
  a lead-in cue before it (ends in `,`/`:` with a speech verb); an open quote carries into at most 3 quote-less cues
  (uncapped, one unbalanced quote flagged 76 % of WoA). Per-voice policy: **Mistborn prose-only in EVERY book** (*"dialogue
  makes it all sound bizarre"*; AoL: *"a lot of jarring character voices"*); Marked Man and 3000 Degrees dialogue out for
  deathstalker; Third Reich keeps everything (Pratt reads no character voices). Why: Higgs tolerates some dialogue
  leakage but is sloppy about WHEN it uses character voices (Owen's ear on 5947: character voice in prose).
- **Also exclude**: chapter openers / epigraphs / headings (the quote filter cannot see a character-voice epigraph such
  as the Lord Ruler's logbook - use the reviewed epigraph list); both sentences around every enhancer-invented sound
  (§3.3); strict-dialogue attributions left behind by older uploads (the 09-09 AoL selection left 61 "Tan said."-type
  cues - the slice-time strict exclusion removes them, no re-upload needed).
- **Heading residue trap**: the library aligner joins a chapter heading into the FIRST prose cue of the chapter ("... Not
  by far. THE END OF PART THREE PART FOUR: DANCERS IN A SEA OF MIST. 26. I ..."), so a cue-level heading regex misses it
  (30 rows dropped by hand on mb_full_rvc1). Repo fix owed: run the heading exclusion over ROW text after the cut, and
  have the s1 check name offending rows.
- **Inline tags added a space before punctuation** (fixed 2026-09-26, bookforge 620d6b4d): BookForge's EPUB extractor
  turned EVERY tag into a space, so `<i>Keep focused</i>.` read "Keep focused ." and "Careful , he told himself" -
  every italic thought that ends at punctuation. It was in every aligned VTT, every corpus and every render before
  that commit: Third Reich at War 2,179 instances, Mistborn ~790 cues across WoA/HoA/AoL. Inline tags (i, em, b,
  span, a, sup ...) now strip with no space; block tags still separate words (tools/test-epub-inline-tags.js).
  Checked on 6 EPUBs: no fused words (the apparent ones were em dashes, now "Mistborn—were" as printed). v16's
  WoA/HoA/AoL VTTs were built before the fix and repaired in place by the same rule (`<bk>_clean.pre-inline-fix.vtt`
  kept); FE, deathstalker and Third Reich come out of the fixed extractor.
- **Book-text VTTs carry NEW cue ids** (2026-09-26, v16): an old exclusion list keyed by library-VTT ids (epigraphs,
  chapter openers) does NOT carry over by id. Map it by TIME - any new cue overlapping an old opener/epigraph cue by
  more than 0.3 s is dropped (`2026-09-26-mb-v16/excl_v16.py`, which also applies the website-span margin and the
  book-text exclusions). WoA check: 5.40 h kept, same as v15's WoA.
- **[HIGH] Punctuation survives book-text (measured 2026-09-29, `2026-09-29-punct-drift/punct_drift.py`).** Every
  training row of four mixes placed in its EPUB (word 6-gram anchors + difflib) and the mark between every agreed word
  pair compared. Lost per 1,000 EPUB marks: tr_v3 0.00, tr_v4 4.02, mb_v16 1.24, mb_v17 0.05; commas intact everywhere;
  "changed" is almost all `. . .` -> ellipsis. Punctuation drift does not explain the v17 / v4 prosody regressions
  (mb_v17's text is closer to the book than v16's). Rows with a >= 40-word unpunctuated run: tr_v3 3.6 %, tr_v4 3.1 %
  (Evans's long sentences), Mistborn 0 %. Run the script on any new corpus: `punct_drift.py <label> <out> <bk>=<epub>
  ... -- <index.jsonl | bk:file.vtt>`.
- **Open book-text defect: a cue swallows its neighbours unpunctuated.** tc cue 1610 (05:23:27-05:23:50) carries the
  reader's words for the next sentences with no punctuation (*"...seemed terrifying money lost its meaning almost
  completely Printing presses..."*) while cues 1611/1612 collapse to 0.05 s. 161 tr_v4 long rows carry at least one
  lost mark (64 carry >= 2); tr_v5 drops all 161. **Fixed in BookForge 7f63b324 (0.1.3334), not yet re-run on any
  book.**
  - Cause: the aligner lost its anchor on "(1,000,000,000,000,000,000)" and stretched sentence 1609's tail over the
    next three sentences. The edge stage then never starts a cue before the previous one ends (so 1611/1612 were
    squeezed to 0.05 s), and correct-to-heard appended the 22 heard words as "reader insertions".
  - Fixes: overlapping placements are trimmed back to their last word before the next start (`overlaps.ts`);
    correct-to-heard writes at most 4 heard words beyond the shared span (a new `drop` op); cues < 0.15 s are flagged
    `tooShort` in discrepancies.json as exclusion candidates.
  - A replay of ~6,900 stored corrections changed 14, including lp's ~300-word cue.
  - Still open (BookForge offered): two-word dialogue tags ("Ham said.") placed on the wrong audio, which can reach
    the 0.3 agreement by chance.
- Third Reich numbers: tr_v3 learned digits in 573 rows ("1933"); the cleaned text spells them out (16 rows with digits).
  The ear test cleared the text as the prosody cause (§3.5.1).

### 3.3 Slicing (was §4n.53, §4n.75, §4n.87, §4n.89, §4n.90, §4n.39.5, §4n.37.19, §4n.95 addendum)

`pipeline/untreated/slice_vtt.py`, via `clipforge slice`. Current mistborn recipe (09-23/09-24, build16_web):
run median 59 s, `--run-sigma 0.35`, long 22-100 s, `--gap-s 4`, `--tail-s 0.25`, `--interp-interior`, tiers `l+s`,
`--min-start 120`. **v16 (Owen 2026-09-25): add `--long-max-chars 1000` and `--run-median-s 45`** (WoA test slice p50
574 / p75 713 / p90 895 / max 1,003). The short tier (single sentences) and its share are unchanged.

- **The post-chunk tail is CUT: `--tail-s 0.25`. [HIGH]** Owen 2026-09-10: *"we trimmed post chunk pauses before. Not
  post sentence. Post chunk ... This is the first model we generated that had NOT trimmed the pauses from the end of the
  chunk."* The only models ever trained with the tail retained (`--tail-s 4.0`) both ran away: mistborn v6 58 % of
  renders (median 37.9 s trailing silence, to the token cap), deathstalker v6 29 %. Same corpus re-sliced at 0.25:
  Owen Morgan v7 **0/24 runaway, trailing silence 0.3 s**; mistborn v7 37.9 s -> 0.2 s; deathstalker v7 29 % -> 0 %.
  The median tail length hid it (0.84 vs 0.94 s); the MAX is the trigger. Internal pauses stay verbatim - this is the
  END of the chunk only. The trim moves the corpus char quartiles (mistborn p75 fell to 869, deathstalker's stayed
  1073) - re-check each voice's cap against the new p75.
- **The END cut must BE a pause** (8a9a97c): `MIN_PAUSE_S 0.20` (an 80 ms quiet run matched the stop closure inside the
  last word, "em-PIRE"); no pause before the cue end -> take the FIRST pause after it within `FWD_S 1.5`; no pause
  either side -> **DROP the clip**. Before: 70/4544 mistborn rows (1.9 % of long) ended mid-word - Owen heard *"a single
  sound at the end of a chunk. just an 'a' sound"*. After: 1/4180, long tier 0.
- **The START cut** (`start_bound`, 2026-09-22): the old blind 0.10 s pad before the aligner's cue start reached into the
  first word (the library aligner puts cue starts 60-100+ ms inside the attack): speech at frame 0 on 28 % of McKinley,
  27 % of HoA rows. Now the clip starts `HEAD_S 0.25` before the end of the last pause (>= 0.20 s) that begins before
  cue start + 0.10, never earlier than that pause's midpoint (the end cut also stops at the midpoint, so a short pause is
  shared); no pause within 1.5 s -> DROP. McKinley: frame-0 speech 25.9 % -> 0.0 %, cost 10/348 rows. **Every corpus cut
  before 2026-09-22 has clipped onsets.**
- **Cuts land in SILENCE, not in a breath** (2026-09-23; Owen: *"it needs to be perfect silence. in fact, cutting in the
  middle of breaths might have been the root cause"*). The old pause test `max(window min + 12 dB, -48 dBFS)` let a
  -50 dBFS breath count as silence on processed audio. Now: floor = LOCAL room tone (p5 of the window's 20 ms frames,
  digital zeros ignored); a pause frame is within `SIL_DB` 10 dB of it **or below speech - 45 dB**; the cut itself must
  land on a frame within `CUT_DB` 6 dB of the floor, else the edge is dropped (`NCAP["breath"]`). Room-tone-only was too
  strict (lost a third of FE's starts); with the speech anchor: FE audit flags 84/5,284 -> 9/4,946, 0 near a word.
- **Edge defects are invisible to every coverage gate.** `row_gate.py` never opens the audio; one clipped word in a 60 s
  row is 0.99 coverage. For any question about a clip's EDGES, measure the waveform (is the 100 ms ending at the cut
  above speech level?). Don't use "speech within 200 ms AFTER the cut" - it flags clips that end correctly just before
  the next onset (inflated 1.5 % to 5.5 %).
- **Enhancer-invented sounds** (`find_invented.py`): compare each export with its own input on one timeline; flag frames
  where the INPUT is quiet (< speech - 40 dB) and the EXPORT is > 20 dB above it and above speech - 45 dB; runs >= 80 ms
  are events (Owen on the listen file: *"yes. thats the sound."*). `invented_to_cues.py` excludes BOTH sentences around
  each event. Rate tracked source editing on Premiere exports (AoL 1/h, HoA 8.1/h) - apparently triggered by
  digital-zero-to-room-tone splices. Run it after any enhancer.
- **Size the corpus from the serving length, then histogram and adjust.** Runs carry ~12.8 chars per second of clip for
  full-tail clips (tr_v3's 0.25 s tails ran 15.4). A book slices in ~11 s - slice, histogram, adjust, re-slice.
  `--served-cap-chars` in the mix only CHECKS mass at the served length; it never trims (v15 trained on rows to 1,709
  chars, p50 644 / p75 875 / p90 1,134) - hence `--long-max-chars`.
- Other flags: `--exclude-cue-ids @<file>` (dialogue lists exceed Windows' 32 KB command line), `--exclude-text-regex`,
  `--min-start` / `--max-end` (front/back matter; see §6.5 for the default trap), `asr-fallback` cues are excluded
  outright. Clips are written at **24 kHz** always (`build_higgs_mix.py` asserts sr == 24000). Pool wavs drop the book
  prefix (`tc_raw_src_l/wavs/tc_000001.wav`) while merged tiers keep it (`tr4t_l/wavs/tc_tc_000001.wav`).
- Merge per book with `merge_corpora.py` (`clipforge merge-tiers`) **under the same build root as every source pool**.
- `clipforge cut-audit` (auto-editor threshold 0.003, ~-50 dBFS) is the strict edge audit.

### 3.4 Gating (was §4n.91, §4n.93 trap 1, §4n.34.8, §4n.40.4, §4n.11-11b)

- `row_gate.py <tier> --min 0.80 --sent-min 0.50` (`clipforge gate`) on EVERY merged tier; the chain refuses a tier with
  no `row_gate.json`. Qwen batch is sized by audio: `max(1, min(32, 800 s // longest clip))` (batch 8 on the long tier,
  16.8 GB) - a fixed 32 on 100 s clips held 24.0/24.5 GB and thrashed the PC (2875f57).
- **Proper nouns are free** (Owen: *"we should be fully relying on the aligned text as it is, and only using whisper to
  determine if there was drift or if its genuinely a completely different/altered section ... if theres paraphrasing, we
  would want to cut it. if its proper nouns, we would want to keep it."*): capitalized non-initial words excluded from
  coverage; sentences with < 4 ordinary words not judged; ASR tokens within difflib 0.80 snapped; digits read as words
  both sides; `--rescore` re-applies the rule without transcribing.
- **Baselines.** Qwen gate on mistborn web (09-24): mbw_l 3/1,248 dropped (coverage median 0.994, p05 0.967), mbw_s
  2/1,509 (median 1.000). Whisper-era prose tiers dropped ~0.6-2 %. A prose tier dropping much above ~2 % is an
  ALIGNMENT problem - never raise the threshold to make it pass. Micro (1-3 word) tiers drop ~14 % by construction.
- **Pairing gate**: decode 3 rows per tier back to audio and ASR them against their own text (coverage >= 0.80;
  `night3/triage/roundtrip.py`). It caught in 60 s two bugs a loss curve never would: the mix read audio from the wrong
  pool (model spoke fluent gibberish), and ids collided across tiers (key everything by tier + id).
- Verify a suspect alignment with `where_align.py <flac> <vtt> <t> [win] [base]` (ASRs a window and slides it over the
  VTT word stream); `verify_align.py` is too strict to judge by.

### 3.5 Bed (was §4f, §4n.61, §4n.69.1, §4n.70, §4n.70.1, §4n.79)

- **Current practice (2026-10-09): NO bed and NO pause cap for new trains.** The stall guard handles runaway
  silences at inference (§3.5.2). Third Reich v4nb and deathstalker v16 are deployed without a bed; Mistborn v16 is
  still the bedded recipe. History below.
- **Former practice: own-source bed (ownbed1) at -70 dBFS under every clip** (median 250 ms-window RMS rule), on both
  deployed voices since 2026-09-13 (Owen 09-13: *"the BED is added to both voices"*; his rule for deathstalker: *"if it
  is better with a bed, use it"*). Third Reich (tr_v3) has no bed and is near flawless. **Under review 2026-09-30:**
  the -70 level is 10-15 dB above the readers' own pause floors and cost Third Reich its prosody (§3.5.1); the
  candidate replacement is a generated own-spectrum bed at -85 with no pause cap.
- **Why, stated correctly** (Owen: *"higgs isnt snac. its a different codec. the same rule doesnt apply evenly with the
  new codec"*): the bed is NOT an EOS/SNAC-floor fix on v3. On v3's own audio no bed criterion fired (coverage 90.9 vs
  90.7, early stops 14 vs 13, runaways 0 vs 0). The measured benefit is on RESYNTHESISED audio: mb_hoa_maxine1 (Maxine +
  bed) screen per 100 renders loops 0.0 / overruns 0.0 / truncations 5.1 vs the RVC no-bed run's 7.1 / 5.4 / 10.2, zero
  `long` ladder failures. Whether the bed helps Adobe-website audio has not been isolated. RVC output leaves pauses at
  digital silence (p05 -89.9 dBFS), so a -70 bed sits ~18 dB above what is there.
- **Hidden variable in every mix**: `build_higgs_mix.py` (nobed path, and the bed test before bedding) applies
  `build_pause_cap.compress` KNEE 1.5 s / CEIL 2.5 s to every INTERIOR silence (tails untouched). Every train since the
  v7s trained on interior pauses capped at 2.5 s. Third Reich trained under the same cap and never looped, so it is not
  the loop cause. Owen: make knee/ceiling explicit mix arguments next time. **`bed_mix_tier.py` (`clipforge bed`)
  applies the same cap by default** - a "bed" change has always been bed + cap (§3.5.1).
- ECAPA/speaker similarity must match the silence on both sides (bedded refs vs digital-zero renders measured the floor,
  not the voice: a +0.036 "bed wins" gap inverted under VAD). Cosine is reported only, never a criterion; VAD-trimmed.

#### 3.5.1 The -70 bed + pause cap cost Third Reich its prosody (2026-09-29/30)

- **[MED] Ear A/B, one variable set.** tr_v4 (bedded) vs tr_v4nb: identical rows, text, clip lengths, recipe and
  checkpoint cadence; tr_v4 had trbed1 at -70 **and** `bed_mix_tier`'s pause cap (knee 1.5 / ceiling 2.5 s, on by default
  - `bed_mix.json` records `compress: true`), tr_v4nb had **neither** (`build_higgs_mix.py --nobed --no-pause-cap`). Owen
  on the same Mutineers' Moon ch. 8 passage + seed: tr_v4nb ckpt-9040 *"superb"*, *"sounds better prosody-wise than
  v3"*, one hiccup on the final *"what the-"* where v3 *"hiccuped a lot"*; tr_v4 10396 audibly worse than v3. So the bed
  and/or its pause cap hurt prosody; text cleanup and the 1,000-char cap are cleared (both shared with the good run).
  One passage, one seed, one listener - [MED], not [HIGH].
- **Why the bed was too loud.** `bed_mix_tier.py` normalises every bed to -70 dBFS median-window: trbed1 (TR's own
  inter-cue tone, median -89.4) got **+19.4 dB**. Measured pause floors of the UN-bedded clips (quietest 15 % of 250 ms
  windows, 150 clips): **Third Reich -84 dB, Mistborn -79 dB**, and **0 % digital-silent windows in either** - the -70
  bed did not cover dead air, it replaced the reader's room with hiss 10-15 dB louder. It also lays the bed as
  consecutive wrapped segments: trbed1 is 27.6 min under ~84 h, so every stretch repeats ~180 times (ownbed1 2 h under
  Mistborn's ~24 h: ~12 times).
- **Without any bed the model over-holds silences.** tr_v4nb screen: every checkpoint 6780-11752 had 2-5
  truncation/repeat flags and 6-14 pauses >= 2 s (tr_v4's clean pair had 0 and 0-1), so the screen found no clean
  checkpoint. Ladder 9040 vs deployed v3 on the same bank (`2026-09-29-tr-v4-nobed/ladder_vs_v3`, band 500-800):
  in-band failure **4.7 % (hi 9.8 %) vs 6.2 % (hi 11.8 %), inside noise**; truncations 7 vs 7; max pause 4.60 vs 2.06
  s; >= 2 s pauses 10.86 vs 0.56 per 10k chars. v3 had the 2.5 s cap. **CORRECTED 2026-10-01: the reference was
  wrong.** A `--nobed` mix wrote the TIER's (bedded + capped) path into `index.jsonl` "wav" while encoding `nobed_wav`,
  and the screen's corpus reference reads "wav" - so tr_v4nb's "corpus" (max 2.16 s, >= 2 s 0.21/10k) was the capped
  audio. The true uncapped reference (tr_v5's, same pools): median 0.64, p90 1.56, **max 3.74 s, >= 2 s 7.24/10k**.
  Rescored (`2026-09-30-tr-v5/rescore_vs_v3.sh`, `ladder_vs_v3/pause_rank_trueref.json`): **9040 69 vs v3 59** - the
  no-bed model pauses closest to the reader; it does not over-hold silences, v3 under-holds them (trained capped). Fixed
  in `build_higgs_mix.py`: "wav" = the encoded file, `tier_wav` keeps the tier path. The screen's "no clean
  checkpoint" verdict on tr_v4nb used the same wrong reference.
- **The generated bed (tr_v5, training 2026-09-30).** `2026-09-30-tr-v5/build_tr5_tiers.py`: Gaussian noise shaped to
  the corpus's own tone (trbed1's average spectrum, matched within 1-2 dB per band), drawn fresh per clip (seed =
  crc32 of the clip name; seeds correlate 0.014), **-85 dBFS** median-window, added to the un-bedded pool clips, no pause
  cap. Equivalent to an overlay on the master (no seams, no repeats) without re-slicing. Result: clip pause floors -84
  to -83 dB (the reader's own floor + ~1 dB). Owen's level ruling: -85; -90 would sit ~6 dB above the 16-bit floor and
  quantise to near-white noise. Slowing a bed to stretch it (proposed) shifts its spectrum an octave down - generate
  instead. **Outcome (2026-10-01, tr_v5):** early stop 9085 (best 3.4971 @ 7900) at 06:20; screen of the newest 5
  found 1 clean; ladder 8295 4.7 % (hi 9.8 %) band 500-800 vs 7505 6.9 % - inside noise; pick **8295**. Pausing vs the
  true reference: 8295 57, 7505 67 (9040 69, v3 59 on the same reference); >= 2 s 10.44/10k (corpus 7.24), max 4.42 s;
  truncations 7 (as 9040 and v3). **Owen's ear (2026-10-01): "the prosody is basically the same. i might favor
  9040".** So a -85 bed at the reader's own floor changes nothing measurable or audible; on natural narration with no
  digital silence (TR -84, Mistborn -79 clip floors) the bed has no job to do. Louder (-70) is what cost prosody.
- **Mistborn bed A/B (2026-10-01/02, `2026-10-01-mb-bed-ab/`)**: v16's exact rows and recipe, no pause cap in either
  arm. Corpus pause map: median 0.74, max 4.44 s, >= 2 s 1.1/10k. **No bed (mb_ab_nb)**: early stop, best 3.7691 @
  3240; the newest 5 checkpoints ALL stall in silence (defects 3-9, max pause 5.6-91.7 s, >= 2 s 10-44/10k); best pauser
  3510 scored 37.5 (3 defects, max 5.60). Its sample runs 8.1 min against v16's 6.1. **-70 ownbed1 (mb_ab_b70)**:
  best 3.9864 @ 2700; defects 0-1, max 2.0-3.5 s, >= 2 s 2.3-13.8/10k; best pauser 2970 scored 46.2 (1 defect, max 3.50).
  So on Mistborn the bed DOES prevent silence stalls: the opposite of Third Reich, where no bed paused like the reader.
  The difference is the source: Mistborn is Adobe-processed (floor ~-79 but presumably gated/smoothed pauses), TR is raw
  narration. **Owen's ear (2026-10-02): "prosody is about the same. Slightly better on the really tough spots with no
  bed. But the no bed render is full of long 10 second pauses, which is untenable."** Rule: raw narration with a real
  room floor (Third Reich) trains with NO bed; enhancer-processed audio (Mistborn/Adobe) KEEPS the bed - it costs little
  or no prosody there and is what prevents silence stalls. **SUPERSEDED (2026-10-09):** the stall guard (§3.5.2) now
  prevents the stalls at inference. Deathstalker v16 (Adobe audio, separated, no bed) screened and laddered with no
  stall and was kept permanently. New trains go without a bed. Mistborn v16 (bedded) stays live, because Owen called it.
- **Render-side protection (Owen 2026-10-02: "cut pauses longer than 1.5 seconds. For protection")**: BookForge
  14becf08 - narrator's render door caps every INTERIOR silence of a Higgs chunk at 1.5 s (pause_lib's definition: 20 ms
  frames, peak - 40 dB; keeps 0.75 s each side, 15 ms raised-cosine splice), after the retake ladder; joins, chapter
  gaps and streaming untouched. `NARRATOR_MAX_INTERIOR_PAUSE_S` (default 1.5, 0 = off). Every cut is logged
  ("PAUSE CUT" on stderr, `pauseCuts` per row). **LIVE 2026-10-02** on both Crucibles (1.0.80, narrator 14becf08);
  BookForge 802e28be records them in each render's job-analytics.json `guard.pauseCuts` {reported, chunksCut, cuts,
  longestS, byChunk, byChapter} - read byChapter to find a stalling voice/chapter. Caveat: real pauses reach 3.7-4.4 s,
  so 1.5 shortens some.
- Levers now explicit: `build_higgs_mix.py --no-pause-cap` (with `--nobed`; default keeps the cap) and
  `bed_mix_tier.py --no-compress`.

#### 3.5.3 The ladder scorer over-counted cut-offs ~10x (fixed 2026-10-02)

- **[HIGH] Text normalisation was undercounting coverage.** Of the deployed Third Reich 9040's 36 renders flagged
  coverage < 0.90 (288-render ladder), only **2 were real cut-offs**. The transcripts of the other 34 ran to the
  last word. Causes, all on the scorer side:
  - the TEXT kept quote marks (`'England` never matched `England`);
  - ordinals never matched Whisper's digits ("eighth" vs "8" -> "eight");
  - British spellings never matched American ones (gaol/jail, fulfil/fulfill, labour/labor);
  - accents and mangled characters in the bank ("H�ss");
  - Whisper's spelling of German names (Gobineau -> "Gobino").
- **Fix** (`pipeline/higgs_ladder.py` `norm_words` + `text_metrics`): both sides now get the same normalisation.
  - digits spelled out, as before;
  - NFKD -> ASCII; apostrophes kept only INSIDE words;
  - ordinals -> cardinals; British -> American spellings (word list plus -our/-ise/-yse/-tre/-lled endings);
  - a REPLACE block whose letters agree >= 0.75 at the same word count (+/- 1) counts as matched, because a misheard
    name is not a missing word. A dropped clause still counts: the "…extended across" cut-off scores 0.31.
  - New `higgs_ladder.py rescore --run <dir>` recomputes from the transcripts already in scored.json (no ASR). It keeps
    `<field>_prev` beside each new value.
- **Third Reich rescored** (copies in `2026-10-02-scorer-fix/`; the original records are untouched). Coverage < 0.90,
  all rungs:

  | model | before | after |
  |---|---|---|
  | 9040 | 36 | 3 |
  | v3 | 33 | 1 |
  | v5 8295 | 35 | 2 |
  | v5 7505 | 34 | 3 |

  In band 500-800 the counts went from 7-8 to 0-1 of 108. The deployed Third Reich's real failure rate is ~1 %, not
  4.7 %. Every pre-2026-10-02 ladder failure rate, band verdict and "truncation" count is inflated by this; compare
  only rescored runs.
- Still owed: Whisper base.en -> Qwen ASR for the ladder/screen scorer (§10). Re-baseline when it lands.

#### 3.5.2 The stall guard: fix runaway silences in the SAMPLER, not with a bed (2026-10-02)

**The problem it solves.** Owen on the Mistborn bed A/B: the bed costs prosody (Mutineers' Moon ch. 8: *"at least 10
hard spots ... it failed 7 or 8 of them with a bed, but only 1 or 2 without"*), but the no-bed model stalls (*"full of
long 10 second pauses, which is untenable"*). The pause screen found silences up to 92 s; one render had a 113 s tail.
The bed was a training-data workaround for an inference failure. Owen: *"id like to find a real solution"*.

**[HIGH] What a Higgs stall is** (`2026-10-01-mb-bed-ab/silence_codes.py`, the 113 s tail of mb_ab_nb MM02 re-encoded):
codebook 0 shows 5 unique codes in 90 s, 98 % of frames identical to the previous frame, and 65 unique whole frames out
of 2,250. Speech: 208 unique cb0 codes in 40 s, 7 % same-as-previous, every frame unique. **A Higgs stall is an
exact-repeat loop.** That is unlike Orpheus/SNAC silence, which is a VARIED token run: that is why repetition_penalty
failed on Orpheus (`SAMPLING_AND_VOICE_QUALITY_FINDINGS.md`) and why a repeat detector works here.

**Why the model cannot leave it.** Deep in a silence the model puts ~97 % on "more silence"; the exits (the next word,
or EOC) sit in the tail, and top-k 50 / top-p 0.95 cut that tail to exactly zero, so the state is absorbing. Adobe-
processed silence (Mistborn) is smooth and identical, so the model learns "silence -> the same silence" with high
confidence. Raw room tone (Third Reich) varies, so it never becomes a perfect loop. That is why no-bed works on TR and
not on Mistborn.

**Prior art** (researched 2026-10-02):
- VALL-E 2 repetition-aware sampling: if the candidate token's share of the last K=10 tokens is > 0.1, re-draw from
  the full distribution; it fixes VALL-E's infinite loops (arXiv 2406.05370). Its threshold fires on 9 % of OUR speech
  frames, and full-distribution sampling is what once gave an 80 s silence here (§2), so it was not copied as-is.
- Chatterbox forces EOS from attention alignment once the text is done (resemble-ai/chatterbox #519).
- Our own Orpheus EOS logit boost (`TESTING_BIBLE.md` "EOS logit boost"): +8 on EOS past 2.0x the expected tokens,
  ramped, validated 8/20 -> 15/20 on a stress model.
- Root fixes (duration/progress-aware models: VoiceStar PM-RoPE, ELLA-V, RALL-E) need a new architecture, not a fine-tune.

**The mechanism (ours).**
- `run` = consecutive frames whose cb0 code equals the previous frame's cb0 code. Counted only for active rows past
  the delay window and not winding down.
- Once `run > FRAMES`, the previous cb0 code's logit is lowered by `min(MAX, RATE * (run - FRAMES))`. This happens
  BEFORE temperature / top-k / top-p (so greedy is covered too).
- The stuck code loses ground and the model's own next choice takes over: the next word if text is left, EOC if it
  is done. Nothing is forced. Pauses shorter than FRAMES/25 s are never touched.
- Tested setting: `FRAMES 37` (1.5 s at 25 fps), `RATE 0.5` logits per frame, `MAX 20`.

**Where it lives (prototype, 2026-10-02).**
- `2026-10-02-stall-guard/patch_stall_guard.py apply|revert` patches the **sglomni test env only** (the env our
  ladders and screens serve from). Crucible's install is untouched; the originals are in `orig/`.
- Files: `sglang_omni/models/higgs_tts/sampler.py` (state `stall_run`, the penalty + counter in `batched_step_direct`,
  the eager wrapper), `model.py` (CUDA-graph buffer `_cg_active_stall_run`), `model_runner.py` (gather/scatter
  beside `step_count`). The code is branchless tensor ops, so it is safe inside the captured CUDA graph.
- Switch: env **`HIGGS_STALL_GUARD="FRAMES,RATE,MAX"`**, read at import, so it is set on the SERVER process.
  Unset or empty = off, and off is the original code path. The patch stays applied; it is inert unless the variable
  is set.
- Unit test `unit_test.py` (synthetic logits, silence code dominant): off = stuck for 391 frames; on = escaped at
  frame 59.

**[HIGH] A/B result** (`ab_test.sh`, `silences.py`; mb_ab_nb ckpt-3510 = the stalling no-bed Mistborn; Mutineers'
Moon bank x 16 seeds = 144 chunks per arm, same seeds; pauses by pause_lib's definition):

| | guard off | guard on (37,0.5,20) |
|---|---|---|
| interior pauses > 5 s | 5 | **0** |
| longest interior pause | 108.6 s | **4.7 s** |
| tails > 2 s | 1 | 0 |
| total audio | 102.8 min | 98.7 min |
| coverage median (ASR) | 0.979 | 0.978 |
| renders with coverage < 0.90 | 5 | 3 |
| median pause | 0.04 s | 0.04 s |

No words lost, normal pausing unchanged. Seed-500 stitched: 7.27 min off -> 6.1 min on, the same as deployed v16's 6.06
min on the same passage (`mb_v16_deployed_mutineers_moon_ch8_2026-10-02.wav`). Owen on the guarded render: *"not bad"*.
Prosody verdict vs v16 pending.

**Through Crucible (2026-10-02, crucible-pc-1, PC arm, Crucible 1.0.85; a maintained sglang-omni patch plus a ring
buffer of recent cb0 codes; settings in `[voice.serving] stall_guard`, default on).** Stalling model, 144 chunks per arm,
pauses as generated (from `pauseCuts.fromS`, because the 1.5 s render cap trims them after generation):

| | off | 37,0.5,20,8 | **37,1,20,16 (accepted)** |
|---|---|---|---|
| generated pauses > 5 s | 5 | 0 | **0** |
| generated pauses > 3 s | 6 | 3 | 2 |
| longest | 35.3 s | 4.2 s | 4.5 s |
| dead air trimmed by the cap | 108 s | 65 s | 48 s |
| coverage median | 0.984 | 0.984 | 0.985 |
| speed | - | −7.3 % | −2.5 % (noise) |

Healthy deployed Mistborn: longest pause 1.9 vs 2.0 s, no harm. The ~2 s ceiling target was **dropped**: the 2.7-4.5 s
leftovers sit inside the narrator's own range (Mistborn corpus max 4.44 s), so pushing lower would cut real pauses.
Settings live in the voice manifest; the Mac arm (BookForge mlx_backend) is pending the Mac GPU.

**Mac arm + the re-stall cycle (2026-10-02 late).**
- Mac (MLX, narrator 2340aab7) at 37,1,20,16 on the stalling model: pauses over 5 s 2 -> 4, all four in one chunk
  (5.3-5.8 s), each with several ENGAGED log lines. No harm on deployed Mistborn (identical off/on, −1.4 %).
- crucible-pc-1 verified the MLX penalty is applied to raw logits before temperature/top-k/top-p, on every ring code,
  with the same resets. My "applied after top-p" theory was wrong.
- Real cause, shared by both arms: run resets to 0 on the first frame OUTSIDE the ring. The model escapes for one
  frame, the penalty drops to 0, and it snaps straight back to the same silence code. It then needs 37 more frames to
  re-engage, so ~3 cycles x ~1.85 s = the 5.5 s stays. The PC's 4.5 s worst case is the same thing at ~2.4 cycles.
- Fix chosen (option b): reset only after **10 consecutive non-ring frames** (0.4 s; speech leaves the ring far
  longer). On a non-ring frame before that, HOLD the run (no increment, no zero) so a return is penalised at once.
  Uncounted frames still reset immediately.
- Pass criteria: unit test reproducing the snap-back cycle; both stalling arms with pauses over 5 s = 0 and longest
  ~3 s; coverage unchanged; no-harm rerun. **v2 shipped in Crucible 1.0.88** (narrator BookForge 94c1f49d, PC patch
  Crucible 7a1fc83): the hold applies only while ENGAGED (run > frames), so rows that never stall are bit-identical to
  v1 and speech cannot ratchet the run upward (confirmed). Unit tests: v1 = 9 stays of ~47 frames, v2 = one stay under
  62 frames (2.5 s). **Mac v2 PASSES** (stalling model, 144 chunks: pauses over 5 s 2 off / 4 v1 / **0 v2**; over 3 s 9 / 4 / **0**;
  longest 6.4 / 5.8 / **3.0 s**; trimmed 69 / 68 / 47 s; 0 failed; no-harm on Mistborn identical, −1.2 %). **PC v2 PASSES**
  (pauses over 5 s 5 off / 0 v2; over 3 s 6 / 0; longest 35.3 / **2.6 s**; trimmed 108 / 42 s; 0 failed). Both arms pass.
  Crucible 1.0.90 also fixed GET /v1/info running `pip list` on the event loop (it froze the server ~3 s in every 4
  during renders). My v1 hand-patch in the sglomni test env was reverted 2026-10-02 (files byte-identical to orig/).
  **ACCEPTED 2026-10-03.** Coverage (CPU base.en, normalised scorer), median / renders under 0.90:

  | arm | off | v2 |
  |---|---|---|
  | PC stalling | 0.985 / 3 | 0.985 / 1 |
  | Mac stalling | 0.985 / 6 | 0.989 / 3 |
  | PC Mistborn | 0.989 / 0 | 0.989 / 1 |
  | Mac Mistborn | 0.989 / 0 | 0.989 / 0 |

  The single PC-Mistborn early stop (MM02_r13: a clean EOC after a 1.56 s pause, over Mistborn's max c/s) prompted a
  targeted check: MM02 x 64 per arm, 0 early stops in either (coverage median 0.992 both). It is a one-off; the guard
  does not cause truncations. Test voices removed and the merged checkpoint sent to the ledger graveyard; the adapter
  stays at 2026-10-01-mb-bed-ab/runs/mb_ab_nb/ckpt-3510.

**Known gaps before shipping (prototype; addressed above).**
- The 4.7 s max: the run resets when the model hops between its ~5 silence codes. Treat a SET of codes (or a
  low-energy frame class) as one run to bring the ceiling to ~2 s.
- Renders ran ~11 % slower with the guard on (516 -> 582 s wall). Likely the full `[B,8,1026]` zeros + scatter per
  step; penalise cb0 only, in place.

**To ship it (both arms, IDENTICAL settings, or PC and Mac pause differently).**
- **Mac:** BookForge `python/narrator/engine/higgs/mlx_backend.py` runs the frame loop itself (`step(model._audio_logits
  (...)[0], state, ...)`, plus the batched `_step_batch_sampler` path). Add the counter + penalty there. BookForge
  owns it and Crucible deploys narrator to the Mac.
- **PC:** the sampler is inside the third-party `sglang_omni` package. Carry it as a maintained patch in Crucible's
  installer (or upstream it); a hand-edit is silently undone by a package update.
- The settings belong in Crucible's voice facts / serving config (one source), not per machine.
- **Then:** Mistborn (Adobe-processed) can train with NO bed and keep the better prosody. The render-side pause cap
  (§3.5.1, BookForge 14becf08) and the retake re-roll stay as later safety nets.

### 3.6 Mix (was §4n.32.5, §4n.37.16, §4n.54, §4n.68.3, §4n.59.2)

`night2/build_higgs_mix.py --out $DATA --device cuda --long <l> --short <s> --build-root <BUILD> --rows-l ... --rows-s ...
--mode-name <MIX> [--max-hours N]`.
- **No `--served-cap-chars` / `--served-cap-strict`** (Owen 2026-10-12): rule 2 (served cap <= train rows' p75) is
  DEPRECATED. It rested on a wrong "horseshoe" theory (best audio in a horseshoe around the max length). Higgs is safe
  to ~800 chars depending on the model, so training rows run to 1000; the served cap is set on the ladder. The strict
  check copied from v16 stopped mb_v18 at 07:52 (p75 692 < 700). Older chains (ds v15/v16, mb bed A/B) still carry it.
- `--short-rows-frac 0.0` does NOT suppress shorts: `--short-frac` defaults to 0.25 of DURATION (~59 % of rows). A
  "<band> only" tier needs both zeros, verified by histogram.
- The mix reads each row's audio from its OWN pool (`pool` field) at `<build-root>/<pool>/wavs/<base>.wav` (book prefix
  stripped) - populate BOTH the tier dir and the pool layout for any rebuilt audio (e.g. RVC), keyed by (tier, basename).
- Holdout by reverse overlap; the leakage check needs >= 1.0 s of real overlap (the cutter's ~20 ms boundary snap is not
  shared audio).

- **The mix's served-cap check vs the 1000-char band** (2026-09-26, v16): `--served-cap-chars N --served-cap-strict`
  exits 3 when N is above the train rows' p75 chars (Owen rule 2, 09-06). It is a CHECK - it drops no rows. With long
  rows cut to <= 1000 chars, p75 lands ~709, so a 1000 check can never pass. v16 checks at 700 (rule-2-safe); the real
  served cap is decided from the ladder's 400-1000 pause curve and Owen's word. OPEN for Owen: rule 2 says serve <= ~709
  for this corpus, the 09-25 band says up to 1000 - the ladder shows which holds. **CLOSED 2026-10-12: rule 2 deprecated (above).**
- **The WSL HF cache is a dependency, not scratch** (2026-09-26): the 09-25 cleanup deleted `~/.cache/huggingface`,
  taking the Higgs v3 base (snapshot 239f63fb, pinned in v3_ft) and the OmniVoice codec; the v16 mix failed on the
  missing path. Restored with `2026-09-26-mb-v16/hf_restore.py` (both public). Clear only named, retired model dirs.

### 3.7 Train (was §4n.32.5, §4n.41.8, §4n.64.1, §4n.73, §4n.69)

`v3_ft/train_lora.py --data $DATA --out $RUN --epochs 12 --patience 3 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
--lr 1e-4 --accum 4 --warmup 20 --max-seq-len 4096 --eval-every Q --save-every Q --seed 1234` (Q ~ rows/16);
expandable_segments default. ~3.8 s/step. Train to early stop (§3.8 explains why the tail matters).
- Checkpoints are `ckpt-<GLOBAL step>` (`ckpt-{step+1+start_step}`), never `checkpoint-<step>`. Resume only through the
  `STEPS.state` ledger (`mb_resume.sh` refuses without it); reading the step off a checkpoint name once ran the cosine LR
  +23..+63 % hot and let legs overwrite each other (cost measured: holdout 3.6255 -> 3.6250, negligible).
- `flush_log` writes `train_log.json` atomically after every eval and save; SIGINT/SIGTERM/SIGBREAK finish the step, save,
  and exit through the normal summary (`status` running/interrupted/complete). `clipforge train` tees stdout into
  `<run>/train_stdout.log` (WSL `/tmp` is wiped on restart). Recover older runs with `clipforge checkpoints --run-dir
  <dir> --rebuild-log --stdout <log>`.
- **Rerunning a chain after its train early-stopped RESUMES the train** (newest ckpt is below the 12-epoch horizon) -
  resume from the screen with a cut-down script (`mb_v15_from_screen.sh`).
- `--init-adapter` warm-starts from an adapter. Community recipe (r16, lr 2e-5, effective batch 8) under-trained on our
  data; reference-conditioned training bought no identity (§8).

### 3.8 Checkpoint pick: pause screen -> ladder, latest clean wins (was §4n.66, §4n.68, §4n.72, §4n.72c, §4n.69)

- **Screen, never pick by loss.** Owen 2026-09-11: *"we changed the ladder ... after TR trained, we changed the ladder to
  use the lowest loss checkpoint ... so the training methodology didnt change, the scoring did."* On mb_v7 the served
  lowest-loss ckpt-704 scored 49 while ckpt-440 (8th by loss) scored 80 with 0 defects; loss anti-correlated with pausing.
- **`clipforge pause-screen`** (`night4/pause_screen.sh` -> `pipeline/higgs_pause_screen.py ref|score|rank`): in-band
  prompts (500-600 chars) x 4 seeds = 16 renders per checkpoint. Defects: loop (tail >= 5 s), overrun (> 1.5x the
  prompt's median length across seeds), truncation (ASR coverage < 0.90), repeat (insert rate >= 0.20). Pausing distance =
  |ln median ratio| + |ln p90 ratio| + 0.5 * max(0, ln((long_r+1)/(long_c+1))), long = interior pauses >= 2 s per 10k
  chars, vs the corpus's own map (`clipforge pause-map --ref`); score = 100 * exp(-d), corpus = 100. Pauses classed by
  the punctuation before them (after-sentence / after-clause / mid-phrase).
- **Rank rule** (Owen: *"the later checkpoint, the better. the voice really settles in the later epochs. but if the only
  clean one is an early one, we should use that one instead."*): every CLEAN checkpoint (0 defects, score >= 70) above
  every unclean one, clean ones LATEST FIRST; below that, defects, score, loss.
- **Screen from the end** (Owen 2026-09-14: *"probably just the later ones, and if there isnt a satisfactory pausing
  checkpoint then we can do the 5 before that ... making sure we clean up the fused models as we go"*): `--batch 5`
  newest first, stop at `--top 2` clean (`--pass-score 70`). The ds_v8 keeper was checkpoint 31 of 31. Merge-test-delete:
  merge only when rendering, `rm -rf` the merge once renders exist, re-merge the picks at the ladder handoff; refuses to
  start under 26 GB free. Point a re-run at the EXISTING `--out-root` so already-scored checkpoints skip.
- **The screen is optimistic** - read the LADDER's pause map before believing a score: ds_v8's screen saw 0.0 mid-phrase
  pauses >= 1 s; the 60-render in-band ladder map found 4.48 per 10k chars vs the corpus's 1.63; ds_v9's screen said 0
  defects, the 120-render map found truncation in both picks.
- **Owen's reading of the defects**: *"the correct pausing checkpoints also have the fewest truncations and loops ... the
  three things are linked"* - one decision (what follows a silence: resume correctly / skip / loop / stop) that loss
  cannot see. Pausing is not monotonic in training (ds_v7 right at epoch 1, off ~1.5-3, back by 3.7). Timbre settles
  early (identity flat across mb_full_rvc1); delivery/phrasing arrive late (*"early epochs sound like the base model with
  a kramer overlay ... prosody is much better in the later checkpoints"*), but *"id rather have a functional model than a
  broken one that sounds like the reader"*.
- Identity column needs speechbrain: score in the env that has it (`orpheus_tts`); speaker refs in
  `night4/speaker_refs/<voice>.json`. ASR on GPU (cuda/float16 0.34 s/render vs cpu 1.46, identical results;
  `--asr-device auto` drops to CPU under 3 GiB free). Record: `temper.sqlite`, report server http://localhost:9876.
- **When nothing is clean the screen walks every checkpoint** (tr_v4nb: 3 batches, ~45 min, 0 clean). For a timed night
  pass `--batch 0 --ckpts "<newest 5>"` - one sweep, then it ranks what it has (tr_v5_chain.sh).
- **A prosody test is not a quality test** (Owen 2026-09-30: *"find one that has normal pausing - just one - and render
  the clip from it. no need for the ladder"*): pick the screened checkpoint whose median pause matches the corpus and
  has the fewest >= 2 s pauses, render the fixed passage, let the ear judge. Ladder only when deployment is in question.
- **Ladder a new model against the DEPLOYED one** for any "does it beat what's live" question: `ln -s
  ~/.crucible/voices/<voice>/cuda-linux /home/telltale/higgs_v3_merged/<voice>_deployed` and pass the link to
  `ladder_chain.sh` beside the merge (it never writes into model dirs; it only `rm -rf`s its own output). Remove the
  LINK with plain `rm` afterwards. Example: `2026-09-29-tr-v4-nobed/ladder_9040_vs_v3.sh`.
- **Owen's ear sample = Mutineers' Moon ch. 8** (`2026-09-21-ds-norvc-10h/mm_sample_bank.json`, 9 chunks, seed 500) -
  hard dialogue. Render every candidate AND the deployed model on it.
- **[HIGH] One render is a noisy grade.** Same model (ds v14), same text and seed, two render days: 4 marks vs 11. A
  6-vs-7-mark difference between models is inside that spread. Weigh Owen's overall impression ("better prosody",
  "impressive in average listening") above the mark count. The marks are in
  `2026-10-02-stall-guard/prosody_marks_mutineers_moon.md`, rounds 1-7.
- **Deathstalker v16's pick:** the ladder's latest clean (3013) beat the screen's best pauser (2489) by ear (6 marks vs
  8, "better prosody"), so the standard rule held.
- **The marks that survive every model are the hard spots, and they are forgivable.** Short, emotionally charged
  dialogue lines ("I'm glad you are!", "get your ass killed, you jerk!") and contrastive stress ("Can you?", "fifth--or
  sixth--time"). Owen 2026-10-09: *"the one or two odd places were very tricky. in average listening, it's
  impressive."*

### 3.9 Ladder and the safe band (was §4n.39, §4n.50, §4n.51, §4n.72b, §4n.72d, §4n.73, §4n.84, §4n.85, §4n.37.14, §4n.37.19-21)

- **Spec** (Owen 2026-09-09): *"the ladder should test two checkpoints, and it should determine which has the fewest
  mistakes in the entire spectrum ... we promote that one, and as part of deployment, we write what the safe range is"*.
  Top two from the screen; 16 rungs 100-1600 x 8 items x **4 seeds** = 512 renders each (2 seeds = n 16/rung cannot
  separate 6 % from 16 %: TR's 900/1000 read 6.2 % at n=16 and 16 % at n=32).
- **Ceiling 1,600 chars, permanent** (Owen 2026-09-21: *"drop it to 1600, make that a permanent piece of the script"*):
  `higgs_ladder.py MAX_RUNG_CHARS = 1600` (items over it skipped by actual chars; `--max-chars` may only lower it),
  `ladder_chain.sh MAXCH=1600` (above exits 2), `make_bank.py` rungs end at 1600, `crucible_render.py
  LADDER_MAX_CHARS = 1600`. **Deathstalker ladders pass `--max-chars 1200`** (*"for deathstalker we can go up to 1,200"*).
  1,600 is a MEASUREMENT range, not a corpus length.
- **The band = cleanest unbroken run of ACTUAL-character bins** (`band_chart.py`, 100-char bins; rung pooling in
  `band.py` is an approximation - a rung is target +-12 % and straddles cliffs), ranked by Wilson 95 % UPPER bound,
  ties to the WIDER band. **Cliff margin**: prefer an edge with a margin even at a slightly worse bound (`band_chart.py`
  warns when a neighbour bin fails > 3x the band). `band.py` measures each run's own pace unless `--pace` is given (a
  fixed 14.5 chars/s once made 5947 look twice as bad).
- **Read two curves**: the failure rate says where the model breaks; the LONGEST-PAUSE curve says where it stops sounding
  like the reader, and that comes first. Deathstalker: median longest pause 0.50 s at 100 chars, 1.38 at 500, 1.54 at 800,
  1.70 at 1400 while failures stayed flat to 1200 - Owen: *"the longer the chunks, the worse prosody gets. lets set the
  safe band to 500 - 800"*. Mistborn's longest-pause curve is flat above 400; its failure rate separates. Use whichever
  discriminates; run the pause map over ALL rungs (`pauses_allrungs.json`, `make_band_report.py`).
- **Served band can be a RULING narrower than the measured band.** ds_v9: measured 400-1000 (0.9 %, hi 3.2 %), served
  500-800 - Owen: *"keep the band at 500-800, for prosody. but record the actual safe band even though it isnt the one we
  use."* Record both.
- `real_difference: false` has been the verdict on almost every ladder - the two finalists are statistically tied and the
  band / screen / Owen's ear decides.
- **The failure curve has two edges** (why a band, not a max): production failed at 323/502/634 chars on TR at target
  700. Working model **[MED]**: the lower edge is the duration prior (~0.85 x corpus p50; short text gets RUN-ONS, not
  early stops - mb_h2lm's 42 production failures were all < 500 chars, 38 of 42 too long); the upper edge is starvation
  where corpus density falls below ~7 % of rows per +-50 chars. Owen 2026-09-25 suspects the curve is not a horseshoe:
  *"i suspect higgs actually just does best around 400 - 1000. above 1000, it starts losing prosody anyway"* - the v16
  corpus (§3.3) tests it.
- Artifacts: `scored.json` is the real output (`higgs_ladder.py table()` prints an empty table for `len#####` banks);
  bank files are `{"items": [...]}`; `--limit-items` is per class; the chain aborts before scoring if fewer than half the
  renders landed. Evidence: `chart.md` per run, appended to `bookforge/electron/data/higgs-safe-bands-evidence.md` at
  deployment (Owen: *"part of deployment is recording the chart/spread that we generate"*).

- **The ladder renders up to 1000 by default** (Owen 2026-09-26: "Change the ladder to only render up to 1000 for
  testing"): `night4/ladder_chain.sh` MAXCH and `higgs_ladder.py render` DEFAULT_RUNG_CHARS = 1000; 1600 stays the hard
  ceiling (`--max-chars 1600` still renders the long rungs). Why: v15 (rows to 1,709) vs v16 (rows <= 1,000) per-rung
  failure - both are safe to ~800 and both fail ~50 % at 1,200; longer training rows did NOT raise the safe ceiling,
  they only softened the collapse past it (1,400-1,500: v15 34-44 %, v16 63-88 %). Inside 400-700 v16 is equal or better.

### 3.10 Pace and assembly gaps (was §4n.65, §4n.72d, §4n.73, §4n.80)

- **Pace** = median chars/s of CLEAN ladder renders (coverage >= 0.95, insert < 0.15, audio ratio 0.8-1.25, >= 150 chars);
  write the IN-BAND figure. Guard edges = median x 1.3 / median / 1.3 (Owen's ruling; the narrator's PaceTracker
  re-centres on the book's running median). Pace is a corpus property, stable across checkpoints (ds 15.91 vs 15.85;
  mistborn 13.29 / 13.33 / 13.76 / 13.83 across retrains). Never inherit pace from a predecessor.
- **Silence definition** (`pause_lib.py`): -40 dB RELATIVE to each clip's own peak, 20 ms hop, pauses >= 0.30 s; tails
  measured with NO floor (`edge_silence()`). The Orpheus `measure_sentence_gaps.py` is -55 dBFS absolute - never mix the
  two tools' numbers. Compare pauses per 10k characters, never per hour.
- **Joins** (`clipforge pause-match`): a Higgs render ends with its own ~0.22 s tail and narrator trims nothing (Owen:
  *"narrator doesnt trim it. the silence of 0.22 is effectively zero. it needs padding."*), so `gapAfter = target_join -
  self_tail`; assembly cannot subtract silence. The gap plumbing exists (manifest `gapBefore/gapAfter` ->
  `assemble/chapters.py`); a per-voice gap SOURCE in the catalog is still missing.

### 3.11 Deploy (was §4n.47, §4n.72d, §4n.74, §4n.80, §4n.84, §4n.92, §4n.94)

**Current path (2026-10-01 TR v4nb, 2026-10-09 ds v16): the voice follows its HF `crucible` tag.** Copy
`2026-10-09-ds-v16-deploy/` with the voice swapped.
1. Write the voice card: `<voice>.crucible-voice.toml`, the export of the live one with pace and band re-measured on
   the shipped weights' own ladder. Pace is the median of clean in-band renders (coverage >= 0.95, insert < 0.15,
   audio ratio 0.8-1.25). The band is Wilson-ranked. Guards are pace x1.3 and /1.3. The cap is the longest training
   row, with `max_chars_basis` kept at the known-valid `"placeholder"`.
2. Check HF private storage (`hf_quota.py`): the peak is current + 8.5 GB until the squash frees it.
3. `hf_deploy_ds.py --dry-run`, then the detached push. Weights, card, NOTICE and crucible-voice.toml go up together,
   then the history is squashed.
4. `verify.sh`: HF `model.safetensors` sha256 must equal the local merge, and `crucible voices check repo@sha` must
   parse the card.
5. `publish.sh` = `crucible/scripts/publish-voice.py <id> <sha>` moves the tag. Then `crucible voices pull <id>` on
   the PC; on the Mac only with Owen's go. Read back the sha256 and `check-updates` = "current at <sha>".
6. BookForge holds no band copy; it reads Crucible (Owen 2026-09-28: voice facts live only where served). Delete local merges once the PC
   copy is verified, and keep the adapters.

Older path, kept for reference: **HuggingFace + both Crucible arms on PC and Mac by pin** (the 09-24 ds v13 / 09-25 mb
v15 deploys; campaign scripts `2026-09-24-ds-v13-deploy`, `2026-09-25-mb-v15-deploy`).
1. Merge the winner; upload to the private repo `owenmorgan/<voice>-higgs-v3` (squashed to one commit); check
   `model.safetensors` sha256 equals the local merge (a 1-min upload was checked for exactly that reason). Keep the repo
   card ours (it had silently become Boson's README).
2. Write the voice manifest `crucible/crucible/voices/<id>.toml`: `[voice.pace]`, `safe_min_chars`/`safe_max_chars`,
   `[voice.serving]`, per backend `hf_repo`/`revision`/`memory_bytes_estimate`/`max_chars`/`sampling`, provenance as
   comments (read `deathstalker.toml` first). Ownership (Owen 2026-09-14): *"let the trainer handle the safety band entry
   and other stuff. it knows how with more specificity."* Contract `crucible/docs/PHASE3-TTS.md`; `voices.py` refuses a
   malformed manifest by name. Repo rules there: commit BY PATH, never `git add -A`, never checkout/reset/stash/switch,
   never push, leave already-dirty files.
3. **Both arms or neither**, pinned to the same `hf_repo` revision (a `path` backend is scratch and splits the arms).
   Pull on PC and Mac (8.50 GB each), verify installed + loadable **by enumeration / read-back** (GET `/v1/voices`) -
   `put_voice.py`'s PUT response has no top-level revision and prints None.
4. `max_chars` is a separate decision from swapping weights (raising it changes what BookForge SENDS). The family cap is
   800 (Owen 2026-09-09).
5. Delete old merges only after the new one is verified; keep adapters.

**Legacy BookForge catalog path** (still used 2026-09-15; `higgs-models.json` still describes an old mistborn):
`night4/promote_voice.py` (`clipforge deploy`) writes `electron/data/higgs-safe-bands.json` + `higgs-models.json`
(arm `backends.served` + `backends.mlx`; `backends.vllm` is Orpheus-only and nothing reads it in the Higgs catalog).
Then `cp electron/data/*.json dist/electron/data/`, `npx tsc -p tsconfig.electron.json` (NEVER `npm run build:electron` -
it `rm -rf`s dist mid-render), `node tools/test-higgs-engine.js` to ALL OK **and exit 0** (a syntax error prints neither),
commit, push, Mac pull + dist refresh. Known gaps promote_voice leaves (all caught by the test): arm/dir notes still
describe the retired dir; `maxCharsSource` must be the enum `catalog|placeholder|length-sweep` (a ruled cap is
`catalog`; prose fails prep by name); `pace` inherited - update IN PLACE with median, p05, p99, n, method, source,
measuredOn; the test hard-codes the production dir/floor. `--band LO HI` records a ruling; supply `holdout` in the
verdict (multi-leg logs cannot answer it, and the tool silently fell through to the wrong checkpoint). Preserve the
JSON format (indent 2, CRLF, non-ASCII literal) - round-trip the untouched file and compare bytes first. `dist/` is
what runs: the app rendered at 0.8 for a day while source said 0.7.

---

## 4. Tools and invocations (was §4n.42, §4n.64.2, §4n.82, §4n.95)

**clipforge is the front door**: `node C:\Users\tellt\Projects\bookforge\cli\clipforge-process.js help`; every verb has
`--help` with the WHY and a field-note reference, and spawns the proven script (reimplements nothing). Owen: *"just make
it the central hub for training tools"*. Machine facts are never guessed: **`--training-root` and `--python` must be
named on every verb** (a chain that named them on `slice` but not `merge-tiers` sat failed 85 min).

    masters -> sentences/align -> cleanup -> slice -> merge-tiers -> gate -> bed -> mix -> train -> checkpoints
            -> merge-lora -> pause-screen -> ladder -> deploy
    diagnostics: pauses, pause-map, pause-match, cut-audit     RVC branch: rvc-dataset, rvc-train

| verb | wraps / notes |
|---|---|
| `slice` | `slice_vtt.py`; passes `--long-max-chars`; warns when `--tail-s` is omitted |
| `merge-tiers` | `merge_corpora.py` (splits `dir:rows` on a colon not followed by `/`) |
| `gate` | `row_gate.py` (Qwen) |
| `book-text` | **The standard text prep, one verb** (2026-09-25; Owen: *"write a verb ... thatll do all of this automatically"*, *"make ai cleanup a part of the process before we train, so what goes in matches what will come out"*): `clipforge book-text --audio <book or web master> --epub <book.epub> --out-dir <dir> --crucible-server <name> --python <py> --training-root <orpheus-finetune> [--name <stem>]` -> generate-sentences on Crucible (silence cut, EPUB diff, alignment, pause-centred edges, cues corrected to the reader) -> the render's AI cleanup (REQUIRED, triaged, model unloaded after) -> strict dialogue on the cleaned text -> `<name>_exclude_ids.txt` (dialogue, barely-matched cues, high pace outliers, cues over non-speech/music) -> `book-text.json`. Slice `<name>_clean.vtt` with `--exclude-cue-ids @<name>_exclude_ids.txt`. Re-runs reuse the cached transcript and cleaner answers. |
| `cleanup` | `clipforge cleanup --vtt <in.vtt> --out <out.vtt> [--crucible-server <name>] [--no-triage]` (also `--lines`): `foundry clean-triage` on the server's CLEAN model (qwen3.5-9b) as act `decide`, then `clean-text --triage <verdicts>`; cue ids/times untouched; `<out>.changes.tsv` lists every changed cue. `clean-lines.js --triage` is the same stage. The model is made resident before EACH act (`loadFirst`, bookforge 70b020c6): releasing the triage's lease empties the card and clean-text then refused 409 not_resident. The triage talks to the Crucible's BASE url (`/v1/decide`), not the `/openai` chat endpoint (2b626804). Answers are cached by text in `<out>.clean-lines/` - a killed run resumes, a re-run over re-aligned cues asks only about new text. Measured on AoL/WoA: triage ~6-7 sentences/s, ~37 % of sentences flagged, cleaning ~30-60 flagged sentences/min - the 9B at bf16 leaves KV room for only ~5 concurrent requests (vLLM `num_requests_waiting` reason `capacity`, `--max-num-seqs 16`); a quantized 9B would roughly triple it (snap's config). Unload after: `bookforge-tts --crucible-unload --server <n> --model qwen3.5-9b` (`--crucible-*` verbs were all broken until 2b626804: an un-awaited async client) |
| `bed` | ownbed at -70 (median-window rule) |
| `train` | `train_lora.py`, tees stdout into the run dir |
| `checkpoints` | ranking + `--rebuild-log` |
| `merge-lora` | `merge_for_serving.py` |
| `pause-screen` / `pause-map` | §3.8 |
| `ladder` | `night4/ladder_chain.sh` -> `higgs_ladder.py`, `band.py`, `band_chart.py` |
| `deploy` | `promote_voice.py` (legacy catalog path) |
| `pauses` / `pause-match` | `corpus_pauses.py` / `pause_match.py` |
| `rvc-dataset` / `rvc-train` | `slice_rvc.py` / `pipeline/rvc/urvc_train.py` (caps `--cpu-cores` at 4) |

**BookForge generate-sentences** (Crucible server on this PC: `crucible@owens-pc-wsl`, 1.0.38 on 09-25):
- Whole book: `generate-sentences --epub <book> --audio <file> --crucible-server <name>` (before fdeef6f3 the CLI never
  reached Crucible and silently used local WhisperX).
- Many short clips of unknown position: `generate-sentences --clips <dir|list.txt> --epub <book> --out-dir <dir>
  --crucible-server <name>` - stitched with 1.5 s gaps into ONE asr job, each clip located by unique 4/3/2-word runs,
  one batched align job; outputs `clips.tsv`, `clips.json` (edge-cut sentences `partial`), `vtt/<id>.vtt`. Not yet run live.
- Windows app + WSL qwen env: the Windows Generate-sentences door refused by name (pre-Crucible note); use the CLI.

**`pipeline/enhance/tone_match.py`** (2026-10-06; Owen: *"id like to do more than just match the tone, id like to make
them sound nearly identical to each other"*, dereverb deliberately left out): makes one recording's CHAIN match
another's, not the performance. `--ref REF --in IN --out DIR [--denoise in|both] [--stages bw,eq,dyn,loud,floor]`.
Stages: RoFormer denoise (roformer_dir.py in the WSL separator env, ~10x realtime on the GPU, ~0.04x on the CPU);
matched bandwidth (each file's speech-over-floor bandwidth on the ORIGINAL audio, both low-passed to the lower);
speech-spectrum EQ (eq_match, 2 passes); dynamics (quantile map of the 50 ms speech envelope, 3 passes - one pass
under-shoots because of the smoothing); BS.1770 loudness (checked: 1 kHz 0 dBFS sine = -3.05 vs -3.01) + -1 dBFS limiter;
shared floor (fills each file's floor UP to the reference's room tone per band, never removes; the target floor gets
the same low-pass). Measures every stage against the reference; `stages/`, `report.json`, `ab.wav`. Floor windows =
those within 6 dB of the quietest 3 % (a fixed quietest-15 % took breaths on dense narration: CoD read -73 by median
but -58 by power). Test (2-min raw Marked Man -> raw CoD, `E:\claude-files\scratch\tone-match`): spectrum diff
1.32 / 3.33 -> 0.33 / 1.04 dB (mean / worst band), loudness exact, dynamics 0.67 -> 0.19, bandwidth both 10.2 kHz,
floor -56.7 vs -79 without denoise (cannot be lowered by adding), -77.8 vs -77.4 with `--denoise both`. Not yet a
clipforge verb.

Other tools worth knowing: `pipeline/temper.py` (+ `temper_db.py`, `temper_report.py`; `npm run temper` / `temper:bg` /
`temper:stop`), `find_invented.py` / `invented_to_cues.py`, `region_edges.py`, `build_web_master.py`, `brighten.py`,
`web_exclusions.py`, `dialogue_cues_strict.py`, `heading_cues.py`, `where_align.py`, `filter_rows_by_pause.py` (drop rows
with an interior pause >= 2 s or tail > 1 s - Owen's fallback lever, unused so far), `eq_master.py` (`--curve`,
per-block, sample-exact), `night4/push_mac.sh`, `night4/probe_vram.py`, `night4/health.sh`, `pipeline/crucible_render.py`
(written against Crucible 1.0.7's wire; see §9).

---

## 5. Serving on SGLang-Omni (was §4n.2-3, §4n.36.6, §4n.37.1-3, §4n.37.10, §4n.44, §4n.45)

- Env `sglomni` (torch 2.13.0+cu130, sglang 0.5.18, sglang-omni 0.1.4, flashinfer 0.6.17 + `flashinfer-jit-cache` cu130).
  JIT needs nvcc: `CUDA_HOME` -> `nvidia/cu13` plus symlinks `lib64 -> lib`, `libcudart.so -> libcudart.so.13`; `sgl-omni`
  dies on a missing `ninja` unless `CUDA_HOME`/`PATH` are exported as `night3/sgl/serve_sgl.sh` does.
- `sgl-omni serve --model-path <merged dir> --model-name <name> --mem-fraction-static 0.48 --tts_engine.factory.max_running_requests N
  --tts_engine.factory.cuda_graph_max_bs N --tts_engine.factory.max_new_tokens <cap>`, port 8200, `POST /v1/audio/speech`.
- **Context is 4096 tokens** - a class attribute (`sglang_omni/models/higgs_tts/engine_builder.py:29`), patched to read
  env **`HIGGS_CONTEXT_LENGTH`** (`night4/patch_ctx.sh`, idempotent, `.orig` kept; `serve_sgl.sh` exports it default 4096;
  `launch_serve_sgl.sh` carries it across tmux). A second cap `MAX_SPEECH_INPUT_CHARS = 4096` (`serve/speech_service.py:58`)
  reads `HIGGS_MAX_INPUT_CHARS`. Stages are `preprocessing, audio_encoder, tts_engine, vocoder` - there is no `thinker`;
  `--thinker.*`, `--tts_engine.factory.context_length` and `--tts_engine.engine.context_length` all FAIL.
- **Symptom**: a render run finishing in ~2 s with `0/640 ok` and `Rejecting request ... before scheduling`; the error
  text's advice (raise mem fraction) is wrong - `kv_capacity=4095` is the per-request context, and the server reserves
  `max_new_tokens` up front.
- **Sizing**: 4096 tokens ~ 2,000 chars at TR pace; 1,600 chars ~ 3,743 tokens (just fits). KV pool at 0.48 = 23,173
  tokens shared: `conc x (max_new_tokens + input) <= 23173` (e.g. `--max-tokens 5000 --conc 4`). Too high refuses every
  request; too low manufactures truncations.
- **Memory**: `--mem-fraction-static` sizes only the LM stage; the other stages allocate on top. 0.48 + render conc 4
  ~ 20 GB total (0.60 summed to 24.2 GB and thrashed). The ladder's server was seen holding 22.7 GB (not traced).
- References: `SpeechReference` takes base64 `data` + `media_type`; 10 MiB, 100 s server ceiling (narrator keeps 30 s);
  context cost ~25 tokens per reference second.
- Why SGLang: vllm-omni 0.28's batched talker damaged the newest batch row at 16 concurrent (truncations, babble,
  sustained voice switches in 6-12 of 50 chunks; 0 at concurrency 1); SGLang-Omni at 16 was clean on those and ~2.5x
  faster.
- `launch_serve_sgl.sh` refuses a missing/non-dir model path (`BAD_MODEL_PATH`, exit 4) and quotes every argument (an
  empty first arg once shifted all of them). Its server guard ignores `pgrep|grep|bash|sh|zsh|tee|xargs|watch` and prints
  what it matched (a watcher's own argv once tripped `A_SERVER_IS_RUNNING`).
- Mac arm = mlx-audio in narrator: loads the merged safetensors directly (no conversion); 0.4.8's higgs_audio_v3 does not
  read generation_config.json and has no repetition penalty - BookForge passes sampling explicitly. A wrong-voice Mac
  render was the app submitting the catalog's `default` voice, not the loader.
- **[HIGH] PC renders hiss on EVERY voice; the cause is SGLang-Omni, not the weights, CUDA or the training data
  (2026-10-24/25, Owen by ear).** Same Mistborn passage (Mutineers' Moon bank chunk 00), same weights, T 0.8:
  - PC Crucible (sglang-omni): hiss. Owen: *"sure enough, the pc has the hiss, the mac does not"*.
  - Mac Crucible (MLX): clean.
  - Plain PyTorch on the SAME PC GPU (`v3_ft/render.py --snapshot <merged dir>`, whole-clip codec decode): clean.
    Owen: *"Torch-mistborn does not have the hiss"*.
  - **The meter misses it:** the 4-6.4 kHz pause floor read PC -82.7, Mac -80.0, PyTorch -92.1 / -88.1 dB. The Mac,
    which is clean by ear, measures LOUDER in pauses than the PC. The audible hiss is under the speech, not in the
    pauses. Judge it by ear, with the Mac as the reference.
  - **Streaming vocoder windows RULED OUT (as replayed):** `vocoder_scheduler.py` decodes 20 frames first, then
    75-frame steps with 8 frames of left context, no look-ahead, butt-joined. Replaying that exactly on clean PyTorch
    codes (fp32, GPU, eager) differs from a whole-clip decode by -49 dBFS (speech -29), but Owen hears NO hiss in
    either (*"Hiss is not present in either of those two renders"*; `2026-10-23-om-v10/hiss_check/stream_sim/`). GPU vs
    CPU decode is also clean.
  - **CAUSE (found by crucible pc 1, 2026-10-25): the sglang vocoder decoded in bf16.** `create_vocoder_executor`
    defaults to bfloat16 and narrator never set it, although `HiggsAudioCodec.from_pretrained` itself defaults to fp32
    (reading that default is why "the codec is fp32" was wrongly assumed here). Same codes through sglang's own codec:
    the bf16 floor is +15-18 dB across 4-11 kHz versus fp32; Owen confirmed by ear. Fix: narrator dc05e7e7 passes
    `--vocoder.factory.dtype float32`; shipped in Crucible 2.0.24 (PC and Mac). The streaming windows were a red herring.
    **Rule: when a decode path sounds wrong, check the dtype the SERVER actually runs, not the class default.**
  - **Cost:** a day of retraining v10/v10b/v10c (EQ, RoFormer separation, Adobe-web audio, top shelf) that could not
    have helped. Every PC-rendered hiss judgement before 2026-10-25 is suspect (§7 Deathstalker v15sep).
- **vllm-omni offline run (for the hiss A/B), not working yet:** `examples/offline_inference/text_to_speech/
  higgs_audio_v3/end2end.py` pattern; the script needs an `if __name__ == "__main__":` guard (spawn). The Higgs deploy
  yaml pins `attention_backend: FLASHINFER`, so `VLLM_ATTENTION_BACKEND` and `VLLM_USE_FLASHINFER_SAMPLER=0` are ignored
  and FlashInfer (0.6.16.post3 in `higgs3`) JIT-compiles at warm-up: no nvcc -> "Could not find nvcc"; with
  `CUDA_HOME=nvidia/cu13` -> "CUDA compiler and CUDA toolkit headers are incompatible". Next step: the prebuilt
  `flashinfer-jit-cache==0.6.16.post3` (cu130) in the clone `higgs3v`. Crucible has not used `higgs3` since 09-13
  (its own envs live in `~/.crucible/envs`).

---

## 6. Pitfalls and traps

### 6.1 WSL / Windows / shells (was §4n.46, §4n.82, §4n.37.8, .11-12, §4n.34.2-3, .7, §4n.56, §4n.64.5, §4n.86, §4n.93, §4n.94, §4n.68.2)
- **Git Bash rewrites `/mnt/...` arguments** (`wsl bash /mnt/e/x.sh` -> `C:/Program Files/Git/mnt/e/...`): prefix
  `MSYS_NO_PATHCONV=1`. It also turns `dir:rows` pairs into `A;B` path lists - pass `cygpath -m` paths.
- **`wsl -- bash -c "..."` loses shell variables** and `$(...)` expands on the WINDOWS side to nothing (a `kill
  $(pgrep ...)` kills nothing; pass PIDs literally); heredocs mangle backslashes (`"\\n"` became a real newline).
  **Write the script to a file, then run the file.**
- Python `open(..., "w")` on Windows writes CRLF and WSL bash dies at line 1; Git Bash's `bash -n` passes it anyway -
  syntax-check with WSL's own bash (project memory, 2026-09-20). `wmic` also emits CRLF (strip `\r`).
- Windows exes launched from WSL get Windows-style ARG paths (`/mnt/e` -> `C:\mnt\e`, ENOENT) - pass Windows paths.
- **Bash reads a running script by byte offset**: never edit a live script except appending/replacing its tail at the
  same byte length; otherwise stop, edit, restart from markers. Don't rename dirs under a running job.
- **`tmux -t name` matches by PREFIX** (`kill-session -t night4` killed `night4b`, the runner issuing it) - use `-t =name`;
  coordinate chained runners by marker files.
- **`pgrep -f` / `pkill -f` match whole command lines**, including the poller's own argv: bracket literals in any polling
  loop (`[s]gl-omni serve`); `pkill -f` once killed its own parent shell.
- **Windows process visibility**: Git Bash `ps` cannot see `python.exe`; `taskkill //PID n //T` silently did nothing -
  use PowerShell `Get-CimInstance Win32_Process` + `Stop-Process -Id n` (from a .ps1 file; an inline pattern matches the
  shell running it). Stopping a background task kills only its wrapper - list and kill the children.
- **Launching long work**: Claude Code's low-memory guard kills its own background shells when Windows free RAM runs low
  (CPU merges, Crucible pulls fill WSL page cache); background jobs launched through the agent's task tracker die with
  it. Launch in WSL tmux, or PowerShell `Start-Process` on a LOGIN bash (`bash -l`; a non-login detached shell had no Unix
  PATH), and prove the launch with a log line.
- **A Claude Code "killed" background notice does not mean the process died** (2026-10-06): the notice cut the agent's
  link to the shell, but the 12:18 `after_sep.sh` waiter was still alive an hour later and would have run prep twice.
  Enumerate (`Get-CimInstance Win32_Process`, `pgrep`) before relaunching. A `wsl -e bash -c "pkill -f <pat>; ..."`
  matched its own command line and killed itself mid-list (exit 15); use separate calls or bracketed patterns.
- `tail -F` on a log another process appends to from PowerShell blocks its `Add-Content` - use `tail -n`.
- **WSL**: kill jobs inside it freely; never `wsl --shutdown`/`--terminate`, never kill vmmem/wsl.exe (Owen: *"force
  killing jobs inside wsl is fine. just be careful about killing wsl itself"*). Settings: `autoMemoryReclaim=dropcache` +
  `wsl-cache-trim.service` (09-25). The vhdx is sparse: `fstrim` returns space to C: live. C: filled once when the vhdx
  hit 757 GB of screen merges - WSL then would not start. The 13 GB WSL cap bus-errored when CPU scoring overlapped a merge.
- **Cross-OS GPU checks must ask the card**: pgrep across the Windows->WSL boundary silently returned "free" mid-render.
  Use Windows-side `nvidia-smi --query-gpu=memory.used`; a non-numeric read = busy; require six quiet reads at 30 s
  (ladder_chain idles ~70 s between servers).
- **Mac from WSL**: WSL needs its own `~/.ssh/config` `Host mac` block (`owens-mac-studio.hs.owenmorgan.com`,
  `~/.ssh/mac_key`). The Mac ships openrsync: `-s` kills the transfer (code 12), and a backslash-escaped `Application\
  Support` path SILENTLY creates a literal dir. Stage into a space-free path (`/Volumes/Callisto/staging/models`) then
  `ssh mac mv` (`night4/push_mac.sh`), and sha-verify at the destination. Mac runs need Owen's go.
- Exclusion lists exceed Windows' 32 KB command line - pass `@file`.
- **ffmpeg can wedge on Windows at exit** (2026-09-26, HoA window 461 of 955): the output file was complete but the
  process never exited, 0 CPU, one thread in Wait, and node waited on it forever with no log line for 10 min. Stop-Process
  marked it exited but the kernel object held 96 handles and node never got `close`; a 09-24 test ffmpeg is wedged the
  same way. It struck AGAIN at 13:24 on The Coming of the Third Reich (window 598 of 1,434). ROOT FIX (bookforge
  43fd4979): the aligner windows are no longer one ffmpeg each - `cutWindowsStreamed` slices every window out of ONE
  streaming decode (sample-equal to the old cuts, tools/test-cut-windows.js; 1,434 windows in under a minute).
  Watchers still flag "no log has grown in 10 min" - GPU-idle checks miss CPU-only stages.
- Never name a scratch file after a stdlib module (`C:\tmp\inspect.py` broke every python run from that dir).
- Ask which machine before diagnosing ("wired memory" was the Mac, Owen meant the PC GPU).

### 6.2 GPU / memory (was §4n.37.10, .15, §4n.93 trap 1, §4n.68.1, §4n.77, §4d)

- **The Qwen ASR worker fragmented the card into shared memory** (2026-09-26, v16 gate): measured peak ALLOCATED 7.4 GB
  (800 s batches of up to 100 s clips) while RESERVED climbed 7.7 -> 10.2 -> 12.8 GB in 110 clips and past 21 GB by
  1,400; the card filled, 3.3 GB spilled into shared system memory and Windows commit hit 69.6/69.9 GB (Owen: "we're
  paging"). Not two jobs - the gate was alone. Fix in `pipeline/asr_qwen.py`: `expandable_segments:True` (the
  trainer's 09-02 fix) + a hard per-process ceiling `--vram-cap-gb 12`; row_gate's batch budget 800 -> 400 s of audio
  (Owen: "no more 800 seconds at a time"). Verified: reserved tracks peak, 5.6-5.9 GB flat. Every asr_qwen run now
  prints alloc / peak / reserved per batch. A detached watcher (`2026-09-26-mb-v16/gpu_shared_watch.ps1`) logs ALERT
  when GPU shared usage rises 1 GB over baseline or dedicated passes 22 GB.
- **The lyrics trainer's cache crept to fill the card** (2026-10-11, lyrics v3, Qwen3.5-4B LoRA, batch 2x8, len 3072):
  peak ALLOCATED stayed 19.8-19.9 GiB, but the process grew from ~21.6 GB at launch to 22.9 GB by step 1170 (card
  24.1/24.6 GB with the desktop's 1.25 GB). At step ~1170 steps went 9.1 -> ~24 s, 100 % util at 170-420 W bursty, no
  throttle reasons, adapter Shared Usage only 0.22 GB - so a near-full card stalls even when the shared counter barely
  moves; the s/step line is a CUMULATIVE average, so it crept 9.15 -> 10.11 while the real rate had more than doubled
  (diff two log lines). Fix in `pipeline/lyrics/train_lyrics.py`: `--max-vram-gib` (set_per_process_memory_fraction;
  v3 resumes at 21.5 GiB) + reserved printed beside peak; `--best-loss` so a resume into the same run dir only moves
  best/ on a better eval (it started at inf and would overwrite with any first eval). Per-process `\GPU Process Memory`
  dedicated numbers (QuickLook 16 GB, dwm 8.6 GB) are artifacts - read `\GPU Adapter Memory` and nvidia-smi.
- **Full VRAM on Windows does not OOM - WDDM pages to host RAM and the whole machine thrashes 4-10x slower with no
  error.** Check `\GPU Adapter Memory(*)\Shared Usage`, not just nvidia-smi; a stage that suddenly runs 3x slower is a
  machine symptom until the throughput line says otherwise (`night4/health.sh`). `mem_get_info` lies under WSL - use
  `night4/probe_vram.py`.
- **GPU lock** `bash $H/v3_ft/gpu_lock.sh acquire` (file on the Windows side: `%APPDATA%\BookForge\external-gpu-job.lock`,
  survives reboots). Our locks carry `pid` + `boot` id; a waiter clears OUR lock only when that pid is dead or the boot
  differs; never judge another owner's lock stale. VRAM < 4 GB gate still applies. Ollama passes take the lock and hold
  the card until the pass EXITS (unloading mid-pass is undone by the next call; a killed pass writes nothing).
- **Never kill a GPU process while another CUDA job shares the card** (a Windows-side kill coincided with `illegal memory
  access` in a WSL job); never SIGKILL a WSL GPU process. Edits wait for a stage boundary.
- Budget the whole pipeline; CPU gates at 2-4 workers, never beside serving at 6 (`mkl_malloc` failure was the only message).
- Qwen asr ~15 GB per Crucible job (weights 4.5 + 3.5 KV + vLLM + aligner), fixed per job.
- RVC: `--cpu-cores` defaults to all 20 and each worker maps CUDA DLLs -> Windows commit limit ("paging file is too
  small"); concurrency 2 on this box (1 = 23.4x at 4.0 GB, 2 = 49.1x, 3 = card full, slower than one).
- **Training commits ~30 GB through vmmemWSL** (2026-09-30, tr_v4nb): vmmemWSL private 29.9 GB while its working set
  sits at the 13 GB cap (GPU mappings count against Windows commit); Windows commit ~67-70 of 72.8 GB for the whole
  run. Model load paged hard for ~20 s (3,900 pages in/s, 0 MB available), then settled (~85/s). Later Windows trimmed
  the VM to 1 GB resident (8.4 GB in Memory Compression, ~3,000 pages in/s sustained) while the step rate stayed normal
  (3.6 s/step, 390 W) - the GPU-bound trainer was unaffected. Closing Owen's desktop apps (Blender, ChatGPT, Discord)
  dropped paging to ~5-50/s and commit to 67.6 GB. Judge a stall by the step rate + GPU watts, not by page-ins alone.
- **Higgs v3 training pace on the 3090 Ti is ~3.55 s/step** (tr_v4 10,396 steps in 10 h 13 min; tr_v4nb 3.57 s/step).
  Estimate a run's end as (early-stop step x 3.55 s); do NOT divide by the 12-epoch horizon (that error gave "1.7 s/step").
- Claude Code's background shells are reaped after ~30 min idle and under memory pressure: overnight checks are cron
  one-shots (Owen 2026-09-30: one halfway, one at the expected end, a third if the chain isn't done).

### 6.3 Serving / ladder (see §5, plus)
- A pipe to `| tail` hides exit codes on both sides - redirect to a file and echo `$?`; ladder scripts use `set -o pipefail`.
- `$(grep -c ... || echo 0)` yields two values when the file is absent - a guard that errors is not a guard.
- **[HIGH] A screening voice left in Crucible broke every app's model pickers (2026-10-25, OM v10).** For the hiss
  A/B, two checkpoints were installed as home voice manifests: `~/.crucible/voices/omv10b-869.toml` on the Mac
  (`[voice.backends.mlx-darwin] path = /Volumes/Callisto/claude-files/scratch/om_v10b_869`) and `omv10c-342.toml` on
  the PC. These are local-source voices (a path, no `hf_repo`). Crucible's catalog skips local-source voices
  (`catalog._voice_subjects`), but the tts candidate list (`fit`) still offers them. So the Mac's `GET /settings` returned
  500 `catalog_incomplete` ("the catalog has no row for 'omv10b-869', which tts offers as a candidate"), and EVERY
  client's model picker went blank (Briefcase: "No models available", the LLM list too). Found by crucible pc 1 and
  briefcase mac 1; fixed by deleting both .toml files (Mac `/v1/settings` back to 200). The catalog gap itself is
  being fixed in Crucible.
  - **Why it happened:** the manifests were written as throwaway test fixtures, with no expiry, no ledger entry and
    no cleanup step. They outlived the test by a day, and nothing tied them to the campaign.
  - **Rules:** (1) Prefer rendering a screening checkpoint WITHOUT installing it as a Crucible voice: bare sgl-omni on
    the PC, or plain-PyTorch `v3_ft/render.py` (§5). (2) If a test needs Crucible's own render path, install the
    manifest, record its path in the campaign folder's notes (and as a temp file in the ledger, 1-2 d expiry), and
    delete it the moment the test is done - before reporting the result. (3) After installing ANY voice on a box,
    check that `GET /v1/settings` still returns 200 before walking away. (4) At campaign end, `ls ~/.crucible/voices`
    on the PC and the Mac; only deployed voices (pinned in `pins.toml`) and `higgs-default` belong there.
- Mac ladder via Crucible (2026-09-27, ds_v14): Crucible 1.0.49's job document has NO per-chunk rows, so
  `crucible_render.fetch` recorded `audio_s: null` and `higgs_ladder.py score` died dividing by it. fetch now measures
  each wav's duration when the row is absent (the wav is the truth). The Mac runner (2026-09-19-crucible-ladder/
  mac_ladder_runner.sh) also now renders only classes <= 1000 chars (handoff `maxch=`), matching the PC's MAXCH -
  it used to render every rung up to 1600, so the two ladders were not comparable.
- PC Crucible ladder (2026-09-27, sigma v2) FAILED on 1.0.43: every take is its own job, the voice unloads between
  jobs, and narrator relaunched its SGLang server into the previous one still leaving port 8200 -> engine_failed
  ("answering /health on port 34xxx, NOT on port 8200") and an orphaned sgl_omni_entry.py holding 13.4 GB (next job:
  409 accelerator_busy). Reap: kill the ppid-1 sgl_omni_entry tree. Crucible main fixes it (fdd903a free port per
  launch, 7f41f03 unload waits for narrator's processes; not on the PC until Owen says). Our fix: crucible_render
  `render` now holds a LEASE for the whole ladder (load-voice job with params.lease {act tts, ttl 600}, heartbeat
  every 200 s, DELETE /v1/leases/{id} in a finally) - UNTESTED LIVE as of this note; smoke it before the next ladder.
  Also: never run our direct SGLang (port 8200) beside a Crucible Higgs job on 1.0.43. Sigma v2 was laddered on
  direct SGLang (Owen's call).

### 6.4 Alignment (was §4n.40, §4n.35.3, §4n.93, §4n.2x)
- **Library VTTs** (pre-Crucible aligner): cue START lands inside the first word, cue END on the next sentence's onset
  (the whole pause lives in the preceding cue). Slicing depends on the waveform-based start/end cuts to fix this.
- qwen3 forced aligner publishes no confidence and never refuses: its misses concentrate on headings and tiny chunks whose
  printed text differs from the spoken words, and the prose right after them - exactly the rows most worth listening to.
- Books assembled from multiple sources drift against the m4b (FE -216 ms at 23.6 h) - locate, don't trust times.
- A concatenated master's asr-fallback loss tracks span COUNT (FE 133 spans lost 2.71 of 5.99 h), not audio quality.
- Footnote markers and heading glue leak into aligned text (§3.2).

- ROW GATE DROPPED FROM THE PROCESS (2026-09-29, Owen: "Drop the gate as part of the process. We already have that worked
  out and don't need it"). book-text already verifies every cue's text against its own audio (Qwen long pass, re-hear,
  Whisper second opinion, misplaced-cue drop at agreement < 0.3), so re-ASRing the sliced clips was redundant and cost
  hours (TR v4: ~6.5 h). Order now: book-text -> slice (+ cut-audit) -> merge-tiers -> bed -> mix -> train. `clipforge bed`
  no longer requires row_gate.json (bookforge 73b7d0f4). First corpus trained ungated: Third Reich v4.
- Book-text DATE defect (2026-09-28, Owen's spot-check of a Third Reich clip): the book's "1 December 1933" read as
  "December first nineteen thirty-three" came out "December December 1933" - month names are capitalised mid-sentence,
  so the name rule let book "December" match any heard word and the day was substituted away. 1,039 TR cues (tc 142,
  tp 323, tw 574); none in deathstalker/Mistborn (no day-month dates). Fixed in bookforge 5a1a30a7 (mergeDates: day +
  month (+ year) is one value-keyed token, exact match only; the reader's order wins in spoken form). TR text re-run
  and v4 re-prepped. Check any corpus with dates: grep -iE '(Month) ' on the clean VTT must be 0.
- Crucible's speech detector (asr `speech_only`, Silero VAD; default ON from Crucible 1.0.52 unless vad_filter is true)
  A/B 2026-09-28 (campaign 2026-09-27-speech-only-ab): 15 min of Third Reich in Power, qwen3-asr-1.7b,
  off 2,008 words vs on 2,009, agreement 0.9963, word starts median 0 ms / p95 24 ms / max 184 ms, 11.8 s removed; the
  only diffs are number spellings. Harmless: book-text runs on 1.0.52 with it on (BookForge c43b3a02 can pin it false).

### 6.5 Data hygiene (was §4n.71, §4n.34.4, §4n.58.5, §4n.93 trap 4, §4n.68.2, §4j)
- **Separators return their own rate**: `audio-separator` always writes 44.1 kHz; `pc_sep_book.py` dropped the rate and
  length-matched, silently keeping 54.4 % of every 24 kHz McKinley part (348/348 rows failed the gate). Carry the returned
  rate and resample (`soxr.resample(y, ysr, sr, quality="VHQ")`) before any sample-exact match. `mac_sep_book.py` has the
  same latent bug.
- **Resampling**: 48k->24k by `scipy.signal.resample_poly(x, 1, 2, window=('kaiser', 12.0))`; `ffmpeg -af
  aresample=resampler=soxr` wrote silently EMPTY files at exit 0.
- **A WAV cannot pass 4 GB**: a 24-bit 9.1 h WAV read back as 8.28 h; a 7.2 h 24-bit stereo separation died writing after
  3 h 14 min. Write long intermediates as FLAC (or per chapter - also the only resumable form).
- `slice_vtt.py --min-start` defaults to 300 (project memory, 2026-09-15): omitting it deletes the first 5 min of every
  master (fatal on per-chapter masters). Name it explicitly.
- `np.convolve` for a sliding energy is O(N x M) (looked hung) - use a cumulative sum.
- Score transcripts against the VTT/corpus text, never the old cutter's pool metadata (it wrote years as cardinals).
- A subprocess that exits 0 with short output is not success - count the rows (normalize_text truncated at 64 KiB).
- **AAC decodes go over 0 dBFS** (2026-10-23, OM v10): God's People's m4a chapters decode to +2.5..+3.3 dBFS float (a
  0 dBFS limited master plus AAC overshoot). Decoding straight to an INTEGER wav (`pcm_s24le`) hard-clips every over.
  Decode lossy sources to `pcm_f32le`, then take a constant gain before anything integer. OM v10 used -6 dB on both books.
- **`eq-match` levelled each block on its own** (fixed 2026-10-23, orpheus-finetune `eq_master.py` + `eq_match.py`): with
  `--ref`, `eq_match.match()` scaled any 9-min block whose EQ'd peak passed 0.99 down to 0.99, silently, so a master
  peaking near full scale came out with a level step at every block seam (the clip warning never fired - the rescale ran
  first). Now blocks are never normalised; `--headroom-db` comes off the whole master in both modes (default 0 for `--ref`,
  2 for `--curve`) and residual clipping is reported: raise `--headroom-db`, re-run. `match()` keeps its per-clip
  normalise for the RVC corpus pass (`normalize=True` default), where one clip is one file.
- **`clipforge mix` dropped `--no-pause-cap`** (fixed 2026-10-23, bookforge `cli/clipforge-process.js` runMix): its pass list
  forwarded only valued flags, so the switch never reached `build_higgs_mix.py` and the 1.5/2.5 s cap applied. Switches
  (`--no-pause-cap`, `--select-only`) now pass through, as do `--short-min`, `--micro-*`, `--long-max-s`. Mixes before this
  date built through the verb (not by calling the script, as mb v18 did) were capped.

---

## 7. Voices

| voice | deployed | HF revision (2026-09-26, carries crucible-voice.toml) | band | pace | cap |
|---|---|---|---|---|---|
| deathstalker | **ds_v14 ckpt-3510** (2026-09-27; v13 removed) | owenmorgan/deathstalker-higgs-v3 @ **28f6c4d3** | 400-700 (own ladder; rescored 2026-10-02: 0/92 in band) | 16.14 | 800 |
| mistborn | **mb_v16 ckpt-3375** (redeployed 2026-09-28 over v17; CALLED 2026-10-02) | owenmorgan/mistborn-higgs-v3 @ **8a8d1bf3** | 300-800 (own ladder) | 13.56 | 800 measured |
| thirdreich | **tr_v4nb ckpt-9040** (2026-10-01; was tr_v3 ckpt-7084 @ f83ce206) | owenmorgan/thirdreich-higgs-v3 @ **3e96bea6** (`crucible` tag; PC + Mac pulled, sha256 f1762425) | 500-800 (own ladder, 6/128) | 14.94 (79 in-band clean) | 1000 placeholder (Owen's ruling) |
| owen | see below | owenmorgan/owen-morgan-higgs-v3 @ **5885e746** | 600-800 (predecessor; method disputed) | 16.32 inherited | 800 placeholder |
| sigma | **sigma v2 ckpt-4092** (2026-09-27) | owenmorgan/sigma-higgs-v3 @ **8f6b3ca2** | 400-800 | 15.94 | 1000 |

**Deploys now go through the HF `crucible` tag** (2026-09-28): push weights + `crucible-voice.toml` (check with
`crucible voices check <repo>@<sha>`), `python scripts/publish-voice.py <id> <sha>` in the crucible repo (moves the tag;
runs from Windows since 2026-10-02 - Owen's fine-grained token is in `~/.cache/huggingface/token` - or from WSL with
`~/.crucible/server/bin/python3`), then `crucible voices pull <id>` on each box. No explicit pins; BookForge copies no
voice facts. Worked example: `2026-10-01-tr-v4nb-deploy/` (hf_deploy_tr.py, verify.sh, publish.sh, mac_pull.sh).

**State 2026-10-02.**
- **Called** (Owen): Mistborn stays on v16 (*"The deployed mistborn sounds fine. I'm calling it on training
  mistborn"*); Third Reich 9040 needs no more work (real failure ~1 % after the scorer fix, §3.5.3); Owen Morgan kept as
  is (*"lets just keep owen morgan as is"*).
- **In progress: deathstalker on Adobe-WEBSITE audio** (`2026-10-02-ds-adobe-full/`). Owen: *"grab the original
  deathstalker corpus (pre adobe-premiere), split them into 59 minute parts, and ill send them through adobe web"*;
  *"lets not train with these right now. we dont have enough yet. but we can process them"*.
  - Inputs: `pipeline/enhance/build_adobe_upload.py` on the RAW masters, with the current book-text VTTs
    (`2026-09-26-ds-text/<bk>_clean.vtt`) and `<bk>_exclude_ids.txt` as the drop list.
  - The raw and Premiere masters have identical durations (0.000 s), so the VTTs fit the raw timeline.
  - Parts 01-04 of the 2026-09-22 upload (all d3k, 3.71 h) were already returned and are reused via
    `already_web_spans.json`.
  - 26.0 h new = 30 parts (`parts/ds_full_partNN.wav` + adobe_map.json), Enhance v2 70 %, background 0, music 0.
  - Returns so far: parts 01-04. **d3k is complete**: `masters/d3k_web_48k.flac` (build_web_master.py, two maps,
    733 spans located, 0 dropped, ±0.5 ms, 6.15 h website audio). Each book gets built when its last part returns
    (build_web_master exits on a missing return).
  - Then: eq-match tone across the 5 books, re-run book-text on the new masters (picks up 7f63b324), slice rows <= 1000
    with NO bed and NO pause cap (the stall guard covers stalls, §3.5.2), v14's recipe, compare by ear to v14.
- **ds_v15 TRAINING 2026-10-06 02:10** (`2026-10-06-ds-v15/`; Owen: *"gpu is free. lets prep the parts to train and
  start it when its ready"*). Returns 01-20 complete three books: d3k, cd (parts 04-11), lp (11-19); Marked Man (needs
  21-23) and McKinley (24-30) are not in it.
  - Masters: `build_web_master.py` per book (cd 61 spans, lp 50, 0 dropped, shift <= 0.1 ms). Adobe darkened every book
    2-3 dB against its raw source (raw -15.7/-15.6/-14.9, web -17.6/-18.6/-17.3 dB), so all three were matched to the
    narrator's RAW tone (`tone/brighten_ds.py` = Mistborn's brighten.py with the raw d3k/cd/lp masters as reference):
    -14.85 / -15.02 / -14.84 dB, `tone/masters_eq/<bk>_web_eq_48k.flac`.
  - Text: the 09-26 book-text VTTs, NOT re-run (Owen asked; the raw and website masters share the timeline exactly).
    The 7f63b324 cue-swallow defect is excluded instead: `excl_v15.py` drops every cue squeezed under 0.15 s with both
    neighbours and any cue over 150 words (lp's 325-word cue) - 8 cues - plus cues outside the website spans (1 s
    margin) and book-text's own exclusions. Kept cue time: d3k 4.84, cd 6.26, lp 6.40 h.
  - Slice: v16's rule (`--long-max-chars 1000 --run-median-s 45`, `--min-start 120`); cut-audit 0/2/2 cuts near a word
    of 1,858/2,420/3,214. Merge ds15_l 1,327 / ds15_s 2,187 train rows. No gate, no bed.
  - Mix: `--nobed --no-pause-cap`, 1,458 train rows, **17.05 h**, chars p50 636 / p75 829 / max 1000. Train: v14
    recipe, horizon 4,374 steps, **3.36 s/step**, 15.9 GiB.
  - Screen, ladder and the Mutineers' Moon sample SERVE WITH THE STALL GUARD (`HIGGS_STALL_GUARD=37,1,20,16`, exported
    by the chain after training). Without it a no-bed Adobe model stalls on every checkpoint (mb_ab_nb) and the screen
    would find nothing clean. Crucible's v2 patch (`crucible/envs/tts/patches/patch_sglang_omni_stall_guard.py
    /home/telltale/anaconda3/envs/sglomni`) is applied to the test env; it is inert unless the variable is set.
    `night3/sgl/launch_serve_sgl.sh` now forwards the variable into the tmux server (tmux does not carry the caller's
    environment; an EMPTY value is a startup error under v2, so it is forwarded only when set), and serve_sgl.sh logs
    `STALL_GUARD=` in serve_events.log.
  - Prep broke once on our side: the n6 -> n18 normalizer re-vendor (6795d3c) had missed `light-gate.js` and
    `caps_acronyms.json`, so `normalize_text.js` could not load and slice_vtt died. Fixed in 4a816da (light-gate.js
    compiled from the PINNED commit's .ts, because BookForge's dist is n19 now; the transpile reproduces HEAD's dist
    byte-for-byte). `run_fixtures.js` passed throughout because it never loads light-gate - smoke `normalize_text.js`
    itself after any re-vendor.
  - **Result (2026-10-06): NOT deployed, v14 stays.** Early stop 2002 (best holdout 3.8218 @ 1729, ~5.5 epochs; v14
    ran to 3510). Screen: newest 5 all clean with the guard on (no stalls; max pause 3.7-4.7 s vs corpus 3.38). Ladder
    2002 400-900 6/192, 1911 400-600 1/96, inside noise -> 2002. But the model holds pauses long: >= 2 s 9.05 per 10k
    chars vs the corpus's 2.88, p90 1.68 vs 1.16. **Owen's ear (Mutineers' Moon, round 3 in
    `2026-10-02-stall-guard/prosody_marks_mutineers_moon.md`): 11 marks** - wrong emphasis on "Can you?", "If that
    thing", "trip any waiting detectors", "sheer south wall", "God, how he hoped he was", "food sitting out"; a long
    pause after "maybe" and after "another security com" (v15's chunk holds 3.32 s there, v14 1.22 s). v14 on the same
    text and server had 4. v15 differs from v14 in several things at once (no bed; Adobe website vs Premiere; three
    books, 2/3 nonfiction, no Marked Man; 17 h; earlier stop). The bed is NOT the suspect: the Mistborn A/B (§3.5.1)
    already tested it on Adobe-website audio with identical rows, and no bed was slightly BETTER on prosody (Owen
    2026-10-06: "didnt we already run the bed test on mistborn"). Lead suspect: book mix / genre (Owen's 10-02 finding
    that genre drives this passage); then hours and the early stop. Next: the full five-book train once Marked Man
    (parts 21-23) and McKinley (24-30) return.
  - **Mark attribution correction:** Owen's first 11 marks were on the FRESH v14 render (he took it for v15); v14's
    10-02 render of the same text drew 4 - one-render ear grades vary a lot between renders of the same model. v15's
    own marks (round 4, preliminary): "let me go" lackluster, long pause after "Maybe" (1.74 s), bad emphasis on
    "Can you?", "But he's not going to", "I'm glad you are!". Verdict pending.
  - **[HIGH] The v15 hiss is the narrator's own RAW room tone, TRAINED IN through Adobe's 70 % strength** (Owen:
    *"background hiss that sounds very much like ownbed"*). This REPLACES an earlier same-day entry, "generated, not
    trained", which was wrong: it measured too few clips and a non-comparable floor. Re-measured, with ONE floor rule on
    every source (50 ms frames, digital zero dropped, frames within 6 dB of the 3rd percentile, third-octave bands; script
    `2026-10-06-ds-v15/floor_shape.py`):
    - **Adobe leaves a scaled copy of the original background.** The `esv2-70p` returns sit 14 dB under the part sent,
      in every band from 100 Hz to 10 kHz, with the same shape (shape distance 0.49 dB on part 12, 0.8 on part 02).
      The 70 % strength mixes the original back in, room noise included.
    - **The brighten EQ lifts it** by 2-3 dB above 2.5 kHz (d3k_web vs d3k_web_eq).
    - **The training clips carry it.** Floors total -82 dB for d3k and lp, -88.5 for cd. The d3k clips' shape matches
      the raw d3k master at 0.60 dB.
    - **The render copies it, louder.** The v15 render floor (-78.5) has the shape of the raw d3k master (1.06) and the
      d3k clips (1.21), NOT ownbed1 (2.9). It runs about 5 dB above the clips at 2.5-6.4 kHz, so the model overstates
      a floor it learned. The v14 render floor does match ownbed1 (1.37): that is its bed.
    - It sounds like ownbed because both are room tone.
    - **It does NOT explain Mistborn v17 < v16.** Both builds were brightened and both used 70 % returns. Their
      un-bedded clip floors match per book within 0.4 dB (FE -85.3/-85.7, WoA -90.0/-90.4, HoA -85.3/-85.1,
      AoL -89.2/-89.3, 60 clips each), and both trained under a -70 ownbed1 bed. Mistborn's leak sits 3-8 dB under
      deathstalker's (-82).
    - **ds_v15sep (armed 2026-10-06, Owen: "lets prep to train the files after they finish"):** v15 with ONE change.
      - **Separation:** the brightened website masters are vocal-stem separated IN PLACE with the RoFormer dry stem
        (`2026-10-02-ds-adobe-full/tone/separate_eq_masters.py`). Each website span is separated on its own, because
        WSL's 12 GB cannot hold a 7 h master. Audio goes 48k -> 44.1k -> separate -> 48k and is refit to the span's
        exact sample count. Speed is about 10x realtime.
      - **Originals untouched:** `parts/*esv2*` and `masters/<bk>_web_48k.flac`. To re-make the EQ'd masters, run
        `brighten_ds.py`.
      - **Floor check:** `floor_check.py` must show the room tone at least 10 dB down before prep runs.
      - **Training:** prep and chain are v15's, generated by `make_v15sep.py`: same excl, slicer, mix (no bed, no cap),
        recipe and screen/ladder/sample.
      - **Crucible:** its `denoise` job (`denoise-roformer`) is the same checkpoint; use it for the next separation.
      - **Result:** pause floor before -> after: d3k -77.9 -> -91.2 (13.3 dB), cd -80.3 -> -92.6 (12.3), lp -76.2 -> -81.6
        (5.3, failed the 10 dB check). lp's remainder sits at 250-2000 Hz, voice-band (breath, room echo), while 4 kHz+
        dropped 3-14 dB. The 10 dB total-floor check misjudges this case; judge the hiss band. Owen listened and
        overrode it: "close enough. sounds good. i dont hear the bg on any of the three".
      - **Mix and train:** the mix matches v15's (1,459 rows, 17.05 h). Training started 15:08.
      - **Render floor after separation:** low rumble fell 2-7 dB below 1 kHz (-95 -> -102 dB under 160 Hz). The 4-6.4
        kHz hiss did NOT move: -91.4/-91.8 in both v15 and v15sep screen renders, although the v15sep training clips lost
        4-8 dB there (-103/-105). That band is model-made in pauses, independent of the training floor (sampling at
        T 0.8, or the base model's prior; untested). Owen: "this level of hiss is fine. i just dont want it to hurt
        prosody is all. if prosody is good, its fine." Don't chase it unless prosody suffers.
      - **Why the 4-6 kHz pause hiss survives (2026-10-06, Owen: "i would like to know why it didnt go away
        completely").** It is the BASE MODEL's idea of silence, and LoRA training does not change it. **CAVEAT
        (2026-10-25):** every render here was on the PC sglang path. The hiss Owen hears was SGLang-Omni's bf16
        vocoder (§5, fixed in Crucible 2.0.24); a plain-PyTorch render of the same weights is clean by ear, and
        pause-floor numbers do not track what he hears. Scripts and
        results are in `2026-10-06-ds-v15sep/hiss_*`; all figures are 4-6 kHz band floors.
        - **Codec ruled out:** a round trip of 30 separated training clips decodes their pauses to -122..-131 dB per
          band (-113 total).
        - **Sampling and guard ruled out:** ckpt-1274 at guard on T 0.8 / 0.5 / 0.3 and guard off T 0.8 all give
          -91.2..-91.4. Lower T only adds more quiet frames, at the same level.
        - **Inherited from the base:** the untrained base model (no LoRA) gives -91.6 / -92.6. So do v15 (2002, -91.4)
          and v14 (bedded). Training only moved the bands below 1 kHz (base -95 -> v15sep -102 under 160 Hz).
        - **Where it sits:** about 92 % of the floor frames are interior pauses, not the forced head/tail padding. Base
          and fine-tune measure the same there (-88.7 / -88.5 dB at 4-6.5 kHz). The separated training pauses are
          mostly digital zero.
        - **Mechanism, inferred:** even near-argmax (T 0.3), the model does not emit the codes the codec gives true
          silence, in the codebooks that carry 4-6 kHz. Untested: a teacher-forced check of the pause-frame
          predictions per codebook against the training codes.
      - **Ear verdict (Owen, ckpt-1274 picked as best pauser, ladder stopped):** "the best prosody ive heard from
        deathstalker so far (not including deployed). it nailed some really hard parts." 7 marks, all
        emphasis/flatness on short dialogue lines plus "fifth--or sixth" pausing (marks file round 5), and "long
        pauses throughout".
        - **Pauses measured on the 9 chunks:** median 0.79 s against v14's 0.62, and 16 pauses of 1.5 s or more
          against v14's 4. v14 was trained under the 1.5/2.5 s pause cap.
        - **Owen's suspects for the long pauses:** the ladder pick, more data, pause capping.
      - **Recipe of record: `recipes/deathstalker_v15sep/`** (README, script copies, inputs), written at Owen's ask:
        "did you write down the formula we used to train this time".
      - **Owen's verdict (2026-10-06 evening):** "i think this new model is better than deployed. lets sit on it for
        now. dont deploy. we'll retrain tonight with the new adobe parts". ckpt-1274 is HELD and v14 stays live. Tonight
        = this formula + the returned mm/mck parts; the steps are in the recipe's §8.
    - **ds_v16 (armed 2026-10-09 00:14, Owen: "we're going to train tonight"; all 30 parts back):** the v15sep recipe
      over five books (+ Marked Man, McKinley). Campaign `2026-10-08-ds-v16`.
      - **Audio:** `masters_book.sh <bk>` = build_web_master, then brighten_ds (same raw d3k/cd/lp reference), then
        RoFormer separation AFTER the EQ. Owen: "we have to re-separate AFTER eq, because we found that EQing amps up
        the background noise". mm: 677/677 spans placed (4.12 h); brightness -17.70 -> -14.97; floor -76.9 -> -87.1
        (10.2 dB). d3k/cd/lp reuse their v15sep separated masters.
      - **Exclusions and prep:** excl_v15.py's rule for all five books. `prep_v16.sh` and `ds_v16_chain.sh` are
        generated by `make_v16.py`. The chain waits for **GO.marker** (Owen's green light) and PREP.DONE.
      - **No ladder:** after the screen, Mutineers' Moon renders for the best pauser and, if different, the latest
        clean.
      - Owen considered dropping the EQ and kept it, so only one thing changes against 1274: the added data.
      - **Overnight (Owen, going to bed 01:40):** "train, then do the pause test, then the ladder on the top 2". The
        ladder runs from a separate script, `ladder_v16.sh` (tmux dsv16l). It waits for the chain's ALL_DONE, because
        a running bash chain cannot be safely edited. Pick rule: the ladder winner if the difference is real, else the
        latest clean. It renders the pick's Mutineers' Moon sample if the chain has not already. Wake checks are set
        at 02:07, 03:47, 04:43, 05:23, 06:27 and 07:33 (morning A/B against 1274).
      - **ds_v16 result (2026-10-09):**
        - **Mix and train:** 2,103 train rows, 24.2 h. Train 01:42-04:43, about 3.6 s/step; early stop at 3013, best
          holdout 3.7039 @ 2620.
        - **Screen (guard on):** scores 3013 81, 2882 84 (1 truncated), 2751 84, 2620 82, 2489 85; all others clean.
          Median pauses 0.72-0.80 (reader 0.58).
        - **Ladder 3013 vs 2751:** inside noise (2751 1/192 fails at 400-900), so the pick is **3013**, the latest
          clean.
        - **Mutineers' Moon pauses (median / >=1.5 s count):**

          | render | median | >=1.5 s |
          |---|---|---|
          | v14 deployed | 0.62 | 4 |
          | v15sep 1274 | 0.79 | 16 |
          | v16 3013 | 0.81 | 12 |
          | v16 2489 | 0.81 | 16 |

          More data did not shorten the pauses. The pause cap is the untested lever.
        - **Owen's ear (marks file rounds 6-7):** 2489 got 8 marks. 3013 got 6 and "had better prosody but slightly longer
          pauses". NOT deployed.
        - **The misses repeat across models.** "fifth--or sixth--time", "I'm glad you are!" and "get your ass killed"
          fail on most renders since 10-02: short, charged dialogue lines and contrastive stress.
        - **Data gap:** the fiction dialogue the narrator delivers is excluded. Marked Man's is 1.64 h, out for good
          (heavy Australian character accent, Owen 09-06). 3000 Degrees' is 0.41 h, excluded as "fiction" and never sent
          to Adobe.
          **RULED OUT (Owen 2026-10-09):** no narrator fiction dialogue in any deathstalker book, 3000 Degrees included.
          "He does Australian character voices ... They sound like different people entirely. It was tested and makes
          the model measurably worse." Emphasis fixes must come from text (italics), context, or checkpoint/pause
          work, not from dialogue data.
      - **DEPLOYED (2026-10-09, Owen: "Let's try deploying the 3013 model. I'll render some stuff with it").**
        - HF owenmorgan/deathstalker-higgs-v3@**3d778ad4** (squashed). The `crucible` tag moved from 28f6c4d3.
        - The PC pulled; sha256 8b052d2a… matches local, HF and the PC.
        - Card: pace 15.69 (99 clean 600-900 renders, own ladder), band 600-900 (1/128), cap 1000 (longest training
          row).
        - Mac: not pulled yet, waiting on Owen's go.
        - Fallback: the v14 ckpt-3510 adapter. Scripts: `2026-10-09-ds-v16-deploy/`. Local merges deleted.
        - **KEPT PERMANENTLY (Owen 2026-10-09, on a real book render):** *"its very good. we can keep this model
          permanently. the one or two odd places were very tricky. in average listening, it's impressive."*
        - **Chunk starts:** only 3 of 16 marked spots sit in the first 10 % of a render chunk ("let me go", "not going
          to", "hope for the best"), so a missing lead-in explains a few at most.
      - **Background jobs:** run long jobs in WSL tmux or a detached Start-Process. A Claude Code background shell can
        lose its link, and the "killed" notice does not mean the process stopped. The 12:18 waiter survived it and had
        to be killed by hand so prep would not run twice.
    - **Root-cause fix:** Enhance Speech at 100 % strength, or a speech-preserving denoise of the website returns
      before slicing. Either fix needs Owen's call: 70 % was chosen for naturalness, and a fix changes the voice's audio.

**Ear test of all deployed voices** (Mutineers' Moon ch. 8, seed 500; marks in
`2026-10-02-stall-guard/prosody_marks_mutineers_moon.md`).
- Owen's ranking on this sci-fi thriller follows each narrator's GENRE: Mistborn *"flawless"* (one word slip, "dash LEDs"
  -> "day LEDs"), deathstalker *"very good"*, sigma *"good"*, Third Reich *"slightly flatter than expected"*.
- The shared weakness is short emotional dialogue: "I'm glad you are!", "I appreciate that", "Can you?" (contrastive
  stress), "You should let me go". Mistborn's corpus excludes dialogue too and was flawless, so the narrator's genre
  matters more than whether dialogue was included. Owen: *"i think they all sound good on nonfiction"* - no nonfiction
  test needed.
- Word slips / tail repeats (sigma "year's there" after a chunk end) are rare. Crucible's chars/sec guard catches them
  and its retake covers them (Owen: *"acceptable job as is"*).

**Text formatting matters to prosody (2026-10-02).**
- The Mutineers' Moon EPUB types em-dashes as unspaced hyphens ("wrong-if he", "button-he's", `thing-"`, `What the-?`)
  and prints ellipses spaced (`. . .`).
- With the dashes restored by hand, deathstalker's marked dash spots passed. On `. . .` (a form NO training corpus
  contains: tr 0, mb 0, ds 0 vs "..." 287/353/121) Third Reich invented words ("torjures night").
- Fixed in BookForge:
  - c8e8dcfe: `normalize_for_tts` turns `. . .` / spaced runs / U+2026 into "...". Clean text already did this for
    cleaned books.
  - narrator 2340aab7 (live on Crucible 1.0.85+): deterministic hyphen -> em-dash only in unambiguous shapes (`x-"`,
    `"-x`, `x-?`/`x-!`, a trailing `x-`, `x-he's`), each logged. Mutineers' Moon: 138 repairs.
  - Foundry a3400e0 / normalizer **n18**: the cleanup MODEL judges the ambiguous cases ("wrong-if", "fifth-or
    sixth-time"), per Owen's 2026-09-06 rule that judgement goes to the model, not to code.
- The test bank `2026-09-21-ds-norvc-10h/mm_sample_bank.json` now has all 11 em-dashes restored. The original is kept
  as `mm_sample_bank.hyphens.json`; samples rendered before 2026-10-02 used it.
- Training corpora carry none of these shapes (0 in tr/mb/ds), so no VTT re-run is needed.
- orpheus-finetune's vendored normalizer was re-vendored **n6 -> n18** (bookforge ec93db0d; 7 fixtures moved to the
  rules stage because BookForge now reads glued decades/years by rule; 111/132 pass, 21 deferred, 0 fail).

**Voices ship their own config** (PHASE21, 2026-09-26, Owen: "i dont want to re-measure. we should have trustworthy
settings"): every voice repo carries `crucible-voice.toml` + a README card rendered from it. A new voice or retrain =
commit weights + `crucible-voice.toml` (`crucible voices export <id> --pace-basis measured|inherited --max-chars-basis
measured|placeholder`), then `crucible voices pin <id> <repo>@<sha>` + `pull` on each machine - no Crucible deploy.
Label every number's basis honestly (measured ON THE SHIPPED WEIGHTS, or not). Squashing deletes old revisions: re-pin
the machines. Scripts: `2026-09-26-voice-tomls/` (export_all.sh, commit_voice_tomls.py).

**Deathstalker (was §4n.92, §4n.84, §4n.72-72d).** Deployed 2026-09-24 on HF + both Crucible arms PC + Mac; Owen:
*"sounds very good. a couple little flaws here or there but the prosody is better than any other so far."* Corpus
build15_premiere (kept): the five-book corpus (Marked Man, 3000 Degrees, McKinley, Discipline, Leadership Pipeline as in
ds_v8) through RoFormer denoise -> Premiere Enhance Speech 70 -> EQ to the books' raw mean tone, cuts in silence, invented
sentences / headings / Marked Man + 3000 Degrees dialogue / Marked Man chapter music excluded, ownbed1, 24.3 h,
Whisper-gated. Early stop 3150 (best holdout 3.8426 @ 2835); screen 91, 0 defects. Pace 16.29 from 16 in-band screen
renders (guard 12.53-21.18) - a small sample. Band 500-800 is Owen's deathstalker ruling, not re-laddered on v13 (ds_v9
measured 400-1000 at 0.9 % and was served 500-800 for prosody). Ladders stop at 1,200 chars. Also kept: `ds_dn_merged*`
exports (Owen: *"produced a really nice model"*). Open: v13's own ladder/pace; ds_v8's mid-phrase hesitation (4.48 vs
1.63 per 10k) and ds_v9's ~1.5 % truncation were never re-measured on v13.

**Mistborn (was §4n.94, §4n.93, §4n.95).** Deployed 2026-09-25; Owen: *"Sounds perfect. Deploy to hf, pc, and Mac"*.
Corpus: Adobe-website FE + WoA + HoA prose (build16_web 19.8 h kept prose + a second 4 h of HoA returns, sliced ~23 h),
mix 1,432 train rows / 19.4 h, ownbed, v14 recipe (r32/a64, lr 1e-4, patience 3). Early stop 1691 (best holdout 3.9837 @
1424). Screen: 1691 = latest clean (score 72, 0 defects; 1157 scored 88). Ladder: 1691 band 200-700, 1/192 in-band
(Wilson hi 2.9 %); 1157 300-800, 5/192 - inside noise. Pace 13.83 (105 clean 200-700 renders). Replaced mb_hp_rvcbed1
ckpt-4257 (10b797e3, 500-800, 13.76). **Next (v16)**: brightened `masters_eq/*_web_eq_48k.flac`, `--long-max-chars 1000
--run-median-s 45`, plus the AoL website returns (4.88 h in `mistborn/build/aol_upload_v2`, located as
`masters/aol_web_48k.flac`, 224 spans) and the full-corpus Adobe upload (`2026-09-25-mb-adobe-full`, 29 parts / 24.1 h:
FE 6.14, WoA 10.52, HoA 5.99, AoL 1.41 h) when returned. AoL was recorded years later and measures differently - measure
its tilt against the trilogy before assuming it blends (brightening now matches all four books to within 0.5 dB).

**Third Reich (was §4n.35.1, §4n.50, §4n.51, §4n.83).** tr_v3: Evans trilogy (Sean Pratt), all clean hours, raw decodes
(no denoise / Adobe / bed), 105.7 h gated (67.5 h runs + 38.2 h sentences), `--tail-s 0.25`, nobed mix (with the 2.5 s
interior pause cap); early stop 8316, best holdout 3.4720 @ 7392; picked ckpt-7084 by fewest flaws. Near flawless and
pauses like the reader. Ladder (620 renders, n=32/rung): band 500-800 at 5.5 % (hi 10.9 %), applied b95a7ce8; maxChars
1000. **UNRESOLVED - ask Owen**: character-bin pooling put 500-700 (2.9 %, hi 10.1 %) ahead (the 700-800 bin 12.5 %,
800-900 18.8 %) and the safe-bands note records "500-800 -> 500-700"; which band the live manifest carries is not recorded.
Owed (09-19): no safe band on HF; the card still carries the retired `higgs_target_chars` and `max_chars` 1623.
**tr_v4 (2026-09-29)**: the book-text text, v4 slice (1,000-char cap, run-median 45 s; long rows median 616 chars vs
v3's 772, p90 59 s vs 83 s), trbed1 at -70 + pause cap, 79 h; early stop 10396, ladder 400-600 at 2.1 %; Owen: worse
prosody than v3, not deployed. **tr_v4nb (2026-09-30)**: the same rows with no bed and no pause cap; early stop 11752
(best 3.4409 @ 10396, paused for a stream at 8588 and resumed - a resumed run's patience counter restarts). Owen on
ckpt-9040: *"superb"*, prosody better than v3; ladder vs v3 equal on errors, worse on long pauses (§3.5.1). Not
deployed. **tr_v5 (training 2026-09-30 21:10)**: tr_v4nb's rows minus 161 lost-punctuation rows, long tier capped to
66.7 h (6,325 train rows, ~70 h), generated -85 dB own-spectrum bed, no cap; screen = newest 5 checkpoints, ladder
top 2. Other v3 -> v4 differences not yet isolated: clip length, post-fix clip edges, spelled-out numbers, +10 h.

**Mistborn v17 vs v16 (2026-09-29)**: text and clip shape match (long rows median 548 vs 558 chars, 37.6 vs 38.2 s,
punctuation as above), so v17's prosody drop is in the AUDIO - the new stitched Adobe parts (FE 5-8, WoA 1-4) lead.
WoA parts 5-8 returned 2026-09-30 (`2026-09-25-mb-adobe-full/woa/`); compare new vs old returns (brightness,
loudness, pace, pauses) before sending more. Owen's direction: the -85 generated bed and a hard quality cut (score every
clip; up to 10 h out).

**Owen Morgan (was §4n.33, §4n.53, §4n.81, §4n.83, §4n.84).** owen_gp2w ckpt-425 (God's People + WHAA, 09-07) shipped to
PC, Mac and HF owen-morgan-higgs-v3; Owen: *"owen morgan model sounds flawless."* ow_v7 ckpt-966 (same corpus at
`--tail-s 0.25`: 0/24 runaway) existed as the legacy merge `ow_v7_prod`. ow_v9_recut1 was unusable (2-7 truncations per
16 renders across all 27 checkpoints; ladder band 200-400). HF pace 16.32 is inherited from owen_gp2w, not measured on the
shipped weights. **UNRESOLVED - ask Owen**: which checkpoint the Crucible owen voice pins. WHAA's pauses were
synthetically quantile-mapped - never use WHAA for a speaker embedding (RVC datasets use God's People only).

**Owen Morgan v10 (2026-10-23, campaign `E:/training/_campaigns/2026-10-23-om-v10`; for YouTube ads etc.).** Three
books, trained together (Owen: *"lets train everything together for now"*; an ads model and an audiobook model are
an option later). Later split into two voices; see "Outcome" below.
- **Sources (MASTERS, KEEP.txt in each):** God's People = the LOSSLESS `Callisto/Shared/gods people/files/mastered/`
  WAVs (`E:/training/owen/gp_mastered`), NOT the Audible m4a: the m4a is AAC ~115 kbps with ~+6 dB added and decodes to
  +3.3 dBFS (the "he had the machinery" crunch); mastered/ peaks at a -3.46 dBFS limiter, flat factor 0. Chapter 8 =
  `files/final/chapter 8 - father kun updated.wav` (the published edit; mastered/ only has the older 42.9-min one);
  ch 5 in mastered/ is 7.6 s longer than the published edit. `originals/` = raw unedited takes (retakes in) - unsafe.
  WHAA = `whaa_extended` (pauses as they are). UJW (Understanding Jehovah's Witnesses) = Audible 320k mp3s + the
  FIRST-edition epub (draft 5; the 2nd edition was never re-recorded); "100 Questions" not trained (Owen).
- **Masters vs working copies (Owen):** masters = originals + anything expensive (Adobe web); working = cheap changed
  copies (EQ'd) in `E:/training/owen/working/<bk>` (`<bk>_eq24.flac` + book-text output); never copy a master into
  working unless it changes.
- **Audio:** one resample to 24 kHz (44.1 -> 24 direct; never via 48), decode lossy sources to FLOAT (§6.5), constant
  per-book gain to -29.7 LUFS (gp -9.5 dB / ch 8 -0.3, whaa -6.9, ujw -10.3), `eq-match` all three to GP's mastered tone
  (ref 24 x 30 s over GP; WHAA needed +9 dB at 6-12 kHz). Result: presence/air within 0.3 dB across books. Owen: close
  enough, but GP sounds "muffled and dark" - NOT spectral (raw GP has the most top end); it is the Adobe-75 % texture
  baked into mastered/, which EQ cannot fix. Trained as is; full Adobe re-run is the fallback if the model shows it.
- **Text:** book-text on the PC Crucible (`--dialogue keep`): WHAA ~10 min, GP ~20 min; the Mac was far slower.
- **Exclusions** (`excl_v10.py`): book-text's list + every cue (<= 12 words) matching an EPUB h1-h6 heading (the slicer's
  heading regex is Sanderson's and missed ~110 GP section titles) - gp 182, whaa 140, ujw 307 cues out.
- **Slice:** v18 flags, `--min-start 0`, `--max-end` = the VTT's true end (no retail credits; -300 would cut the Garbe
  appendix). cut-audit in THREE passes (`cutaudit_to_cues.py`: the cue at each flagged edge joins the exclusions, re-slice
  into a fresh root): gp 1 -> 0 -> 0, whaa 43 -> 10 -> 0, ujw 97 -> 36 -> 6 flagged. The slicer's "no clean edge" drop
  took 16 % of GP long rows, 29 % WHAA, 39 % UJW (UJW's sentence pauses: median 0.24 s vs ~0.35 s for GP/WHAA).
- **Corpus:** `owen/build10_p3` om10_l / om10_s: 1,147 + 62 holdout long, 113 short picked -> 1,260 rows, 14.55 h; rows
  p50 657 / max 1000 chars; leakage PASS. Mix called directly (`--nobed --no-pause-cap`, no served cap).
- **Base model moved:** a 2026-10-19 C: cleanup in another project deleted the Higgs base from WSL `~/.cache` again.
  It now lives permanently at WSL `~/training/models/higgs-audio-v3-tts-4b@239f63fb` (+ `omnivoice-audio_tokenizer@c5fdb5cc`);
  v3_ft higgs_v3_model.py / encode_dataset.py / merge_for_serving.py / higgs_v3_codec.py point there.
- **Outcome (2026-10-25):** two voices, deployed by crucible pc 1 on Owen's word (*"Deploy the om models"*): HF
  (private) `owenmorgan/owen-audiobook-higgs-v3` @ 26e635cc and `owenmorgan/owen-youtube-higgs-v3` @ a2f930ff, pinned
  in Crucible 2.0.26 and pulled on PC + Mac. Old `owen` retired (pin, weights and the owen-morgan-higgs-v3 repo deleted).
  Card paces from the clean in-band renders: audiobook 16.37 c/s (n=77), youtube 16.49 (n=75). Deploy folder
  `E:/training/_campaigns/2026-10-25-om-v10-deploy`.
  - **owen-youtube = v10b ckpt 948** (GP + WHAA + UJW, RoFormer-separated). Ladder 948 vs 869 inside noise; band
    200-500, 0 % in band (the 16.2 c/s pace quoted during the campaign was 869's).
  - **owen-audiobook = v10d ckpt 931** (GP + WHAA only). Replaces the old `owen` voice. Early stop, best holdout 3.7294
    at step 784. Pause screen top two 931 (clean, 88.7) / 392 (clean, 89); ladder 931 0/116 vs 392 1/116 in band 200-500
    (inside noise); pace 16.3 c/s.
  - The "hiss" that drove the v10 -> v10b/c retrains (RoFormer separation, Adobe-web parts in v10c) was SGLang-Omni's
    (§5), not the data. The separated books are masters anyway (RECORD.md beside each).
  - Mac v10b render of a 740-char passage came back ~30 s against ~45 s expected (possible MLX truncation); not
    investigated.

**Sigma (was §4n.59, §4n.60, §4n.76, §4n.83, §4n.85).** An RVC voice with no real narrator. Live: `sigma_prod` =
ds_sigma_full ckpt-1426 (deathstalker -> Sigma Male Narrator, 998 train rows, cap 1000 strict, ds_v7 recipe, nobed).
A new TR -> Sigma corpus is prepped (88.3 h converted, `third_reich_raw/build_sigma`, `trs_l` 4516 / `trs_s` 10538 train
rows, bedded - not a like-for-like A/B with sigma_prod); training held. Settings per pair in
`pipeline/rvc/conversion_settings.json` (§8).

---

## 8. RVC (voice conversion for sigma) (was §4n.58-60, §4n.64.3-4, §4n.76-79)

- **RVC-converted audio trains to normal quality on Higgs** (Owen on the 2 h sigma quick train: *"it worked flawlessly"*);
  the old Orpheus-era ban on RVC training audio is dead. RVC resynthesises the voice (100 % synthetic), it does not clean.
- **Semitone offset is a property of the PAIR**: the shift that lands the source's median F0 on the target corpus's.
  Sigma corpus 92.0 Hz: deathstalker 104.8 Hz -> **-2**; Pratt 116.9 Hz -> **-4**; Kramer 78.9 Hz -> +3. Integer only.
  Third Reich -> Sigma ruled **-4 / index 0.6 / protect 0.1 / rmvpe** (index = how much of the model's feature index;
  protect LOWER protects more; BookForge's code defaults index 0.5 / protect 0.5 / semitones 0 are none of these, and
  `nSemitones` is persisted nowhere - read Owen's Conversion-voice panel). `cli/rvc-convert.js` ignored `--n-semitones`
  until bookforge 4b532f2d / ours 5aecf30. Do not "fix" per-render pitch with semitones - a fixed offset cannot follow a
  passage; the lever is index rate or more hours.
- **Running it**: PC rvc-env `%APPDATA%\BookForge\components\rvc-env` (not `runtime/rvc-env`); `python -m
  ultimate_rvc.cli.main generate convert-dir IN OUT "<model>" --f0-method rmvpe --n-semitones N --index-rate R
  --protect-rate P --input-glob '*.wav' --output-ext wav` (args positional; the glob defaults to `*.flac` and a wrong one
  "converts" 0 files at exit 0). Env REQUIRED or it segfaults: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
  URVC_SKIP_INIT=1 URVC_MODELS_DIR=<runtime/rvc-models> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (without
  `URVC_MODELS_DIR` it writes `models/` into the CWD). Training: `pipeline/rvc/urvc_train.py` (the fork doesn't register
  `train`); `noisereduce` must be installed. Whole masters: `rvc_master_pass.py` (sample-exact, resamples each chunk back
  to the master rate, `--shard I/N`, `--resume`, `--plan-only`; stitches only when every chunk exists). No CPU path.
- **Speed**: PC ~23-32x realtime on long clips per process (2 processes 49.1x); Mac ~8x slower (RMVPE's GRU has no fused
  Metal kernel). Timing preserved (worst drift 34 ms on 74.8 s), so existing VTT/rows stay valid.
- **48 kHz RVC output into the 24 kHz codec is transparent for a 24 kHz source** (everything RVC does is < 12 kHz;
  12-24 kHz sits at -58..-63 dB). Does NOT transfer to a 48 kHz source - re-measure. RVC lifts level +3..+8 dB uniformly -
  level before blending with unconverted audio. Bed after RVC (§3.5).
- **Through Crucible (2026-09-26, sigma v2):** `crucible api --pairing <line> job submit --type rvc --model sigma --params
  '{"f0_method":"rmvpe","n_semitones":-2,"index_rate":0.3,"protect_rate":0.1}' --input c000.flac=<path> ...` (input
  NAMES keep their extension; one job at a time - a busy server refuses 409, it does not queue). Masters are split to
  <= 600 s chunks and joined back sample-exact (`2026-09-26-sigma-v2-rvc/rvc_chunks.py`: each chunk resampled to the
  master rate and trimmed/padded to its input's frames - urvc returns 48 k, 16-bit, ~20 ms short). **Keep jobs SMALL:**
  urvc's converting process grows ~1.5 GB per 10-min file and Crucible recycles it only every 96 FILES - a 35-file job
  swapped the Mac 28.7 GB deep (Owen: "is it running away") and a 44-file job pushed kylies-pc (16 GB) to 20.2/23.7 GB
  commit; a cancelled job's outputs are lost. Jobs of 8 files on the Mac, 3 on kylies-pc. Speeds: kylies-pc GTX 1660
  SUPER ~10x realtime (batched), the Mac ~2.5x.
- RVC voice-model datasets (`clipforge rvc-dataset`): the 09-10 recipe was ~45 min of narration, quote filter on,
  temporal strata x 6 pitch bins, no silence-trim/normalisation; the 09-11 deathstalker RVC model used a 15-min dataset.
  Mistborn trained at 48k (true 48 kHz master); voices that exist only as 24 kHz masters train at 32k (no higher-rate
  copy with a MATCHING VTT - m4b rips are on their own timeline). Speaker material from unedited audio only.

---

## 9. Retired / tried-and-failed (one line each - do not retry without new evidence)

- **vLLM-Omni serving + our patches** (token -100 fix, sentinel/tail-trim filter, frames7500 profile, 0.35/0.10 fractions,
  CUDA-graph talker patch, FlashInfer) - replaced by SGLang-Omni 2026-09-05: its batched talker damaged the newest batch
  row (truncations, babble, voice switches).
- **Temperature 1.0 / "use server defaults"** - retired 2026-09-06; every measurement before 2026-09-06 18:30 (scores,
  pace 17.11 / 15.16, Führer census) was at T1.0 and is not comparable.
- **`targetChars` / a single point target, "largest clean rung", "safe band = corpus IQR / p25"** - retired 2026-09-09
  for a measured two-edged band (a p25-derived floor was wrong in production within a day).
- **`--tail-s 4.0` (retained post-chunk tail)** - retired 2026-09-10: runaway 58 % / 29 %. S3c's "no penalty" came from a
  135-step quick train.
- **Lowest-holdout checkpoint pick (`pick_two.py`, 09-09 to 09-11)** - replaced by the pause screen 2026-09-11.
- **repetition_penalty 1.1** - refuted 2026-09-10 (absent on the Mac arm, same pauses).
- **The bed as a general v3 / SNAC-floor fix** - refuted 2026-09-05/09-19; only the RVC/resynthesis benefit is measured.
- **A -70 dBFS bed on natural (non-resynthesised) narration** - cost Third Reich's prosody 2026-09-30 (§3.5.1; with the
  bed step's pause cap). Readers' own floors sit at -79..-84.
- **Punctuation drift as the v17 / v4 prosody cause** - measured and refuted 2026-09-29 (§3.2).
- **Pause normalisation / compression of the corpus (`normalize-pauses` quantile map, capped-pause T2 probe)** - not used:
  Owen 2026-09-11 *"id like to use kramer's real pauses"*; compression edits the reader's pauses (ruled out 2026-09-08).
  The model stretches the corpus's pause median (mistborn 0.78 -> 0.96 s) and there is no inference-side lever; picking
  the checkpoint that pauses right is the fix.
- **Mistborn pause hypotheses** - parts 16-18, inserted digital silence, noise gate, Adobe, slicer config, bed, sampling:
  all refuted 2026-09-10 (measure the master, not upload parts).
- **Mixed-source corpora (Adobe + denoised)** - rejected by ear 2026-09-08.
- **Maxine Studio Voice** - out 2026-09-13: restores but does not converge books (between-book cosine 0.755 -> 0.756) and
  shifts timbre (same-clip 0.71-0.75 vs 0.84 ceiling).
- **RVC + EQ as a same-voice normaliser** (mistborn 09-13 to 09-19, `mistborn_owen_2026-09-11.json` curve) - superseded
  by the Adobe website; RVC converges books but "flattens" the voice.
- **Resemble Enhance** (09-22 candidate: nfe 64, tau 0.75, lambd 0.1, 5-seed median, anchor; spread 3.05 -> 0.78 dB) -
  out by 2026-09-23 in favour of Adobe.
- **Premiere Enhance Speech, and RoFormer-denoise-then-enhancer pipelines** - retired 2026-09-24 (*"premiere files are
  unnecessary now"*). Premiere local != Adobe online (floor stays raw, darkens > 5 kHz); one export returned the input x
  0.21 three times. Deployed ds v13 still comes from this pipeline.
- **Whisper everywhere** (faster-whisper, whisperx/wav2vec2 backend, base.en gates, large-v3 cross-checks, `rough_cache.py`)
  - retired 2026-09-23/25 for Qwen (§3.2). Whisper word times ran 0.3-1.0 s late on this audio.
- **Ollama residue pass / Foundry `--clean-lines` on qwen3.8:27b** - replaced by `clipforge cleanup` (triaged, the render's
  own stage).
- **Old silence/cut rules**: -48 dBFS pause cap, 80 ms end pause, blind 0.10 s start pad, `--lead-s` region padding -
  replaced 2026-09-19..24.
- **Micro/heading tier (`--tiers m`, 09-09 requirement)** - not in the current recipe (tiers l+s, headings excluded).
  **UNRESOLVED - ask Owen** whether his 2026-09-09 requirement for one-word entries / chapter titles in the corpus still
  stands (a micro tier also drags the corpus median, and the lower band edge, down).
- **Band-saturation training** (S3c: 780-1250 for 3 epochs did not clear 900-1400) and **giant rows / 15k renders**
  (tr_v3 derails from the first sentences past its training length) - dropped 2026-09-09.
- **Community recipe** (r16, lr 2e-5, effective batch 32/8) under-trains; **reference-conditioned training** buys no
  identity and costs context - kept as a tool (`make_ref_index.py`), not adopted (2026-09-06).
- **Shape bank as a grade** - it ranks within a run and names failing shapes; a `RETRAIN-CANDIDATE` line is never a
  retrain decision (Owen on two such models: *"flawless"*, *"very impressive"*).
- **Orpheus-only rules** that do not transfer: SNAC bed/EOS floor, `backends.vllm` in the Higgs catalog, `-55 dBFS`
  absolute gap measurement, 86 tok/s sequence math.
- **Named safe/danger-zone chunking** - dropped 2026-09-09.

---

## 10. Open questions / TODO (still open 2026-09-25)

- **Migrate the remaining Whisper call sites** (Owen 09-25): `higgs_ladder.py` / `higgs_pause_screen.py` scoring (base.en),
  `row_gate.py`'s default, `loop_gate.py`, `asr_text_qc.py`, `xcheck_vtt.py`, `rough_cache.py`, `spot_clips.py`,
  `rvc/asr_vtt_window.py`, BookForge `generate-sentences`' Whisper mode and local-WhisperX fallback, clipforge `sentences`
  anchor mode. Then re-baseline a Qwen-scored ladder against a known model.
- **Different-reader and music detection as Crucible verbs** (speaker embeddings / an AudioSet classifier) for
  discrepancies.json.
- **Quantized 9B for triage KV capacity** (raised in the 2026-09-25 prune request; not yet measured here).
- **First live run of `generate-sentences --clips`.**
- **Mistborn v16** (§7); whether 400-1000 is the natural band (Owen's hypothesis).
- **Temperature for served voices** (0.7 vs 0.8, §2); Third Reich band; Owen Morgan's pinned checkpoint (§7).
- Heading exclusion over ROW text + s1 check naming rows (§3.2); explicit pause-cap knee/ceiling mix args (§3.5);
  per-voice join-gap source in the catalog (§3.10).
- Crucible ladder path: 1.0.6 blocked `crucible_render.py` (`[voice.pace]` unconditionally required, no per-request seed);
  1.0.7 was specified with bare renders, seed lanes, applied sampling + weights identity on the result, and
  `[voice.serving] mem_fraction` / `context_length`. Status after 09-19 not recorded here.
- Truncation class: deathstalker's ~1.5 % vs Owen Morgan's 2-7/16 - same fault at different rates or different faults.
- Em-dash parenthetical skips on mistborn (61 % of long rows carry em-dashes, so not scarcity).
- The ladder SGLang server's 22.7 GB (vs ~20 GB budget) and 98-99 % Windows commit - not traced.
- Owen's next-recipe levers (listed, untested): checkpoint/adapter averaging, lower LR with more saves, silence-token loss
  weighting, a speaker-identity column in the screen (now live).
- **(2026-09-30)** tr_v5's verdict: does a -85 generated bed keep tr_v4nb's prosody AND fix its long pauses? Then the
  pause cap on its own (bed fixed, cap on vs off). Then Mistborn: the same bed + a scored quality cut (pace, pitch
  range, brightness/loudness vs the book, edge quality, text flags; Owen: up to 10 h out, "only the highest quality
  clips"), and the new-vs-old Adobe return comparison. The book-text cue-swallow defect (§3.2) to BookForge.
  Owen's view: *"we can make a model sound superb with just 30 hours"*.
- **(2026-10-02) Stall guard (§3.5.2): LIVE on both arms** at `37,1,20,16`. PC Crucible 1.0.86 was verified with
  `crucible api voices`, which shows `serving.stall_guard.env` per voice; the Mac is on 1.0.86 with the MLX guard in
  narrator 2340aab7. Owed: the Mac A/B (when its GPU is free), then removing the abtest-* voices and
  `/home/telltale/higgs_v3_merged/mb_ab_nb_3510`. My test-env hand-patch is reverted.
- Mistborn is called, so its no-bed retrain is dropped. The no-bed idea now applies to the deathstalker Adobe-web
  retrain (§7).
- BookForge's offered fix for two-word dialogue tags placed on the wrong audio (§3.2) is still open.
- Ladder scorer: Whisper base.en -> Qwen ASR is still owed. The normalisation fix (§3.5.3) covers the known biases
  meanwhile.
- **(2026-10-09) Deathstalker v16 follow-ups**:
  - Pull the Mac (on Owen's go) and run an mlx ladder; the card's 600-900 band is cuda-only.
  - **Pause-cap-only retrain:** v16's typical pause runs 0.81 s against the reader's 0.58, with 12 pauses of 1.5 s
    or more against v14's 4. More data did not shorten them.
  - **Ebook italics as emphasis markers** in training and render text (parked by Owen) - the most direct lever for
    wrong stress.
  - A small hard-line bank rendered at several seeds, so a change is judged on more than one render.
  - The 4-6 kHz pause hiss is the base model's silence (§7): Owen ruled it fine. The mechanism (the model never emits
    the codec's true-silence codes in the high codebooks) is untested. The AUDIBLE PC hiss was a separate problem,
    FIXED 2026-10-25: the sglang vocoder ran in bf16 (§5; Crucible 2.0.24).

---

## Old section map

| old § | now |
|---|---|
| Current guidance, §0 | header, §0, §3 |
| §1, §4, §4c | §1 |
| §2, §4e-§4e addendum, §4i, §4k, §4l, §4m, §8 | §5 (SGLang), §9 (vLLM-Omni retired) |
| §3, §4b, §4g, §4h | §3.9 band theory; rest history (git) |
| §4d, §4j | §6.2 lock/Ollama, §3.2 text parity, §6.5 |
| §4f | §3.5 |
| §5, §6, §7 | history (git) |
| §4n.1-.10 (engine night) | §5, §9 |
| §4n.11-11b | §3.4 pairing gate, §3.6 |
| §4n.12-.29 | §3.2 (em-dash, caps, text), §6.4 (library VTT edges), §9 |
| §4n.30, .31, .43, .61.1 | §2 |
| §4n.32, .41 (runbook) | §3 (whole recipe) |
| §4n.33, .35, .38, .62, .67, .83 (state snapshots) | §7 decisions; rest history |
| §4n.34 | §6.1, §6.5, §3.2 |
| §4n.36, .37 | §3.9, §5, §6.1, §6.2 |
| §4n.39, .50, .51, .72b, .72d, .84, .85 | §3.9 |
| §4n.40 | §6.4, §3.4 |
| §4n.42, .64.2, .82 | §4 |
| §4n.44, .45 | §5, §6.3 |
| §4n.46, .56, .64.5, .86 | §6.1 |
| §4n.47, .48, .74, .80, .92, .94 (deploy) | §3.11, §7 |
| §4n.49, .64.6 | §0 |
| §4n.52, .53, .54 | §3.3 tail cut |
| §4n.57 | §7 mistborn |
| §4n.58, .59, .60, .64.3-4, .76, .77, .78 | §8 |
| §4n.61, .63 | §9 (pause hypotheses), §3.10 |
| §4n.64.1, .73 | §3.7 |
| §4n.65 | §3.10 |
| §4n.66, .68, .69, .72, .72c | §3.8 |
| §4n.69.1, .70, .70.1, .79 | §3.5 |
| §4n.71 | §6.5 |
| §4n.75, .87, .89, .90 | §3.3 |
| §4n.81 | §0 |
| §4n.88 | §9 |
| §4n.91, .95 | §3.2, §4, §10 |
| §4n.93 | §3.1, §3.2, §6 |
| §4n.95 addendum | §3.3, §3.9 |
