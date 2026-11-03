# Capacity scenario (H1) — 2026-10-11

অবস্থা: **DONE**। 10-11-এর reboot-এর পর কাজ পুনরায় শুরু করে 2026-10-16-তে শেষ হয়েছে। শুধু বিশ্লেষণ ও স্থানীয় পরীক্ষা; কোনো kernel push বা submission হয়নি।

**মূল ফলাফল**
1. **H1 প্রত্যাখ্যাত।** Train ও validation/private-এর মধ্যে ×1.2 বা 1/1.2 capacity factor-এর পার্থক্য নেই। তিন split-এই bottleneck discharge প্রকাশিত fd capacity-এর সমান; একই bottleneck-এর discharge-ও split বদলালে একই থাকে।
2. **Scenario-তে প্রকৃত পার্থক্য:**
   - প্রতি ramp-এর demand নতুন করে নির্ধারিত হয়েছে: 0.2–2.7×; mainline flow ±3%-এর মধ্যে;
   - capacity অপরিবর্তিত থাকলেও সক্রিয় bottleneck-এর location বদলেছে;
   - সামগ্রিক demand level বা peak timing বদলায়নি।
3. **আগে নির্ধারিত ongoing ramp correction—noramp ও ramprel—সব gate-এ FAIL।** FM-PM validation-এ −0.0004 এবং private-এ −0.0008; pseudo-March-ও negative। নতুন scenario-তেও ramp column উপকারী। কোনো kernel লেখা হয়নি।
4. **Public-LB selection bias**, ধারা 4: 48–100টি submission করা দলের public score প্রকৃত স্তরের চেয়ে +0.005..+0.02 বেশি হতে পারে। আমাদের 0.88198-ও bias-মুক্ত নয়।

## 0. পরীক্ষার আগের নির্ধারিত পরিকল্পনা

লেখা হয়েছে 2026-10-11 01:35 +06-এ, H1 data test-এর পরে—ধারা 1—কিন্তু কোনো model result পাওয়ার আগে।

H1, অর্থাৎ train ও validation/private-এর মধ্যে প্রায় 1.2 বা 1/1.2 capacity factor-এর পার্থক্য, **data দিয়ে প্রত্যাখ্যাত**, ধারা 1। তাই capacity rescaling (a), x1.2 augmentation (b) বা FD constant recalibration (c) পরীক্ষা করা হয়নি; প্রয়োগ করার মতো ওই factor নেই।

Data-তে পাওয়া **সম্পর্কিত scaling** হলো নতুন scenario-তে প্রতি ramp-এর demand নতুন করে নির্ধারণ:
- প্রতি link-এর ramp ON/OFF mean 0.2–2.7 গুণ বদলেছে। 82–90% ramp-এ পরিবর্তন 15%-এর বেশি; ফেব্রুয়ারি বনাম অন্য train month-এ তা 0–6%।
- Mainline flow ±3%-এর মধ্যে থাকে।

Production ongoing model-এ 5টি ramp column আছে: `ramp_on3`, `ramp_off3`, `ramp_on_surge`, `ramp_on_pm3`, `ramp_off_pm3`। সবগুলো link capacity দিয়ে scale করা। তাই তাদের per-link level scenario-নির্ভর। Production onset model ইতিমধ্যে NORAMP।

**Correction arm: ongoing; lab np recipe = og_np_base; F03 row, 255 leaves, 500 trees, lr 0.05, fold 0 ও 1:**
- `og_np_noramp`: 5টি ramp column বাদ।
- `og_np_ramprel`: 5টি column-এর বদলে scenario বদলালেও একই অর্থ বহন করা একটি ratio, `rel_on = on3 / on12`। একই window-তে 4–60 min আগের flow-এর তুলনায় সাম্প্রতিক on-ramp flow। হিসাব: `ramp_on3 / (ramp_on3 − ramp_on_surge)`। Denominator ≤ 1e-6 হলে NaN; অন্যথায় [0, 5]-এ clip করা।

**বাধ্যতামূলক gate: og_np_base-এর তুলনায়; fold 0+1-এর গড় probability:**
- Ongoing: FM-PM validation ≥ +0.0015 LB **এবং** FM-PM private ≥ +0.0005 LB।
  - FM-PM setting TRANSFER_ANALYSIS §5.0 অনুযায়ী: একই panel, ±2 h, সব cell-এর predicted count, pseudo-March offlike_m + runstart_m lab pair।
  - Footprint: 40 + 40টি official ongoing window-এ দুই lab model-এর decision-এর পরিবর্তন; lab input ও all-train prior ব্যবহার।
- জানানো হবে, কিন্তু বাধ্যতামূলক নয়: train offlike / runstart OOF, fold 0 ও 1; pseudo-March offlike_m / runstart_m, সঙ্গে CI ও কত panel positive।
- Onset: কোনো arm নেই। Production onset-এ ramp বাদ আছে; সংশোধন করার মতো capacity scaling-ও নেই। Gate—pseudo-March ≥ +0.01 IoU, CI > 0, official-40-এ sign negative নয়—লিপিবদ্ধ, কিন্তু ব্যবহার হয়নি।
- T1: কোনো FD-constant arm নেই। প্রতি split-এ bottleneck discharge fd capacity-এর সমান, ধারা 1.3। Gate J ≥ +0.0003 ও 7/8 লিপিবদ্ধ, কিন্তু ব্যবহার হয়নি।

## 1. H1 data test: REJECTED

Data: শুধু released masked observation, `data/t2local/t2data_<P>.npz`।
- Sample size-নির্ভর statistic-এর জন্য train-কে মার্চ/এপ্রিলের visibility অনুযায়ী subsample করা হয়েছে।
- Script: `analysis/t2lab/capscen_stats.py`, `capscen_links.py`, `capscen_heads.py`, `capscen_ramps.py`।
- Log: `analysis/t2lab/logs/capscen_*.log`। Table: `analysis/t2lab/results/capscen/`-এ `stats_panel_month.csv`, `tod_share.csv`, `links.csv`, `heads.csv`, `heads_links.csv`, `ramps.csv`।

### 1.1 বেশি flow-এর quantile বনাম fd capacity

প্রতি panel-এ train-month median-এর সঙ্গে MAR / APR-এর অনুপাত:

| Panel | q99 lane flow / cap: train median | MAR / train | APR / train | Fitted capacity: top 1%-এর median; MAR / APR বনাম train |
|---|---:|---:|---:|---|
| D12_I5_N | 0.770 | 0.95 | 0.96 | 0.97 / 0.97 |
| D12_I5_S | 0.789 | 0.95 | 0.94 | 0.95 / 0.94 |
| D7_I10_E | 0.765 | 0.98 | 1.00 | 0.98 / 1.00 |
| D7_I10_W | 0.792 | 0.98 | 1.00 | 0.98 / 1.00 |
| D7_I210_E | 0.656 | 0.98 | 0.95 | 0.99 / 0.96 |
| D7_I210_W | 0.649 | 0.98 | 0.97 | 0.99 / 0.97 |
| D7_I405_N | 0.962 | 1.00 | 0.98 | 1.00 / 0.99 |
| D7_I405_S | 0.773 | 1.01 | 1.05 | 1.03 / 1.06 |

1,450টি link-এর q99.5 lane-flow ratio:

| তুলনা | Median | q05–q95 | [1.15, 1.25]-এর মধ্যে link | [0.80, 0.87]-এর মধ্যে link |
|---|---:|---|---:|---:|
| MAR / train | 0.998 | 0.957–1.031 | 0.5 % | 1.4 % |
| APR / train | 0.993 | 0.907–1.049 | – | – |
| Feb / অন্য train month: null তুলনা | 0.998 | 0.978–1.008 | – | – |

**1.2 বা 1/1.2-এ কোনো cluster নেই।** Train month-গুলো প্রায় সমান, বিস্তার ±1%। তাই train-এ অন্যদের তুলনায় আলাদা factor আগে থেকেই আছে—এমন প্রমাণ নেই।

### 1.2 Speed–flow ও flow–density সম্পর্ক

- Free-flow speed / vf: সব split-এ 0.998–1.002।
- Congested branch: queued cell-এ lane flow / cap বনাম density / per-lane kcrit:
  - Slope −0.15..−0.21; train, MAR, APR-এ একই রকম। যেমন D7_I405_N-এ −0.183 / −0.182 / −0.181।
  - Intercept থেকে অনুমিত jam density 5.5–7.5 kcrit; D12_I5_S-এর +6% বাদে train-month range-এর মধ্যে।

### 1.3 Breakdown ও bottleneck discharge: নির্ণায়ক পরীক্ষা

**Discharge** হলো queue head-এর ঠিক downstream-এর free link-এর flow-কে সেই link-এর নিজস্ব fd capacity দিয়ে ভাগ করা:
- q90: 24টি panel × split cell-এর 20টিতে 1.00–1.03; train-month median, MAR ও APR ধরে। ব্যতিক্রম: D7_I10_W-এ তিন split-এই 0.94–0.98, queue কম; D7_I10_E MAR-এ 0.90। Simulator train, MAR, APR-এ প্রকাশিত fd capacity অনুযায়ী discharge করে; তার 1.2× বা 1/1.2× নয়।
- একই bottleneck-এ split বদলালেও discharge একই: D12_I5_S link 39-এ 0.500 / 0.509 / 0.500; D7_I10_E link 81-এ 0.438 / 0.437 / 0.441; D12_I5_N link 239-এ 0.785 / 0.792 / 0.793।

**Breakdown-এর ঠিক আগের flow**, t−1; t-তে free → queued, cap দিয়ে ভাগ করা:
- MAR / train median 0.85–1.05; APR 0.91–1.04।
- সবগুলো train-month range-এর ভেতরে বা কাছাকাছি। সেই range যথেষ্ট চওড়া; যেমন D7_I210_E-এ 0.52–0.72।

### 1.4 দিনের সময়ভেদে queued share

8টি panel একত্রে; observed cell-এর শতাংশ:

| ঘণ্টা | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6.5 | 11.7 | 16.2 | 18.4 | 17.5 | 14.6 | 9.4 |
| MAR | 6.4 | 10.0 | 14.1 | 15.5 | 14.8 | 12.2 | 8.6 |
| APR | 7.3 | 11.2 | 14.7 | 15.9 | 14.8 | 12.0 | 7.8 |

- Timing অপরিবর্তিত: flow peak ±1–2 slot-এর মধ্যে; pooled queue peak slot বদলায়নি।
- Level panel-নির্ভর; 3টি panel-এর 15–18 h share train-month range-এর বাইরে:
  - D12_I5_N: MAR 0.099 / APR 0.132, বনাম train 0.152–0.250।
  - D7_I210_W: 0.188 / 0.147, বনাম 0.218–0.241।
  - D7_I405_S: 0.221 / 0.223, বনাম 0.148–0.189।
- Demand অপরিবর্তিত। সব slot, AM, PM-এ mean v/c বদলায় −4..+4%; per-link PM mainline flow ratio-এর median 0.985–0.997।

**H1-এর রায়: প্রত্যাখ্যাত।**
- Train-এর তুলনায় capacity-জাতীয় প্রতিটি statistic ≈1.00। Panel-স্তরের 3–6% পরিবর্তন 1.2 factor নয়।
- fd_parameters-এর তুলনায় সব split-এ bottleneck discharge fd capacity-এর সমান।
- `capacity_multiplier 1.2` সম্ভবত synthetic fd capacity-তে আগেই অন্তর্ভুক্ত। Manifest-এ `replaced_with_synthetic`-এর অধীনে FD capacity parameter আছে; সাধারণ link-এর observed flow সর্বোচ্চ প্রায় 0.65–0.95 × cap। যাই হোক, এটি train ও validation/private উভয়ের জন্য প্রযোজ্য।
- Train দিয়ে শেখানো model, আমাদের FD feature ও recon constant capacity factor-এর জন্য **miscalibrated নয়**।

## 2. Train → মার্চ/এপ্রিলের অন্য systematic পার্থক্য

শুধু data বিশ্লেষণ; ≤ T data দিয়ে এসবের ইঙ্গিত পাওয়া যেতে পারে।

| পার্থক্য | প্রমাণ | ≤ T data দিয়ে সংশোধন করা যায়? |
|---|---|---|
| **প্রতি ramp-এর demand নতুন করে নির্ধারণ** | Per-link ramp ON/OFF mean, AM/PM: MAR/train q05–q95 0.16–2.7, median 0.78–0.91; 82–90% ramp 15%-এর বেশি বদলায়। Null হিসেবে ফেব্রুয়ারি বনাম অন্য train month-এ 0–6%। একই link-এর mainline flow q05–q95 0.93–1.04; `capscen_ramps.py` | Ramp-level feature-এর input shift। Ramp flow mainline-এর প্রায় 1%; queue কোথায় সরেছে তা ব্যাখ্যা করে না। corr(Δ downstream on-ramp, Δ queued share) −0.40..+0.39; sign সঙ্গতিপূর্ণ নয়। ধারা 3-এ পরীক্ষা হয়েছে |
| **সক্রিয় bottleneck-এর সেট** | একই bottleneck-এর discharge অপরিবর্তিত, ধারা 1.3; কিছু train bottleneck বন্ধ, কিছু নতুনটি সক্রিয়। বন্ধ: D12_I5_N link 114; train-এ 3,319 head cell, MAR/APR-এ 0; link 111–114-এ queued share 17% → 0%, অথচ flow 5,944 / 5,753 / 5,930 vph। সক্রিয়: D12_I5_S link 91/95-এ 0 → 8–25%; D7_I10_W link 20 শুধু MAR; D7_I10_E link 19/23 train + APR-এ সক্রিয়, MAR-এ নয় | Scenario-নির্ভর location effect। একই scenario-এর history, অর্থাৎ মার্চ/এপ্রিলের আগের দিন, এটি চেনায়। sp4 / pv-তে তা ব্যবহৃত। নতুন scenario-র non-train-recurrent location-এর 10% onset-এর মধ্যে মাত্র 15%-এ একই দিন ওই link ±1-এ আগে queue ছিল; `capscen_onsetloc.py` |
| **Onset বেশিরভাগ train-recurrent location-এ** | MAR/APR onset-এর 90% train prior ≥ 0.05 থাকা cell-এর ±1 link / ±30 min-এর মধ্যে; train-এ 97%। আগের ≤ 4 same-split দিনে ওইখানে queue ছিল 84–89% ক্ষেত্রে; train-এ 87% | পরিচিত candidate-এর মধ্যে সঠিকটি বাছাই করা প্রধান সমস্যা; অদেখা location নয়। একই মাসের সাম্প্রতিক prior blend উপযোগী; capacity fix নয় |
| **Incident-এর মতো queue** | Train prior ≤ 0.02 থাকা link-এ ≥ 30-min queued run-যুক্ত দিন: train median 4–10%। MAR/APR-এ D12_I5_S 94% / 97%, D7_I405_S 87% / 87%, D7_I10_E 48% / 53%; অন্য panel-এ 0.3–13× | প্রধানত ওপরের নতুন recurrent bottleneck; শুধু 5–7টি incident নয়। Same-split recurrence দিয়েই সংশোধনের সুযোগ |
| **Queue level** | Pooled PM queued share −15%; 16 h-এ 18.4 → 15.5 / 15.9। D12_I5_N ও D7_I210_W train-month range-এর নিচে; D7_I405_S ওপরে | Same-split prior দিয়ে আংশিক সমাধান সম্ভব। Global level shift-এর সমর্থন নেই |
| **Demand level ও peak timing** | v/c −4..+4%; flow peak ±1–2 slot; pooled queue peak অপরিবর্তিত | সংশোধনের মতো পার্থক্য নেই |

## 3. আগে নির্ধারিত দুই ramp-correction arm: FAIL

- Code: arm-এর জন্য `analysis/t2lab/capscen_ramp_arm.py`, gate-এর জন্য `capscen_ramp_report.py`।
- Log: `logs/capscen_ramp_{noramp,ramprel,report}.log`।
- Output:
  - `data/t2local/cache/ogfeat/arms/og_np_{noramp,ramprel}/`-এ fold pickle, pm_f0/1, off_f0/1 এবং base model-এর official probability;
  - `results/capscen/ramp_fm.csv` ও `official_footprint_*.csv`।
- Reference: og_np_base; fold 0 + 1, বিদ্যমান model।

| Gate / set | noramp | ramprel |
|---|---:|---:|
| Train official-40; fold 0 | −0.0046 [−0.015, +0.005] 1/8 | −0.0079 [−0.019, +0.002] 2/8 |
| Train offlike; 1,473, fold 0+1 | −0.0053 [−0.0074, −0.0032] 1/8 | −0.0061 [−0.0090, −0.0037] 1/8 |
| Train pseudo; 1,055 | −0.0067 [−0.0097, −0.0036] 0/8 | −0.0070 [−0.0118, −0.0020] 3/8 |
| **Pseudo-March offlike_m; 298** | **−0.0059 [−0.0113, −0.0004] 3/8** | **−0.0105 [−0.0165, −0.0044] 1/8** |
| Pseudo-March runstart_m; 1,124 | −0.0020 [−0.0053, +0.0014] 3/8 | −0.0007 [−0.0039, +0.0026] 4/8 |
| Official validation footprint: window / cell / Δn | 28 / 155 / +7 | 30 / 166 / +18 |
| Official private footprint: window / cell / Δn | 31 / 172 / +28 | 31 / 168 / +28 |
| **FM-PM validation; gate ≥ +0.0015** | **−0.00039** | **−0.00039** |
| **FM-PM private; gate ≥ +0.0005** | **−0.00080** | **−0.00079** |
| সিদ্ধান্ত | **FAIL** | **FAIL** |

ব্যাখ্যা:
- Ramp column-এর সময়ভিত্তিক তথ্য—নিজস্ব level-এর তুলনায় সাম্প্রতিক surge এবং neighbour sum—নতুন scenario-তেও signal বহন করে। বাদ দিলে বা শুধু within-window ratio রাখলে pseudo-March offlike_m-এ 0.006–0.011 IoU ক্ষতি হয়।
- Per-ramp level shift-এ production ongoing model-এর পরিমাপযোগ্য ক্ষতি নেই। আগের +ramp gate −0.007, TRANSFER_ANALYSIS §1a row 27, বর্তমান recipe-তে পুনরুত্পাদিত হয়নি।
- t2_occ_v26 / t2_ongoing_pvogf kernel বদলানোর যথেষ্ট কারণ নেই। কোনো kernel লেখা হয়নি।
- পরিকল্পনা অনুযায়ী onset ও T1-এর arm নেই। প্রয়োগের মতো capacity factor নেই; production onset ইতিমধ্যে NORAMP।

## 4. Public-LB sampling noise ও selection bias

Script: `analysis/t2lab/capscen_selbias.py`; log: `logs/capscen_selbias.log`।

- **Sampling noise-এর আকার:** প্রতি split-এ 40 + 40 official window থেকে একটি submission-এর S_total SD প্রায় 0.004–0.006। আয়োজকের persistence baseline-এর হিসাবে 0.0126; per-window IoU SD 0.375। Public ও private স্বাধীন sample, তাই কোনো selection না হলেও একটি দলের public-to-private পরিবর্তনের SD ≈ 0.006–0.008।
- **Nটির সেরা বাছাই:** Nটি submission থেকে সেরা নেওয়ার bias σ·√(1−ρ)·E[max_N]; ρ একই দলের submissionগুলোর public noise-এর correlation। N = 48 হলে E[max] 2.23; N = 100 হলে 2.51।
  - আমাদের probe measurement, TRANSFER_ANALYSIS §1, অনুযায়ী দুটি variant-এর difference SD ongoing-এ 0.0012–0.0027 LB এবং onset-এ 0.003–0.005; ফলে ρ ≈ 0.6–0.95।
  - তাই best-of-N bias প্রায় **+0.003..+0.007**; σ 0.004–0.006, ρ 0.8–0.9। Variant অনেক ভিন্ন হলে, ρ ≤ 0.5, তা +0.010..+0.015 পর্যন্ত হতে পারে।
- **Public Δ > 0 হলেই রেখে দেওয়া**, অর্থাৎ greedy hill-climbing-এ accepted noise জমতে থাকায় সমস্যা আরও বাড়ে।
  - নির্দিষ্ট 80টি public window-এ bounded IoU-সহ window-level simulation: অর্ধেক onset-type probe-এ 3–7 window নতুন করে নেওয়া, অর্ধেক ongoing-type probe-এ 25 window পরিবর্তন; প্রকৃত গড় gain প্রতিটি probe-এ শূন্য।
  - Public-only gain আসে N = 48-এ +0.025 থেকে N = 100-এ +0.035।
  - এটি upper bound; বাস্তব probe-এর বেশিরভাগেই সত্যিকারের পরিবর্তন ও correlated window effect থাকে। প্রতি probe-এর noise স্বাধীন ধরে সরল যোগ করলে অবাস্তব +0.05..+0.10 আসে।
- **অনুমান:** 48–100টি submission-এ 0.90 পাওয়া দলের public selection bias **+0.005..+0.02**, মাঝামাঝি ≈ +0.01 হতে পারে। প্রকৃত স্তর সেক্ষেত্রে প্রায় 0.88–0.895।
- **আমাদের 0.88198-এও positive bias আছে বলে অনুমান**, প্রায় +0.003..+0.01। প্রায় 70টি submission ও +0.0001 KEEP rule ছোট পরিবর্তনে noise দিয়ে সিদ্ধান্ত নিয়েছে। তাই 0.90 দলের সঙ্গে প্রকৃত ব্যবধান সম্ভবত 0.005–0.015; 0.018 নয়। ব্যবধান শূন্যও নয়; public LB দিয়ে তা নির্ভুল মাপা যায় না।
- **Final pick-এর জন্য তাৎপর্য:** প্রায় +0.003-এর নিচের public delta-র বদলে lab / pseudo-March / FM-PM-সমর্থিত পরিবর্তনকে অগ্রাধিকার দেওয়া উচিত। দুই pick-এর পার্থক্য public noise থেকে স্বাধীন অনুমানে হওয়া উচিত; A/B scenario hedge ইতিমধ্যে তা করে।

## 5. নিয়ম মেনে 0.89-এর সম্ভাবনা: এই বিশ্লেষণের মূল্যায়ন

- **Physics constant ভুল calibrate হওয়ার প্রমাণ নেই**; H1 প্রত্যাখ্যাত। সহজ global correction নেই।
- **অবশিষ্ট loss scenario-নির্ভর location / extent-এ**। TRANSFER_ANALYSIS §3 অনুযায়ী onset location প্রায় 0.02 LB, ongoing extent প্রায় 0.009 LB। একই scenario-এর history এগুলো চেনাতে পারে। Public window মার্চের 1–5 দিনে; আগের মার্চের দিন থাকে মাত্র 0–4টি।
  - তাই এই বিশ্লেষণে **ভাগ্য বা public-LB fitting ছাড়া public 0.89 সম্ভবত অর্জনযোগ্য নয়** বলে মূল্যায়ন করা হয়েছে; ধারা 4।
  - Private-এর অধিকাংশ window-এর আগে মার্চের 30+ দিন ও আগের এপ্রিলের দিন থাকে। সেখানে একই scenario-এর pv / pvo / sp4, অর্থাৎ pick A, কাজে লাগার বাস্তব সুযোগ আছে। বহু দিনের pseudo-March-এ lab onset IoU +0.03..+0.09 ও ongoing IoU +0.009..+0.014 অনুমান করে।
- **সবচেয়ে সম্ভাবনাময় অবশিষ্ট private লাভ**, সবগুলো ≤ T এবং নিয়মসম্মত:
  1. Candidate link-এ একই split-এর আগের দিন থেকে scenario-recurrence location feature। নতুন scenario-র প্রায় 90% onset পরিচিত train-recurrent location-এ, তবে candidate-এর মধ্যে সঠিকটি বাছাই করতে হয়। বহু দিনের pseudo-March-এ উন্নতি ও validation দরকার; pv2 a1–a3-তে +0.007..+0.014 পাওয়া যায়, মূলত 1টি panel-এ।
  2. নতুন-scenario growth prediction-এর per-step extent calibration; released মার্চ window দিয়ে shrink শেখানো, এপ্রিলের জন্য। এটি decode calibration-এর অন্তর্ভুক্ত; FM-PM private ≥ +0.0005 লাগবে।
  3. Capacity, demand level বা timing correction-এর পক্ষে data নেই।
- **প্রত্যাশিত private লাভ +0.002..+0.006 LB**, +0.01 নয়। শুধু শেষ public score 0.89 করাকে লক্ষ্য রাখা উচিত নয়।
