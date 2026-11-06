> বাংলা ভাষান্তর: স্থানীয় অনুবাদ মডেলের সহায়তায় অনূদিত। মূল ইংরেজি নথি সংরক্ষিত ব্যাকআপে আছে।

# চূড়ান্ত পিকের বংশধর A3 = tfb-k9 এবং B3 = tfb-k10 (দুইটিই পাবলিক LB  0.88281 , জমা দেওয়া 2026-11-03 )

ট্র্যাক করা হয়েছে 2026-11-03 সঙ্গে `kaggle kernels pull -m` (মেটাডেটা) `kernel_sources` + source) থেকে দুটি শিকড় থেকে ফিরে
যেসব কোর `consolidation/` ইতিমধ্যে পুনরুত্পাদন করে ( `tfb-i4` , `tfb-j4` এর অংশ, টাস্ক 2 প্রযোজক, 31 টাস্ক 1 সদস্য
`lineage_A2.json` এর অংশ) ।
-18 kernelগুলি পিকের উপরে নতুন, প্লাস 2 রেফারেন্স kernelগুলি ( `tfb-k5` , `tfb-k7` ) । গ্রাফ, রানটাইমস, পিক RAM এবং
  SHA -1 : `lineage_K9.json` . সঠিক সূত্রঃ `lineage_src/<slug>.py` (সমস্ত বাইট `submissions/` এবং `kaggle_jobs/` এর সাথে একক , `cmp` দিয়ে যাচাই করা হয়েছে) ।
- `tests/test_static.py` প্রতিটি উৎস, বিক্রিত পর্যায়ে এবং বেস চেইন প্রতিটি ধ্রুবক এর SHA -1 পুনরায় চেক করে।

## আগের পিক থেকে চেইন

```
tfb-j4 (= A2, 0.88176) ──base──> tfb-k3d ──> tfb-k3 ──base──> tfb-k5d ──base──> tfb-k9 (= A3, 0.88281) ──base──> tfb-k10 (= B3, 0.88281)
            31 old members ──────────┤                  ├── 31 old members           │                     ├── tfb-t2osp  base
            t1catb a-d, t1catb2 a-d ─┘                  ├── t1catb a-d, t1catb2 a-d  │                     └── tfb-t2ogveto veto10
                                                        └── t1xgb a-d                └ (released data only: LRTC of the month)
```

| kernel | kernel উত্স | এটা কি পরিবর্তন করে | পিক্সে পৌঁছানোর জন্য সারি | উৎস ( `lineage_src/` ) = স্থানীয় ফাইল |
|---|---|---|---|---|
| `tfb-t1catb{a,b,c,d}` | `tfb-t2lab-full` | **নতুন টাস্ক 1 সদস্য**: ক্যাটবস্ট অবশিষ্ট মডেল (গভীরতা) 8 × 2000 এটি + গভীরতা 10 × 1000 গড়; 0.12, l2 10, বার্নোল্লি 0.85, সীমান্ত 64, `random_seed` 20261005, 4 (কোনও ক্ষেত্রে) 94 v13 FD কলাম; v13 রায় (রায় rng) 9, ট্রান্সডাক্টিভ রিং 10000+), ≤2.4 M সারি (rng) 21প্যানেলঃ a = D7_I405_N, D7_I10_W; b = D7_I210_E, D7_I210_W, D12_I405_S; c = D12_I5_N, D12_I5_S; d = D7_I10_E, D7_I405_S, D12_I405_N | `state_submission_FTA.csv` → k3d / k5d | `tfb-t1catba..d.py` = `kaggle_jobs/t1_catb_member_a..d.py` |
| `tfb-t1catb2{a,b,c,d}` | `tfb-t2lab-full` | দ্বিতীয় CatBoost বীজ সেটঃ `CB_SEED`  31 , `random_seed`  20261016 , সারি rng 19 , ট্রান্সডাক্টিভ rng 12000 + | একই | `tfb-t1catb2a..d.py` = `kaggle_jobs/t1_catb2_member_a..d.py` |
| `tfb-k3d` | 31 পুরানো সদস্য অংশ, 8 catb অংশ, `tfb-j4` | T1 মিশ্রণ = tfb-i4d মিশ্রণ, তারপর 0.65 মিশ্রণ +0.35 mean(catb, catb2 ), তারপর ব্ল্যাকআউট ওভাররাইড; বেস `tfb-j4/submission_J4.csv` | **শুধুমাত্র ক্যু / path_flow সারি** (কে 3 → কে 5 d এর ভিত্তি হিসাবে) | `tfb-k3d.py` = `kaggle_jobs/t1_catb2_swap.py` |
| `tfb-k3` | `tfb-k3d` | recon + TV (tfb-i4 কোড) = 10-16 ( 0.88201 ) এর পুরানো নির্বাচন | **শুধুমাত্র ক্যো / path_flow সারি** | `tfb-k3.py` = `kaggle_jobs/swap_fdrecon_k3.py` |
| `tfb-t1xgb{a,b,c,d}` | `tfb-t2lab-full` | **নতুন টাস্ক 1 সদস্য** : XGBoost hist ( max_depth  9 , eta 0.06 , 1500 রউন্ড, সাবসেম্পল / colsample 0.8 , min_child_weight  50 , lambda 10 , max_bin  256 , বীজ 20261016 , 4 থ্রেড); t1catb হিসাবে সারি এবং অংশ | `state_submission_FTA.csv` → k5d | `tfb-t1xgba..d.py` = `kaggle_jobs/t1_xgb_member_a..d.py` |
| `tfb-k5d` | 31 পুরানো অংশ, 8 catb অংশ, 4 xgb অংশ, `tfb-k3` | T1 মিশ্রণ K5D : i4d মিশ্রণ → 0.65 মিশ্রণ +0.35 mean(catb, catb2 ) → 0.65 মিশ্রণ +0.35 xgb → ব্ল্যাকআউট সারি 0.25 mix +0.75 t1bspec. **প্রতিটি স্ট্যাটের সারি পুনরায় গণনা করা হয়** ; ক্রম / path_flow সারিগুলি বেস থেকে অনুলিপি করা হয় (আসারিত) | সমস্ত স্ট্যাটাস লাইন ( M/B/R/T এর আগে); tfb-j4 এর ক্যু লাইন | `tfb-k5d.py` = `kaggle_jobs/t1_div2_swap.py` |
| `tfb-k9` | `tfb-k5d` | ধাপ **M**: প্রবাহ LRTC প্রকাশিত মাসের (প্যানেল × বিভক্ত, পদ (24, 40, 10), 60 EM পুনরাবৃত্তি, স্টেশন-মিডিয়ামে স্বাভাবিক), `q ← max(0.65 q + 0.35 max(q_lrtc, 20), 20)` সাধারণ রাজ্য সারি; ধাপ **B**: ব্ল্যাকআউট লাইন `v ← clip(0.85 v + 0.15 v_lrtc(8,20,20), 3, 1.05 vf)`, `q ← max(0.85 q + 0.15 q_lrtc(8,40,10), 20)`পদক্ষেপ **R** (FD- গেইটড রিকন) এবং **T** (1-D TV) থেকে tfb-i4 / tfb-k5. সারিতে / path_flow রংগুলি অপরিবর্তিত বলে দাবি করা হয়েছে | **নির্বাচন A3** = `submission_FDRTV.csv` | `tfb-k9.py` = `kaggle_jobs/swap_lrtc_k9.py` = `submissions/76_K9/tfb-k9.py` |
| `tfb-k10` | `tfb-k9`, `tfb-t2osp`, `tfb-t2ogveto` | ক্যো স্ওয়াপ (tfb-j3 কোড +4 গেট): প্রাইভেট শুরু ← osp `base` , প্রাইভেট চলমান ← ogveto `veto10` ; দাবি করে যে মার্চ রায় পরিবর্তন হয়নি | **নির্বাচন B3** = `submission_K10.csv` | `tfb-k10.py` = `kaggle_jobs/swap_k10.py` = `submissions/77_K10/tfb-k10.py` |
| `tfb-j4` | `tfb-i4`, `tfb-t2ogpvogf` | (এখনই A2 হিসাবে পুনর্ব্যবহৃত) | তার সারিতে / path_flow সারি = A3 সারি / path_flow সারি | `tfb-j4.py` = `kaggle_jobs/swap_tfb-j4.py` |
| `tfb-k5` (রেফারেন্স) | `tfb-k5d` | R + T on k5d ( 0.88225 ) = k9 without M, B | – | `tfb-k5.py` |
| `tfb-k7` (রেফারেন্স) | `tfb-k5d` | M এর র্যাঙ্ক ( 8 , 40 , 10 ), কোন B, + R + T ( 0.88259 ) নেই | – | `tfb-k7.py` |

উপরের দিকে নয়ঃ `tfb-k1d` , `tfb-k1` (একটি CatBoost বীজ; k3d / k3 দ্বারা প্রতিস্থাপিত), `tfb-k2` , `tfb-k4` , `tfb-k6` , `tfb-k8` (পুরোনো পিকের হেজ) ।

## রং ব্লক অনুযায়ী কার্যকর productionকারী

| সারি | বিভক্ত | A3 = tfb-k9 | B3 = tfb-k10 | বনাম A2 / B2 |
|---|---|---|---|---|
| টাস্ক 1 অবস্থা | ভ্যাল + প্রাইভেট | tfb-k9 M + B + R + T on tfb-k5d ( 18 সদস্যঃ 8 পুরানো FTA , 6  FD , t1bspec, 2 × CatBoost, XGBoost) | = A3 | **পরিবর্তন** (A2/B2: টিএফবি-আই4 R + T on tfb-i4d, 15 সদস্য) |
| টাস্ক 2 শুরু | বৈধতা | tfb-t2osp sp4 | = A3 | একই |
| 2 কাজ চলছে | বৈধতা | tfb-t2occ fs4occ | = A3 | একই |
| টাস্ক 2 শুরু | ব্যক্তিগত | tfb-t2opv pv | tfb-t2osp বেস | A2 / B2 এর সমান |
| 2 কাজ চলছে | ব্যক্তিগত | tfb-t2ogpvogf pvogfv | tfb-t2ogveto ভেতো 10 | A2 / B2 এর সমান |
| কাজ 4  path_flow | ভ্যাল + প্রাইভেট | tfb-final (প্রতিটি kernelে অপরিবর্তিত) | = A3 | একই |

সুতরাং নতুন পিকগুলি A2 / B2 থেকে পৃথক হয় শুধুমাত্র টাস্ক 1 রাষ্ট্র সারি; প্রতিটি ক্যু এবং ওডম সারি একই।

## কিভাবে এন্ট্রি স্ক্রিপ্ট এটি পুনরায় তৈরি করে (স্তর, `tfb/config.py` )

- `t1m:<part>` জন্য 12 নতুন অংশঃ টেমপ্লেট `tfb/stages/t1_catb.py` (catb + catb)2ক্রমাগত; `cb_seed`, `cb_random_seed`, `row_rng`, `rng_base`, `lgb_seeds` মৃতদের `pF` লাইন, প্যানেল) এবং `tfb/stages/t1_xgb.py` (শুধুমাত্র প্যানেল) । `tests/test_templates.py` তাদের সকলের জন্য 12 অংশ এবং তুলনা AST প্রতিটি মূলের সাথে।
- `t1_mix_k5d` = `tfb-k5d` (ভিত্তিকঃ T4 স্কেলেট পরিবর্তে tfb-k3, যেমন `t1_mix` প্রতিস্থাপন tfb-i2 জন্য tfb-i4d).
- `t1_lrtc` = `tfb-k9` (বেসঃ আমাদের tfb-k5d) ।
- `assemble` (মডিউল) `assemble_k10` = `tfb-k10`) আমাদের টিএফবি-কে9 FDRTV, সঙ্গে 4 শর্ত × বিভক্ত অংশ A2 (A3) বা B2 (B3). A3 ভার্চুয়াল kernel ফোল্ডারে লিখুন `tfb-k9-asm` (প্রথম চেইন কোন পৃথক ছিল না A3 assembly; এর queue rows j এর মাধ্যমে এসেছিল4 → k3d → k3 → k5d → k9).

## গণনা (ক্যাগল CPU , পর্যবেক্ষণ)

| উপাদান | চালানোর সময় | পিক RSS |
|---|---|---|
| t1catb a / b / c / d | 3.5 / 4.1 / 3.2 / 4.9 h (15.7 h) | 26.1 / 22.0 / 19.3 / 10.9 GB |
| t 1 catb 2 a / b / c / d | 3.4 / 4.2 / 3.5 / 4.8 h (15.8 h) | 26.1 / 22.2 / 19.2 / 10.9 GB |
| t1xgb a / b / c / d | 1.0 / 1.2 / 0.8 / 1.2 h (4.2 h) | 26.0 / 22.2 / 19.2 / 10.8 GB |
| tfb-k5d | ≤15.6 min (পশ → সম্পূর্ণ) | লগ করা হয়নি (≈ tfb-i4d +12 ফাইল) |
| tfb-k9 | ট্র্যাক করা হয়নি; প্রত্যাশিত 16–20 min (tfb-k7 , প্রতি প্যানেল-মাসে এক LRTC : 771 s; k9 তিন রান) | ≈ tfb-k5 + < 0.4  GB |
| tfb-k10 | ট্র্যাক করা হয়নি (tfb-j3 , একই কোডঃ 97 s) | ছোট |
