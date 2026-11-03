# বর্তমান 0.88281 pipeline-এর সঙ্গে discussionগুলোর পর্যালোচনা

পর্যালোচনার তারিখ: 2026-11-03, Asia/Dhaka। কাজের পরিসর: শুধু বিশ্লেষণ; কোনো training, kernel push, submission বা নির্বাচিত prediction পরিবর্তন করা হয়নি।

## প্রধান ফলাফল

স্থানীয়ভাবে সংরক্ষিত 19টি discussion thread-এ Task 2 সম্পর্কে আয়োজকদের নিশ্চিত করা কিছু উপকারী তথ্য আছে। তবে **বর্তমান pipeline-কে 0.90-এর ওপরে নিয়ে যাওয়ার নতুন কোনো পরীক্ষিত পদ্ধতি পাওয়া যায়নি**। প্রকাশিত সবচেয়ে কার্যকর সুবিধাগুলো বর্তমান সমাধানে ইতিমধ্যে আছে। এটি প্রমাণের মূল্যায়ন; 0.90 অসম্ভব—এমন দাবি নয়।

বিশেষ করে, discussion 742328-এ উল্লিখিত **0.93** হলো আলাদা রাখা training date-এ শুধু onset-এর IoU; এটি সম্মিলিত public leaderboard score নয়। discussion 742761-এর **0.9** হলো S_queue নিয়ে একটি প্রশ্ন; অর্জিত ফল বা প্রমাণিত সর্বোচ্চ সীমা নয়।

বেসলাইন: `consolidation/README.md` ও `consolidation/LINEAGE_K9.md` অনুযায়ী A3 = tfb-k9 এবং B3 = tfb-k10; দুটির public LB **0.88281**। মূল README-এর 0.85309 এবং `research/SCORE_GAP_ANALYSIS.md`-এর শুরুর স্কোর পুরোনো, বর্তমান নয়।

## তথ্যসূত্র ও হালনাগাদ অবস্থা

- `discussions/`-এর 19টি topic file ও সংরক্ষিত উত্তর পুরো পড়া হয়েছে। সূচি অনুযায়ী forum সংগ্রহের সময় **2026-11-03 16:41 UTC / বাংলাদেশ সময় 22:41**।
- তথ্যগুলো বর্তমান stage implementation, experiment tracker, selector replay, transfer analysis, capacity-scenario analysis এবং অক্টোবর 30 পর্যন্ত সাম্প্রতিক পরীক্ষার ফলের সঙ্গে মিলিয়ে দেখা হয়েছে।
- ওয়েব অনুসন্ধানে competition-এর অফিসিয়াল পৃষ্ঠা পাওয়া গেছে। বিস্তারিত discussion পৃষ্ঠা সরাসরি পড়তে গেলে কিছু পৃষ্ঠা অনুপলব্ধ বা ফাঁকা ছিল; shell network-এ DNS অনুপলব্ধ থাকায় আলাদা anonymous API refresh-ও ব্যর্থ হয়। তাই বিস্তারিত discussion-এর প্রমাণ এখানে তারিখযুক্ত স্থানীয় archive থেকে নেওয়া; স্বাধীনভাবে নতুন করে সংগ্রহ করা live forum থেকে নয়।
- বর্তমান অফিসিয়াল নিয়মে online/offline পার্থক্য ও সেপ্টেম্বরের rebuild-এর সমর্থন পাওয়া যায়: <https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules>।
- আয়োজকের বক্তব্য, অংশগ্রহণকারীর দাবি, আমাদের পরিমাপ করা ফল এবং নতুন hypothesis নিচে আলাদা করে দেখানো হয়েছে। অপরীক্ষিত hypothesis-এর জন্য কোনো লাভ ধরে নেওয়া হয়নি।

## গুরুত্বপূর্ণ threadগুলো আসলে কী দেয়

| থ্রেড | প্রমাণ | তাৎপর্য | আমাদের সমাধানে অবস্থা |
|---|---|---|---|
| [742350](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350), [742328](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742328) | আয়োজক নিশ্চিত করেছেন: 120টি onset window-এর প্রথম পাঁচ horizon step-এ queue নেই; queue আসে শুধু T+30-এ। এই নিয়ম ব্যবহার করা অনুমোদিত | ছয় step-এ positive prediction ছড়িয়ে না দিয়ে step 6-এ location predict করা | **ব্যবহার হচ্ছে।** onset-এর জন্য `to_matrix`-এ `[6]` আছে। অংশগ্রহণকারীর 0.57 → 0.93 ফলটি train onset-এর; আমাদের baseline-এর ওপরে নতুন লাভ নয় |
| [745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700) | T সময়ের observation ব্যবহার করা যায়। প্রকাশিত 360-minute spacing ও maximum-persistence-IoU selection rule-ও ব্যবহার করা যায়। শুধু T বা তার আগের origin ব্যবহার করা যাবে | Origin-slot feature ও selector মেনে training করা গুরুত্বপূর্ণ | **অনেকটাই ব্যবহার হয়েছে।** Slot T পড়া হয়; F03-এ faithful candidate ও run-start weighting আছে। ঠিক 360-minute ব্যবধানের window-তে model বদলানোর পরীক্ষা public লাভ প্রতিষ্ঠা করেনি |
| [749278](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749278) | Task 2-এর history span Task 1 masking থেকে সুরক্ষিত; train-এ unmasked state আছে | সাধারণ masked calendar window-এর history coverage অফিসিয়াল window-এর মতো নয় | **জানা আছে এবং আংশিক সমাধান হয়েছে।** Raw train history ও selection-faithful set আছে। বিশেষ করে ছোট onset পরিবর্তন যাচাইয়ে pseudo-March এখনও অসম্পূর্ণ proxy |
| [748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789) | একই split-এর আগের observation দিয়ে model refit করা যায়। মার্চের সব released traffic এপ্রিলের forecast-এ কাজে লাগানো যায়; উল্টোটি নয় | একই scenario-এর তথ্য দিয়ে adaptation বৈধ; শুরুর public window-এর তুলনায় private window-তে সুযোগ বেশি | **ব্যবহার হচ্ছে।** Scenario prior sp4, মার্চ দিয়ে বাড়ানো onset pv ও ongoing pvogf আছে। A3/B3-তে এই অনুমান নিয়ে দুটি ভিন্ন নির্বাচন রাখা হয়েছে |
| [748784](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748784) | 80টি অফিসিয়াল train window সব জুনে। Label তৈরি হয় underlying state থেকে, noisy observation থেকে নয়। Train-এর আরও সময় থেকে window তৈরি যুক্তিসংগত | শুধু অফিসিয়াল train window দিয়ে validation করলে পরিসর ছোট হয়; observed-cell proxy IoU হুবহু অফিসিয়াল metric নয় | **জানা আছে।** নয় মাসজুড়ে faithful train set আছে; noisy ও আংশিক label এখনও validation-এর সীমাবদ্ধতা |
| [749601](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749601) | প্রকাশিত descriptive metadata দেখা বৈধ। T-এর পরের traffic fitting, selection বা forecast-এর কোনো মান তৈরিতে ব্যবহার করা যাবে না। আয়োজক প্রতিটি T-এর পরের row সরিয়ে code আবার চালাবেন | শুধু feature extraction নয়, পুরো forecasting পদ্ধতিকে সময়ের সীমা মানতে হবে | বর্তমান provenance/truncation কাজ প্রাসঙ্গিক। পুরো মার্চের pseudo-label diagnostic দিয়ে আগের মার্চ window-এর জন্য method বাছাইয়ের বৈধতা নিজে থেকে প্রতিষ্ঠিত হয় না |
| [742761](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742761) | আয়োজক অর্জনযোগ্য score বা forecasting-এর কঠিনতা নির্দিষ্ট করেননি। Label-এ observation noise থেকে আসা কোনো বাধ্যতামূলক সীমা নেই | অংশগ্রহণকারীর onset প্রায় 0.7-এ আটকে যায়—এই দাবিকে প্রমাণিত সীমা ধরা যাবে না | **কোনো algorithm প্রকাশ হয়নি।** Noise-free label থাকলেই অনুমোদিত history থেকে ভবিষ্যৎ state নির্ভুল জানা যাবে, এমন নয় |
| [743819](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/743819) | Ordering হবে milepost/link_id দিয়ে; connectivity হবে incoming/outgoing দিয়ে; ramp attachment আলাদা। শুধু detector link রাখলে graph বিচ্ছিন্ন হওয়া স্বাভাবিক | জোর করে graph connected করা বা from_node/to_node থেকে scorer-এর order অনুমান করা ঠিক নয় | **আগেই audit করা হয়েছে।** নতুন +0.01719 লাভের প্রমাণ নেই; scorer topology এবং data থেকে শেখা propagation direction আলাদা বিষয় |
| [742980](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742980) | অংশগ্রহণকারী 0.809 pipeline, regularized NNLS ও soft FD projection দিয়েছেন। হঠাৎ onset হলে শুধু speed বেশি দেখে queue বাদ দেওয়া ব্যর্থ হয় | Baseline ও ODME scaling-এর উপকারী শিক্ষা আছে; সরল onset suppression ক্ষতি করতে পারে | **ইতিমধ্যে অতিক্রম করেছি।** আমাদের pipeline এই baseline থেকে তৈরি হলেও reconstruction ও queue model এখন আরও উন্নত |
| [742068](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068), [742910](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742910) | Task 2-তে T বা তার আগের traffic নেওয়া যায়; Task 1 offline। Synthetic তারিখ বাস্তব traffic archive-এর সঙ্গে মেলে না | Offline T1 reconstruction বা T-এর পরের ramp দিয়ে online queue forecast বানানো যাবে না। বাইরের traffic data উত্তর পাওয়ার উৎস নয় | **নিয়মের ব্যাখ্যা; নতুন predictive signal নয়** |

বাকি threadগুলোও পড়া হয়েছে:

| থ্রেড | 0.90 পদ্ধতির সঙ্গে সম্পর্ক |
|---|---|
| 740522, 740536 | আগে সরাসরি horizon এবং overlapping history থেকে উত্তর দেখা যেত। এসব disclosure-এর পর repair/rebuild হয়েছে। এগুলো বর্তমান forecasting পদ্ধতি নয় |
| 740844 | সেপ্টেম্বর 11-এর rebuild-এ scored traffic ও ID বদলেছে; আগের public score তুলনাযোগ্য নয়। পুরোনো বেশি score দিয়ে বর্তমান সক্ষমতা বিচার করা যাবে না |
| 744187, 749480 | Award, registration, reproducibility ও report-এর নিয়ম; modeling পদ্ধতি নেই |
| 741898 | Student award-এর পুরোনো ব্যাখ্যা; পরের award announcement সেটি বদলেছে। Modeling পদ্ধতি নেই |
| 743549 | দ্বিতীয় account থেকে accidental submission-এর ঘটনা; modeling পদ্ধতি নেই |

## বর্তমান baseline থেকে স্কোরের নির্ভুল হিসাব

Competition-এর weight:

`S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`।

অন্য output অপরিবর্তিত থাকলে:

`gap = 0.90000 - 0.88281 = 0.01719`।

| যে অংশ বদলাবে | সেই component-এ প্রয়োজনীয় absolute উন্নতি |
|---|---:|
| Queue: দুই condition-এর গড় | **+0.05730 S_queue** |
| শুধু onset | **+0.11460 onset IoU** |
| শুধু ongoing | **+0.11460 ongoing IoU** |
| শুধু state | +0.04911 S_state |

স্থানীয় Task 3 specification অনুযায়ী effective weight FD-এর জন্য 0.05 এবং LWR-এর জন্য 0.10। শুধু LWR দিয়ে ঘাটতি পূরণ করতে হলে +0.17190 LWR লাগবে। এগুলো হিসাবের শর্ত; অর্জনযোগ্যতার সীমা নয়।

`TRANSFER_ANALYSIS_2026-10-08.md`-এর পুরোনো decomposition-এ onset ≈0.765 এবং ongoing ≈0.863 ধরা হয়েছে। বর্তমান নির্বাচনেও একই public queue block আছে। **এই আনুমানিক মান সঠিক হলে**, S_queue প্রায় 0.814; অন্য অংশ স্থির রেখে লক্ষ্য পূরণে তা প্রায় **0.8713** হতে হবে। কয়েকটি উদাহরণ:

| Onset | Ongoing | গড় S_queue | ওই আনুমানিক decomposition অনুযায়ী মোট স্কোর |
|---:|---:|---:|---:|
| 0.7650 | 0.8630 | 0.8140 | 0.88281 |
| 0.8200 | 0.9200 | 0.8700 | 0.89961 |
| 0.8300 | 0.9200 | 0.8750 | 0.90111 |

এই মোটগুলো সম্ভাব্য পরিস্থিতির হিসাব; কোনো candidate-এর পরিমাপ করা score নয়। শুধু composite leaderboard থেকে বর্তমান componentগুলোর নির্ভুল score জানা যায় না।

Public-এ প্রতি condition-এ 40টি window আছে। সমান weight-এর aggregation অনুযায়ী একটি window-এর অবদান তার IoU-এর `0.15 / 40 = 0.00375` গুণ। Queue দিয়েই 0.01719 তুলতে হলে সব window মিলিয়ে **4.584 units per-window IoU উন্নতি** লাগবে; যেমন প্রায় দশটি window-এর প্রতিটিতে 0.4584 উন্নতি। এটি বড় আচরণগত পরিবর্তন; সামান্য কয়েকটি threshold edit নয়।

## সম্ভাবনাময় মনে হওয়া কোন ধারণাগুলো আগে পরীক্ষা হয়েছে

মিলিয়ে দেখা হয়েছে: `experiments/TRACKER.md`, `analysis/TRANSFER_ANALYSIS_2026-10-08.md`, `analysis/CAPACITY_SCENARIO_2026-10-11.md` ও সংশ্লিষ্ট স্থানীয় result file।

- **Selector মেনে training:** F03-এ ইতিমধ্যে public +0.00474 এসেছে। এটি baseline-এ আছে; ভবিষ্যতের লাভ হিসেবে আবার যোগ করা যাবে না।
- **360-minute spacing flag:** EXP_T2_G72-তে flagged window-এ model বদলানো হয়েছিল; public delta −0.00022। সব ধরনের selector modeling ব্যর্থ—এটি তা প্রমাণ করে না, কিন্তু নতুন লাভও প্রতিষ্ঠা করে না।
- **মাসের শুরুর same-split adaptation:** sp4 public +0.00104 দিয়েছে। আরও জটিল LODO ও mix9 যথাক্রমে −0.00496 ও −0.00144 হারিয়েছে। মাসের শেষের proxy-তে বড় লাভ মানেই মার্চের প্রথম পাঁচ দিনে লাভ নয়।
- **Ongoing model-এর আরও ভাগ:** ছোট model-এ per-step, step-6 specialist ও queue/free split-এর লাভ 511 leaves-এ টেকেনি; পরীক্ষা EXP_T2_OGCAP।
- **Extent calibration আগেও ভাবা হয়েছে:** `extcal_lib.py`-তে per-step Platt/isotonic calibration, queue-এর ভেতর/বাইরের calibration, Poisson size model ও size-rescaling arm আছে। নতুন প্রস্তাবকে এসব পরীক্ষা থেকে আলাদা হতে হবে।
- **একই দিনের context:** OGCLOCK train-এ লাভ দেখিয়েছে; নতুন scenario-তে লাভ ছোট এবং seed অনুযায়ী বদলেছে। এটি deploy করা হয়নি।
- **Physics/shock filtering:** EXP_T2_SHOCK transfer criteria পাস করেনি। পরের occupancy/vehicle-count feature এবং raw space-time patch-ও নির্ধারিত deployment gate পাস করেনি; পরীক্ষা EXP_T2_PHYS / PATCH।
- **Private pvogf-এ occupancy:** সাম্প্রতিক PVOCC verdict-এ দুই arm-এই `adopt: false`। বহু দিনের +0.0031 IoU estimate-এর confidence interval-এ শূন্য আছে; private footprint অনুযায়ী মোট লাভ প্রায় +0.00015। এটি +0.01719-এর প্রমাণ নয়।
- **Onset label noise:** EXP_onset_noise-এর sample-এ positive onset cell-এর মাত্র 0.2% cutoff-এর ±2 km/h-এর মধ্যে ছিল। মূল onset gap মিটিয়ে দেবে বলে label denoising-কে এখনই কৃতিত্ব দেওয়া যায় না।
- **Capacity ×1.2 correction:** capacity-scenario study-তে train/evaluation-এর মধ্যে এমন miscalibration পাওয়া যায়নি। প্রকাশিত factor কোনো অব্যবহৃত score multiplier নয়।
- **আরও T1 model ও smoothing:** কয়েকটি সংযোজন ইতিমধ্যে K9-এ আছে। সাম্প্রতিক লাভ অবশিষ্ট ঘাটতির তুলনায় অনেক ছোট; আগের probe অনুযায়ী T4-ও প্রায় saturated। তবে এগুলো T1/3-এর কোনো চূড়ান্ত সর্বোচ্চ সীমা প্রমাণ করে না।

## 0.90-এর সমাধান দাবি না করে আর কী পরীক্ষা করা যায়

### 1. Selection শর্ত মেনে queue site ও extent-এর যৌথ সম্ভাব্যতা

**এটি hypothesis; এই review-তে পরীক্ষা হয়নি।** বর্তমান `t2_occ.py:decode` cell probability rank করে এবং `sum(top p) / (m + sum(rest p))` proxy সর্বোচ্চ করে। এটি পুরো window-এর বিকল্প queue trajectoryগুলো আলাদাভাবে infer করে না।

একটি পৃথক পরীক্ষায় pre-origin evidence দিয়ে সম্ভাব্য onset site ও ongoing queue boundary trajectory যৌথভাবে model করা যায়। এরপর সম্ভাব্য trajectoryগুলো জুড়ে গড় IoU অনুযায়ী prediction বাছাই করা যায়। Trajectory distribution-এ প্রকাশিত onset/persistence selector-এর শর্ত রাখতে হবে। তুলনা হবে প্রকৃত production bag-এর সঙ্গে; দুর্বল single model-এর সঙ্গে নয়।

এটি শুধু এক বা দুই site-এ prediction সীমিত করা, per-step size scaling বা shock-speed rule দিয়ে cell বাদ দেওয়ার সমান নয়; সেসবের কিছু variant আগেই পরীক্ষা হয়েছে। লক্ষ্য transfer study-তে পাওয়া দুটি ত্রুটি:

- Onset: বিশ্লেষিত নতুন-scenario proxy-তে loss-এর 57–78% এসেছে zero-overlap window থেকে; অর্থাৎ location ভুল ছিল।
- Ongoing: study-র per-step size oracle প্রায় +0.05–0.06 IoU ফেরত পায়। অর্থাৎ extent সম্পর্কে কার্যকর তথ্য পাওয়া গেলে লাভ হতে পারে। Inference-এ oracle পাওয়া যায় না; এই লাভ অর্জনযোগ্য বলে প্রতিষ্ঠিত হয়নি।

**অনুমানের গুরুত্বপূর্ণ সীমা:** selection-এ true-state persistence IoU ≤0.9 লাগে। কিন্তু T-তে আমাদের observed/filled queue noisy ও অসম্পূর্ণ। এই proxy-এর সঙ্গে final prediction-এর IoU জোর করে 0.9-এর নিচে নামালেই উন্নতি হবে, এমন নিশ্চয়তা নেই। Selector-কে probabilistic model-এর তথ্য হিসেবে নিয়ে যাচাই করতে হবে; noisy reference-এর ওপর নিশ্চিত উত্তর-শর্ত হিসেবে চাপানো যাবে না। এমনকি Bayes-optimal prediction-ও প্রতিটি সম্ভাব্য true trajectory-এর সব বৈশিষ্ট্য মানতে বাধ্য নয়।

F03 training-এ selection-এর প্রভাব ইতিমধ্যে আছে, তাই বাড়তি লাভ ছোটও হতে পারে। এই প্রস্তাবের কোনো পরিমাপ করা লাভ নেই।

### 2. শুরুর public ও পরের private সময়ের প্রমাণ আলাদা করে দেখা

**এটি প্রমাণসমর্থিত দিক; বেশিরভাগই ব্যবহার হচ্ছে।** মার্চের প্রথম official window-গুলোর আগে একই scenario-এর খুব অল্প data থাকে। এপ্রিলের আগে পুরো আগের মাস থাকে। আয়োজক সেই history ব্যবহার অনুমোদন করেছেন। তাই বর্তমান A3/B3 নির্বাচন যুক্তিসংগত; তবে সুবিধাটি বর্তমান সমাধানেই আছে।

নতুন adaptation পরীক্ষায় সময়ক্রমে বাড়তে থাকা data prefix ও protected-history-এর মতো input ব্যবহার করা উচিত। Queue location ও extent আলাদা করে মাপতে হবে। পুরো মার্চের diagnostic এপ্রিলের method selection-এ কাজে লাগতে পারে; কিন্তু তা দিয়ে আগের মার্চ forecast-এর fitting/selection-এ T-এর পরের traffic আনা যাবে না। আগেই স্থির করা backtest মূল্যায়নে held-out ভবিষ্যৎ label দেখা যায়; সেই একই আগের forecast fit বা বাছাই করতে তা ব্যবহার করা যাবে না।

### 3. ছোট লাভ বিশ্বাস করার আগে মাপার পদ্ধতি উন্নত করা

**Validation উন্নতির প্রয়োজনের পক্ষে প্রমাণ আছে।** Protected history, মাসের শুরুর origin নির্বাচন, আংশিক observed proxy target ও hidden underlying official target—সবই গুরুত্বপূর্ণ। History coverage-এর distribution এবং সময়ক্রমিক month-start evaluation মিলিয়ে নিতে হবে; এরপর একাধিক panel ও আলাদা scenario-তে লাভ চাইতে হবে। Retrospective FM-PM diagnostic হিসেবে উপকারী; শুধু পুরো মার্চের ফল দিয়ে আগের মার্চ forecast-এর method selection সময়সীমা মেনেছে, তা প্রতিষ্ঠা হয় না।

Transfer study-তে পরীক্ষা করা পরিবর্তনগুলোর onset public-probe variability মোট score-এ প্রায় 0.003–0.005। একই file-এর public score deterministic। এখানে variability বলতে sampling এবং variant বাছাইয়ের প্রভাব বোঝানো হয়েছে; server-এর randomness নয়। Public-এ কয়েক দশ-হাজার ভাগের পুনরাবৃত্ত লাভ দিয়ে নির্ভরযোগ্য private লাভ বা 0.90-এর পথ ধরে নেওয়া উচিত নয়।

## সিদ্ধান্ত

**Discussion থেকে 0.90+ দিতে পারে—এমন কোনো candidate পরীক্ষিত হয়নি।** প্রকাশিত উপকারী সুবিধার বেশিরভাগ ইতিমধ্যে ব্যবহার হয়েছে। পরের যুক্তিসংগত গবেষণা Task 2-এর site/extent prediction, সঙ্গে সময়সীমা মেনে নির্ভরযোগ্য evaluation। Joint trajectory model আলাদা একটি hypothesis; কিন্তু এর পরিমাপ করা লাভ নেই, তাই 0.90-এর প্রতিশ্রুত সমাধান বলা যাবে না।

এই review-তে production code, model output বা final pick পরিবর্তন করা হয়নি।

## পরে পাঠানো বিকল্প বিশ্লেষণের তথ্য যাচাই

পাঠানো বিশ্লেষণে onset timing, T-সহ input rule, মার্চ দিয়ে এপ্রিলের adaptation এবং topology contract ঠিকভাবে চিহ্নিত হয়েছে। তবে এসব তথ্য থেকে নিয়মসম্মত 0.90+ পদ্ধতি প্রতিষ্ঠিত হয়েছে—এই সিদ্ধান্ত আসে না।

1. **পুরোনো component estimate-কে বর্তমান সর্বোচ্চ সীমা বলা হয়েছে।** 0.940–0.945 state ও প্রায় 0.778 queue এসেছে 0.86217 সময়ের score-gap analysis থেকে। এগুলো বর্তমান নির্ভুল পরিমাপ বা গাণিতিক upper bound নয়। বিশেষ করে 0.530 LWR ceiling সব panel-এর জন্য ওই স্থানীয় prototype থেকে প্রতিষ্ঠিত হয় না।
2. **Non-queue সর্বোচ্চ স্কোরের হিসাব নিজের table-এর সঙ্গে মেলে না।** উল্লিখিত সর্বোচ্চ মান বসালে `0.35*0.950 + 0.05*0.991 + 0.10*0.530 + 0.20*0.995 = 0.63405`, 0.631 নয়। ওই table ধরলে 0.90 পেতে S_queue = 0.8865 লাগে; 0.631 ধরলে প্রায় 0.89667 লাগে। কোনো অনুমানই অর্জনযোগ্যতা বা প্রকৃত সর্বোচ্চ সীমা প্রমাণ করে না।
3. **Roadmap-এর ন্যূনতম target যথেষ্ট নয়।** Onset 0.85 ও ongoing 0.88 হলে S_queue = 0.865। Table-এর অনুমিত সেরা non-queue অংশসহ মোট **0.89355**। উপরের বর্তমান আনুমানিক decomposition ধরলে **0.89811**। আরও বেশি score অর্জিত হলে 0.90 ছাড়ানো যেতে পারে; কিন্তু দেওয়া threshold-গুলো নিজেরা সেটি প্রতিষ্ঠা করে না।
4. **Onset-এর 0.93 একজন অংশগ্রহণকারীর held-out train ফল।** আয়োজক timing property ও তার অনুমোদিত ব্যবহার নিশ্চিত করেছেন; প্রতিটি model বা scored split ওই IoU পাবে বলেননি। আমাদের pipeline-এ timing property আছে।
5. **Direction reversal নতুন অপরীক্ষিত fix নয়।** `experiments/TRACKER.md:89`-এ EXP_021 দিয়ে W/S panel উল্টে overall pseudo-window delta খারাপ পাওয়ার পর বাদ দেওয়ার তথ্য আছে। আবার `analysis/t2lab/ogphys_lib.py`-এর অক্টোবর 30-এর data check-এ আটটি synthetic panel-এই stable queue head বেশি index-এর দিকে পাওয়া গেছে। Synthetic dynamics যাচাই ছাড়া পাঠানো geographic-direction rule চাপানো উচিত নয়। পরে direction ঠিক করা occGd arm-ও সম্মিলিত deployment gate পাস করেনি।
6. **বর্তমান baseline-এর অংশকে ভবিষ্যৎ লাভ বলা যাবে না।** Transductive training, CatBoost/LightGBM member, ODME regularization ও উন্নত FD-aware reconstruction আছে। সাধারণ 15% FD blend পুরোনো baseline setting; K9-এর ওপরে লাভের প্রমাণ নয়।

মূল্যায়ন: বিকল্প লেখাটি কিছু উপকারী তথ্য সংক্ষেপ করেছে, কিন্তু এই review-এর সিদ্ধান্ত বদলানোর মতো নতুন পরিমাপ করা candidate বা যুক্তিসংগত score budget দেয়নি। Task 2 এখনও গবেষণার সবচেয়ে শক্তিশালী অগ্রাধিকার। একে একমাত্র সম্ভাব্য পথ বলা বা এসব পরীক্ষা দিয়ে 0.90 অর্জনযোগ্য ঘোষণা করা প্রমাণের চেয়ে বেশি দাবি করা হয়।
