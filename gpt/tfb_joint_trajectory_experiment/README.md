# TrafficFlowBench Task 2 — correlated-trajectory decoder: শুধু গবেষণার জন্য

**তৈরির তারিখ:** 2026-11-03। **এই package তৈরির সময় প্রকৃত competition data সরবরাহ বা চালানো হয়নি।**
এই পরীক্ষা Kaggle leader-কে হারিয়েছে—এমন দাবি করে না, competition CSV তৈরি করে না,
এবং tfb-k9, tfb-k10 বা আপনার বর্তমান দুটি final pick বদলায় না।

## মূল hypothesis

বর্তমান production `consolidation/tfb/stages/t2_occ.py:decode` P6 থেকে স্বাধীন
cell বেছে নেয় `sum(top p) / (m + sum(rest p))` সর্বোচ্চ করে। কিন্তু প্রকৃত queue
cell স্থান ও সময়জুড়ে পরস্পর সম্পর্কিত। নতুন এই *post-processing* arm:

1. মূল্যায়নের origin-এর তুলনায় STRICTLY আগের দিন এবং held-out training fold-এর
   বাইরে থাকা observed TRAIN queue trajectory সংগ্রহ করে।
2. একই directional panel-এর মধ্যে candidate trajectory template সবচেয়ে সম্ভাব্য
   3টি anchor link-এ সরায়; বাইরে zero padding, কোনো wrapping নয়।
3. Marginal probability-এর মিল অনুযায়ী সবচেয়ে সম্ভাব্য 64টি candidate রাখে।
4. Candidate সম্পূর্ণ trajectoryগুলোর ওপর probability-weighted posterior বানায়।
5. Posterior অনুযায়ী সর্বোচ্চ **expected full-window IoU** থাকা candidate বেছে নেয়।
6. Posterior expected improvement 0.005 না ছাড়ালে হুবহু baseline top-m output রাখে;
   prediction-এর সময় বর্তমান window-এর post-T truth পড়ে না।

এটি calibrated posterior probability বা physical correctness প্রমাণিত—এমন দাবি করে না।
Temperature 0.02, anchor 3টি, bank template 128টি এবং decision margin 0.005
**hypothesis-এর setting**; tuned বা optimal বলে প্রমাণিত নয়। এপ্রিলের label বা
এপ্রিলের leaderboard result দিয়ে এসব মান বাছাই করা যাবে না।

## স্থানীয়ভাবে সম্পন্ন কাজ

- Decoder, zero-padded translation, baseline equivalence, horizon truth-এর ওপর
  নির্ভর না করা ও আগের দিন/ভিন্ন fold-এর bank যাচাইয়ের 6টি unit/synthetic test পাস।
- প্রকৃত OOF runner-এর synthetic integration test পাস: 64টি window ও দুই panel।
- দুই panel-এর synthetic fixture যথাযথভাবেই >=8-panel research gate-এ FAILED।
- প্রকৃত TrafficFlowBench train window মূল্যায়িত 0টি; Kaggle submission 0টি।

## কোন স্কোর তুলনাযোগ্য

আপনার জানানো A3/B3 public composite **0.88281**; screenshot-এর সেরা **0.93124**।
কঠোরভাবে ছাড়াতে হলে **0.93125** লাগবে; ঘাটতি **+0.04844**। শুধু S_queue বদলালে
তা +0.1614667 বাড়তে হবে। একটি queue condition-এর মোট score coefficient 0.15;
তাই পরীক্ষার পরিমাপ করা `delta_condition_iou` মোট score-এ
`0.15 * delta_condition_iou` হবে, যদি তা নতুন scenario-তে টেকে এবং অন্য কিছু না বদলায়।
এখান থেকে এপ্রিলের অপ্রকাশিত official public score নির্ণয় করা যায় না।

## প্রকৃত OOF পরীক্ষা: বিদ্যমান competition data ও cache প্রয়োজন

Package যেকোনো folder-এ unzip করুন। তিনটি .py file সুবিধামতো যেকোনো জায়গায় রাখুন।
Script শুধু বিদ্যমান TRAIN lab cache পড়ে; macOS-এর absolute path-নির্ভর `lab.py`
import করে না।

Cache আগে থেকেই আছে ধরে, *নিজের repository-এর root* থেকে:

```bash
python /path/to/run_oof_experiment.py \
  --cache-root data/t2local/cache \
  --arm A_np255 \
  --set offlike \
  --bank-set faith3 \
  --condition queue_ongoing \
  --output analysis/t2lab/JOINT_OFFLIKE.json
```

**প্রতিটি panel-এর** জন্য প্রয়োজনীয় cache:

```
data/t2local/cache/oof/<ARM>/probs__offlike__<PANEL>.npz
     keys P6, done, d, t (original OOF probability output)
data/t2local/cache/rows/<PANEL>__offlike__queue_ongoing/
     aux.npz with truth, ev; win.pkl with d,t,fold
data/t2local/cache/rows/<PANEL>__faith3__queue_ongoing/
     aux.npz with truth, ev; win.pkl with d,t,fold
```

`A_np255` উদাহরণটি **পুরোনো lab baseline**; বর্তমান k9/k10-এর production fs4occ bag নয়।
প্রকৃত বর্তমান submission-এর তুলনায় বাড়তি লাভ প্রতিষ্ঠা করতে **বর্তমান production bag**
থেকে একই fold, eligibility ও decoder-সহ সমতুল্য OOF marginal P6 export করুন;
সেই arm-এর নাম ব্যবহার করুন। শুধু `A_np255`-কে হারানো দিয়ে arm গ্রহণ করা যাবে না।

Runner JSON ও `<output>.windows.csv` লেখে। এতে থাকে:
- Eligible cell-এ প্রতি window-এর নির্ভুল space-time IoU: baseline ও বিকল্প;
- Panel/family অনুযায়ী গড় IoU এবং প্রতি panel-এর delta;
- Day-block bootstrap confidence interval;
- বদলানো window-এর সংখ্যা, আগেই নির্ধারিত diagnostic gate ও deployment gate।

**Research gate; নিজে থেকে Kaggle submission নয়:**
`>=8 panels`, অন্তত `7`টি positive panel, aggregated condition IoU
`>=+0.01`, day-block bootstrap `95% lower > 0`; সবই TRAIN OOF-এ।
এই হিসাব অনুযায়ী domain shift-এর অনিশ্চয়তা ধরার আগে ওই এক condition থেকে
আনুমানিক সর্বোচ্চ +0.0015 composite লাভ; +0.04844-এর ঘাটতির তুলনায় অনেক কম।

## Leakage এড়ানোর গুরুত্বপূর্ণ সীমা

- TRAIN template bank-এ `bank_day < eval_day` এবং `bank_fold != eval_fold` filter আছে।
- বিদ্যমান `analysis/t2lab/lab.py:run_arm` OOF model নিজেই simulated OOF origin-এর
  তুলনায় *ভবিষ্যতের TRAIN fold* ব্যবহার করে fitted হতে পারে। তাই পুরোনো cache-এর
  OOF score দিয়ে **পুরো পদ্ধতি online সময়সীমা মানে—তা প্রমাণ হয় না**।
- Codeটি decoder lab; production-এর সঙ্গে integrate করা হয়নি।
- Award eligibility-এর জন্য সময়ক্রমে model আবার train করুন। শুধু origin T-এর আগে
  জানা train history ব্যবহার করুন। আয়োজকের **post-T truncation replay** চালিয়ে
  নিশ্চিত করুন যে *হুবহু একই queue prediction* পাওয়া যায়।
- এপ্রিলের label, leaderboard output বা post-T observation দিয়ে training, tuning,
  selection, filtering কিংবা কোনো parameter বাছাই করা যাবে না।
- মার্চের আগের window-তে backtest করার সময় পরের মার্চের output দিয়ে decoder train
  করা যাবে না। মার্চের জন্য train label ব্যবহারযোগ্য; এপ্রিলের origin-এর আগে পাওয়া
  মার্চের history এপ্রিলের জন্য ব্যবহারযোগ্য।
- বর্তমান lab RowCache-এর `truth` হলো observed eligible proxy truth;
  অপ্রকাশিত নির্ভুল official truth নয়। Synthetic test শুধু software test।

## ব্যবহারিক সিদ্ধান্তের সুপারিশ

k9/k10 স্থির রাখুন। Deadline-এর আগে *বর্তমান production*-এর OOF P6 cache পাওয়া গেলে
শুধু train-ভিত্তিক একটি diagnostic চালান। ব্যর্থ হলে Kaggle submission খরচ করবেন না।
পাস করলে কোনো submission বানানোর **আগে** chronological replay ও মার্চের মতো
transfer check করুন। শুধু এই artifact দিয়ে 0.93124-কে হারাবে বলে নিশ্চিত করা যায় না।

## Smoke test

```bash
python test_joint_decoder.py
python smoke_integration.py
```

প্রয়োজন `numpy`, `pandas`; Kaggle credential বা বাইরের network লাগে না।
