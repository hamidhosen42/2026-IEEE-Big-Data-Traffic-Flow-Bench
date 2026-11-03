> বাংলা ভাষান্তর: স্থানীয় অনুবাদ মডেলের সহায়তায় অনূদিত। মূল ইংরেজি নথি সংরক্ষিত ব্যাকআপে আছে।

# ধাপ ধাপঃ T1 CatBoost সদস্য ( EXP_T1_CATB )

এই পদক্ষেপগুলি Task 1 মিশ্রণে একটি CatBoost সদস্য যোগ করে।
- পরীক্ষাগার ফলাফলঃ +0.00071 J 0.35 ওজন, 8 এর মধ্যে ভাল 8 প্যানেল-মাস।
- জনসাধারণের প্রত্যাশিত LB লাভঃ +0.0003 থেকে +0.0005 পর্যন্ত ।

## শুরু করার আগে

- **hosen42** অ্যাকাউন্ট থেকে `~/.kaggle/access_token` এর টোকেন ব্যবহার করে ধাক্কা। প্রতিটি উত্স kernel একটি ব্যক্তিগত hosen42 kernel।
- `kaggle_jobs/` থেকে সব কমান্ড চালান।

## ধাপ 1 : চারটি সদস্যের kernel

চারটি kernel সমান্তরালভাবে চলবে। প্রতিটি kernel প্রায় 3–5 ঘন্টা সময় নেয়।

```
python -c "import kjob; [print(p, kjob.push(f'tfb-t1catb{p}', f't1_catb_member_{p}.py', kernel_sources=['hosen42/tfb-t2lab-full'])) for p in 'abcd']"
```

যখন তারা শেষ হয়, প্রতিটি kernelের লগ চেক করুনঃ
- কোন ট্র্যাকব্যাক নেই;
- `state rows ... NaN 0` লাইন উপস্থিত থাকে;
- পিক RSS  30  GB এর নিচে।

## ধাপ 2 : swap kernel tfb-k1d

1 ধাপ থেকে চারটি kernel সম্পূর্ণ হওয়ার পরে এটি শুরু করুন। এটি T1 মিশ্রণে tfb-j4 এর ওজন 0.35 এর সাথে CatBoost যোগ করে।

```
python -c "import kjob; O=['tfb-t1v11','tfb-t1v12']+[f'tfb-t1v12s{i}' for i in range(2,8)]; F=['tfb-t1v13','tfb-t1v13s10','tfb-t1v13s11','tfb-t1dark','tfb-t1ramp','tfb-t1ramp2']; S=['tfb-j4']+[k+p for k in O+F for p in 'ab']+['tfb-t1bspeca','tfb-t1bspecb','tfb-t1bspecc']+[f'tfb-t1catb{p}' for p in 'abcd']; print(kjob.push('tfb-k1d','t1_catb_swap.py',kernel_sources=['hosen42/'+s for s in S]))"
```

## ধাপ 3 : recon + TV kernel tfb-k1

```
python -c "import kjob; print(kjob.push('tfb-k1','swap_fdrecon_k1.py',kernel_sources=['hosen42/tfb-k1d']))"
```

## ধাপ 4 : জমা দিন

tfb-k1 এর `submission_FDRTV.csv` আউটপুট জমা দিন।
- যদি এটি 0.88176 (tfb-j4 এর পাবলিক স্কোর) কে পরাজিত করে, তাহলে এটি নতুন চূড়ান্ত পিক এ।
- এছাড়াও এটিতে একটি হেজ বি তৈরি করুনঃ এপ্রিলের ক্যু লাইনগুলি tfb-t2osp `queue_submission_base.csv` (আগমন) এবং tfb-t2ogveto `queue_submission_veto10.csv` (আগন্ত), tfb-j3 এর সাথে একই।
