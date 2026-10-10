# Push steps: T1 CatBoost member (EXP_T1_CATB)

These steps add a CatBoost member to the Task 1 mix.
- Lab result: +0.00071 J at weight 0.35, better in 8 of 8 panel-months.
- Expected public LB gain: +0.0003 to +0.0005.

## Before you start

- Push from the **hosen42** account, using the token in `~/.kaggle/access_token`. Every source kernel is a private hosen42 kernel.
- Run all commands from `kaggle_jobs/`.

## Step 1: four member kernels

The four kernels run in parallel. Each takes about 3–5 h.

```
python -c "import kjob; [print(p, kjob.push(f'tfb-t1catb{p}', f't1_catb_member_{p}.py', kernel_sources=['hosen42/tfb-t2lab-full'])) for p in 'abcd']"
```

When they finish, check each kernel's log:
- there is no Traceback;
- the `state rows ... NaN 0` line is present;
- peak RSS is below 30 GB.

## Step 2: swap kernel tfb-k1d

Start this only after all four kernels from step 1 are complete. It adds CatBoost to the T1 mix of tfb-j4 at weight 0.35.

```
python -c "import kjob; O=['tfb-t1v11','tfb-t1v12']+[f'tfb-t1v12s{i}' for i in range(2,8)]; F=['tfb-t1v13','tfb-t1v13s10','tfb-t1v13s11','tfb-t1dark','tfb-t1ramp','tfb-t1ramp2']; S=['tfb-j4']+[k+p for k in O+F for p in 'ab']+['tfb-t1bspeca','tfb-t1bspecb','tfb-t1bspecc']+[f'tfb-t1catb{p}' for p in 'abcd']; print(kjob.push('tfb-k1d','t1_catb_swap.py',kernel_sources=['hosen42/'+s for s in S]))"
```

## Step 3: recon + TV kernel tfb-k1

```
python -c "import kjob; print(kjob.push('tfb-k1','swap_fdrecon_k1.py',kernel_sources=['hosen42/tfb-k1d']))"
```

## Step 4: submit

Submit the `submission_FDRTV.csv` output of tfb-k1.
- If it beats 0.88176 (tfb-j4's public score), it is the new final pick A.
- Also build a hedge B on it: April queue rows from tfb-t2osp `queue_submission_base.csv` (onset) and tfb-t2ogveto `queue_submission_veto10.csv` (ongoing), the same as tfb-j3.
