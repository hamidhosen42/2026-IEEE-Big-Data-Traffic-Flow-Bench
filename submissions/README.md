> বাংলা ভাষান্তর: স্থানীয় অনুবাদ মডেলের সহায়তায় অনূদিত। মূল ইংরেজি নথি সংরক্ষিত ব্যাকআপে আছে।

# জমা দেওয়া নোটবুক (স্থানীয়ভাবে রাখা)

| আউটপুট ফাইল | উপস্থাপনা রেফ | জনসাধারণ LB | বিবরণ |
|---|---|---:|---|
| submission_K6.csv | 57281653 | 0.88225 | K6 : FINAL  CANDIDATE B (হেক) on tfb-k5 (XGBoost T1 ): এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = ভেতো 10 ; মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 57275411 | 0.88225 | K5 : tfb-k3 + T1 XGBoost FD সদস্য (W 0.35 ; ল্যাব J +0.00056  8/8 , নিয়ন্ত্রণ-সংক্রান্ত +0.00061 ); ব্ল্যাকআউট বিশেষজ্ঞ, পুনর্বিবেচনা + TV অপরিবর্তিত |
| submission_K4.csv | 57263111 | 0.88201 | K4 : FINAL  CANDIDATE B (হেক) on tfb-k3 ( 2 CatBoost seeds): এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = ভেতো 10 ; মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 57263000 | 0.88201 | K3 : যেমন K1 কিন্তু T1 CatBoost সদস্য দুটি বীজের মধ্যম (tfb-t1catb + tfb-t1catb2 ), W 0.35 ; tfb-j4 ( A2 ); ব্ল্যাকআউট বিশেষজ্ঞ, recon + TV অপরিবর্তিত |
| submission_K2.csv | 57049844 | 0.88198 | K2 : FINAL  CANDIDATE B (হেক) on tfb-k1 (CatBoost T1 ): এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = ভeto 10 (ogfeat bag with growth veto); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 57049756 | 0.88198 | K1 : T1 মিশ্রণ + CatBoost FD সদস্য (গভীরতা 8 +10 , W 0.35 ; ল্যাব J +0.00071  8/8 ) tfb-j4 ( A2 ); ব্ল্যাকআউট বিশেষজ্ঞ, পুনর্বিবেচনা + TV অপরিবর্তিত |
| submission_J4.csv | 57004474 | 0.88176 | Q4 : FINAL  CANDIDATE  A2 (একই দৃশ্যের বাজি) tfb-i4 : এপ্রিল চলমান = মার্চ-বৃদ্ধি F03  np/id/full ব্যাগ + দৃশ্যপট পূর্ববর্তী +103 ওগফেট কলাম (pvogfv; বহুদিনের ছদ্ম মার্চ +0.020 IoU 8/8 , FM - PM প্রাইভেট +0.0015 ); মার্চ সারি অপরিবর্তিত |
| submission_J3.csv | 56989460 | 0.88176 | Q3 : FINAL  CANDIDATE  B2 (হেক) on tfb-i4 : এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = ogfeat F03 bag with a growth veto (veto10 : window fs4occ রাখে যদি ogfeat T এ ক্রমের বাইরে > 10 সেল যোগ করে; FM - PM private +0.0036 vs fs4occ); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56981375 | 0.88179 | Q1 : T1  FD ব্যাগ ALL সারি + ব্ল্যাকআউট বিশেষজ্ঞ ( 7  FD সদস্যঃ v13 , s10 , s11 , t 1 অন্ধকার, t 1 র্যাম্প, t 1 র্যাম্প 2 , t1bspec; W_FD  0.5 ); ব্ল্যাকআউট সারি 0.25 মিশ্রণ +0.75 বিশেষজ্ঞ; recon + TV |
| submission_I5.csv | 56956515 | 0.88176 | Q5 : FINAL  CANDIDATE B (হেক) on tfb-i4 (blackout specialist): এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = সব-সমাপ্ত F03 bag (only train models); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56956277 | 0.88176 | Q4 : T1 ব্ল্যাকআউট সারি = 0.25 মিশ্রণ +0.75  BLACKOUT  SPECIALIST (অফিসিয়াল ব্ল্যাকআউট প্লেসমেন্টের উপর প্রশিক্ষিতঃ শুরু হওয়া ক্যো শুরু হয় T+30 , ক্রমাগত ক্যো ভিতরে; 30 টাইমিং কলামঃ onset/ongoing পতাকা, অফসেট, সীমানা মান before/after ফাঁক, ফাঁকের উপর র্যাম্প প্রবাহ); FD ব্যাগ অপরিবর্তিত; recon + TV |
| submission_I3.csv | 56940941 | 0.88156 | Q3 : FINAL  CANDIDATE B (হেক) tfb-i2 (দুটি র্যাম্প সদস্য): এপ্রিল শুরু = train-এর আগে production CatBoost; এপ্রিল চলমান = সব-সমাপ্ত F03 ব্যাগ (শুধুমাত্র train-এর মডেল); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56940650 | 0.88156 | Q2 : T1 : দ্বিতীয় র্যাম্প সদস্য (বিম 17 ): ব্ল্যাকআউট সারি 0.25 মিশ্রণ +0.75 র্যাম্প সদস্যদের মধ্যম; FD সমস্ত সারি 6 ব্যাগ (v13 , s10 , s11 , t 1 অন্ধকার, t 1 র্যাম্প, t 1 র্যাম্প 2 ; W_FD  0.5 ); recon + TV |
| submission_I1.csv | 56929045 | 0.88145 | Q1 : FINAL  CANDIDATE B (হেক) on tfb-h5 ( 5 -member FD bag + ramp blackout): এপ্রিল শুরু = production CatBoost with the train prior; এপ্রিল চলমান = সম্পূর্ণ সমাপ্ত F03 bag (train-only models); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56908727 | 0.88145 | Q5 : T1  FD ব্যাগ ALL সারি + ব্ল্যাকআউট সদস্য এবং র্যাম্প সদস্য ( 5  FD সদস্যঃ v13 , s10 , s11 , t 1 অন্ধকার, t 1 র্যাম্প; W_FD  0.5 ); ব্ল্যাকআউট সারি 0.25 মিশ্রণ +0.75 র্যাম্প সদস্য; recon + TV |
| submission_H4.csv | 56908135 | 0.88008 | Q4 : F03 ব্যাগ থেকে মার্চ চলমান সারি (সব শেষ + কর্মক্ষমতা) 103 অতিরিক্ত ONGOING বৈশিষ্ট্যগুলি 60 -min ইতিহাস এবং স্লট T (কোরোরোর প্রসঙ্গে, প্রতি লিঙ্ক কিনেমিস, ক্যু-এন্ড demand/supply , ভোল্যাটি-স্বাভাবিক গতি; T এর পরে কিছুই নেই); এপ্রিল অপরিবর্তিত [ব্যাগ 4] |
| submission_H3.csv | 56898576 | 0.88129 | Q3 : FINAL  CANDIDATE B (হেক) র্যাম্প-ম্যান বেসেঃ এপ্রিল শুরু = production CatBoost train-এর আগে; এপ্রিল চলমান = সম্পূর্ণ সমাপ্ত F03 ব্যাগ (শুধুমাত্র train মডেল); মার্চ সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56898478 | 0.88120 | Q2 : T1 ব্ল্যাকআউট সারি (প্রতিটি অফিসিয়াল টাস্ক 2 উৎপত্তি পরে 90 min) = 0.5  T1 মিশ্রণ +0.5 সদস্য সিমুলেটেড ব্ল্যাকআউট সারি +9  RAMP বৈশিষ্ট্য (র্যাম্প-অ্যাঙ্করিত প্রবাহ স্থানান্তর; র্যাম্পগুলি ব্ল্যাকআউটগুলিতে মুক্তি পায়); recon + TV |
| submission_FDRTV.csv | 56897994 | 0.88129 | Q1 : T1 ব্ল্যাকআউট সারি (প্রতিটি অফিসিয়াল টাস্ক 2 উৎপত্তি পরে 90 min) = 0.25  T1 মিশ্রণ +0.75 সদস্য সিমুলেটেড ব্ল্যাকআউট সারি +9  RAMP বৈশিষ্ট্য (র্যাম্প-অ্যাঙ্করিত প্রবাহ স্থানান্তর; র্যাম্পগুলি ব্ল্যাকআউটগুলিতে মুক্তি পায়); recon + TV |
| submission_G5.csv | 56865768 | 0.88092 | Q5 : FINAL  CANDIDATE B (হেক): এপ্রিল শুরু = production আগাম train-এর সাথে CatBoost; এপ্রিল চলমান = সব-সমাপ্ত F03 ব্যাগ (শুধুমাত্র train-এর মডেল, পাবলিক সেরা হিসাবে একই রেসিপি); মার্চ লাইন অপরিবর্তিত |
| submission_G4.csv | 56865673 | 0.88092 | Q4 : FINAL  CANDIDATE A (একই দৃশ্যের বাজি): এপ্রিল শুরু = CatBoost ব্যাগ প্রশিক্ষণ মুক্ত-মার্চ window + দৃশ্য আগে; এপ্রিল চলমান = মার্চ-বৃদ্ধি F03  np/id/full + দৃশ্য আগে, সব শেষ ইনপুট; মার্চ সারি অপরিবর্তিত |
| submission_G3.csv | 56865571 | 0.88092 | Q3 : F03 ব্যাগ থেকে মার্চ চলমান সারিগুলি সব-সমাপ্ত ইনপুট +13  OCCUPANCY বৈশিষ্ট্য (অপকুয়েন্সি / প্রতি লিঙ্কের সমালোচনামূলক আবাস, প্রতিবেশী, পর্বত, ঘনত্ব চুক্তি); এপ্রিল অপরিবর্তিত |
| submission_G2.csv | 56865482 | 0.87883 | Q2 : CatBoost থেকে মার্চ শুরু হওয়া সারিগুলি ALL - FINITE ইনপুট (অযোগ্য কিন্তু বৈশিষ্ট্য হিসাবে মানের cell), একই sp4 দৃশ্য আগে; এপ্রিল অপরিবর্তিত |
| submission_FDRTV.csv | 56865369 | 0.87990 | Q1 : T1 ব্ল্যাকআউট লাইন ( 90 min after each official Task 2 origin) = 0.5  T1 mix +0.5 member trained with simulated-blackout transductive rows; recon + TV |
| submission_F5.csv | 56846866 | 0.88051 | P5 : ALL - FINITE  F03 ব্যাগ থেকে চলমান লাইন মার্চ (অযোগ্য কিন্তু বৈশিষ্ট্য হিসাবে মূল্যবান cell); এপ্রিল অপরিবর্তিত |
| submission_F4.csv | 56844607 | 0.87939 | P4 : F10 জোড় থেকে (g72 ) দূরত্ব-ব্লক windowের চলমান সারি মার্চ (np+id 511 , সমস্ত বিশ্বস্ত window 1.0 ); অন্যান্য window এবং এপ্রিল অপরিবর্তিত |
| submission_F3.csv | 56840738 | 0.87817 | P3 : মার্চ শুরুঃ production CatBoost + প্রথম মাসের মিশ্রণ প্রাক মিশ্রণ 9 ( P2  LODO সংরক্ষিত নয়) (প্রতি সেল দুই-হ্যোপোটিজ প্রাক 1-4 মার্চ প্রতিটি window আগে দিন, W= 6 , pi= 0.9 ); এপ্রিল সারি অপরিবর্তিত |
| submission_F2.csv | 56840649 | 0.87465 | P2 : LODO CatBoost ব্যাগ থেকে মার্চ শুরু হওয়া সারি (প্রশিক্ষণ সারিগুলি দিনের বাইরে পূর্বসূরি বহন করেঃ এক দিনের বাইরে যাওয়া x3 + ছুটি সপ্তাহের গ্রুপ আউট x3 ), একই sp4 দৃশ্য আগে; এপ্রিল সারি অপরিবর্তিত |
| submission_FDRTV.csv | 56840469 | 0.87963 | P1 : T1 = 0.5  FD - ফিচার ব্যাগ ( 4 সদস্য) +0.5 পুরানো ব্যাগ( 8 ), recon + TV |
| submission_E5.csv | 56817463 | 0.87961 | S5 : EXP_COMBO_OSP: চেইন বেস + একই সিনারি পূর্ববর্তী সঙ্গে শুরু সারি (sp4 : train পূর্ববর্তী প্রতিটি সেল সঙ্গে মিশ্রিত March/April মাস্ক স্তর T আগে দিন, মসৃণ +-2 স্লট, a=n/(n+4 ) |
| submission_E4.csv | 56817377 | 0.87857 | S4 : EXP_COMBO_v22: চেইন বেস + v22 fs4 থেকে চলমান সারি (নির্বাচন-নিষ্ঠ F03 ব্যাগঃ np/id x 2 বীজ) |
| submission_E3.csv | 56814685 | 0.87817 | S3 : EXP_COMBO_v21: চেইন বেস + v21 fs থেকে চলমান সারি 2 (নির্বাচন-নিষ্ঠার চলমান প্রশিক্ষণঃ বিশ্বস্ত window 0.3 + রান-স্টার্ট 1.0 ) |
| submission_E2.csv | 56810594 | 0.87343 | EXP_COMBO_24: চেইন বেস + v20 ও 8 ব্যাগ থেকে চলমান সারি (np511r+id511r x 4 বীজ) |
| submission_FDRTV.csv | 56810522 | 0.87334 | EXP_COMBO_23:  T1 = 0.5  FD - ফিচার ব্যাগ ( 3 ) +0.5 পুরানো ব্যাগ( 8 ), তারপর FD recon + TV |
| submission_TV.csv | 56787882 | 0.87271 | EXP_COMBO_22: চেইন বেস +1 -D TV টাস্ক 1 ঘনত্বের মসৃণতা লক্ষ্য রানগুলির মধ্যে |
| submission_COMBO19.csv | 56787752 | 0.85942 | EXP_COMBO_21: চেইন বেস + v19  ABC_st03 থেকে শুরু হওয়া সারি (CatBoost ব্যাগ + স্টেজ- 2 ; CV -গেটেড) |
| submission_COMBO18.csv | 56787709 | 0.87245 | EXP_COMBO_20: চেইন বেস + v18 ও 4 ব্যাগ থেকে চলমান সারি (np511r+id511r x 2 বীজ) |
| submission_FDR.csv | 56787648 | 0.87176 | EXP_COMBO_19: চেইন বেস + FD -gated reconciliation of Task 1 congested cells (v< 0.6 vf) |
| submission_COMBO16.csv | 56787531 | 0.87009 | EXP_COMBO_18: টাস্ক 1 ব্যাগ 8  FTA সদস্যদের (t 1 v 11 ,t 1 v 12 ,t 1 v 12 s 2 ,t 1 v 12 s 3 ,t 1 v 12 s 4 ,t 1 v 12 s 5 ,t 1 v 12 s 6 ,t 1 v 12 s 7 ) |
| submission_COMBO15.csv | 56764679 | 0.86944 | EXP_COMBO_17:  COMBO13 + টাস্ক 1 ব্যাগ 4  FTA সদস্য (বীর্য 0-3 ) |
| submission_COMBO14.csv | 56762903 | 0.86692 | EXP_COMBO_16:  COMBO13 + শুরু সারি = 50/50 CatBoost / LGB  70-30 (v17 , র্যাম্প নেই; ছদ্ম-মার্চ +0.018 বনাম বিড়াল) |
| submission_COMBO13.csv | 56762852 | 0.86899 | EXP_COMBO_15:  COMBO12 + v15 np511r থেকে চলমান সারি (পূর্ববর্তী 511 + র্যাম্প; সন্ড বনাম আইডি 511 r) |
| submission_COMBO12.csv | 56761353 | 0.86852 | EXP_COMBO_14:  COMBO11 + টাস্ক 1 ব্যাগ (ভয়সন্মানের মধ্যে v11  FTA বীজ 0 এবং v12  FTA বীজ 1 ) |
| submission_COMBO11.csv | 56761216 | 0.86768 | EXP_COMBO_13:  COMBO10 + টাস্ক 1  T1 v11  FTA থেকে সারি (ট্রান্সডাক্টিভ + ASM ; হোল্ডআউট +0.0005 বনাম FT ) |
| submission_COMBO10.csv | 56741839 | 0.86737 | EXP_COMBO_12:  COMBO9 + v15 থেকে চলমান সারি (id511 + র্যাম্প প্রবাহ T- 3 ..T- 1 ) |
| submission_COMBO9.csv | 56741513 | 0.86707 | EXP_COMBO_11:  COMBO8 + টাস্ক 1  T1 v9  FT থেকে সারি (প্রতিস্থাপক লক্ষ্য-মাস সারি; হোল্ডআউট 0.9406 বনাম 0.9386 ) |
| submission_COMBO8.csv | 56738051 | 0.86537 | EXP_COMBO_10:  COMBO7 + v14 id511 থেকে চলমান সারি (পূর্ববর্তী পরিচয় মুক্ত; ছদ্ম মার্চ +0.0036 ) |
| submission_COMBO7.csv | 56732225 | 0.86509 | EXP_COMBO_9:  COMBO6 + চলমান LGB  511  leaves/lr0.03/1800 গাছ (t 2 v 13 ) |
| submission_COMBO6.csv | 56722713 | 0.86327 | EXP_COMBO_8:  COMBO5 + T1 F ( 511 leaves, ভ্যাল-মাস হোল্ডআউট 0.9386 বনাম 0.9374 ) |
| submission_COMBO2D.csv | 56692851 | 0.85860 | EXP_COMBO_4: v9 ( 0.85639 ) + CatBoost শুরু + T1 প্রসারিত নিম্ন- র্যাঙ্ক (v6 D) |
