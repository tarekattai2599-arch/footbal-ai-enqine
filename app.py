import datetime as dt
import difflib
import io
import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as _real_st
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson


# ======================= ARABIC / ENGLISH INTERFACE =======================
# Every visible text goes through tr() when the language is Arabic. Anything without a translation stays English.
def lang():
    try:
        return _real_st.session_state.get("lang", "ar")
    except Exception:
        return "ar"


def LX(en, ar):
    return ar if lang() == "ar" else en


_AR_LINES = r"""⚽ Pro Football AI Engine ⟹ ⚽ محرك كرة القدم الاحترافي بالذكاء الاصطناعي
Page ⟹ الصفحة
🗂️ Match board (multi-league) ⟹ 🗂️ لوحة المباريات (دوريات متعددة)
🎯 Single match deep-dive ⟹ 🎯 تحليل معمّق لمباراة واحدة
👥 Players ⟹ 👥 اللاعبون
Settings ⟹ الإعدادات
Years of history ⟹ سنوات من التاريخ
Recency half-life (days) ⟹ نصف عمر الحداثة (بالأيام)
xG / shots weight (0 = goals only) ⟹ وزن الأهداف المتوقعة والتسديدات (0 = الأهداف فقط)
Fuse a second data source (openfootball) ⟹ دمج مصدر بيانات ثانٍ (openfootball)
Competition ⟹ البطولة
Home team ⟹ الفريق المضيف
Away team ⟹ الفريق الضيف
Neutral venue (no home advantage) ⟹ ملعب محايد (بدون أفضلية الأرض)
**Bookmaker odds (optional)** ⟹ **أسعار السوق (اختياري)**
Home win odds ⟹ سعر فوز المضيف
Draw odds ⟹ سعر التعادل
Away win odds ⟹ سعر فوز الضيف
🎯 Prediction ⟹ 🎯 التوقع
📋 Team analysis ⟹ 📋 تحليل الفريقين
🧪 Backtest ⟹ 🧪 الاختبار الرجعي
📈 Ratings ⟹ 📈 التصنيفات
Estimates only. The model does not know injuries, line-ups or motivation. ⟹ تقديرات فقط. النموذج لا يعرف الإصابات ولا التشكيلات ولا الحافز.
Result correct, all matches ⟹ النتيجة الصحيحة، كل المباريات
Result correct, most confident third ⟹ النتيجة الصحيحة، أكثر ثلث ثقة
Goals for / match ⟹ أهداف مسجلة / مباراة
Goals against / match ⟹ أهداف مستقبَلة / مباراة
Points / match ⟹ نقاط / مباراة
xG for / match ⟹ xG لصالح / مباراة
xG against / match ⟹ xG ضد / مباراة
**Top 10 correct scores** ⟹ **أعلى 10 نتائج دقيقة احتمالًا**
Even the best single score is usually 10-15%. Use it as a guide, not a certainty. ⟹ حتى أفضل نتيجة دقيقة غالبًا لا تتجاوز 10-15%. اعتبرها إرشادًا لا يقينًا.
A safer pick pays less. It is only worth it if the bookmaker's odds are above the fair odds. ⟹ الاختيار الأكثر أمانًا عائده أقل. يستحق فقط إذا كان سعر السوق أعلى من السعر العادل.
**European (3-way) handicap** ⟹ **الهاندكاب الأوروبي (ثلاثي)**
Refund % = stake returned on whole-number lines. Fair odds assume zero bookmaker margin. ⟹ نسبة الاسترداد % = ما يُرد من المبلغ في الخطوط الصحيحة. السعر العادل يفترض عدم وجود هامش للسوق.
1st half goals ⟹ أهداف الشوط الأول
2nd half goals ⟹ أهداف الشوط الثاني
Both halves total ⟹ إجمالي الشوطين
**Team goals** ⟹ **أهداف كل فريق**
Upload the players CSV created by `fetch_understat.py` (Understat season stats). ⟹ ارفع ملف اللاعبين CSV الذي أنشأه `fetch_understat.py` (إحصائيات Understat للموسم).
Players CSV ⟹ ملف اللاعبين CSV
Role ⟹ المركز
All ⟹ الكل
Forwards ⟹ المهاجمون
Midfielders ⟹ لاعبو الوسط
Defenders ⟹ المدافعون
Min minutes played ⟹ الحد الأدنى للدقائق
Teams ⟹ الفرق
#### Add matches ⟹ #### إضافة مباريات
League ⟹ الدوري
Match date ⟹ تاريخ المباراة
Neutral venue ⟹ ملعب محايد
#### Match list (database) ⟹ #### قائمة المباريات (قاعدة البيانات)
#### Predictions for all matches ⟹ #### توقعات كل المباريات
#### Pick quality and risk ⟹ #### جودة الاختيارات ومخاطرها
⬇️ Download predictions (CSV) ⟹ ⬇️ تحميل التوقعات (CSV)
#### Match detail ⟹ #### تفاصيل المباراة
Choose a match ⟹ اختر مباراة
Data grade ⟹ درجة البيانات
**What each source says (home / draw / away %)** ⟹ **ماذا يقول كل مصدر (مضيف / تعادل / ضيف %)**
🎯 Correct score ⟹ 🎯 النتيجة الدقيقة
🛡️ Double chance ⟹ 🛡️ الفرصة المزدوجة
⚖️ Handicap ⟹ ⚖️ الهاندكاب
⏱️ Halves ⟹ ⏱️ الشوطان
⚽ Goals & BTTS ⟹ ⚽ الأهداف وتسجيل الفريقين
#### 🤖 Smart research assistant ⟹ #### 🤖 المساعد الذكي للبحث
##### ⚡ Automatic injury and absence feed ⟹ ##### ⚡ التغذية التلقائية بالإصابات والغيابات
Matches in the next N days ⟹ المباريات خلال الأيام القادمة
Refresh even if recent ⟹ تحديث حتى لو كانت البيانات حديثة
Run the automatic feed now ⟹ شغّل التغذية التلقائية الآن
##### Match research and manual adjustment ⟹ ##### البحث عن المباراة والتعديل اليدوي
Match ⟹ المباراة
Source ⟹ المصدر
📰 Free headlines (no key) ⟹ 📰 عناوين الأخبار المجانية (بدون مفتاح)
🔎 Tavily AI search (free key, 1,000 searches a month) ⟹ 🔎 محرك بحث Tavily AI (مفتاح مجاني، 1000 بحث شهريًا)
🧠 Claude with web search (paid key) ⟹ 🧠 Claude مع البحث في الإنترنت (مفتاح مدفوع)
**Adjust the model for this match** (your judgement after reading the news) ⟹ **عدّل النموذج لهذه المباراة** (تقديرك بعد قراءة الأخبار)
Note ⟹ ملاحظة
Save adjustment ⟹ حفظ التعديل
#### Track record ⟹ #### سجل الإنجازات
#### Data sources and backup ⟹ #### مصادر البيانات والنسخ الاحتياطي
Show status for leagues ⟹ عرض حالة الدوريات
**Auto-tune a league** ⟹ **ضبط تلقائي للدوري**
Leagues to tune ⟹ دوريات للضبط
**Calibrate confidence (recommended)** ⟹ **معايرة الثقة (موصى بها)**
Leagues to calibrate ⟹ دوريات للمعايرة
⬇️ Backup match list (CSV) ⟹ ⬇️ نسخة احتياطية من قائمة المباريات (CSV)
Restore match list from a backup CSV ⟹ استعادة قائمة المباريات من نسخة احتياطية CSV
### 🗂️ Match board: many leagues, one analysis ⟹ ### 🗂️ لوحة المباريات: دوريات كثيرة، تحليل واحد
➕ Matches ⟹ ➕ المباريات
📊 Analysis ⟹ 📊 التحليل
🤖 AI assistant ⟹ 🤖 المساعد الذكي
📈 Track record ⟹ 📈 سجل الإنجازات
🧰 Data & backup ⟹ 🧰 البيانات والنسخ الاحتياطي
Download CSV template ⟹ تحميل قالب CSV
Upload results CSV ⟹ ارفع ملف النتائج CSV
Could not load match data (check the file columns or the host's internet connection). ⟹ تعذر تحميل بيانات المباريات (تحقق من أعمدة الملف أو اتصال الإنترنت).
Very little data: predictions will be unreliable. ⟹ بيانات قليلة جدًا: التوقعات لن تكون موثوقة.
Not enough teams found in the data. ⟹ لا توجد فرق كافية في البيانات.
Choose two different teams. ⟹ اختر فريقين مختلفين.
 (neutral venue) ⟹  (ملعب محايد)
Draw ⟹ تعادل
**Head to head (last 5)** ⟹ **المواجهات المباشرة (آخر 5)**
Matches to test on ⟹ عدد مباريات الاختبار
Run backtest ⟹ تشغيل الاختبار الرجعي
This league is not calibrated yet. ⟹ لم تتم معايرة هذا الدوري بعد.
No data ⟹ لا توجد بيانات
football-data.org key ⟹ مفتاح football-data.org
Could not load data for this league (check the internet connection of the host and any key it needs). ⟹ تعذر تحميل بيانات هذا الدوري (تحقق من الإنترنت وأي مفتاح يحتاجه).
Try again now ⟹ جرّب مرة أخرى الآن
Open the 'Data & backup' tab and press 'Test all sources now' to see exactly what is blocked. ⟹ افتح تبويب 'البيانات والنسخ الاحتياطي' وادوس 'اختبر كل المصادر الآن' لتعرف بالضبط ما المحجوب.
Home odds (optional) ⟹ سعر المضيف (اختياري)
Draw odds (optional) ⟹ سعر التعادل (اختياري)
Away odds (optional) ⟹ سعر الضيف (اختياري)
Add this match ⟹ أضف هذه المباراة
⚡ Paste a list of matches (fastest way) ⟹ ⚡ الصق قائمة مباريات (أسرع طريقة)
Matches ⟹ المباريات
Date for these matches ⟹ تاريخ هذه المباريات
Add pasted matches ⟹ أضف المباريات الملصوقة
📡 Import upcoming matches automatically ⟹ 📡 استيراد المباريات القادمة تلقائيًا
Leagues ⟹ الدوريات
The list is empty. Add matches above. ⟹ القائمة فارغة. أضف مباريات بالأعلى.
Remove matches (by id) ⟹ حذف مباريات (بالرقم)
Remove ALL ⟹ حذف الكل
Add upcoming matches in the Matches tab. The analysis then runs automatically. ⟹ أضف المباريات القادمة في تبويب المباريات. بعدها يعمل التحليل تلقائيًا.
Markets to consider ⟹ الأسواق المطلوب النظر فيها
Minimum fair odds (skips trivial picks) ⟹ أدنى سعر عادل (لتخطي الاختيارات التافهة)
Include ClubElo (free external ratings) ⟹ إضافة ClubElo (تصنيفات خارجية مجانية)
Use each league's calibrated settings (best Elo weight + honest confidence) ⟹ استخدام إعدادات كل دوري المعايَرة (أفضل وزن Elo + ثقة صادقة)
Safe slip: only picks where all sources agree ⟹ السلبة الآمنة: الاختيارات التي تتفق عليها كل المصادر فقط
Maximum risk per match % ⟹ أقصى مخاطرة للمباراة %
Could not analyse:  ⟹ تعذر تحليل: 
No selection passes the filters. Lower the minimum probability or fair odds. ⟹ لا يوجد اختيار يجتاز الفلاتر. خفّض الحد الأدنى للاحتمال أو للسعر العادل.
#### Recommended safe slip ⟹ #### السلبة الآمنة الموصى بها
Copy-ready summary ⟹ ملخص جاهز للنسخ
How reliable has this league's model been? (measured on its recent matches) ⟹ ما مدى موثوقية نموذج هذا الدوري؟ (مقاسة على مبارياته الأخيرة)
 ⚠️ short rest ⟹  ⚠️ راحة قصيرة
Add matches first. ⟹ أضف مباريات أولًا.
Free keys for the other leagues (optional) ⟹ مفاتيح مجانية لباقي الدوريات (اختياري)
Tavily key (kept only in this session) ⟹ مفتاح Tavily (يُحفظ في هذه الجلسة فقط)
AI that reads the news ⟹ الذكاء الاصطناعي الذي يقرأ الأخبار
Groq (free) ⟹ Groq (مجاني)
Claude (paid) ⟹ Claude (مدفوع)
AI key (kept only in this session) ⟹ مفتاح الذكاء الاصطناعي (يُحفظ في هذه الجلسة فقط)
AI model (change if the provider renames it) ⟹ نموذج الذكاء الاصطناعي (غيّره لو غيّرت الشركة اسمه)
Fetch latest headlines for both teams ⟹ اجلب آخر عناوين الأخبار للفريقين
Saved. The Analysis tab now uses it. ⟹ تم الحفظ. تبويب التحليل يستخدمه الآن.
Nothing yet. Picks are saved automatically when you open the Analysis tab for upcoming matches. ⟹ لا شيء بعد. تُحفظ الاختيارات تلقائيًا عند فتح تبويب التحليل للمباريات القادمة.
Graded picks ⟹ اختيارات تم تقييمها
Hit rate ⟹ نسبة الإصابة
Average model probability ⟹ متوسط احتمال النموذج
**Calibration** (hit rate should be close to the model percentage; small samples are noisy) ⟹ **المعايرة** (يجب أن تقترب نسبة الإصابة من نسبة النموذج؛ العينات الصغيرة غير مستقرة)
Connection check (press if a league fails to load) ⟹ فحص الاتصال (اضغط إن فشل تحميل دوري)
Test all sources now ⟹ اختبر كل المصادر الآن
Tune selected leagues ⟹ اضبط الدوريات المختارة
Calibrate selected leagues ⟹ عايِر الدوريات المختارة
Reset all calibration ⟹ مسح كل المعايرة
Reset all tuning ⟹ مسح كل الضبط
Restore ⟹ استعادة
football-data.org free key ⟹ مفتاح football-data.org المجاني
Paste your free football-data.org key in the sidebar (register at football-data.org/client/register). ⟹ الصق مفتاح football-data.org المجاني في الشريط الجانبي (سجّل في football-data.org/client/register).
Main source was unreachable: using the backup source (openfootball). ⟹ المصدر الأساسي لا يستجيب: يتم استخدام المصدر الاحتياطي (openfootball).
**Model vs market** ⟹ **النموذج مقابل السوق**
A positive edge means the model rates the outcome higher than the bookmaker. Double-check injuries and line-ups before trusting it. ⟹ الميزة الإيجابية تعني أن النموذج يقيّم النتيجة أعلى من السوق. راجع الإصابات والتشكيلات قبل الاعتماد عليها.
No meetings in the loaded data. ⟹ لا توجد مواجهات في البيانات المحمّلة.
Reason:  ⟹ السبب: 
The internet source did not answer: using the last saved copy of the data (may be a bit old). ⟹ مصدر الإنترنت لا يستجيب: يتم استخدام آخر نسخة محفوظة من البيانات (قد تكون قديمة قليلًا).
Fetch and add ⟹ اجلب وأضف
Remove selected ⟹ حذف المحدد
Removed. Press R to refresh the list. ⟹ تم الحذف. اضغط R لتحديث القائمة.
List cleared. ⟹ تم مسح القائمة.
Minimum probability % ⟹ أدنى احتمال %
Weight of the Elo sources % ⟹ وزن مصادر Elo %
Weight of our model vs market odds % (only if odds exist) ⟹ وزن نموذجنا مقابل أسعار السوق % (فقط لو توفرت الأسعار)
Safe slip: minimum combined chance % (stress-tested) ⟹ السلبة الآمنة: أدنى احتمال مركب % (بعد اختبار الضغط)
Updating injury and absence data for the next 3 days... ⟹ جارٍ تحديث بيانات الإصابات والغيابات للأيام الثلاثة القادمة...
Chance every pick wins ⟹ احتمال فوز كل الاختيارات
Stress-tested chance ⟹ الاحتمال بعد اختبار الضغط
Chance at most one fails ⟹ احتمال فشل اختيار واحد على الأكثر
Build your own slip ⟹ ابنِ سلبتك بنفسك
Matches in the slip ⟹ المباريات في السلبة
No relevant absences found. ⟹ لا توجد غيابات مؤثرة.
Collecting injury and absence data... ⟹ جارٍ جمع بيانات الإصابات والغيابات...
No matches in that window. ⟹ لا توجد مباريات في هذه الفترة.
Headlines only, from Google News. Open the links to verify before adjusting anything. ⟹ عناوين فقط من Google News. افتح الروابط للتأكد قبل أي تعديل.
Search the web for this match ⟹ ابحث في الإنترنت عن هذه المباراة
Model ⟹ النموذج
Max web searches per match ⟹ أقصى عدد عمليات بحث لكل مباراة
 (changed) ⟹  (تم التغيير)
 (default kept) ⟹  (تم الإبقاء على الافتراضي)
Calibration cleared. ⟹ تم مسح المعايرة.
Tuning cleared. ⟹ تم مسح الضبط.
Backtesting... ⟹ جارٍ الاختبار الرجعي...
Not enough data. ⟹ بيانات غير كافية.
Could not understand:  ⟹ لم أفهم: 
No upcoming matches were published for these leagues in that window. Add them manually. ⟹ لم تُنشر مباريات قادمة لهذه الدوريات في هذه الفترة. أضفها يدويًا.
Not calibrated yet:  ⟹ لم تتم المعايرة بعد: 
. Calibration tests the model on each league's latest matches and keeps the settings that predicted best (about a minute per league, done once). ⟹ . تختبر المعايرة النموذج على أحدث مباريات كل دوري وتحتفظ بالإعدادات الأفضل (حوالي دقيقة لكل دوري، مرة واحدة).
Calibrate these leagues now ⟹ عايِر هذه الدوريات الآن
Even the safest single pick is below your target. No slip meets it. ⟹ حتى أأمن اختيار منفرد أقل من هدفك. لا توجد سلبة تحققه.
Exactly one fails ⟹ يفشل اختيار واحد فقط
Two or more fail ⟹ يفشل اختياران أو أكثر
No Tavily key found. Save it in secrets.toml (then restart the app) or paste it in the 'Free keys' box in the AI assistant tab. ⟹ لم يتم العثور على مفتاح Tavily. احفظه في secrets.toml (ثم أعد تشغيل البرنامج) أو الصقه في خانة 'المفاتيح المجانية' في تبويب المساعد الذكي.
Summaries are machine-generated from web pages. Open the sources before relying on them. ⟹ الملخصات مولّدة آليًا من صفحات الويب. افتح المصادر قبل الاعتماد عليها.
Anthropic API key (kept only in this session) ⟹ مفتاح Anthropic API (يُحفظ في هذه الجلسة فقط)
This text comes from web pages and a language model. Check it before relying on it. ⟹ هذا النص مأخوذ من صفحات ويب ونموذج لغوي. تحقق منه قبل الاعتماد عليه.
This match is already in the list. ⟹ هذه المباراة موجودة بالفعل في القائمة.
No recent headlines about injuries or line-ups. ⟹ لا توجد عناوين حديثة عن الإصابات أو التشكيلات.
Enter your Tavily key first. ⟹ أدخل مفتاح Tavily أولًا.
Enter an API key first. ⟹ أدخل مفتاح API أولًا.
**Sources:**  ⟹ **المصادر:** 
Searching... ⟹ جارٍ البحث...
Last 10 matches:  ⟹ آخر 10 مباريات: 
Best score inside each result:  ⟹ أفضل نتيجة داخل كل احتمال: 
Model confidence ⟹ ثقة النموذج
sources disagree on the result ⟹ المصادر مختلفة على النتيجة
❌ not found ⟹ ❌ غير موجود
real xG from the file ⟹ xG حقيقي من الملف
not available (goals only) ⟹ غير متوفر (الأهداف فقط)
measured in this league's data ⟹ مقاس من بيانات هذا الدوري
default - no half-time data ⟹ افتراضي - لا توجد بيانات الشوط الأول"""
AR_EXACT = {}
for _ln in _AR_LINES.split("\n"):
    if " ⟹ " in _ln:
        _k, _v = _ln.split(" ⟹ ", 1)
        AR_EXACT[_k] = _v
AR_PREFIX = [('Time-weighted Dixon-Coles model trained on real results',
  'نموذج ديكسون-كولز موزون بالزمن، مدرَّب على نتائج وتسديدات وأهداف متوقعة (xG) حقيقية. للتحليل فقط، بدون مراهنات.'),
 ("Measured on this league's most recent matches",
  "مقاسة على أحدث مباريات هذا الدوري، وكل توقع تم دون معرفة النتيجة. إذا اقترب 'ما قاله النموذج' من 'ما حدث فعلًا' فيمكن الوثوق بالنسب. الصفوف ذات "
  'المباريات القليلة غير مستقرة.'),
 ('Score = npxG/90 + xA/90',
  "الدرجة = npxG/90 + xA/90 (للمدافعين: xGChain/90 + xGBuildup/90، المساهمة بالاستحواذ فقط). 'حصة الفريق %' هي نصيب اللاعب من مجموع npxG + xA "
  'لفريقه: الرقم العالي يعني أن هجوم الفريق يعتمد عليه، فغيابه أكثر تأثيرًا.'),
 ('The database file lives next to the app.',
  'ملف قاعدة البيانات بجوار البرنامج. على Streamlit Cloud يُمسح عند إعادة تشغيل البرنامج، فاستخدم النسخ الاحتياطي والاستعادة من تبويب البيانات.'),
 ('Risk % = chance the pick fails',
  'المخاطرة % = احتمال فشل الاختيار + عقوبات للبيانات الضعيفة أو اختلاف المصادر أو الراحة القصيرة أو التعديلات المبنية على الأخبار. منخفضة ≤ 25، '
  'متوسطة ≤ 40، وما فوق ذلك عالية. درجة البيانات: A = أكثر من 30 مباراة حديثة للفريقين، B = أكثر من 18، C = أقل. الحكم: قوي = مخاطرة منخفضة ومصادر '
  "متفقة وبيانات كافية ودوري معايَر؛ متوسط الخطر = مخاطرة متوسطة بلا اختلاف؛ تجنّبه = غير ذلك. 'معايَر' = تم اختبار إعدادات الدوري على مبارياته الأخيرة (إعدادات "
  'التحليل أو تبويب البيانات).'),
 ('The model knows results, not news.',
  'النموذج يعرف النتائج لا الأخبار. اختر خيارًا مجانيًا أدناه للبحث عن الإصابات والإيقافات وأخبار التشكيل، ثم عدّل المباراة بنفسك.'),
 ('Premier League matches need nothing',
  'مباريات الدوري الإنجليزي لا تحتاج شيئًا: بيانات Fantasy Premier League الرسمية تسرد المصابين والموقوفين والمشكوك فيهم. وللدوريات الأخرى يبحث '
  'البرنامج في الأخبار (Tavily مجانًا) ويقرأ ذكاء اصطناعي مجاني (Groq) العناوين ويستخرج الغائبين. كل شيء يُحفظ ويُستخدم تلقائيًا في تبويب التحليل.'),
 ("Multiplies a team's expected goals.",
  'يضرب الأهداف المتوقعة للفريق. مثال: غياب المهاجم الأساسي حوالي 0.92؛ تشكيلة احتياطية أضعف 0.9. المباراة المعدّلة تأخذ عقوبة مخاطرة صغيرة لأنها '
  'مبنية على أخبار.'),
 ('Sources: football-data.co.uk',
  'المصادر: football-data.co.uk (النتائج والتسديدات والأسعار)، openfootball على GitHub (مصدر نتائج ثانٍ يُدمج بمطابقة أسماء الفرق)، نتائج martj42 '
  'الدولية (المنتخبات)، ClubElo (تصنيفات خارجية)، FPL (إصابات الدوري الإنجليزي).'),
 ('Tries a few settings',
  'يجرّب عدة إعدادات (نصف عمر الحداثة ووزن xG) ويحتفظ بما توقّع أحدث المباريات بشكل أفضل. لا يترك الإعداد الافتراضي إلا إذا كان التحسن واضحًا، فلا '
  'يجري وراء الصدفة. يستغرق حوالي 30-60 ثانية لكل دوري.'),
 ("Tests the model on each league's latest",
  'يختبر النموذج على أحدث حوالي 300 مباراة في كل دوري، توقعًا بتوقع دون النظر إلى النتيجة، ثم يحتفظ بوزن Elo وتصحيح الثقة الأفضل. كما يقيس كم مرة '
  "تحدث فعلًا توقعات النموذج 40% و50% و60%... افعل ذلك بعد 'الضبط' (إن استخدمته)."),
 ('Add matches from any league',
  'أضف مباريات من أي دوري، فيدرّب المحرك نموذج كل دوري، ويتوقع كل مباراة، ويرتب أقوى الاختيارات، ويوضح مدى خطورة السلبة المركبة. للتحليل فقط، بدون '
  'مراهنات في هذا الموقع.'),
 ('Walk-forward test',
  'اختبار تدريجي: يُعاد تدريب النموذج على الماضي فقط ثم يتوقع مباريات لم يرها من قبل. مع تفعيل xG والتسديدات تتم مقارنته بنسخة الأهداف فقط.'),
 ('Free sources only give attacking',
  'المصادر المجانية تعطي بيانات الهجوم والبناء فقط (xG وxA والتسديدات والتمريرات المفتاحية وxGChain وxGBuildup). بيانات الالتحامات والاعتراضات '
  'والهجمات المرتدة غير متاحة مجانًا لأن FBref أوقف إحصائياته المتقدمة في يناير 2026.'),
 ('This competition needs a free football-data.org key',
  'هذه البطولة تحتاج مفتاحًا مجانيًا من football-data.org: سجّل في football-data.org/client/register وانسخ المفتاح من الإيميل والصقه بالأسفل (أو '
  'احفظه باسم FOOTBALL_DATA_ORG_KEY في Secrets).'),
 ('One match per line',
  'مباراة في كل سطر، مثل `Arsenal - Chelsea` أو `Premier League: Arsenal vs Chelsea`. تتم مطابقة أسماء الفرق تلقائيًا ولا يشترط الإملاء الدقيق.'),
 ('Pulls the published fixtures',
  'يجلب المباريات المنشورة (مع أسعار السوق إن وُجدت) للدوريات الرئيسية. الدوريات بدون مصدر مجاني تحتاج إدخالًا يدويًا بالأعلى.'),
 ('Sources combined:',
  'المصادر المدمجة: نموذج ديكسون-كولز + Elo الخاص بنا + ClubElo + أسعار السوق (إن وُجدت). الأسعار اختيارية؛ وبدونها تُستخدم الثلاثة الأولى. راجع '
  'صفحة الاختبار الرجعي (صفّا Elo والمزج) لمعرفة أفضل الأوزان لكل دوري.'),
 ('Apply the automatic injury/absence feed',
  'تطبيق التغذية التلقائية للإصابات والغيابات (الدوري الإنجليزي: بيانات رسمية؛ باقي الدوريات: تحتاج المفاتيح المجانية في تبويب المساعد الذكي)'),
 ('Impact is a cautious estimate',
  'التأثير تقدير حذر: غياب مهاجم يُسقط نحو ثلث نصيبه من هجوم الفريق؛ وغياب مدافعين أساسيين أو الحارس يرفع أهداف الخصم قليلًا. بحد أقصى حوالي -18% / '
  '+10%.'),
 ('- **Tavily** (search)',
  '- **Tavily** (البحث): سجّل في tavily.com وانسخ المفتاح الذي يبدأ بـ `tvly-` (1000 بحث مجاني شهريًا)\n'
  '- **Groq** (ذكاء اصطناعي مجاني): سجّل في console.groq.com ثم API Keys ثم Create key'),
 ('Matches are still pending.', 'المباريات ما زالت معلّقة. تظهر النتائج تلقائيًا بعد نشرها من مصدر البيانات (عادة خلال يوم).'),
 ('No free automatic data feed',
  'لا يوجد مصدر بيانات مجاني تلقائي لهذه البطولة. ارفع ملف CSV بالأعمدة Date, HomeTeam, AwayTeam, FTHG, FTAG (حمّل القالب من الشريط الجانبي). أعمدة '
  'اختيارية ترفع الدقة: HS, AS, HST, AST (التسديدات) أو HxG, AxG (xG حقيقي).'),
 ('Main source unreachable: this league uses',
  'المصدر الأساسي لا يستجيب: هذا الدوري يستخدم المصدر الاحتياطي (openfootball)، لذا لا تتوفر التسديدات/xG ولا أسعار السوق.'),
 ('Built greedily from the lowest-risk picks',
  'تُبنى تدريجيًا من أقل الاختيارات مخاطرة ما دام الاحتمال المركب بعد اختبار الضغط فوق هدفك. اختبار الضغط = احتمال كل اختيار ناقص عقوبات مخاطر '
  'النموذج. لا نتيجة مضمونة: حتى سلبة 70% تفشل 3 مرات من كل 10.'),
 ('Free key without a card',
  'مفتاح مجاني بدون بطاقة: سجّل في tavily.com وانسخ المفتاح الذي يبدأ بـ `tvly-`. كل مباراة تستهلك 3 من رصيدك الشهري البالغ 1000.'),
 ("A 404 for 'this season'",
  "ظهور 404 لـ'هذا الموسم' طبيعي في أول أسابيع الموسم الجديد، وقتها يستخدم البرنامج المواسم السابقة. لو ظهر FAILED للكل فالمشكلة في إنترنتك أو "
  'الجدار الناري أو مضاد الفيروسات.'),
 ('Random guessing scores Brier', 'التخمين العشوائي يعطي Brier ≈ 0.667. كلما قلّ Brier / Log loss كان أفضل. أسعار السوق عادةً أصعب معيار للمقارنة.')]
AR_VALS = {'Low': 'منخفضة',
 'Medium': 'متوسطة',
 'High': 'عالية',
 'Strong': 'قوي ✅',
 'OK': 'متوسط الخطر',
 'Skip': 'تجنّبه ⛔',
 'all agree': 'كلهم متفقون',
 'split': 'منقسمون',
 'n/a': '—',
 'single source': 'مصدر واحد',
 'yes': 'نعم',
 'no': 'لا',
 'pending': 'معلّق',
 'Dixon-Coles': 'ديكسون-كولز',
 'Elo (own)': 'Elo (خاص بنا)',
 'ClubElo': 'ClubElo',
 'Market': 'السوق',
 'ok': 'تم',
 'already fresh': 'محدّث بالفعل',
 'team not found in FPL data': 'الفريق غير موجود في بيانات FPL',
 'needs free Tavily + AI keys (AI assistant tab)': 'يحتاج مفاتيح Tavily والذكاء الاصطناعي المجانية (تبويب المساعد الذكي)',
 'no data': 'لا توجد بيانات',
 'FPL official data': 'بيانات FPL الرسمية',
 'news search + AI': 'بحث الأخبار + ذكاء اصطناعي',
 'out': 'غائب',
 'doubtful': 'مشكوك فيه',
 'injured': 'مصاب',
 'suspended': 'موقوف',
 'unavailable': 'غير متاح',
 'GK': 'حارس',
 'DEF': 'مدافع',
 'MID': 'وسط',
 'FWD': 'مهاجم',
 'under 40%': 'أقل من 40%',
 'Our model + xG/shots': 'نموذجنا + xG/التسديدات',
 'Our model (goals only)': 'نموذجنا (الأهداف فقط)',
 'Naive (league averages)': 'ساذج (متوسطات الدوري)',
 'Model + Elo 20%': 'النموذج + Elo 20%',
 'Our model (matches with odds)': 'نموذجنا (المباريات ذات الأسعار)',
 'Bookmaker odds': 'أسعار السوق',
 'Blend 50/50 (model + market)': 'مزج 50/50 (النموذج + السوق)',
 'Home': 'المضيف',
 'Away': 'الضيف',
 'Neutral': 'محايد',
 'Over 1.5': 'أكثر من 1.5 هدف',
 'Over 2.5': 'أكثر من 2.5 هدف',
 'Under 2.5': 'أقل من 2.5 هدف',
 'Under 3.5': 'أقل من 3.5 هدف',
 'BTTS Yes': 'الفريقان يسجلان: نعم',
 'BTTS No': 'الفريقان يسجلان: لا',
 '1X': 'المضيف أو تعادل (1X)',
 'X2': 'تعادل أو الضيف (X2)',
 '12 (no draw)': '12 (بدون تعادل)',
 'football-data.co.uk (Serie A, this season)': 'football-data.co.uk (الدوري الإيطالي، هذا الموسم)',
 'football-data.co.uk (Serie A, last season)': 'football-data.co.uk (الدوري الإيطالي، الموسم الماضي)',
 'football-data.co.uk (fixtures)': 'football-data.co.uk (المباريات القادمة)',
 'openfootball (GitHub)': 'openfootball (GitHub)',
 'International results (GitHub)': 'النتائج الدولية (GitHub)',
 'FPL official API': 'واجهة FPL الرسمية',
 'Google News RSS': 'أخبار Google (RSS)'}
AR_LEAGUES = {'Premier League (England)': 'الدوري الإنجليزي الممتاز (إنجلترا)',
 'Championship (England)': 'تشامبيونشيب (إنجلترا)',
 'La Liga (Spain)': 'الدوري الإسباني (إسبانيا)',
 'Serie A (Italy)': 'الدوري الإيطالي (إيطاليا)',
 'Bundesliga (Germany)': 'الدوري الألماني (ألمانيا)',
 'Ligue 1 (France)': 'الدوري الفرنسي (فرنسا)',
 'Eredivisie (Netherlands)': 'الدوري الهولندي (هولندا)',
 'Liga Portugal': 'الدوري البرتغالي',
 'Super Lig (Turkey)': 'الدوري التركي (تركيا)',
 'Belgian Pro League': 'الدوري البلجيكي',
 'Scottish Premiership': 'الدوري الاسكتلندي الممتاز',
 'Greek Super League': 'الدوري اليوناني',
 'Norway Eliteserien': 'الدوري النرويجي',
 'Sweden Allsvenskan': 'الدوري السويدي',
 'Russian Premier League': 'الدوري الروسي الممتاز',
 'Brazil Serie A': 'الدوري البرازيلي',
 'USA MLS': 'الدوري الأمريكي MLS',
 'Argentina Primera': 'الدوري الأرجنتيني',
 'Mexico Liga MX': 'الدوري المكسيكي',
 'Japan J1 League': 'الدوري الياباني J1',
 'Denmark Superliga': 'الدوري الدنماركي',
 'Austria Bundesliga': 'الدوري النمساوي',
 'Switzerland Super League': 'الدوري السويسري',
 'Poland Ekstraklasa': 'الدوري البولندي',
 'League One (England)': 'الدرجة الأولى الإنجليزية (League One)',
 'League Two (England)': 'الدرجة الثانية الإنجليزية (League Two)',
 'Scottish Championship': 'تشامبيونشيب اسكتلندا',
 'Bundesliga 2 (Germany)': 'بوندسليجا 2 (ألمانيا)',
 'Serie B (Italy)': 'الدوري الإيطالي الثاني (سيري B)',
 'La Liga 2 (Spain)': 'الدوري الإسباني الثاني',
 'Ligue 2 (France)': 'الدوري الفرنسي الثاني',
 'China Super League': 'الدوري الصيني الممتاز',
 'Finland Veikkausliiga': 'الدوري الفنلندي',
 'Ireland Premier Division': 'الدوري الأيرلندي الممتاز',
 'Romania Liga 1': 'الدوري الروماني',
 'Africa Cup of Nations (national teams)': 'كأس الأمم الأفريقية (منتخبات)',
 'UEFA Nations League (national teams)': 'دوري أمم أوروبا (منتخبات)',
 'Champions League (football-data.org free key)': 'دوري أبطال أوروبا (مفتاح football-data.org مجاني)',
 'Egyptian Premier League (upload CSV)': 'الدوري المصري الممتاز (رفع CSV)',
 'Saudi Pro League (upload CSV)': 'الدوري السعودي للمحترفين (رفع CSV)',
 'Any other competition (upload CSV)': 'أي بطولة أخرى (رفع CSV)'}
AR_COLS = {'1 %': 'فوز المضيف %',
 'X %': 'تعادل %',
 '2 %': 'فوز الضيف %',
 '1st half over %': 'أكثر من في الشوط الأول %',
 '2nd half over %': 'أكثر من في الشوط الثاني %',
 'Absences': 'الغيابات',
 'Added by fusion': 'مضاف بالدمج',
 'Agreement': 'الاتفاق',
 'Away %': 'ضيف %',
 'Away goals x': 'معامل أهداف الضيف',
 'BTTS %': 'الفريقان يسجلان %',
 'Best pick': 'أفضل اختيار',
 'Calibrated': 'معايَر',
 'Chance all win %': 'احتمال فوز الكل %',
 'Chance of playing %': 'احتمال اللعب %',
 'Chance slip fails %': 'احتمال فشل السلبة %',
 'Combined fair odds': 'السعر العادل المركب',
 'Data': 'البيانات',
 'Date': 'التاريخ',
 'Detail': 'التفاصيل',
 'Draw %': 'تعادل %',
 'Fair odds': 'السعر العادل',
 'From': 'من',
 'Goals exp.': 'الأهداف المتوقعة',
 'Half-life': 'نصف العمر',
 'Handicap': 'الهاندكاب',
 'Hit': 'أصاب',
 'Hit rate %': 'نسبة الإصابة %',
 'Home %': 'مضيف %',
 'Home goals x': 'معامل أهداف المضيف',
 'League': 'الدوري',
 'Legs': 'عدد الاختيارات',
 'Likely score': 'النتيجة المرجحة',
 'Line': 'الخط',
 'Log loss': 'Log loss',
 'Long-term (ignored)': 'طويل الأمد (متجاهل)',
 'Lose %': 'خسارة %',
 'Market': 'السوق',
 'Match': 'المباراة',
 'Matches': 'المباريات',
 'Model %': 'النموذج %',
 'Model confidence': 'ثقة النموذج',
 'Model said %': 'ما قاله النموذج %',
 'News': 'الخبر',
 'Over 2.5 %': 'أكثر من 2.5 %',
 'Pick': 'الاختيار',
 'Pick %': 'احتمال الاختيار %',
 'Picks': 'الاختيارات',
 'Player': 'اللاعب',
 'Pos': 'المركز',
 'Probability %': 'الاحتمال %',
 'Really happened %': 'ما حدث فعلًا %',
 'Refund %': 'الاسترداد %',
 'Result': 'النتيجة',
 'Risk': 'المخاطرة',
 'Risk %': 'المخاطرة %',
 'Score': 'النتيجة',
 'Seconds': 'ثوانٍ',
 'Share of attack': 'حصة الهجوم',
 'Source': 'المصدر',
 'Sources': 'المصادر',
 'Status': 'الحالة',
 'Stress-tested %': 'بعد اختبار الضغط %',
 'Team': 'الفريق',
 'Teams': 'الفرق',
 'To': 'إلى',
 'Upcoming (2nd source)': 'قادمة (المصدر الثاني)',
 'Verdict': 'الحكم',
 'Why': 'السبب',
 'Win %': 'فوز %',
 'xG source': 'مصدر xG',
 'xG weight': 'وزن xG',
 'adj_a': 'تعديل الضيف',
 'adj_h': 'تعديل المضيف',
 'away': 'الضيف',
 'home': 'المضيف',
 'id': 'الرقم',
 'kickoff': 'موعد المباراة',
 'league': 'الدوري',
 'neutral': 'محايد',
 'note': 'ملاحظة',
 'o1': 'سعر 1',
 'ox': 'سعر X',
 'o2': 'سعر 2',
 'created': 'تاريخ الإضافة',
 'half_life': 'نصف العمر',
 'wx': 'وزن xG',
 'loss': 'الخسارة',
 'base_loss': 'الخسارة الأساسية',
 'tuned_at': 'وقت الضبط',
 'w_elo': 'وزن Elo',
 'temp': 'درجة الحرارة',
 'n': 'العدد',
 'Accuracy %': 'الدقة %',
 'Brier (lower=better)': 'Brier (الأقل أفضل)',
 'Log loss (lower=better)': 'Log loss (الأقل أفضل)',
 'Attack (goals vs avg)': 'الهجوم (الأهداف مقابل المتوسط)',
 'Defence (conceded vs avg, lower=better)': 'الدفاع (المستقبَل مقابل المتوسط، الأقل أفضل)',
 'Overall': 'الإجمالي',
 'Outcome': 'النتيجة',
 'Market %': 'السوق %',
 'Edge (pp)': 'الميزة (نقطة مئوية)',
 'Venue': 'الملعب',
 'Opponent': 'الخصم',
 'Res': 'النتيجة',
 'Min': 'الدقائق',
 'Goals': 'الأهداف',
 'Goals-xG': 'الأهداف-xG',
 'Shots/90': 'تسديدات/90',
 'KeyPass/90': 'تمريرات مفتاحية/90',
 'Team share %': 'حصة الفريق %',
 'Model': 'النموذج',
 'Hit_rate': 'نسبة الإصابة',
 'Tournament': 'البطولة'}
AR_PICK = {'1': 'فوز المضيف (1)', 'X': 'تعادل (X)', '2': 'فوز الضيف (2)', '12': 'فوز أي فريق (12)'}
AR_PICK_COLS = ['Best pick', 'Pick', 'Selection']
AR_COLVALS = {'Venue': {'H': 'أرض', 'A': 'خارج'}, 'Res': {'W': 'ف', 'D': 'ت', 'L': 'خ'}, 'Result': {'OK': 'يعمل ✅', 'FAILED': 'فشل ❌'}}
AR_LEAD = [('**Sources:** ', '**المصادر:** '), ('Could not analyse: ', 'تعذر تحليل: '), ('Could not understand: ', 'لم أفهم: '), ('Reason: ', 'السبب: ')]
AR_TPL_SRC = [('Version {}', 'الإصدار {1}'),
 ('xG source: {}', 'مصدر xG: {1}'),
 ('estimated from shots (on target x{}, other shots x{})', 'مقدَّر من التسديدات (على المرمى ×{1}، الأخرى ×{2})'),
 ('{} matches | {} → {} | home advantage ×{} | xG: {}', '{1} مباراة | {2} ← {3} | أفضلية الأرض ×{4} | xG: {5}'),
 ('last {} matches', 'آخر {1} مباراة'),
 ('model confidence {}%+', 'ثقة النموذج {1}%+'),
 ('temperature {}', 'درجة الحرارة {1}'),
 ('**Score grid (%)** - rows: {} goals, columns: {} goals', '**شبكة النتائج (%)** - الصفوف: أهداف {1}، الأعمدة: أهداف {2}'),
 ('Highest-probability pick: **{}** - {}% (fair odds {})', 'الاختيار الأعلى احتمالًا: **{1}** - {2}% (السعر العادل {3})'),
 ('Draw no bet: **{}** {}% (fair odds {}) | **{}** {}% (fair odds {})',
  'التعادل يرد المبلغ: **{1}** {2}% (السعر العادل {3}) | **{4}** {5}% (السعر العادل {6})'),
 ('**Asian handicap** (line added to {}; the {} line is the opposite)', '**الهاندكاب الآسيوي** (الخط يُضاف إلى {1}؛ وخط {2} هو العكس)'),
 ('Safest cover: **{} +1.5** - {}%', 'أأمن تغطية: **{1} +1.5** - {2}%'),
 ('1st half share of goals: {}% ({})', 'نصيب الشوط الأول من الأهداف: {1}% ({2})'),
 ('Half with more goals: 1st **{}%** | equal **{}%** | 2nd **{}%**', 'الشوط الأكثر أهدافًا: الأول **{1}%** | متساويان **{2}%** | الثاني **{3}%**'),
 ('Half-time result: {} **{}%** | Draw **{}%** | {} **{}%**', 'نتيجة الشوط الأول: {1} **{2}%** | تعادل **{3}%** | {4} **{5}%**'),
 ('Expected goals (1st / 2nd half) - {}: {} / {} | {}: {} / {}', 'الأهداف المتوقعة (الشوط الأول / الثاني) - {1}: {2} / {3} | {4}: {5} / {6}'),
 ('BTTS Yes: `{}%` | No: `{}%` | Expected total: **{}**', 'الفريقان يسجلان: نعم `{1}%` | لا `{2}%` | إجمالي الأهداف المتوقع: **{3}**'),
 ('{} expected goals', 'الأهداف المتوقعة لـ{1}'),
 ('{} / {} recent matches', '{1} / {2} مباريات حديثة'),
 ('Keys found: Tavily {} | AI ({}) {}. Keys saved in the secrets.toml file are loaded when the app starts.',
  'المفاتيح الموجودة: Tavily {1} | الذكاء الاصطناعي ({2}) {3}. المفاتيح المحفوظة في ملف secrets.toml تُحمّل عند تشغيل البرنامج.'),
 ('{} attack x', 'معامل هجوم {1}'),
 ('### 🏟️ {} vs {}', '### 🏟️ {1} ضد {2}'),
 ('{} win', 'فوز {1}'),
 ('fair odds {}', 'السعر العادل {1}'),
 ('Strongest line for the favourite: **{} -{}** - {}% (fair odds {})', 'أقوى خط للمرشح: **{1} -{2}** - {3}% (السعر العادل {4})'),
 ('No half-goal line where {} is above 55%: the match looks tight.', 'لا يوجد خط نصف هدف يتجاوز فيه {1} نسبة 55%: المباراة تبدو متقاربة.'),
 ('- Over **{}**: `{}%` | Under: `{}%`', '- أكثر من **{1}**: `{2}%` | أقل: `{3}%`'),
 ('**Absences from the automatic feed** ({}): goals x{} for {}, x{} for {}',
  '**الغيابات من التغذية التلقائية** ({1}): الأهداف ×{2} لـ{3}، ×{4} لـ{5}'),
 ('**{}**: form {} | {} scored, {} conceded | ', '**{1}**: الحالة {2} | سجّل {3}، استقبل {4} | '),
 ('{} days since last match', '{1} يوم منذ آخر مباراة'),
 ('Added {} match(es).', 'تمت إضافة {1} مباراة.'),
 ('Added {} vs {}.', 'تمت إضافة {1} ضد {2}.'),
 ('No match is below {}% risk. Raise the limit or lower the probability filters.',
  'لا توجد مباراة مخاطرتها أقل من {1}%. ارفع الحد أو خفّض فلاتر الاحتمال.'),
 ('**{}** -> half-life {} days, xG weight {}', '**{1}** ← نصف العمر {2} يوم، وزن xG {3}'),
 ('**{}**: Elo weight {}%, temperature {}, {} matches tested', '**{1}**: وزن Elo {2}%، درجة الحرارة {3}، تم اختبار {4} مباراة'),
 ('Restored {} match(es).', 'تمت استعادة {1} مباراة.'),
 ('Database problem: {}', 'مشكلة في قاعدة البيانات: {1}'),
 ('Fusion: +{} matches from the second source', 'الدمج: +{1} مباراة من المصدر الثاني'),
 (' {} skipped (team not recognised).', ' تم تخطي {1} (فريق غير معروف).'),
 ('Excluded as too risky ({})', 'مستبعد لأنه شديد المخاطرة ({1})'),
 ('Monte Carlo with {} rounds; matches treated as independent.', 'محاكاة مونت كارلو بـ {1} جولة؛ المباريات تُعامل كأحداث مستقلة.'),
 ('{}: no data', '{1}: لا توجد بيانات'),
 ('Tuning {}...', 'جارٍ ضبط {1}...'),
 ('Calibrating {}...', 'جارٍ معايرة {1}...'),
 ('Could not read the backup: {}', 'تعذر قراءة النسخة الاحتياطية: {1}'),
 ('Could not fetch news: {}', 'تعذر جلب الأخبار: {1}'),
 ('- **{}** ({}, risk {}%): {}', '- **{1}** ({2}، مخاطرة {3}%): {4}'),
 ('only {}% likely', 'احتمالها {1}% فقط'),
 ('little recent data for one team (+8)', 'بيانات حديثة قليلة لأحد الفريقين (+8)'),
 ('moderate recent data (+3)', 'بيانات حديثة متوسطة (+3)'),
 ('sources disagree by {} points (+{})', 'المصادر مختلفة بمقدار {1} نقطة (+{2})'),
 ('a team has 3 days or less of rest (+4)', 'أحد الفريقين لديه 3 أيام راحة أو أقل (+4)'),
 ('manual adjustment applied, news-driven (+3)', 'تم تطبيق تعديل يدوي مبني على الأخبار (+3)'),
 ('injury/absence news applied (+3)', 'تم تطبيق أخبار الإصابات/الغيابات (+3)'),
 ('4+ matches in the last 14 days for a team (+3)', '4 مباريات أو أكثر خلال آخر 14 يومًا لأحد الفريقين (+3)'),
 ('Model + Elo best (model {}%)', 'النموذج + Elo الأفضل (النموذج {1}%)'),
 ('Blend best weight (model {}%)', 'أفضل وزن للمزج (النموذج {1}%)'),
 ('1X ({} or draw)', '{1} أو تعادل (1X)'),
 ('X2 ({} or draw)', '{1} أو تعادل (X2)'),
 ("Not calibrated yet: {}. Calibration tests the model on each league's latest matches and keeps the settings that predicted best (about a minute "
  'per league, done once).',
  'لم تتم المعايرة بعد: {1}. تختبر المعايرة النموذج على أحدث مباريات كل دوري وتحتفظ بالإعدادات الأفضل (حوالي دقيقة لكل دوري، مرة واحدة).')]
AR_EXACT.update(AR_VALS)
# ---- v13 additions to the Arabic tables ----
AR_LEAGUES.update({
    "Egyptian Premier League": "الدوري المصري الممتاز", "Egyptian Premier League (upload CSV)": "الدوري المصري الممتاز (رفع CSV)",
    "Saudi Pro League": "الدوري السعودي للمحترفين",
    "UEFA Champions League": "دوري أبطال أوروبا",
    "UEFA Europa League": "الدوري الأوروبي (يوروبا ليج)",
    "UEFA Conference League": "دوري المؤتمر الأوروبي",
    "CAF Champions League": "دوري أبطال أفريقيا",
})
_EXT_COLS = {
    "Corners exp.": "الركنيات المتوقعة", "Total corners exp.": "إجمالي الركنيات المتوقع",
    "Over 8.5 %": "أكثر من 8.5 %", "Over 9.5 %": "أكثر من 9.5 %", "Over 10.5 %": "أكثر من 10.5 %",
    "Best corner pick": "أفضل اختيار ركنيات", "Confidence %": "الثقة %", "Grade": "التصنيف",
    "Exp. corners": "الركنيات المتوقعة", "Market": "السوق", "Probability %": "الاحتمال %", "Fair odds": "الأودز العادل",
    "Opposite": "العكس", "Opposite %": "احتمال العكس %", "Line": "الخط", "Home team %": "المضيف %",
    "Away team %": "الضيف %", "Model": "النموذج", "Log loss (Over 9.5)": "خطأ اللوغاريتم (أكثر من 9.5)",
    "Avg error (corners)": "متوسط الخطأ (ركنيات)", "Measure": "المقياس", "Statistic": "الإحصائية",
    "Understat xG matches": "مباريات xG من Understat", "Corners data": "بيانات الركنيات", "w_slow": "وزن النموذج البطيء",
    "Team": "الفريق", "Player": "اللاعب", "Status": "الحالة",
}
for _k, _v in _EXT_COLS.items():
    AR_COLS.setdefault(_k, _v)
_EXT_EXACT = {
    "Team corner model": "نموذج ركنيات الفرق", "League average only": "متوسط الدوري فقط",
    "Home more corners": "المضيف يحصل على ركنيات أكثر", "Away more corners": "الضيف يحصل على ركنيات أكثر",
    "Dixon-Coles (slow)": "ديكسون-كولز (بطيء)", "Understat": "Understat",
    "real xG (Understat)": "xG حقيقي (من Understat)", "yes": "نعم", "no": "لا",
    "Goals for": "الأهداف المسجلة", "Goals against": "الأهداف المستقبلة", "Shots for": "التسديدات",
    "Shots against": "تسديدات الخصم", "On target for": "التسديدات على المرمى",
    "On target against": "تسديدات الخصم على المرمى", "Goals per shot": "الأهداف لكل تسديدة",
    "Corners for": "الركنيات المحتسبة له", "Corners against": "الركنيات ضده", "Clean sheets %": "شباك نظيفة %",
    "Failed to score %": "مباريات بلا تسجيل %", "Both scored %": "تسجيل الفريقين %", "Over 2.5 %": "أكثر من 2.5 %",
    "ESPN (Egypt league scoreboard)": "ESPN (نتائج الدوري المصري)", "Understat (EPL)": "Understat (الدوري الإنجليزي)",
}
for _k, _v in _EXT_EXACT.items():
    AR_EXACT.setdefault(_k, _v)
AR_TPL_SRC.extend([
    ("Corners Over {}", "ركنيات أكثر من {1}"), ("Corners Under {}", "ركنيات أقل من {1}"),
    ("Home corners Over {}", "ركنيات المضيف أكثر من {1}"), ("Away corners Over {}", "ركنيات الضيف أكثر من {1}"),
    ("Over {}", "أكثر من {1}"),
])

AR_EXACT.update(AR_LEAGUES)
AR_TPL = [(re.compile("^" + re.escape(_e).replace(r"\{\}", "(.+?)") + "$", re.S), _a) for _e, _a in AR_TPL_SRC]


def tr(s):
    """English -> Arabic for one text (only when the interface language is Arabic)."""
    if not isinstance(s, str) or not s or lang() != "ar" or not re.search(r"[A-Za-z]", s):
        return s
    r = AR_EXACT.get(s)
    if r is not None:
        return r
    for p, a in AR_PREFIX:
        if s.startswith(p):
            return a
    for rx, tpl in AR_TPL:
        m = rx.match(s)
        if m:
            g = m.groups()
            return re.sub(r"\{(\d+)\}", lambda k: tr(g[int(k.group(1)) - 1]), tpl)
    for p, a in AR_LEAD:
        if s.startswith(p):
            return a + s[len(p):]
    if s.startswith("Best score inside each result: "):
        rest = re.sub(r"\*\*(.+?) win\*\*", r"**فوز \1**", s[len("Best score inside each result: "):])
        return "أفضل نتيجة داخل كل احتمال: " + rest.replace("**Draw**", "**تعادل**")
    m = re.match(r"^(#\d+ )(.+)$", s)
    if m:
        return m.group(1) + tr(m.group(2))
    if " | " in s:
        parts = s.split(" | ")
        out = [tr(p) for p in parts]
        if out != parts:
            return " | ".join(out)
    if "; " in s:
        parts = s.split("; ")
        out = [tr(p) for p in parts]
        if out != parts:
            return "؛ ".join(out)
    if ", " in s:
        parts = s.split(", ")
        if all(p in AR_EXACT for p in parts):
            return "، ".join(AR_EXACT[p] for p in parts)
    return s


def _fmt_opt(o):
    if isinstance(o, str):
        return AR_PICK.get(o) or tr(o)
    return str(o)


def _cell(v, pick=False):
    if not isinstance(v, str):
        return v
    if pick and v in AR_PICK:
        return AR_PICK[v]
    return tr(v)


def tr_df(df):
    """Translate column names and the text cells of a table that is about to be shown."""
    if lang() != "ar" or not isinstance(df, pd.DataFrame):
        return df
    try:
        d = df.copy()
        for c in list(d.columns):
            if not isinstance(c, str):
                continue
            if c in AR_COLVALS:
                d[c] = d[c].map(lambda v, m=AR_COLVALS[c]: (m[v] if v in m else tr(v)) if isinstance(v, str) else v)
            elif pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
                d[c] = d[c].map(lambda v, p=(c in AR_PICK_COLS): _cell(v, p))
        d.columns = [AR_COLS.get(c) or tr(c) if isinstance(c, str) else c for c in d.columns]
        if (pd.api.types.is_object_dtype(d.index) or pd.api.types.is_string_dtype(d.index)) and len(d.index):
            d.index = d.index.map(lambda v: _cell(v))
        return d
    except Exception:
        return df


_TXT1 = {"markdown", "caption", "info", "warning", "error", "success", "subheader", "header", "title", "spinner",
         "expander", "button", "checkbox", "toggle", "text_input", "text_area", "number_input", "slider", "date_input",
         "file_uploader", "download_button", "text"}
_CONTAINERS = {"expander", "container", "empty", "popover"}


def _wrap(name, fn):
    def inner(*a, **k):
        if lang() == "ar":
            a = list(a)
            if name in _TXT1 and a and isinstance(a[0], str) and not k.get("unsafe_allow_html"):
                a[0] = tr(a[0])
            elif name == "write":
                a = [tr(x) if isinstance(x, str) else x for x in a]
            elif name == "metric":
                if a:
                    a[0] = tr(a[0])
                if len(a) > 2 and isinstance(a[2], str):
                    a[2] = tr(a[2])
                if isinstance(k.get("delta"), str):
                    k["delta"] = tr(k["delta"])
            elif name == "tabs" and a:
                a[0] = [tr(x) if isinstance(x, str) else x for x in a[0]]
            elif name in ("selectbox", "multiselect", "radio") and a:
                a[0] = tr(a[0])
                if len(a) <= 3 and "format_func" not in k:
                    k["format_func"] = _fmt_opt
            elif name in ("dataframe", "table") and a:
                a[0] = tr_df(a[0])
            for kw in ("help", "placeholder"):
                if isinstance(k.get(kw), str):
                    k[kw] = tr(k[kw])
        res = fn(*a, **k)
        if name in ("columns", "tabs") and isinstance(res, (list, tuple)):
            return [_TrST(x) for x in res]
        if name in _CONTAINERS and res is not None:
            return _TrST(res)
        return res
    return inner


_WRAPPED = _TXT1 | {"write", "metric", "tabs", "selectbox", "multiselect", "radio", "dataframe", "table", "columns",
                    "container", "empty", "popover"}


class _TrST:
    """Thin proxy around streamlit: same API, but texts and labels are translated in Arabic mode."""
    def __init__(self, real):
        object.__setattr__(self, "_r", real)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_r"), name)
        if name == "sidebar":
            return _TrST(attr)
        if name in _WRAPPED and callable(attr):
            return _wrap(name, attr)
        return attr

    def __enter__(self):
        object.__getattribute__(self, "_r").__enter__()
        return self

    def __exit__(self, *exc):
        return object.__getattribute__(self, "_r").__exit__(*exc)


st = _TrST(_real_st)


def apply_language_style():
    """Right-to-left layout for Arabic, and tell the browser not to auto-translate the page (that breaks Streamlit)."""
    if lang() == "ar":
        st.markdown("<style>"
                    "[data-testid='stAppViewContainer'],[data-testid='stSidebar']{direction:rtl;}"
                    "[data-testid='stMarkdownContainer'],[data-testid='stCaptionContainer'],[data-testid='stAlert'],"
                    "[data-testid='stMetric'],label{text-align:right;}"
                    "[data-testid='stSlider'] [data-baseweb='slider']{direction:ltr;}"
                    "code,pre,input,textarea{direction:ltr;text-align:left;}"
                    "[data-testid='stDataFrame'],[data-testid='stTable']{direction:ltr;}"
                    "</style>", unsafe_allow_html=True)
    try:
        import streamlit.components.v1 as _cv1
        _cv1.html("<script>try{const d=window.parent.document;d.documentElement.setAttribute('lang','%s');"
                  "d.documentElement.setAttribute('translate','no');d.documentElement.classList.add('notranslate');"
                  "if(!d.querySelector('meta[name=google]')){const m=d.createElement('meta');m.name='google';"
                  "m.content='notranslate';d.head.appendChild(m);}}catch(e){}</script>" % lang(), height=0)
    except Exception:
        pass


APP_VERSION = "v13"
MAX_G = 8
RIDGE = 0.05
FD = "https://www.football-data.co.uk"
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

SOURCES = {
    "Premier League (England)": ("fd", "E0"), "Championship (England)": ("fd", "E1"),
    "La Liga (Spain)": ("fd", "SP1"), "Serie A (Italy)": ("fd", "I1"),
    "Bundesliga (Germany)": ("fd", "D1"), "Ligue 1 (France)": ("fd", "F1"),
    "Eredivisie (Netherlands)": ("fd", "N1"), "Liga Portugal": ("fd", "P1"),
    "Super Lig (Turkey)": ("fd", "T1"), "Belgian Pro League": ("fd", "B1"),
    "Scottish Premiership": ("fd", "SC0"), "Greek Super League": ("fd", "G1"),
    "Norway Eliteserien": ("new", "NOR"), "Sweden Allsvenskan": ("new", "SWE"),
    "Russian Premier League": ("new", "RUS"), "Brazil Serie A": ("new", "BRA"),
    "USA MLS": ("new", "USA"), "Argentina Primera": ("new", "ARG"),
    "Mexico Liga MX": ("new", "MEX"), "Japan J1 League": ("new", "JPN"),
    "Denmark Superliga": ("new", "DNK"), "Austria Bundesliga": ("new", "AUT"),
    "Switzerland Super League": ("new", "SWZ"), "Poland Ekstraklasa": ("new", "POL"),
    "League One (England)": ("fd", "E2"), "League Two (England)": ("fd", "E3"),
    "Scottish Championship": ("fd", "SC1"), "Bundesliga 2 (Germany)": ("fd", "D2"),
    "Serie B (Italy)": ("fd", "I2"), "La Liga 2 (Spain)": ("fd", "SP2"), "Ligue 2 (France)": ("fd", "F2"),
    "China Super League": ("new", "CHN"), "Finland Veikkausliiga": ("new", "FIN"),
    "Ireland Premier Division": ("new", "IRL"), "Romania Liga 1": ("new", "ROU"),
    "Africa Cup of Nations (national teams)": ("intl", "African Cup of Nations"),
    "UEFA Nations League (national teams)": ("intl", "UEFA Nations League"),
    "Champions League (football-data.org free key)": ("fdorg", "CL"),
    "Egyptian Premier League (upload CSV)": ("upload", ""),
    "Saudi Pro League": ("espn", "ksa.1"),
    "UEFA Champions League": ("espn", "uefa.champions"),
    "UEFA Europa League": ("espn", "uefa.europa"),
    "UEFA Conference League": ("espn", "uefa.europa.conf"),
    "CAF Champions League": ("espn", "caf.champions"),
    "Any other competition (upload CSV)": ("upload", ""),
}
# the competitions Tarek follows most come first in every list
_PRIORITY = ["Premier League (England)", "La Liga (Spain)", "Egyptian Premier League (upload CSV)", "UEFA Champions League",
             "Serie A (Italy)", "Bundesliga (Germany)", "Ligue 1 (France)", "UEFA Europa League",
             "UEFA Conference League", "UEFA Nations League (national teams)", "Africa Cup of Nations (national teams)",
             "CAF Champions League", "Russian Premier League", "Eredivisie (Netherlands)", "Liga Portugal",
             "Super Lig (Turkey)", "Brazil Serie A", "Belgian Pro League", "Norway Eliteserien", "Saudi Pro League",
             "Sweden Allsvenskan", "USA MLS", "Switzerland Super League"]
SOURCES = {**{k: SOURCES[k] for k in _PRIORITY if k in SOURCES}, **SOURCES}
ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
             ("PH", "PD", "PA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
             ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
ALIASES = {"Date": ["date"], "HomeTeam": ["Home", "home_team", "Home Team"],
           "AwayTeam": ["Away", "away_team", "Away Team"],
           "FTHG": ["HG", "home_score", "home_goals", "home_goal"],
           "FTAG": ["AG", "away_score", "away_goals", "away_goal"],
           "HxG": ["home_xg", "xG_home", "xGH", "HomeXG", "home_xG"],
           "AxG": ["away_xg", "xG_away", "xGAway", "AwayXG", "away_xG", "team_b_xg"],
           # FootyStats / other CSV exports
           "HC": ["home_team_corner_count", "HomeCorners", "home_corners"],
           "AC": ["away_team_corner_count", "AwayCorners", "away_corners"],
           "HS": ["home_team_shots", "HomeShots"], "AS": ["away_team_shots", "AwayShots"],
           "HST": ["home_team_shots_on_target", "HomeShotsOnTarget"],
           "AST": ["away_team_shots_on_target", "AwayShotsOnTarget"]}
ALIASES["HomeTeam"] += ["home_team_name", "Home Team Name", "team_home", "HomeTeamName"]
ALIASES["AwayTeam"] += ["away_team_name", "Away Team Name", "team_away", "AwayTeamName"]
ALIASES["FTHG"] += ["home_team_goal_count", "HomeGoals", "FTHG_"]
ALIASES["FTAG"] += ["away_team_goal_count", "AwayGoals", "FTAG_"]
ALIASES["HxG"] += ["team_a_xg"]
ALIASES["Date"] += ["date_GMT", "Date (GMT)", "match_date", "Match Date"]
TEMPLATE = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST\n"
            "15/08/2025,Team A,Team B,2,1,14,9,6,3\n22/08/2025,Team B,Team C,0,0,8,10,2,4\n")


# ======================= DATA =======================
# ---------- robust downloading: browser-like header, retries, saved copy on disk, error log ----------
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "*/*"}
CACHE_DIR = "data_cache"
NET_LOG = {}          # key -> (status, detail)
HOST_DEAD = {}        # host -> time of the last connection failure (skip it for a short while)


def fetch_bytes(url, tries=2, timeout=20, headers=None):
    last = ""
    host = url.split("/")[2] if "//" in url else url
    if time.time() - HOST_DEAD.get(host, 0) < 60:
        return None, f"{host} did not answer a moment ago (skipped, press 'Try again now')"
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers or UA, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content, ""
            last = f"HTTP {r.status_code}"
            if r.status_code in (403, 404):
                break
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:110]}"
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                HOST_DEAD[host] = time.time()
                break
        time.sleep(0.8 * (i + 1))
    return None, last


def fetch_csv(url, key):
    """Download a CSV. On success keep a copy on disk; on failure fall back to the last saved copy."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    path = os.path.join(CACHE_DIR, safe + ".csv")
    raw, err = fetch_bytes(url)
    if raw is not None:
        NET_LOG[key] = ("ok", url)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
        except OSError:
            pass
    elif os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            NET_LOG[key] = ("saved copy", f"{err} - using the last saved copy")
        except OSError:
            raw = None
    if raw is None:
        NET_LOG[key] = ("failed", f"{err}  [{url}]")
        return None
    try:
        try:
            txt = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            txt = raw.decode("latin-1")
        return pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
    except Exception as e:
        NET_LOG[key] = ("failed", f"unreadable file: {str(e)[:100]}  [{url}]")
        return None


def net_failures(prefix):
    return [f"{k}: {v[1]}" for k, v in NET_LOG.items() if k.startswith(prefix) and v[0] == "failed"]

def normalize(df):
    df = df.copy()
    if "status" in df.columns:        # exports that also list unplayed matches: keep the finished ones
        st_ = df["status"].astype(str).str.lower()
        if st_.isin(["complete", "finished", "ft", "full-time"]).any():
            df = df[st_.isin(["complete", "finished", "ft", "full-time"])]
    if "timestamp" in df.columns and not any(c in df.columns for c in ("Date", "date", "date_GMT")):
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            df["Date"] = pd.to_datetime(ts, unit="s", errors="coerce").dt.strftime("%Y-%m-%d")
    for std, alts in ALIASES.items():
        if std not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: std})
                    break
    nc = next((c for c in df.columns if str(c).lower() == "neutral"), None)
    df["Neutral"] = df[nc].astype(str).str.upper().eq("TRUE").astype(int) if nc else 0
    tc = next((c for c in df.columns if str(c).lower() == "tournament"), None)
    df["Tournament"] = df[tc].astype(str) if tc else ""
    df["Weight"] = np.where(df["Tournament"].eq("Friendly"), 0.5, 1.0)
    return df


def clean(df, dayfirst=True):
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    df = df.dropna(subset=need).copy()
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=LX("Downloading historical match data...", "جارٍ تحميل بيانات المباريات التاريخية..."))
def _load_data_raw(kind, code, years, key=""):
    """Raises when nothing could be loaded, so a failure is never cached."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=int(365.25 * years))
    if kind == "fd":
        y0 = today.year if today.month >= 7 else today.year - 1
        frames = []
        for y in range(y0, y0 - years - 1, -1):
            tag = f"{y % 100:02d}{(y + 1) % 100:02d}"
            d = fetch_csv(f"{FD}/mmz4281/{tag}/{code}.csv", f"{code}-{tag}")
            if d is not None and len(d):
                frames.append(d)
        if not frames:
            raise RuntimeError("football-data.co.uk: no season file could be downloaded")
        df = clean(normalize(pd.concat(frames, ignore_index=True)))
    elif kind == "new":
        d = fetch_csv(f"{FD}/new/{code}.csv", f"new-{code}")
        if d is None:
            raise RuntimeError("football-data.co.uk: file could not be downloaded")
        df = clean(normalize(d))
    elif kind == "fdorg":
        df = load_fdorg(code, years, key)[0]
    elif kind == "espn":
        df = load_espn_history(code, years)
    else:
        d = fetch_csv(INTL_URL, "intl-results")
        if d is None:
            raise RuntimeError("international results file could not be downloaded")
        df = clean(normalize(d), dayfirst=False)
    df = df[df.Date >= cutoff].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("the file was downloaded but has no recent matches")
    return df


BACKUP_USED = {}


def load_data(kind, code, years, key=""):
    """Main source first; for the big European leagues fall back to the openfootball source if it fails."""
    try:
        return _load_data_raw(kind, code, years, key)
    except Exception:
        pass
    if kind == "fd" and code in OF:
        res, _ = load_openfootball(code, years)
        if not res.empty:
            BACKUP_USED[code] = len(res)
            return res
    return pd.DataFrame()


def add_xg(df):
    """Real xG if the file has it; otherwise an xG estimate learned from shots; else goals."""
    df = df.copy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    if "HxG" in df and "AxG" in df and num("HxG").notna().mean() > 0.5:
        df["XGH"], df["XGA"] = num("HxG").fillna(df.FTHG), num("AxG").fillna(df.FTAG)
        return df, "real xG from the file"
    if all(c in df for c in ["HS", "AS", "HST", "AST"]):
        hs, aw, hst, ast = num("HS"), num("AS"), num("HST"), num("AST")
        ok = hs.notna() & aw.notna() & hst.notna() & ast.notna()
        if ok.sum() >= 100:
            fh = np.column_stack([hst[ok], (hs - hst)[ok].clip(lower=0)])
            fa = np.column_stack([ast[ok], (aw - ast)[ok].clip(lower=0)])
            coef, *_ = np.linalg.lstsq(np.vstack([fh, fa]),
                                       np.concatenate([df.FTHG[ok], df.FTAG[ok]]), rcond=None)
            coef = np.clip(coef, 0, None)
            if coef.sum() > 0:
                df["XGH"] = (hst * coef[0] + (hs - hst).clip(lower=0) * coef[1]).fillna(df.FTHG)
                df["XGA"] = (ast * coef[0] + (aw - ast).clip(lower=0) * coef[1]).fillna(df.FTAG)
                return df, f"estimated from shots (on target x{coef[0]:.2f}, other shots x{coef[1]:.2f})"
    df["XGH"], df["XGA"] = df.FTHG.astype(float), df.FTAG.astype(float)
    return df, "not available (goals only)"


def half_share(df):
    if "HTHG" in df and "HTAG" in df:
        h, a = pd.to_numeric(df.HTHG, errors="coerce"), pd.to_numeric(df.HTAG, errors="coerce")
        ok = h.notna() & a.notna()
        tot = (df.FTHG + df.FTAG)[ok].sum()
        if ok.sum() >= 100 and tot > 0:
            return float(np.clip((h[ok] + a[ok]).sum() / tot, 0.35, 0.55)), "measured in this league's data"
    return 0.45, "default - no half-time data"


def current_teams(df, min_gap=40):
    d = np.sort(df.Date.dt.normalize().unique())
    gaps = np.diff(d).astype("timedelta64[D]").astype(int)
    big = np.where(gaps > min_gap)[0]
    cur = df[df.Date >= d[big[-1] + 1]] if len(big) else df
    if cur.empty:
        cur = df
    return set(cur.HomeTeam) | set(cur.AwayTeam)


def eligible_teams(df, model, min_matches, recent_days):
    rec = df[df.Date >= df.Date.max() - pd.Timedelta(days=recent_days)]
    rec_t = set(rec.HomeTeam) | set(rec.AwayTeam)
    cnt = pd.concat([df.HomeTeam, df.AwayTeam]).value_counts()
    return sorted(t for t in model["teams"] if t in rec_t and cnt.get(t, 0) >= min_matches)


# ======================= MODEL =======================
def fit_dc(df, xi, wx=0.0):
    """Time-weighted Dixon-Coles (MLE). Scoring target = blend of goals and xG/shots-xG."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
    xg_, yg_ = df.FTHG.values, df.FTAG.values
    if wx > 0 and "XGH" in df:
        x = (1 - wx) * xg_ + wx * df.XGH.values
        y = (1 - wx) * yg_ + wx * df.XGA.values
    else:
        x, y = xg_.astype(float), yg_.astype(float)
    hv = 1.0 - (df.Neutral.values.astype(float) if "Neutral" in df else 0.0)
    bw = df.Weight.values if "Weight" in df else 1.0
    w = bw * np.exp(-xi * (df.Date.max() - df.Date).dt.days.values)
    m00, m01 = (xg_ == 0) & (yg_ == 0), (xg_ == 0) & (yg_ == 1)
    m10, m11 = (xg_ == 1) & (yg_ == 0), (xg_ == 1) & (yg_ == 1)

    def unpack(p):
        return p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean(), p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def fun(p):
        att, dfn, home, rho, c0 = unpack(p)
        el = c0 + att[hi] + dfn[ai] + home * hv
        em = c0 + att[ai] + dfn[hi]
        lam, mu = np.exp(el), np.exp(em)
        tau, dtl, dtm, dtr = np.ones_like(lam), np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
        lm = lam * mu
        tau[m00] = 1 - lm[m00] * rho
        dtl[m00] = dtm[m00] = -lm[m00] * rho
        dtr[m00] = -lm[m00]
        tau[m01] = 1 + lam[m01] * rho
        dtl[m01] = lam[m01] * rho
        dtr[m01] = lam[m01]
        tau[m10] = 1 + mu[m10] * rho
        dtm[m10] = mu[m10] * rho
        dtr[m10] = mu[m10]
        tau[m11] = 1 - rho
        dtr[m11] = -1.0
        tau = np.clip(tau, 1e-6, None)
        ll = np.log(tau) + x * el - lam + y * em - mu
        f = -(w * ll).sum() + RIDGE * (p[:2 * n] ** 2).sum()
        gl = w * (x - lam + dtl / tau)
        gm = w * (y - mu + dtm / tau)
        ga = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
        gd = np.bincount(ai, gl, n) + np.bincount(hi, gm, n)
        g = np.concatenate([ga - ga.mean(), gd - gd.mean(), [(gl * hv).sum(), (w * dtr / tau).sum(), (gl + gm).sum()]])
        g = -g
        g[:2 * n] += 2 * RIDGE * p[:2 * n]
        return f, g

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = 0.25
    p0[2 * n + 2] = float(np.log(max(0.3, (x.mean() + y.mean()) / 2)))
    bounds = [(-3, 3)] * (2 * n) + [(-0.5, 1.0), (-0.3, 0.3), (-2.0, 2.0)]
    res = minimize(fun, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    att, dfn, home, rho, c0 = unpack(res.x)
    return {"teams": teams, "idx": idx, "att": att, "dfn": dfn, "home": home, "rho": rho, "c": float(c0)}


@st.cache_data(show_spinner="Training model...")
def get_model(df, xi, wx):
    return fit_dc(df, xi, wx)


def lambdas(model, h, a, neutral=False):
    i, j = model["idx"][h], model["idx"][a]
    hadv = 0.0 if neutral else model["home"]
    c0 = model.get("c", 0.0)
    return (np.exp(c0 + model["att"][i] + model["dfn"][j] + hadv),
            np.exp(c0 + model["att"][j] + model["dfn"][i]))


def score_matrix(lam, mu, rho):
    g = np.arange(MAX_G + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def outcome_probs(m):
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


# ======================= BACKTEST =======================
def market_probs(row):
    for cols in ODDS_SETS:
        if all(c in row.index for c in cols):
            o = np.array([row[c] for c in cols], dtype=float)
            if np.all(np.isfinite(o)) and np.all(o > 1):
                inv = 1 / o
                return inv / inv.sum()
    return None


def metrics(P, y):
    P, y = np.array(P), np.array(y)
    O = np.eye(3)[y]
    return {"Accuracy %": (P.argmax(1) == y).mean() * 100,
            "Brier (lower=better)": ((P - O) ** 2).sum(1).mean(),
            "Log loss (lower=better)": -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean(),
            "Matches": len(y)}


# ======================= ELO (second, independent model) =======================
def elo_to_1x2(E, d0):
    """Win expectancy -> (home, draw, away). The draw share is largest between equal teams."""
    d = float(np.clip(d0 * 4 * E * (1 - E), 0.03, 0.45))
    ph, pa = max(E - d / 2, 0.01), max(1 - E - d / 2, 0.01)
    t = ph + d + pa
    return ph / t, d / t, pa / t


def _elo_pass(df, K, hfa, record_from=None):
    r, out = {}, []
    neu = df["Neutral"].values if "Neutral" in df else np.zeros(len(df))
    for i, (h, a, hg, ag) in enumerate(zip(df.HomeTeam.values, df.AwayTeam.values, df.FTHG.values, df.FTAG.values)):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        E = 1 / (1 + 10 ** (-(rh - ra + (0 if neu[i] else hfa)) / 400))
        y = 0 if hg > ag else (1 if hg == ag else 2)
        if record_from is not None and i >= record_from:
            out.append((h, a, E, y))
        gd = abs(hg - ag)
        G = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = K * G * ((1.0 if y == 0 else 0.5 if y == 1 else 0.0) - E)
        r[h], r[a] = rh + delta, ra - delta
    return r, out


def fit_elo(df):
    """Pick the Elo speed (K) and home bonus that best predicted this league's own history."""
    d0 = float((df.FTHG == df.FTAG).mean()) if len(df) else 0.26
    if len(df) < 80:
        return {"K": 20, "hfa": 80, "d0": d0, "ratings": _elo_pass(df, 20, 80)[0]}
    best = None
    for K in (10, 20, 35):
        for hfa in (50, 80, 110):
            _, out = _elo_pass(df, K, hfa, record_from=int(0.3 * len(df)))
            ll = -np.mean([np.log(max(elo_to_1x2(E, d0)[y], 1e-9)) for _, _, E, y in out])
            if best is None or ll < best[0]:
                best = (ll, K, hfa)
    return {"K": best[1], "hfa": best[2], "d0": d0, "ratings": _elo_pass(df, best[1], best[2])[0]}


def elo_probs(elo, h, a, neutral=False):
    if not elo:
        return None
    r = elo["ratings"]
    E = 1 / (1 + 10 ** (-(r.get(h, 1500.0) - r.get(a, 1500.0) + (0 if neutral else elo["hfa"])) / 400))
    return elo_to_1x2(E, elo["d0"])


def elo_walk(df, n_test):
    """Walk-forward Elo forecasts for the last n_test matches (ratings update only after each match)."""
    cut = len(df) - n_test
    cfg = fit_elo(df.iloc[:cut])
    _, out = _elo_pass(df, cfg["K"], cfg["hfa"], record_from=cut)
    return {cut + i: elo_to_1x2(E, cfg["d0"]) for i, (_, _, E, _) in enumerate(out)}


def blend_with_market(p_model, mk, w):
    """w = weight on the model, (1 - w) on the de-vigged market probabilities."""
    p = w * np.asarray(p_model, dtype=float) + (1 - w) * np.asarray(mk, dtype=float)
    return p / p.sum(axis=-1, keepdims=True)


def tilt_matrix(m, target):
    """Rescale a score matrix so its 1X2 probabilities equal `target`, keeping the score shape inside each outcome."""
    n = m.shape[0]
    i, j = np.indices((n, n))
    out = m.copy()
    for mask, t in ((i > j, target[0]), (i == j, target[1]), (i < j, target[2])):
        tot = m[mask].sum()
        if tot > 0:
            out[mask] = m[mask] * t / tot
    return out / out.sum()


def wf_logloss(df, xi, wx, n_test=150, chunks=2):
    """Walk-forward log loss of the 1X2 forecast on the most recent matches (lower = better)."""
    n_test = min(n_test, len(df) // 3)
    ll = []
    for s in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        md = fit_dc(df.iloc[:s[0]], xi, wx)
        for k in s:
            r = df.iloc[k]
            if r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"]:
                continue
            lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            p = outcome_probs(score_matrix(lam, mu, md["rho"]))
            y = 0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2)
            ll.append(-np.log(max(p[y], 1e-9)))
    return (float(np.mean(ll)) if ll else float("nan")), len(ll)


def backtest(df, xi, n_test, wx, chunks=3):
    n_test = min(n_test, len(df) // 3)
    names = {"main": "Our model + xG/shots" if wx > 0 else "Our model (goals only)"}
    wxs = {"main": wx}
    if wx > 0:
        names["base"], wxs["base"] = "Our model (goals only)", 0.0
    P, Y, MK, EL = {k: [] for k in names}, [], [], []
    elo_tbl = elo_walk(df, n_test)
    for s in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        models = {k: fit_dc(df.iloc[:s[0]], xi, wxs[k]) for k in names}
        for k in s:
            r = df.iloc[k]
            if any(r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"] for md in models.values()):
                continue
            neu = bool(r.get("Neutral", 0))
            for key, md in models.items():
                lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, neu)
                P[key].append(outcome_probs(score_matrix(lam, mu, md["rho"])))
            Y.append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            MK.append(market_probs(r))
            EL.append(elo_tbl[k])
    if not Y:
        return None
    trn = df.iloc[:len(df) - n_test]
    freq = [(trn.FTHG > trn.FTAG).mean(), (trn.FTHG == trn.FTAG).mean(), (trn.FTHG < trn.FTAG).mean()]
    out = {names[k]: metrics(P[k], Y) for k in names}
    out["Naive (league averages)"] = metrics([freq] * len(Y), Y)
    El, Pm_all = np.array(EL), np.array(P["main"])
    out["Elo (own)"] = metrics(El, Y)
    out["Model + Elo 20%"] = metrics(blend_with_market(Pm_all, El, 0.8), Y)
    bw = min(np.arange(0, 1.01, 0.1), key=lambda w: metrics(blend_with_market(Pm_all, El, w), Y)["Brier (lower=better)"])
    out[f"Model + Elo best (model {bw * 100:.0f}%)"] = metrics(blend_with_market(Pm_all, El, bw), Y)
    sub = [i for i, m in enumerate(MK) if m is not None]
    if sub:
        out["Our model (matches with odds)"] = metrics([P["main"][i] for i in sub], [Y[i] for i in sub])
        out["Bookmaker odds"] = metrics([MK[i] for i in sub], [Y[i] for i in sub])
        Pm, Pk, Ys = np.array([P["main"][i] for i in sub]), np.array([MK[i] for i in sub]), [Y[i] for i in sub]
        out["Blend 50/50 (model + market)"] = metrics(blend_with_market(Pm, Pk, 0.5), Ys)
        best = min(np.arange(0, 1.01, 0.1), key=lambda w: metrics(blend_with_market(Pm, Pk, w), Ys)["Brier (lower=better)"])
        out[f"Blend best weight (model {best * 100:.0f}%)"] = metrics(blend_with_market(Pm, Pk, best), Ys)
    return out


# ======================= CALIBRATION: make the confidence honest =======================
CAL_DEF = {"w_elo": 0.20, "temp": 1.0, "pa": 0.0, "pb": 1.0, "ba": 0.0, "bb": 1.0, "w_slow": 0.0}


def _temp(p, T):
    """Temperature: T > 1 makes forecasts less extreme, T < 1 more extreme."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None) ** (1.0 / T)
    return p / p.sum(axis=-1, keepdims=True)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _platt(p, a, b):
    return 1 / (1 + np.exp(-(a + b * _logit(np.asarray(p, dtype=float)))))


def _ll3(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None))))


def _ll2(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def wf_collect(df, xi, wx, n_test=300, chunks=3):
    """Walk-forward forecasts (model refitted on the past only) for the most recent matches of a league."""
    n_test = min(n_test, len(df) // 3)
    elo_tbl = elo_walk(df, n_test)
    o = {k: [] for k in ("dc", "dc2", "elo", "y", "over", "ov", "btts", "bt")}
    for s_ in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        md = fit_dc(df.iloc[:s_[0]], xi, wx)
        md2 = fit_dc(df.iloc[:s_[0]], xi / 2.5, wx)
        for k in s_:
            r = df.iloc[k]
            if r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"]:
                continue
            lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            m = score_matrix(lam, mu, md["rho"])
            ii, jj = np.indices(m.shape)
            o["dc"].append(outcome_probs(m))
            l2, u2 = lambdas(md2, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            o["dc2"].append(outcome_probs(score_matrix(l2, u2, md2["rho"])))
            o["elo"].append(elo_tbl[k])
            o["y"].append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            o["over"].append(float(m[(ii + jj) > 2].sum()))
            o["ov"].append(int(r.FTHG + r.FTAG > 2))
            o["btts"].append(float(m[(ii > 0) & (jj > 0)].sum()))
            o["bt"].append(int(r.FTHG > 0 and r.FTAG > 0))
    return {k: np.array(v) for k, v in o.items()}


def fit_calibration(W):
    """Pick the Elo weight and a confidence correction that would have predicted this league's latest matches best.
    A setting only replaces the default when the gain is statistically clear (total log-likelihood gain of 4+,
    roughly a 1-in-40 chance of being noise), so the app does not chase luck.
    Returns (settings, reliability info or None)."""
    y = W["y"]
    n = len(y)
    res = {**CAL_DEF, "n": int(n)}
    if n < 120:
        return res, None

    dc2 = W.get("dc2")
    if dc2 is None or len(dc2) != n:
        dc2 = W["dc"]

    def final(we, T, ws=0.0):
        return _temp((1 - we - ws) * W["dc"] + ws * dc2 + we * W["elo"], T)
    base = _ll3(final(0.2, 1.0), y)
    best = (base, 0.2, 1.0, 0.0)
    for ws in (0.0, 0.25, 0.5):
        for we in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            if we + ws > 0.9:
                continue
            for T in (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
                ll = _ll3(final(we, T, ws), y)
                if ll < best[0] - 1e-12:
                    best = (ll, we, T, ws)
    if (base - best[0]) * n >= 5.0:
        res["w_elo"], res["temp"], res["w_slow"] = best[1], best[2], best[3]
        use_ll = best[0]
    else:
        use_ll = base
    res["loss"], res["base_loss"] = use_ll, base
    for key, pk, yk in (("p", "over", "ov"), ("b", "btts", "bt")):
        p, t = W[pk], W[yk]
        b0 = _ll2(p, t)
        bb = (b0, 0.0, 1.0)
        for a in np.arange(-0.4, 0.41, 0.1):
            for bsl in (0.8, 0.9, 1.0, 1.1, 1.2):
                l2 = _ll2(_platt(p, a, bsl), t)
                if l2 < bb[0] - 1e-12:
                    bb = (l2, float(a), float(bsl))
        if (b0 - bb[0]) * len(t) >= 4.0:
            res[key + "a"], res[key + "b"] = bb[1], bb[2]
    P = final(res["w_elo"], res["temp"], res["w_slow"])
    top, hit = P.max(1), (P.argmax(1) == y)
    rel = []
    edges = [0, 0.40, 0.50, 0.60, 0.70, 1.01]
    for lo, hi, lb in zip(edges[:-1], edges[1:], ["under 40%", "40-50%", "50-60%", "60-70%", "70%+"]):
        sel = (top >= lo) & (top < hi)
        if sel.sum():
            rel.append({"Model confidence": lb, "Matches": int(sel.sum()),
                        "Model said %": round(float(top[sel].mean()) * 100, 1),
                        "Really happened %": round(float(hit[sel].mean()) * 100, 1)})
    k = max(1, n // 3)
    order = np.argsort(-top)
    info = {"rel": rel, "all_acc": round(float(hit.mean()) * 100, 1), "n": int(n),
            "top_acc": round(float(hit[order[:k]].mean()) * 100, 1), "top_n": int(k),
            "top_min": round(float(top[order[k - 1]]) * 100, 1)}
    return res, info


def eff_params(src, half_life, wx_user):
    try:
        p = db_read("SELECT half_life, wx FROM league_params WHERE league=?", (src,))
        if len(p):
            return int(p.half_life[0]), float(p.wx[0])
    except Exception:
        pass
    return half_life, wx_user


def run_calibration(lg, b, half_life, wx_user):
    has_xg = not b["xg"].startswith("not available")
    W = wf_collect(b["df"], float(np.log(2) / half_life), wx_user if has_xg else 0.0)
    res, info = fit_calibration(W)
    db_exec("INSERT OR REPLACE INTO league_calib(league,w_elo,temp,pa,pb,ba,bb,loss,base_loss,n,rel,tuned_at,w_slow) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (lg, res["w_elo"], res["temp"], res["pa"], res["pb"], res["ba"], res["bb"], res.get("loss"),
             res.get("base_loss"), res["n"], json.dumps(info) if info else None, dt.datetime.now().isoformat(),
             res["w_slow"]))
    return res, info


def calib_load(league):
    try:
        d = db_read("SELECT * FROM league_calib WHERE league=?", (league,))
    except Exception:
        return None
    if not len(d):
        return None
    r = d.iloc[0]
    out = {k: (float(r[k]) if pd.notna(r[k]) else CAL_DEF[k]) for k in CAL_DEF}
    out["n"] = int(r["n"]) if pd.notna(r["n"]) else 0
    out["loss"], out["base_loss"] = r["loss"], r["base_loss"]
    try:
        out["info"] = json.loads(r["rel"]) if r["rel"] else None
    except Exception:
        out["info"] = None
    return out


def _cal_sel(k, p, c):
    """Apply the saved correction to goal-market probabilities."""
    if k == "Over 2.5":
        return float(_platt(p, c["pa"], c["pb"]))
    if k == "Under 2.5":
        return 1 - float(_platt(1 - p, c["pa"], c["pb"]))
    if k == "BTTS Yes":
        return float(_platt(p, c["ba"], c["bb"]))
    if k == "BTTS No":
        return 1 - float(_platt(1 - p, c["ba"], c["bb"]))
    return p


def render_reliability(lg, calib):
    info = calib.get("info") if calib else None
    if not info:
        st.caption("This league is not calibrated yet.")
        return
    c = st.columns(3)
    c[0].metric("Result correct, all matches", f"{info['all_acc']}%", f"last {info['n']} matches", delta_color="off")
    c[1].metric("Result correct, most confident third", f"{info['top_acc']}%",
                f"model confidence {info['top_min']}%+", delta_color="off")
    c[2].metric("Settings", f"Elo {calib['w_elo'] * 100:.0f}%", f"temperature {calib['temp']:.2f}", delta_color="off")
    st.dataframe(pd.DataFrame(info["rel"]), hide_index=True, use_container_width=True)
    st.caption("Measured on this league's most recent matches, each forecast made without knowing the result. "
               "If 'Model said' and 'Really happened' are close, the percentages can be trusted. Rows with few matches are noisy.")


def agreement(f, pick):
    """Do all result sources (Dixon-Coles, Elo, ClubElo, market) point the same way as the pick?"""
    srcs = list(f["sources"].values())
    sel = pick["sel"]
    if len(srcs) < 2 or sel not in ("1", "X", "2", "1X", "X2", "12"):
        return "n/a"
    if sel in ("1", "X", "2"):
        i = {"1": 0, "X": 1, "2": 2}[sel]
        return "all agree" if all(int(np.argmax(v)) == i for v in srcs) else "split"
    cols = {"1X": (0, 1), "X2": (1, 2), "12": (0, 2)}[sel]
    return "all agree" if all(sum(v[c] for c in cols) >= pick["p"] - 0.08 for v in srcs) else "split"


def verdict(rk, agree, f):
    if rk["risk"] <= 25 and agree != "split" and f["grade"] != "C" and f.get("calibrated"):
        return "Strong"      # 'Strong' is only given when this league's model was calibrated on its own recent matches
    if rk["risk"] <= 40 and agree != "split":
        return "OK"          # shown as 'medium risk' in Arabic
    return "Skip"


# ======================= v13: confidence score, corners / line-ups / style screens =======================
def confidence(f, pick, rk, agree):
    """One 0-99 confidence figure and an A/B/C grade for a pick: the pick's probability minus the risk penalties,
    minus the amount by which this league's calibration showed the model to be over-confident."""
    c = rk["safe_p"] * 100
    if not f.get("calibrated"):
        c -= 3
    cal = f.get("cal") or {}
    info = cal.get("info") or {}
    top = max(f["sels"][0]["p"], f["sels"][1]["p"], f["sels"][2]["p"]) * 100
    for row in info.get("rel", []) if isinstance(info, dict) else []:
        lab = row.get("Model confidence", "")
        lo, hi = {"under 40%": (0, 40), "40-50%": (40, 50), "50-60%": (50, 60), "60-70%": (60, 70), "70%+": (70, 101)}.get(lab, (None, None))
        if lo is not None and lo <= top < hi and row.get("Matches", 0) >= 15:
            gap = row["Model said %"] - row["Really happened %"]
            if gap > 3:
                c -= 0.5 * gap
    if agree == "split":
        c -= 5
    c = float(max(0.0, min(99.0, c)))
    grade = "A" if c >= 80 and agree != "split" and f["grade"] != "C" else "B" if c >= 68 else "C"
    return c, grade


def ui_corners(years, half_life, wx_user, fuse_on):
    st.markdown(LX("#### 🚩 Corners market", "#### 🚩 سوق الركنيات"))
    st.caption(LX("Expected corners per team from each team's corners won and conceded (recent matches count more), "
                  "turned into probabilities for total corners, team corners and who wins more corners. Corner "
                  "counts are noisier than goals, so every pick carries an extra risk margin. Needs corner data: "
                  "available for the football-data.co.uk leagues (England, Spain, Italy, Germany, France, "
                  "Netherlands, Portugal, Turkey, Belgium, Scotland, Greece and the lower divisions) and for any CSV with HC/AC columns.",
                  "الركنيات المتوقعة لكل فريق من عدد الركنيات التي يحصل عليها وتُسجَّل ضده (المباريات الأحدث أثقل)، "
                  "وتتحول إلى احتمالات لإجمالي الركنيات وركنيات كل فريق ومن يحصل على ركنيات أكثر. أعداد الركنيات "
                  "أكثر تذبذبًا من الأهداف، لذلك كل اختيار يحمل هامش خطر إضافيًا. تحتاج بيانات ركنيات: متوفرة "
                  "لدوريات football-data.co.uk (إنجلترا وإسبانيا وإيطاليا وألمانيا وفرنسا وهولندا والبرتغال وتركيا "
                  "وبلجيكا واسكتلندا واليونان والدرجات الأدنى) ولأي ملف CSV فيه أعمدة HC/AC."))
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    fxdf = fxdf[(fxdf.kickoff.isna()) | (fxdf.kickoff >= (dt.date.today() - dt.timedelta(days=1)).isoformat())]
    if fxdf.empty:
        st.info(LX("Add upcoming matches in the Matches tab first.", "أضف المباريات القادمة في تبويب المباريات أولًا."))
        return
    c = st.columns(2)
    min_p = c[0].slider(LX("Minimum probability %", "أقل احتمال %"), 50, 90, 60, 1, key="co_p") / 100
    min_fair = c[1].slider(LX("Minimum fair odds (skips trivial picks)", "أقل أودز عادل (لتجنب الاختيارات التافهة)"),
                           1.05, 2.5, 1.25, 0.05, key="co_o")
    rows, detail, nodata, bundles = [], [], [], {}
    for r in fxdf.itertuples():
        if r.league not in bundles:
            bundles[r.league] = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        b = bundles[r.league]
        cp = corner_probs(b.get("corners"), r.home, r.away) if b else None
        if cp is None:
            nodata.append(f"{r.home} vs {r.away}")
            continue
        grade = data_quality(b, r.home, r.away)[2]
        bp = best_corner_pick(cp, min_p, min_fair)
        rk = corner_risk(cp, bp, grade) if bp else None
        rows.append({"League": r.league, "Date": r.kickoff, "Match": f"{r.home} vs {r.away}",
                     "Corners exp.": f"{cp['lh']:.1f}-{cp['la']:.1f}", "Total corners exp.": round(cp["total"], 1),
                     "Over 8.5 %": round(cp["over_8.5"] * 100), "Over 9.5 %": round(cp["over_9.5"] * 100),
                     "Over 10.5 %": round(cp["over_10.5"] * 100),
                     "Best corner pick": bp["sel"] if bp else "-", "Pick %": round(bp["p"] * 100) if bp else None,
                     "Risk %": round(rk["risk"]) if rk else None, "Risk": rk["label"] if rk else "-", "Data": grade})
        detail.append((r, cp))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(LX("Risk % = chance the pick fails + 6 (noisier market) + 4 if a team has fewer than 25 matches with "
                      "corner data + 4 for thin recent data. Low <= 25, Medium <= 40, High above. Corner picks are "
                      "not mixed into the goals slip because their results are not tied to the score.",
                      "الخطر % = احتمال فشل الاختيار + 6 (سوق أكثر تذبذبًا) + 4 إذا كان لأحد الفريقين أقل من 25 مباراة "
                      "ببيانات ركنيات + 4 لقلة البيانات الحديثة. منخفض ≤ 25، متوسط ≤ 40، وما فوق ذلك مرتفع. لا تُدمج "
                      "اختيارات الركنيات في قسيمة الأهداف لأن نتائجها غير مرتبطة بالنتيجة النهائية."))
    if nodata:
        st.warning(LX("No corner data for: ", "لا توجد بيانات ركنيات لـ: ") + " | ".join(nodata))
    for r, cp in detail:
        with st.expander(f"{r.home} vs {r.away}"):
            lines = []
            for L in CORNER_LINES:
                lines.append({"Market": f"Corners Over {L}", "Probability %": round(cp[f"over_{L}"] * 100, 1),
                              "Fair odds": round(1 / max(cp[f"over_{L}"], 1e-6), 2),
                              "Opposite": f"Corners Under {L}", "Opposite %": round((1 - cp[f"over_{L}"]) * 100, 1)})
            st.dataframe(pd.DataFrame(lines), hide_index=True, use_container_width=True)
            tl = []
            for L in CORNER_TEAM_LINES:
                tl.append({"Line": f"Over {L}", "Home team %": round(cp[f"h_over_{L}"] * 100, 1),
                           "Away team %": round(cp[f"a_over_{L}"] * 100, 1)})
            st.dataframe(pd.DataFrame(tl), hide_index=True, use_container_width=True)
            c3 = st.columns(3)
            c3[0].metric(LX("Home team wins more corners", "المضيف يحصل على ركنيات أكثر"), f"{cp['home_more'] * 100:.0f}%")
            c3[1].metric(LX("Same number of corners", "نفس عدد الركنيات"), f"{cp['equal'] * 100:.0f}%")
            c3[2].metric(LX("Away team wins more corners", "الضيف يحصل على ركنيات أكثر"), f"{cp['away_more'] * 100:.0f}%")
    with st.expander(LX("Does the corner model beat a plain average? (test)", "هل نموذج الركنيات أفضل من المتوسط العادي؟ (اختبار)")):
        lgs = sorted({r.league for r in fxdf.itertuples()})
        pick = st.selectbox(LX("League", "الدوري"), lgs, key="co_bt_lg")
        if st.button(LX("Run corner test", "شغّل اختبار الركنيات"), key="co_bt_btn"):
            b = bundles.get(pick)
            res = corners_backtest(b["df"], float(np.log(2) / eff_params(pick, half_life, wx_user)[0])) if b else None
            if res is None:
                st.warning(LX("Not enough corner data in this league to test.", "لا توجد بيانات ركنيات كافية في هذا الدوري للاختبار."))
            else:
                better = res["ll_model"] < res["ll_base"] - 0.002
                st.dataframe(pd.DataFrame([
                    {"Model": "Team corner model", "Log loss (Over 9.5)": round(res["ll_model"], 4),
                     "Avg error (corners)": round(res["mae_model"], 2)},
                    {"Model": "League average only", "Log loss (Over 9.5)": round(res["ll_base"], 4),
                     "Avg error (corners)": round(res["mae_base"], 2)}]), hide_index=True, use_container_width=True)
                (st.success if better else st.info)(
                    LX(f"Tested on {res['n']} matches. " + ("The team model predicts corners better than a plain average."
                       if better else "No clear gain over a plain average in this league: treat corner picks with extra care."),
                       f"اختُبر على {res['n']} مباراة. " + ("نموذج الفرق يتوقع الركنيات أفضل من المتوسط العادي."
                       if better else "لا مكسب واضح على المتوسط العادي في هذا الدوري: تعامل مع اختيارات الركنيات بحذر إضافي.")))


STYLE_ROWS = ["Goals for", "Goals against", "Shots for", "Shots against", "On target for", "On target against",
              "Goals per shot", "Corners for", "Corners against", "Clean sheets %", "Failed to score %",
              "Both scored %", "Over 2.5 %"]


def render_style(df, home, away):
    sh, sa = team_style(df, home), team_style(df, away)
    if not sh or not sa:
        return
    rows = []
    for k in STYLE_ROWS:
        a, b = sh.get(k), sa.get(k)
        if a is None and b is None:
            continue
        fm = (lambda v: "-" if v is None else f"{v:.2f}") if k == "Goals per shot" else (lambda v: "-" if v is None else f"{v:.1f}")
        rows.append({"Measure": k, home: fm(a), away: fm(b)})
    st.markdown(LX("**Attack, defence and style (last 10 matches)**", "**الهجوم والدفاع والأسلوب (آخر 10 مباريات)**"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(LX("Shots, corners and 'goals per shot' come from the data file when it has them. Counter-attack and "
                  "pressing data are not available in any free feed, so they are not shown.",
                  "التسديدات والركنيات و'الأهداف لكل تسديدة' تأتي من ملف البيانات إن كانت متوفرة. بيانات المرتدات "
                  "والضغط غير متاحة في أي مصدر مجاني، لذلك لا تُعرض."))


def ui_lineups(years, half_life, wx_user, fuse_on):
    st.markdown(LX("#### 🔎 Line-ups, injuries and match info", "#### 🔎 التشكيلات والإصابات ومعلومات المباراة"))
    st.caption(LX("Pulled from ESPN's free match page. Line-ups usually appear about an hour before kick-off; "
                  "injuries and statistics appear only when ESPN publishes them for that competition.",
                  "تُسحب من صفحة المباراة المجانية في ESPN. التشكيلات تظهر عادةً قبل البداية بنحو ساعة؛ "
                  "الإصابات والإحصائيات تظهر فقط إذا نشرتها ESPN لتلك البطولة."))
    fx = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    fx = fx[(fx.kickoff.isna()) | (fx.kickoff >= (dt.date.today() - dt.timedelta(days=1)).isoformat())]
    if fx.empty:
        st.info(LX("Add upcoming matches in the Matches tab first.", "أضف المباريات القادمة في تبويب المباريات أولًا."))
        return
    labels = {int(r.id): f"{r.league}: {r.home} vs {r.away} ({r.kickoff})" for r in fx.itertuples()}
    sel = st.selectbox(LX("Match", "المباراة"), list(labels), format_func=lambda i: labels[i], key="li_fx")
    r = next(x for x in fx.itertuples() if int(x.id) == int(sel))
    slug = espn_slug(r.league)
    if not slug:
        st.info(LX("This competition has no ESPN page in the app.", "هذه البطولة ليس لها صفحة ESPN في البرنامج."))
        return
    if not st.button(LX("Fetch line-ups and info", "اجلب التشكيلات والمعلومات"), key="li_go"):
        return
    try:
        day = pd.Timestamp(r.kickoff) if r.kickoff else pd.Timestamp.today()
    except Exception:
        day = pd.Timestamp.today()
    ev = espn_find_event(slug, r.home, r.away, day)
    if not ev:
        st.warning(LX("The match was not found on ESPN for that date (check the date and the team names).",
                      "لم يتم العثور على المباراة في ESPN في هذا التاريخ (راجع التاريخ وأسماء الفرق)."))
        return
    info = espn_event_info(slug, ev)
    if not info:
        st.warning(LX("ESPN did not return match details.", "لم ترجع ESPN تفاصيل المباراة."))
        return
    if info["lineups"]:
        cols = st.columns(len(info["lineups"]))
        for col, lu in zip(cols, info["lineups"]):
            col.markdown(f"**{lu['team']}** {('· ' + lu['formation']) if lu['formation'] else ''}")
            col.write(", ".join(lu["starters"]) if lu["starters"] else LX("Line-up not published yet.", "لم يُنشر التشكيل بعد."))
            if lu["bench"]:
                col.caption(LX("Bench: ", "البدلاء: ") + ", ".join(lu["bench"]))
    else:
        st.info(LX("Line-ups are not published yet.", "لم تُنشر التشكيلات بعد."))
    if info["injuries"]:
        st.markdown(LX("**Injuries / absences listed by ESPN**", "**الإصابات والغيابات المدرجة في ESPN**"))
        st.dataframe(pd.DataFrame(info["injuries"]), hide_index=True, use_container_width=True)
    else:
        st.caption(LX("ESPN lists no injuries for this match. Use the AI assistant tab to search news for absences.",
                      "لا تُدرج ESPN إصابات لهذه المباراة. استخدم تبويب مساعد الذكاء الاصطناعي للبحث في الأخبار عن الغيابات."))
    if info["stats"]:
        names = list(info["stats"])
        keys = sorted({k for v in info["stats"].values() for k in v})
        st.dataframe(pd.DataFrame({"Statistic": keys, **{n: [info["stats"][n].get(k, "-") for k in keys] for n in names}}),
                     hide_index=True, use_container_width=True)


def render_corners_single(df, xi, home, away):
    cm = fit_corners(df, xi)
    if cm is None:
        st.info(LX("This competition's data has no corner columns (HC/AC) or fewer than 150 matches with them.",
                   "بيانات هذه البطولة لا تحتوي أعمدة الركنيات (HC/AC) أو فيها أقل من 150 مباراة بها."))
        return
    cp = corner_probs(cm, home, away)
    if cp is None:
        st.info(LX("One of the teams has no corner data.", "أحد الفريقين ليس له بيانات ركنيات."))
        return
    c = st.columns(3)
    c[0].metric(LX(f"{home} expected corners", f"ركنيات {home} المتوقعة"), f"{cp['lh']:.1f}")
    c[1].metric(LX(f"{away} expected corners", f"ركنيات {away} المتوقعة"), f"{cp['la']:.1f}")
    c[2].metric(LX("Total expected corners", "إجمالي الركنيات المتوقع"), f"{cp['total']:.1f}")
    rows = [{"Market": s["sel"], "Probability %": round(s["p"] * 100, 1), "Fair odds": round(1 / max(s["p"], 1e-6), 2)}
            for s in corner_options(cp)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(LX("Corner counts are noisy: even a 70% corner pick fails often. Use the corner test in the Corners tab "
                  "of the match board to see whether the model beats a plain average in this league.",
                  "أعداد الركنيات متذبذبة: حتى اختيار الركنيات بنسبة 70% يفشل كثيرًا. استخدم اختبار الركنيات في "
                  "تبويب الركنيات بلوحة المباريات لترى هل النموذج أفضل من المتوسط العادي في هذا الدوري."))


def render_corners_board(f):
    cp = f.get("corners")
    if not cp:
        st.info(LX("No corner data for this league or one of the teams.", "لا توجد بيانات ركنيات لهذا الدوري أو لأحد الفريقين."))
        return
    c = st.columns(3)
    c[0].metric(LX("Home corners", "ركنيات المضيف"), f"{cp['lh']:.1f}")
    c[1].metric(LX("Away corners", "ركنيات الضيف"), f"{cp['la']:.1f}")
    c[2].metric(LX("Total corners", "إجمالي الركنيات"), f"{cp['total']:.1f}")
    rows = [{"Market": s["sel"], "Probability %": round(s["p"] * 100, 1), "Fair odds": round(1 / max(s["p"], 1e-6), 2)}
            for s in corner_options(cp)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ======================= TEAM ANALYSIS =======================
def last_matches(df, team, n):
    d = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    rows = []
    for _, r in d.iterrows():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        xf, xa = (r.XGH, r.XGA) if home else (r.XGA, r.XGH)
        rows.append({"Date": r.Date.strftime("%d/%m/%y"), "Venue": "H" if home else "A",
                     "Opponent": r.AwayTeam if home else r.HomeTeam, "Score": f"{gf}-{ga}",
                     "xG": f"{xf:.1f}-{xa:.1f}", "Res": "W" if gf > ga else ("D" if gf == ga else "L"),
                     "GF": gf, "GA": ga, "XF": xf, "XA": xa})
    return pd.DataFrame(rows)


def team_panel(df, team, has_xg):
    st.markdown(f"**{team}**")
    d10 = last_matches(df, team, 10)
    if d10.empty:
        st.write("No data")
        return
    ppg = ((d10.Res == "W") * 3 + (d10.Res == "D")).mean()
    c = st.columns(3)
    c[0].metric("Goals for / match", f"{d10.GF.mean():.2f}")
    c[1].metric("Goals against / match", f"{d10.GA.mean():.2f}")
    c[2].metric("Points / match", f"{ppg:.2f}")
    if has_xg:
        c = st.columns(2)
        c[0].metric("xG for / match", f"{d10.XF.mean():.2f}")
        c[1].metric("xG against / match", f"{d10.XA.mean():.2f}")
    st.caption("Last 10 matches: " + " ".join(d10.Res))
    cols = ["Date", "Venue", "Opponent", "Score"] + (["xG"] if has_xg else []) + ["Res"]
    st.dataframe(d10.tail(6)[cols], hide_index=True, use_container_width=True)


# ======================= PREDICTION VIEWS =======================
def render_correct_score(m, home, away):
    flat = sorted(((m[i, j], i, j) for i in range(MAX_G + 1) for j in range(MAX_G + 1)), reverse=True)
    top = flat[:10]
    st.markdown("**Top 10 correct scores**")
    st.dataframe(pd.DataFrame({"Score": [f"{home} {i}-{j} {away}" for _, i, j in top],
                               "Probability %": [round(p * 100, 1) for p, _, _ in top],
                               "Fair odds": [round(1 / p, 1) for p, _, _ in top]}),
                 hide_index=True, use_container_width=True)
    parts = []
    for name, cond in [(f"{home} win", lambda i, j: i > j), ("Draw", lambda i, j: i == j),
                       (f"{away} win", lambda i, j: i < j)]:
        p, i, j = next(t for t in flat if cond(t[1], t[2]))
        parts.append(f"**{name}** → {i}-{j} ({p * 100:.1f}%)")
    st.write("Best score inside each result: " + " | ".join(parts))
    st.caption("Even the best single score is usually 10-15%. Use it as a guide, not a certainty.")
    n = 6
    mx = m[:n, :n].max()
    head = "<tr><th></th>" + "".join(f"<th>{j}</th>" for j in range(n)) + "</tr>"
    body = "".join(
        f"<tr><th>{i}</th>" + "".join(
            f"<td style='background:rgba(231,76,60,{0.08 + 0.85 * m[i, j] / mx:.2f});"
            f"text-align:center;padding:4px 7px'>{m[i, j] * 100:.1f}</td>" for j in range(n)) + "</tr>"
        for i in range(n))
    st.markdown(f"**Score grid (%)** - rows: {home} goals, columns: {away} goals")
    st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px'>"
                f"{head}{body}</table></div>", unsafe_allow_html=True)


def render_double_chance(m, home, away):
    ph, pdw, pa = outcome_probs(m)
    opts = {f"1X ({home} or draw)": ph + pdw, f"X2 ({away} or draw)": pa + pdw, "12 (no draw)": ph + pa}
    st.dataframe(pd.DataFrame({"Market": list(opts), "Probability %": [round(v * 100, 1) for v in opts.values()],
                               "Fair odds": [round(1 / v, 2) for v in opts.values()]}),
                 hide_index=True, use_container_width=True)
    best = max(opts, key=opts.get)
    st.success(f"Highest-probability pick: **{best}** - {opts[best] * 100:.1f}% (fair odds {1 / opts[best]:.2f})")
    st.write(f"Draw no bet: **{home}** {ph / (ph + pa) * 100:.1f}% (fair odds {(ph + pa) / ph:.2f}) | "
             f"**{away}** {pa / (ph + pa) * 100:.1f}% (fair odds {(ph + pa) / pa:.2f})")
    st.caption("A safer pick pays less. It is only worth it if the bookmaker's odds are above the fair odds.")


def render_handicap(m, lam, mu, home, away):
    ks = np.arange(-MAX_G, MAX_G + 1)
    dp = np.array([np.trace(m, offset=-int(k)) for k in ks])
    res, rows = {}, []
    for L in [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
        r = ks + L
        w_, p_, l_ = dp[r > 1e-9].sum(), dp[np.abs(r) < 1e-9].sum(), dp[r < -1e-9].sum()
        res[L] = (w_, p_, l_)
        rows.append({"Handicap": f"{home} {L:+g}", "Win %": round(w_ * 100, 1), "Refund %": round(p_ * 100, 1),
                     "Lose %": round(l_ * 100, 1), "Fair odds": round(1 + l_ / w_, 2) if w_ > 0 else None})
    st.markdown(f"**Asian handicap** (line added to {home}; the {away} line is the opposite)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fav = home if lam >= mu else away
    fav_win = lambda k: res[-k][0] if fav == home else res[k][2]
    cands = [k for k in (0.5, 1.5, 2.5) if fav_win(k) >= 0.55]
    if cands:
        k = max(cands)
        st.success(f"Strongest line for the favourite: **{fav} -{k:g}** - {fav_win(k) * 100:.1f}% "
                   f"(fair odds {1 / fav_win(k):.2f})")
    else:
        st.info(f"No half-goal line where {fav} is above 55%: the match looks tight.")
    dog = away if fav == home else home
    st.write(f"Safest cover: **{dog} +1.5** - {(1 - fav_win(1.5)) * 100:.1f}%")
    st.markdown("**European (3-way) handicap**")
    er = []
    for h in (-2, -1, 1, 2):
        r = ks + h
        er.append({"Handicap": f"{home} {h:+d}", "Home %": round(dp[r > 0].sum() * 100, 1),
                   "Draw %": round(dp[r == 0].sum() * 100, 1), "Away %": round(dp[r < 0].sum() * 100, 1)})
    st.dataframe(pd.DataFrame(er), hide_index=True, use_container_width=True)
    st.caption("Refund % = stake returned on whole-number lines. Fair odds assume zero bookmaker margin.")


def render_halves(lam, mu, home, away, s, s_src):
    t, g = lam + mu, np.arange(11)
    p1, p2 = poisson.pmf(g, s * t), poisson.pmf(g, (1 - s) * t)
    p1, p2 = p1 / p1.sum(), p2 / p2.sum()
    joint = np.outer(p1, p2)
    c = st.columns(3)
    c[0].metric("1st half goals", f"{s * t:.2f}")
    c[1].metric("2nd half goals", f"{(1 - s) * t:.2f}")
    c[2].metric("Both halves total", f"{t:.2f}")
    st.caption(f"1st half share of goals: {s * 100:.0f}% ({s_src})")
    rows = [{"Line": ln, "1st half over %": round((1 - p1[:int(ln) + 1].sum()) * 100, 1),
             "2nd half over %": round((1 - p2[:int(ln) + 1].sum()) * 100, 1)} for ln in (0.5, 1.5, 2.5)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(f"Half with more goals: 1st **{np.tril(joint, -1).sum() * 100:.1f}%** | "
             f"equal **{np.trace(joint) * 100:.1f}%** | 2nd **{np.triu(joint, 1).sum() * 100:.1f}%**")
    hh, hd, ha = outcome_probs(np.outer(poisson.pmf(g, s * lam), poisson.pmf(g, s * mu)))
    st.write(f"Half-time result: {home} **{hh * 100:.1f}%** | Draw **{hd * 100:.1f}%** | {away} **{ha * 100:.1f}%**")
    st.write(f"Expected goals (1st / 2nd half) - {home}: {s * lam:.2f} / {(1 - s) * lam:.2f} | "
             f"{away}: {s * mu:.2f} / {(1 - s) * mu:.2f}")


def render_goals(m, lam, mu, home, away):
    btts = m[1:, 1:].sum() * 100
    st.write(f"BTTS Yes: `{btts:.1f}%` | No: `{100 - btts:.1f}%` | Expected total: **{lam + mu:.2f}**")
    tot = np.zeros(2 * MAX_G + 1)
    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            tot[i + j] += m[i, j]
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        u = tot[:int(line) + 1].sum() * 100
        st.write(f"- Over **{line}**: `{100 - u:.1f}%` | Under: `{u:.1f}%`")
    st.markdown("**Team goals**")
    for name, marg in [(home, m.sum(axis=1)), (away, m.sum(axis=0))]:
        parts = [f"O{ln}: `{100 - marg[:int(ln) + 1].sum() * 100:.0f}%`" for ln in [0.5, 1.5, 2.5]]
        st.write(f"**{name}** - " + " | ".join(parts))



SEL = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]
N_SIMS = 20000


# ======================= SLIP ENGINE =======================
def masks(n):
    i, j = np.indices((n, n))
    t = i + j
    return {"1": i > j, "X": i == j, "2": i < j, "1X": i >= j, "X2": i <= j, "12": i != j,
            "Over 1.5": t > 1, "Over 2.5": t > 2, "Under 2.5": t < 3, "Under 3.5": t < 4,
            "BTTS Yes": (i > 0) & (j > 0), "BTTS No": (i == 0) | (j == 0)}


def slip_stats(legs):
    P = float(np.prod([l["p"] for l in legs]))
    O = float(np.prod([l["odds"] or l["fair"] for l in legs]))
    real = all(l["odds"] for l in legs)
    ev = P * O - 1 if real else None
    kelly = max(0.0, (P * O - 1) / (O - 1)) if real and O > 1 else 0.0
    return {"P": P, "O": O, "real_odds": real, "ev": ev, "kelly": kelly, "fair": 1 / P}


def simulate(fx, legs, n_sims=N_SIMS, seed=7):
    """Monte Carlo: sample a scoreline for every match and check which legs win."""
    rng = np.random.default_rng(seed)
    wins = np.zeros((n_sims, len(legs)), dtype=bool)
    for c, l in enumerate(legs):
        m = fx[l["fi"]]["m"]
        n = m.shape[0]
        idx = rng.choice(n * n, size=n_sims, p=(m / m.sum()).ravel())
        i, j = divmod(idx, n)
        wins[:, c] = masks(n)[l["sel"]][i, j]
    lost = (~wins).sum(axis=1)
    return np.bincount(lost, minlength=len(legs) + 1) / n_sims


PALIAS = {"player": ["player", "Player", "player_name", "name"], "team": ["team", "Team", "team_title"],
          "pos": ["position", "pos", "Pos"], "minutes": ["minutes", "time", "Min"], "goals": ["goals"],
          "xg": ["xg", "xG"], "npxg": ["np_xg", "npxG", "npxg"], "assists": ["assists"], "xa": ["xa", "xA"],
          "shots": ["shots"], "kp": ["key_passes", "kp"], "chain": ["xg_chain", "xGChain"],
          "build": ["xg_buildup", "xGBuildup"]}


def map_players(raw):
    out = pd.DataFrame(index=raw.index)
    for std, alts in PALIAS.items():
        col = next((a for a in alts if a in raw.columns), None)
        out[std] = raw[col] if col else np.nan
    for c in out.columns.difference(["player", "team", "pos"]):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["team"] = out["team"].fillna("-")
    out["pos"] = out["pos"].fillna("").astype(str)
    return out


def player_table(d, min_minutes, role, teams_sel):
    tt = (d.npxg.fillna(0) + d.xa.fillna(0)).groupby(d.team).transform("sum")
    d = d.assign(share=(d.npxg.fillna(0) + d.xa.fillna(0)) / tt.replace(0, np.nan) * 100)
    d = d[d.minutes >= min_minutes]
    if teams_sel:
        d = d[d.team.isin(teams_sel)]
    pat = {"Forwards": r"\bF\b", "Midfielders": r"\bM\b", "Defenders": r"\bD\b"}.get(role)
    if pat:
        d = d[d.pos.str.contains(pat, regex=True)]
    p90 = lambda c: d[c] / d.minutes * 90
    t = pd.DataFrame({"Player": d.player, "Team": d.team, "Pos": d.pos, "Min": d.minutes,
                      "Goals": d.goals, "xG": d.xg, "Goals-xG": d.goals - d.xg,
                      "npxG/90": p90("npxg"), "xA/90": p90("xa"), "Shots/90": p90("shots"),
                      "KeyPass/90": p90("kp"), "xGChain/90": p90("chain"), "xGBuildup/90": p90("build"),
                      "Team share %": d.share})
    t["Score"] = (t["xGBuildup/90"] + t["xGChain/90"]) if role == "Defenders" else (t["npxG/90"] + t["xA/90"])
    return t.sort_values("Score", ascending=False).round(2)


# ======================= PLAYERS TAB =======================
def render_players_tab():
    st.write("Upload the players CSV created by `fetch_understat.py` (Understat season stats).")
    up = st.file_uploader("Players CSV", type="csv", key="players_up")
    if up is None:
        st.info("Free sources only give attacking and build-up data (xG, xA, shots, key passes, xGChain, xGBuildup). "
                "Tackles, interceptions and counter-attack data are not available for free since FBref removed its "
                "advanced stats in January 2026.")
        return
    d = map_players(pd.read_csv(up))
    st.session_state["players_df"] = d
    c = st.columns(3)
    role = c[0].radio("Role", ["All", "Forwards", "Midfielders", "Defenders"], key="pl_role")
    mins = c[1].slider("Min minutes played", 0, 3000, 450, 50, key="pl_min")
    tsel = c[2].multiselect("Teams", sorted(d.team.unique()), key="pl_teams")
    t = player_table(d, mins, role, tsel)
    st.dataframe(t, hide_index=True, use_container_width=True)
    st.caption("Score = npxG/90 + xA/90 (defenders: xGChain/90 + xGBuildup/90, in-possession contribution only). "
               "'Team share %' is the player's share of his team's npxG + xA: a high number means the team's attack "
               "depends on him, so his absence matters more.")



# ======================= MATCH BOARD (database + multi-league analysis) =======================
DB_PATH = "football_board.db"
BOARD_LEAGUES = [k for k, (kd, _) in SOURCES.items() if kd in ("fd", "new", "intl", "fdorg", "espn")]
FOCUS = ["Premier League (England)", "La Liga (Spain)", "Serie A (Italy)", "Bundesliga (Germany)", "Ligue 1 (France)",
         "Norway Eliteserien", "Sweden Allsvenskan", "Russian Premier League", "UEFA Champions League",
         "UEFA Europa League", "UEFA Conference League", "Champions League (football-data.org free key)",
         "UEFA Nations League (national teams)", "Africa Cup of Nations (national teams)"]


def focus_on():
    try:
        return bool(st.session_state.get("focus_on", True))
    except Exception:
        return True


def board_leagues():
    return [k for k in BOARD_LEAGUES if k in FOCUS] if focus_on() else list(BOARD_LEAGUES)


def source_options():
    return [k for k in SOURCES if k in FOCUS] if focus_on() else list(SOURCES)


FD_CODE_TO_LEAGUE = {code: name for name, (kd, code) in SOURCES.items() if kd == "fd"}
OFB = "https://raw.githubusercontent.com/openfootball/football.json/master"
OF = {"E0": "en.1", "E1": "en.2", "SP1": "es.1", "I1": "it.1", "D1": "de.1", "F1": "fr.1", "N1": "nl.1",
      "P1": "pt.1", "T1": "tr.1", "B1": "be.1", "SC0": "sco.1", "G1": "gr.1",
      "E2": "en.3", "D2": "de.2", "SP2": "es.2", "I2": "it.2", "F2": "fr.2"}
FX_ODDS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
DEFAULT_MARKETS = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]


# ---------- football-data.org (free key): Champions League results and fixtures ----------
def org_key():
    k = _secret("FOOTBALL_DATA_ORG_KEY")
    return k or st.session_state.get("fdorg_key", "")


def parse_fdorg(matches):
    res, fut = [], []
    for m in matches:
        try:
            row = {"Date": str(m["utcDate"])[:10], "HomeTeam": m["homeTeam"]["name"], "AwayTeam": m["awayTeam"]["name"]}
        except (KeyError, TypeError):
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        if m.get("status") == "FINISHED" and ft.get("home") is not None and ft.get("away") is not None:
            res.append({**row, "FTHG": ft["home"], "FTAG": ft["away"]})
        elif m.get("status") in ("TIMED", "SCHEDULED"):
            fut.append(row)
    return res, fut


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_fdorg(code, years, key):
    """Results (and upcoming fixtures) from api.football-data.org. Needs a free key."""
    if not key:
        return pd.DataFrame(), pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fut = [], []
    for y in range(y0, y0 - min(years, 3) - 1, -1):
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{code}/matches", params={"season": y},
                             headers={"X-Auth-Token": key}, timeout=25)
            if r.status_code != 200:
                continue
            a, b = parse_fdorg(r.json().get("matches", []))
        except Exception:
            continue
        res += a
        fut += b
    rdf = clean(normalize(pd.DataFrame(res)), dayfirst=False) if res else pd.DataFrame()
    fdf = pd.DataFrame(fut)
    if len(fdf):
        fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
        fdf = fdf.dropna(subset=["Date"])
        fdf = fdf[fdf.Date >= today]
    return rdf, fdf


# ---------- free external Elo ratings (ClubElo) ----------
CLUBELO_COUNTRY = {"Premier League (England)": "ENG", "Championship (England)": "ENG", "La Liga (Spain)": "ESP",
                   "Serie A (Italy)": "ITA", "Bundesliga (Germany)": "GER", "Ligue 1 (France)": "FRA",
                   "Eredivisie (Netherlands)": "NED", "Liga Portugal": "POR", "Super Lig (Turkey)": "TUR",
                   "Belgian Pro League": "BEL", "Scottish Premiership": "SCO", "Greek Super League": "GRE",
                   "Norway Eliteserien": "NOR", "Sweden Allsvenskan": "SWE", "Russian Premier League": "RUS",
                   "Denmark Superliga": "DEN", "Austria Bundesliga": "AUT", "Switzerland Super League": "SUI",
                   "Poland Ekstraklasa": "POL", "League One (England)": "ENG", "League Two (England)": "ENG",
                   "Scottish Championship": "SCO", "Bundesliga 2 (Germany)": "GER", "Serie B (Italy)": "ITA",
                   "La Liga 2 (Spain)": "ESP", "Ligue 2 (France)": "FRA", "Finland Veikkausliiga": "FIN",
                   "Ireland Premier Division": "IRL", "Romania Liga 1": "ROU"}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_clubelo(day):
    raw, err = fetch_bytes(f"http://api.clubelo.com/{day}", tries=2, timeout=20)
    if raw is None:
        raise RuntimeError(f"ClubElo: {err}")
    d = pd.read_csv(io.StringIO(raw.decode("utf-8", "replace")))
    return d[["Club", "Country", "Elo"]].dropna()


def clubelo_probs(league, home, away, neutral, d0, hfa=100.0):
    """1X2 from ClubElo ratings (free, cross-league, long history). None if teams are not found."""
    try:
        ce = load_clubelo(dt.date.today().isoformat())
    except Exception:
        return None
    cc = CLUBELO_COUNTRY.get(league)
    pool = ce[ce.Country == cc] if cc else ce
    if pool.empty:
        return None
    names = pool.Club.tolist()
    h, a = match_team(home, names), match_team(away, names)
    if not h or not a or h == a:
        return None
    eh, ea = float(pool[pool.Club == h].Elo.iloc[0]), float(pool[pool.Club == a].Elo.iloc[0])
    return elo_to_1x2(1 / (1 + 10 ** (-(eh - ea + (0 if neutral else hfa)) / 400)), d0)


# ======================= v13: ESPN feed, Understat xG, corners model =======================
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
UNDERSTAT_URL = "https://understat.com/getLeagueData"
UNDERSTAT = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1", "RUS": "RFPL"}
ESPN_BAD_STATUS = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_ABANDONED", "STATUS_SUSPENDED", "STATUS_FORFEIT"}
ESPN_CHUNK = 40
ESPN_EPOCH = pd.Timestamp("2020-01-01")


def fetch_json(url, key, max_age_h=None, headers=None):
    """Download JSON and keep a copy on disk. A copy younger than max_age_h is reused without downloading;
    if the download fails the last saved copy is used."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    path = os.path.join(CACHE_DIR, safe + ".json")

    def _disk():
        try:
            with open(path, "rb") as f:
                return json.loads(f.read().decode("utf-8", "replace"))
        except Exception:
            return None
    if max_age_h is not None and os.path.exists(path):
        try:
            fresh = time.time() - os.path.getmtime(path) < max_age_h * 3600
        except OSError:
            fresh = False
        if fresh:
            js = _disk()
            if js is not None:
                NET_LOG[key] = ("ok", "from the saved copy")
                return js
    raw, err = fetch_bytes(url, headers=headers)
    if raw is not None:
        try:
            js = json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:
            js, err = None, f"not JSON ({str(e)[:50]})"
        if js is not None:
            NET_LOG[key] = ("ok", url)
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(raw)
            except OSError:
                pass
            return js
    js = _disk()
    if js is not None:
        NET_LOG[key] = ("saved copy", f"{err} - using the last saved copy")
        return js
    NET_LOG[key] = ("failed", f"{err}  [{url}]")
    return None


# ---------- ESPN (free, no key): results, fixtures, line-ups of almost every competition ----------
def _espn_rows(js):
    out = []
    for ev in ((js or {}).get("events") or []):
        try:
            comp = ev["competitions"][0]
            cs = comp["competitors"]
            h = next(c for c in cs if c.get("homeAway") == "home")
            a = next(c for c in cs if c.get("homeAway") == "away")
            stt = (ev.get("status") or comp.get("status") or {}).get("type") or {}
            done = bool(stt.get("completed")) and stt.get("name", "") not in ESPN_BAD_STATUS

            def _sc(c):
                try:
                    return int(float(c.get("score")))
                except (TypeError, ValueError):
                    return None
            hs, as_ = _sc(h), _sc(a)
            dte = pd.to_datetime(ev.get("date"), errors="coerce", utc=True)
            if pd.isna(dte):
                continue
            out.append({"EventId": str(ev.get("id", "")), "Kick": dte.tz_convert(None),
                        "HomeTeam": h["team"].get("displayName") or h["team"].get("name"),
                        "AwayTeam": a["team"].get("displayName") or a["team"].get("name"),
                        "FTHG": hs, "FTAG": as_, "Done": bool(done and hs is not None and as_ is not None),
                        "Neutral": int(bool(comp.get("neutralSite", False)))})
        except (KeyError, IndexError, StopIteration, TypeError, AttributeError):
            continue
    return [r for r in out if r["HomeTeam"] and r["AwayTeam"]]


ESPN_STATE = {"dead": 0}


def _espn_fetch(slug, d0, d1, depth=0):
    """One date range. ESPN sometimes answers 400 to a request shape, so try several shapes, then halve the range."""
    today = pd.Timestamp.today().normalize()
    old = d1 < today - pd.Timedelta(days=21)
    age = 24 * 3650 if old else 2
    base = f"{ESPN}/{slug}/scoreboard"
    rng = f"{d0:%Y%m%d}-{d1:%Y%m%d}" if d1 > d0 else f"{d0:%Y%m%d}"
    for suffix, tag in ((f"?dates={rng}&limit=500", "a"), (f"?dates={rng}", "b")):
        js = fetch_json(base + suffix, f"espn-{slug}-{d0:%Y%m%d}-{d1:%Y%m%d}-{tag}", max_age_h=age)
        if isinstance(js, dict):
            return _espn_rows(js)
    span = (d1 - d0).days
    if span >= 6 and depth < 2 and ESPN_STATE["dead"] < 3:
        mid = d0 + pd.Timedelta(days=span // 2)
        a = _espn_fetch(slug, d0, mid, depth + 1)
        b = _espn_fetch(slug, mid + pd.Timedelta(days=1), d1, depth + 1)
        return None if a is None and b is None else (a or []) + (b or [])
    return None


def espn_range(slug, d0, d1):
    r = _espn_fetch(slug, d0, d1)
    if r is None:
        ESPN_STATE["dead"] += 1
        return []
    ESPN_STATE["dead"] = 0
    return r


def load_espn_history(slug, years):
    """Finished matches of one ESPN competition. Raises when nothing could be downloaded."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=int(365.25 * min(years, 3)))
    k0 = (cutoff - ESPN_EPOCH).days // ESPN_CHUNK
    k1 = (today - ESPN_EPOCH).days // ESPN_CHUNK
    rows, seen = [], set()
    ESPN_STATE["dead"] = 0
    for k in range(k0, k1 + 1):
        if ESPN_STATE["dead"] >= 3:
            break
        d0 = ESPN_EPOCH + pd.Timedelta(days=k * ESPN_CHUNK)
        for r in espn_range(slug, d0, d0 + pd.Timedelta(days=ESPN_CHUNK - 1)):
            if r["Done"] and r["EventId"] not in seen:
                seen.add(r["EventId"])
                rows.append(r)
    if not rows:
        raise RuntimeError(f"ESPN: no finished matches downloaded for {slug}")
    df = pd.DataFrame(rows).rename(columns={"Kick": "Date"})[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Neutral"]]
    df["Tournament"] = ""
    df["Weight"] = 1.0
    return clean(df, dayfirst=False)


def espn_upcoming(slug, days=14):
    """Upcoming matches (next N days) of an ESPN competition: Date, HomeTeam, AwayTeam, EventId."""
    today = pd.Timestamp.today().normalize()
    try:
        rows = espn_range(slug, today, today + pd.Timedelta(days=int(days)))
    except Exception:
        rows = []
    fut = [{"Date": r["Kick"].normalize(), "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
            "EventId": r["EventId"]} for r in rows if not r["Done"] and r["Kick"].normalize() >= today]
    return pd.DataFrame(fut, columns=["Date", "HomeTeam", "AwayTeam", "EventId"])


def espn_recent_results(slug, days=45):
    today = pd.Timestamp.today().normalize()
    try:
        rows = espn_range(slug, today - pd.Timedelta(days=int(days)), today)
    except Exception:
        rows = []
    d = [{"Date": r["Kick"].normalize(), "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
          "FTHG": r["FTHG"], "FTAG": r["FTAG"]} for r in rows if r["Done"]]
    return pd.DataFrame(d, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def espn_find_event(slug, home, away, day):
    """Event id of a match (teams matched by name, date +-1 day), or None."""
    day = pd.Timestamp(day).normalize()
    try:
        rows = espn_range(slug, day - pd.Timedelta(days=1), day + pd.Timedelta(days=1))
    except Exception:
        return None
    hn, an = [r["HomeTeam"] for r in rows], [r["AwayTeam"] for r in rows]
    if not rows:
        return None
    h, a = match_team(home, sorted(set(hn))), match_team(away, sorted(set(an)))
    for r in rows:
        if r["HomeTeam"] == h and r["AwayTeam"] == a:
            return r["EventId"]
    return None


def espn_event_info(slug, event_id):
    """Line-ups, formations, injuries and match statistics from ESPN's match summary (whatever is published)."""
    js = fetch_json(f"{ESPN}/{slug}/summary?event={event_id}", f"espn-sum-{slug}-{event_id}", max_age_h=0.5)
    if not isinstance(js, dict):
        return None
    out = {"lineups": [], "injuries": [], "stats": {}}
    for ro in js.get("rosters") or []:
        try:
            team = (ro.get("team") or {}).get("displayName", "?")
            ps = ro.get("roster") or []
            start = [(p.get("athlete") or {}).get("displayName", "?") + " (" + ((p.get("position") or {}).get("abbreviation", "?")) + ")"
                     for p in ps if p.get("starter")]
            bench = [(p.get("athlete") or {}).get("displayName", "?") for p in ps if not p.get("starter")]
            out["lineups"].append({"team": team, "formation": ro.get("formation") or "", "starters": start, "bench": bench})
        except (AttributeError, TypeError):
            continue
    for blk in js.get("injuries") or []:
        try:
            team = (blk.get("team") or {}).get("displayName", "?")
            for it in blk.get("injuries") or []:
                out["injuries"].append({"Team": team, "Player": (it.get("athlete") or {}).get("displayName", "?"),
                                        "Status": it.get("status") or (it.get("type") or {}).get("description", "")})
        except (AttributeError, TypeError):
            continue
    for t in ((js.get("boxscore") or {}).get("teams") or []):
        try:
            nm = (t.get("team") or {}).get("displayName", "?")
            out["stats"][nm] = {s.get("name"): s.get("displayValue") for s in (t.get("statistics") or []) if s.get("name")}
        except (AttributeError, TypeError):
            continue
    return out


# ---------- Understat (free): real expected goals (xG) for the top-5 leagues and Russia ----------
def load_understat(ucode, years):
    """Understat league pages as JSON. Returns (finished matches with xG, upcoming matches with their forecast)."""
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fc = [], []
    for y in range(y0, y0 - min(max(int(years), 1), 4), -1):
        hdr = {**UA, "X-Requested-With": "XMLHttpRequest", "Referer": f"https://understat.com/league/{ucode}/{y}"}
        js = fetch_json(f"{UNDERSTAT_URL}/{ucode}/{y}", f"understat-{ucode}-{y}",
                        max_age_h=24 * 3650 if y < y0 else 6, headers=hdr)
        ms = js.get("dates") if isinstance(js, dict) else None
        for m in ms or []:
            try:
                d = pd.to_datetime(m["datetime"], errors="coerce")
                h, a = m["h"]["title"], m["a"]["title"]
                if pd.isna(d):
                    continue
                if m.get("isResult"):
                    res.append({"Date": d, "HomeTeam": h, "AwayTeam": a, "HxG": float(m["xG"]["h"]),
                                "AxG": float(m["xG"]["a"])})
                elif m.get("forecast"):
                    f = m["forecast"]
                    fc.append({"Date": d, "HomeTeam": h, "AwayTeam": a, "pw": float(f["w"]), "pd": float(f["d"]),
                               "pl": float(f["l"])})
            except (KeyError, TypeError, ValueError):
                continue
    return pd.DataFrame(res), pd.DataFrame(fc)


def merge_understat(df, us):
    """Write Understat's real xG into HxG/AxG of our results table (teams matched by name, date +-1 day)."""
    if us is None or us.empty:
        return df, 0
    names = set(df.HomeTeam) | set(df.AwayTeam)
    cache = {}

    def mp(n):
        if n not in cache:
            cache[n] = match_team(n, names)
        return cache[n]
    look = {}
    for r in us.itertuples(index=False):
        h, a = mp(r.HomeTeam), mp(r.AwayTeam)
        if h and a:
            look[(h, a, r.Date.normalize())] = (r.HxG, r.AxG)
    hx, ax = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i, (h, a, d) in enumerate(zip(df.HomeTeam, df.AwayTeam, df.Date.dt.normalize())):
        for off in (0, -1, 1):
            v = look.get((h, a, d + pd.Timedelta(days=off)))
            if v:
                hx[i], ax[i] = v
                break
    n = int(np.isfinite(hx).sum())
    if n == 0:
        return df, 0
    out = df.copy()
    if "HxG" in out and "AxG" in out:
        out["HxG"] = pd.to_numeric(out["HxG"], errors="coerce").fillna(pd.Series(hx, index=out.index))
        out["AxG"] = pd.to_numeric(out["AxG"], errors="coerce").fillna(pd.Series(ax, index=out.index))
    else:
        out["HxG"], out["AxG"] = hx, ax
    return out, n


def understat_forecasts(fc, names):
    """{(home, away): [P(home), P(draw), P(away)]} for Understat's own forecasts of upcoming matches."""
    out = {}
    if fc is None or fc.empty:
        return out
    cache = {}

    def mp(n):
        if n not in cache:
            cache[n] = match_team(n, names)
        return cache[n]
    for r in fc.sort_values("Date").itertuples(index=False):
        h, a = mp(r.HomeTeam), mp(r.AwayTeam)
        s = r.pw + r.pd + r.pl
        if h and a and s > 0 and (h, a) not in out:
            out[(h, a)] = [r.pw / s, r.pd / s, r.pl / s]
    return out


# ---------- corners: team corner rates (attack = corners won, defence = corners conceded) ----------
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CORNER_TEAM_LINES = (3.5, 4.5, 5.5, 6.5)
CORNER_MIN_MATCHES = 150


def corner_frame(df):
    if df is None or "HC" not in df.columns or "AC" not in df.columns:
        return None
    d = df[["Date", "HomeTeam", "AwayTeam", "HC", "AC"]].copy()
    d["HC"] = pd.to_numeric(d["HC"], errors="coerce")
    d["AC"] = pd.to_numeric(d["AC"], errors="coerce")
    d = d.dropna()
    d = d[(d.HC >= 0) & (d.AC >= 0) & (d.HC <= 30) & (d.AC <= 30)]
    return d.reset_index(drop=True) if len(d) >= CORNER_MIN_MATCHES else None


def fit_corners(df, xi, k=4.0, iters=60):
    """Multiplicative corner model: home corners = base_home x attack[home] x defence[away], and the same for
    away; fitted with recency weights and shrinkage towards 1. alpha = extra spread (negative binomial)."""
    d = corner_frame(df)
    if d is None:
        return None
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = d.HomeTeam.map(idx).values, d.AwayTeam.map(idx).values
    ch, ca = d.HC.values.astype(float), d.AC.values.astype(float)
    w = np.exp(-xi * (d.Date.max() - d.Date).dt.days.values)
    att, dfn = np.ones(n), np.ones(n)
    mh = float((w * ch).sum() / w.sum())
    ma = float((w * ca).sum() / w.sum())
    c0 = (mh + ma) / 2
    for _ in range(iters):
        num = np.bincount(hi, w * ch, n) + np.bincount(ai, w * ca, n)
        den = np.bincount(hi, w * mh * dfn[ai], n) + np.bincount(ai, w * ma * dfn[hi], n)
        att = (num + k * c0) / (den + k * c0)
        num = np.bincount(ai, w * ch, n) + np.bincount(hi, w * ca, n)
        den = np.bincount(ai, w * mh * att[hi], n) + np.bincount(hi, w * ma * att[ai], n)
        dfn = (num + k * c0) / (den + k * c0)
        att, dfn = att / att.mean(), dfn / dfn.mean()
        mh = float((w * ch).sum() / (w * att[hi] * dfn[ai]).sum())
        ma = float((w * ca).sum() / (w * att[ai] * dfn[hi]).sum())
    eh, ea = mh * att[hi] * dfn[ai], ma * att[ai] * dfn[hi]
    num = (w * ((ch - eh) ** 2 - eh)).sum() + (w * ((ca - ea) ** 2 - ea)).sum()
    den = (w * eh ** 2).sum() + (w * ea ** 2).sum()
    alpha = float(np.clip(num / den if den > 0 else 0.0, 0.0, 0.4))
    cnt = pd.concat([d.HomeTeam, d.AwayTeam]).value_counts()
    return {"teams": teams, "idx": idx, "att": att, "dfn": dfn, "mh": mh, "ma": ma, "alpha": alpha,
            "n": int(len(d)), "cnt": {t: int(cnt.get(t, 0)) for t in teams}}


def corner_rates(cm, h, a):
    i, j = cm["idx"][h], cm["idx"][a]
    return cm["mh"] * cm["att"][i] * cm["dfn"][j], cm["ma"] * cm["att"][j] * cm["dfn"][i]


def _nb_pmf(mean, alpha, kmax=45):
    k = np.arange(kmax + 1)
    if alpha < 0.01:
        p = poisson.pmf(k, mean)
    else:
        r = 1.0 / alpha
        p = nbinom.pmf(k, r, r / (r + mean))
    return p / p.sum()


def corner_probs(cm, h, a, rates=None):
    """Totals, team totals and 'who wins more corners' for one match. None if a team is unknown."""
    if not cm or h not in cm["idx"] or a not in cm["idx"]:
        return None
    lh, la = rates if rates else corner_rates(cm, h, a)
    ph, pa = _nb_pmf(lh, cm["alpha"]), _nb_pmf(la, cm["alpha"])
    tot = np.convolve(ph, pa)
    joint = np.outer(ph, pa)
    out = {"lh": float(lh), "la": float(la), "total": float(lh + la), "alpha": cm["alpha"],
           "nh": cm["cnt"].get(h, 0), "na": cm["cnt"].get(a, 0)}
    for L in CORNER_LINES:
        out[f"over_{L}"] = float(tot[int(L + 0.5):].sum())
    for L in CORNER_TEAM_LINES:
        out[f"h_over_{L}"] = float(ph[int(L + 0.5):].sum())
        out[f"a_over_{L}"] = float(pa[int(L + 0.5):].sum())
    out["home_more"] = float(np.tril(joint, -1).sum())
    out["equal"] = float(np.trace(joint))
    out["away_more"] = float(np.triu(joint, 1).sum())
    return out


def corner_options(cp):
    """Every corner market of a match as a list of {sel, p}."""
    o = []
    for L in CORNER_LINES:
        p = cp[f"over_{L}"]
        o += [{"sel": f"Corners Over {L}", "p": p}, {"sel": f"Corners Under {L}", "p": 1 - p}]
    for L in CORNER_TEAM_LINES:
        o += [{"sel": f"Home corners Over {L}", "p": cp[f"h_over_{L}"]},
              {"sel": f"Away corners Over {L}", "p": cp[f"a_over_{L}"]}]
    o += [{"sel": "Home more corners", "p": cp["home_more"]}, {"sel": "Away more corners", "p": cp["away_more"]}]
    return o


def best_corner_pick(cp, min_p, min_fair):
    c = [s for s in corner_options(cp) if s["p"] >= min_p and (1 / s["p"] if s["p"] > 0 else 99) >= min_fair]
    return max(c, key=lambda s: s["p"]) if c else None


def corner_risk(cp, pick, grade):
    """Risk % of a corner pick: the chance it fails + an extra for the noisier corner model + thin data."""
    base = (1 - pick["p"]) * 100
    extra, why = 6.0, [("corner counts are noisier than goals (+6)")]
    if min(cp["nh"], cp["na"]) < 25:
        extra += 4
        why.append("fewer than 25 matches with corner data for a team (+4)")
    if grade == "C":
        extra += 4
        why.append("little recent data (+4)")
    total = min(95.0, base + extra)
    return {"risk": total, "why": why, "label": "Low" if total <= 25 else "Medium" if total <= 40 else "High"}


def corners_backtest(df, xi, n_test=200, chunks=2, line=9.5):
    """Walk-forward: corner model vs 'league average only', on the latest matches. Lower log loss = better."""
    d = df[["Date", "HomeTeam", "AwayTeam", "HC", "AC"]].copy() if "HC" in df.columns and "AC" in df.columns else None
    if d is None:
        return None
    d["HC"] = pd.to_numeric(d.HC, errors="coerce")
    d["AC"] = pd.to_numeric(d.AC, errors="coerce")
    d = d.dropna().reset_index(drop=True)
    n_test = min(n_test, len(d) // 3)
    if len(d) < CORNER_MIN_MATCHES + 20 or n_test < 40:
        return None
    ll_m, ll_b, ae_m, ae_b, cnt = 0.0, 0.0, 0.0, 0.0, 0
    for s_ in np.array_split(np.arange(len(d) - n_test, len(d)), chunks):
        past = d.iloc[:s_[0]]
        cm = fit_corners(past, xi)
        if cm is None:
            continue
        w = np.exp(-xi * (past.Date.max() - past.Date).dt.days.values)
        base_tot = float((w * (past.HC + past.AC)).sum() / w.sum())
        pb = _nb_pmf(base_tot / 2, cm["alpha"])
        tb = np.convolve(pb, pb)
        p_base = float(tb[int(line + 0.5):].sum())
        for k in s_:
            r = d.iloc[k]
            if r.HomeTeam not in cm["idx"] or r.AwayTeam not in cm["idx"]:
                continue
            cp = corner_probs(cm, r.HomeTeam, r.AwayTeam)
            y = 1 if r.HC + r.AC > line else 0
            for p, which in ((cp[f"over_{line}"], "m"), (p_base, "b")):
                p = min(max(p, 1e-4), 1 - 1e-4)
                l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
                if which == "m":
                    ll_m += l
                else:
                    ll_b += l
            ae_m += abs(cp["total"] - (r.HC + r.AC))
            ae_b += abs(base_tot - (r.HC + r.AC))
            cnt += 1
    if cnt < 30:
        return None
    return {"n": cnt, "line": line, "ll_model": ll_m / cnt, "ll_base": ll_b / cnt, "mae_model": ae_m / cnt,
            "mae_base": ae_b / cnt}


# ---------- team style profile (shots, corners, scoring habits) ----------
def team_style(df, team, n=10):
    d = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    if d.empty:
        return None
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    home = (d.HomeTeam == team)
    pick = lambda hc, ac: pd.Series(np.where(home, num(hc), num(ac)), index=d.index)
    gf = pd.Series(np.where(home, d.FTHG, d.FTAG), index=d.index).astype(float)
    ga = pd.Series(np.where(home, d.FTAG, d.FTHG), index=d.index).astype(float)
    sh, sa = pick("HS", "AS"), pick("AS", "HS")
    sf, sg = pick("HST", "AST"), pick("AST", "HST")
    cf, ca = pick("HC", "AC"), pick("AC", "HC")
    mean = lambda s: float(s.mean()) if s.notna().any() else None
    shots_n = int(sh.notna().sum())
    return {"Matches": len(d), "Goals for": mean(gf), "Goals against": mean(ga),
            "Shots for": mean(sh), "Shots against": mean(sa), "On target for": mean(sf), "On target against": mean(sg),
            "Goals per shot": (float(gf[sh.notna()].sum() / sh.sum()) if shots_n and sh.sum() > 0 else None),
            "Corners for": mean(cf), "Corners against": mean(ca),
            "Clean sheets %": float((ga == 0).mean() * 100), "Failed to score %": float((gf == 0).mean() * 100),
            "Both scored %": float(((gf > 0) & (ga > 0)).mean() * 100),
            "Over 2.5 %": float(((gf + ga) > 2).mean() * 100)}


ESPN_FUT = {
    "Premier League (England)": "eng.1", "Championship (England)": "eng.2", "La Liga (Spain)": "esp.1",
    "Serie A (Italy)": "ita.1", "Bundesliga (Germany)": "ger.1", "Ligue 1 (France)": "fra.1",
    "Eredivisie (Netherlands)": "ned.1", "Liga Portugal": "por.1", "Super Lig (Turkey)": "tur.1",
    "Belgian Pro League": "bel.1", "Scottish Premiership": "sco.1", "Greek Super League": "gre.1",
    "Norway Eliteserien": "nor.1", "Sweden Allsvenskan": "swe.1", "Russian Premier League": "rus.1",
    "Brazil Serie A": "bra.1", "USA MLS": "usa.1", "Argentina Primera": "arg.1", "Mexico Liga MX": "mex.1",
    "Japan J1 League": "jpn.1", "Denmark Superliga": "den.1", "Austria Bundesliga": "aut.1",
    "Switzerland Super League": "sui.1", "Bundesliga 2 (Germany)": "ger.2", "Serie B (Italy)": "ita.2",
    "La Liga 2 (Spain)": "esp.2", "Ligue 2 (France)": "fra.2",
    "Africa Cup of Nations (national teams)": "caf.nations",
    "UEFA Nations League (national teams)": "uefa.nations",
}


def espn_slug(league):
    kd, code = SOURCES.get(league, ("", ""))
    return code if kd == "espn" else ESPN_FUT.get(league)


# ---------- database ----------
def _con():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS fixtures(id INTEGER PRIMARY KEY AUTOINCREMENT, league TEXT, home TEXT, "
                "away TEXT, kickoff TEXT, neutral INTEGER DEFAULT 0, o1 REAL, ox REAL, o2 REAL, "
                "adj_h REAL DEFAULT 1, adj_a REAL DEFAULT 1, note TEXT DEFAULT '', created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS league_params(league TEXT PRIMARY KEY, half_life INTEGER, wx REAL, "
                "loss REAL, base_loss REAL, tuned_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS league_calib(league TEXT PRIMARY KEY, w_elo REAL, temp REAL, pa REAL, "
                "pb REAL, ba REAL, bb REAL, loss REAL, base_loss REAL, n INTEGER, rel TEXT, tuned_at TEXT)")
    try:
        con.execute("ALTER TABLE league_calib ADD COLUMN w_slow REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    con.execute("CREATE TABLE IF NOT EXISTS context(fixture_id INTEGER PRIMARY KEY, data TEXT, fetched TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS picks(fixture_id INTEGER, sel TEXT, p REAL, created TEXT, "
                "PRIMARY KEY(fixture_id, sel))")
    con.execute("CREATE TABLE IF NOT EXISTS app_flags(k TEXT PRIMARY KEY)")
    if not con.execute("SELECT 1 FROM app_flags WHERE k='dc_intercept_v13'").fetchone():
        con.execute("DELETE FROM league_calib")
        con.execute("INSERT INTO app_flags VALUES('dc_intercept_v13')")
        con.commit()
    for _c in ("verdict TEXT", "grade TEXT", "agree TEXT", "risk REAL"):
        try:
            con.execute(f"ALTER TABLE picks ADD COLUMN {_c}")
        except sqlite3.OperationalError:
            pass
    return con


def db_read(sql, params=()):
    con = _con()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def db_exec(sql, params=()):
    con = _con()
    try:
        with con:
            cur = con.execute(sql, params)
        return cur.lastrowid
    finally:
        con.close()


def db_add_fixture(league, home, away, kickoff, neutral=0, o1=None, ox=None, o2=None):
    dup = db_read("SELECT id FROM fixtures WHERE league=? AND home=? AND away=? AND kickoff=?",
                  (league, home, away, str(kickoff)))
    if len(dup):
        return False
    nz = lambda v: float(v) if v is not None and pd.notna(v) and float(v) > 1 else None
    db_exec("INSERT INTO fixtures(league,home,away,kickoff,neutral,o1,ox,o2,created) VALUES(?,?,?,?,?,?,?,?,?)",
            (league, home, away, str(kickoff), int(neutral), nz(o1), nz(ox), nz(o2), dt.datetime.now().isoformat()))
    return True


# ---------- team-name matching and source fusion ----------
SYN = [(r"\bman\b", "manchester"), (r"\butd\b", "united"), (r"\bnott m\b", "nottingham"), (r"\bwolves\b", "wolverhampton wanderers"),
       (r"\bsheff\b", "sheffield"), (r"\bath\b", "athletic"), (r"\bspurs\b", "tottenham hotspur"),
       (r"\bbrighton hove albion\b", "brighton"), (r"\batletico\b", "atletico"), (r"\binternazionale\b", "inter"),
       (r"\bbayern munchen\b", "bayern munich"), (r"\bparis saint germain\b", "paris sg"), (r"\bpsg\b", "paris sg"),
       (r"\bm gladbach\b", "monchengladbach"), (r"\bmgladbach\b", "monchengladbach")]


def _key(s):
    s = re.sub(r"\b(fc|cf|afc|sc|ac|as|cd|sk|if|fk|bk|ik|the)\b", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    for pat, rep in SYN:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\b(wanderers|hotspur|albion|hove|city|united)\b", lambda m: m.group(1), s)
    return s


def match_team(name, teams):
    """Map a team name from any source onto the names used by the model. Returns None if nothing close enough."""
    teams = list(teams)
    if name in teams:
        return name
    keys = {_key(t): t for t in teams}
    k = _key(name)
    if k in keys:
        return keys[k]
    cand = [t for kk, t in keys.items() if min(len(k), len(kk)) >= 4 and (k in kk or kk in k)]
    if len(cand) == 1:
        return cand[0]
    best = difflib.get_close_matches(k, list(keys), n=1, cutoff=0.74)
    return keys[best[0]] if best else None


def fuse(primary, others):
    """Merge result tables from several sources. Team names of every extra source are mapped onto the primary
    names; a match already in the primary (same teams, date within one day) is kept as is, and only matches the
    primary does not have are added. Returns (merged_df, number_of_added_matches)."""
    names = set(primary.HomeTeam) | set(primary.AwayTeam)
    known = set()
    for r in primary[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False):
        d = r.Date.normalize()
        for off in (-1, 0, 1):
            known.add((d + pd.Timedelta(days=off), r.HomeTeam, r.AwayTeam))
    cache, add = {}, []
    for o in others:
        if o is None or o.empty:
            continue
        for r in o.itertuples(index=False):
            h = cache.setdefault(r.HomeTeam, match_team(r.HomeTeam, names))
            a = cache.setdefault(r.AwayTeam, match_team(r.AwayTeam, names))
            if h is None or a is None:
                continue
            key = (r.Date.normalize(), h, a)
            if key in known:
                continue
            known.add(key)
            add.append({"Date": r.Date, "HomeTeam": h, "AwayTeam": a, "FTHG": r.FTHG, "FTAG": r.FTAG,
                        "Neutral": 0, "Tournament": "", "Weight": 1.0})
    if not add:
        return primary, 0
    out = pd.concat([primary, pd.DataFrame(add)], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out, len(add)


def load_openfootball(code, years):
    try:
        return _load_openfootball_raw(code, years)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_openfootball_raw(code, years):
    """Second free source (openfootball on GitHub): results and, when published, upcoming matches.
    Raises when nothing was downloaded, so a failure is never cached."""
    f = OF.get(code)
    if not f:
        return pd.DataFrame(), pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fut = [], []
    for y in range(y0, max(y0 - years, y0 - 3) - 1, -1):
        raw, err = fetch_bytes(f"{OFB}/{y}-{(y + 1) % 100:02d}/{f}.json", tries=2)
        if raw is None:
            NET_LOG[f"openfootball-{code}-{y}"] = ("failed", err)
            continue
        try:
            matches = json.loads(raw.decode("utf-8", "replace")).get("matches", [])
        except Exception:
            continue
        for m in matches:
            sc = m.get("score")
            ft = sc.get("ft") if isinstance(sc, dict) else sc if isinstance(sc, list) else None
            row = {"Date": m.get("date"), "HomeTeam": m.get("team1"), "AwayTeam": m.get("team2")}
            if isinstance(ft, list) and len(ft) == 2 and all(isinstance(v, (int, float)) for v in ft):
                res.append({**row, "FTHG": ft[0], "FTAG": ft[1]})
            else:
                fut.append(row)
    if not res and not fut:
        raise RuntimeError("openfootball: nothing downloaded")
    rdf = clean(normalize(pd.DataFrame(res)), dayfirst=False) if res else pd.DataFrame()
    fdf = pd.DataFrame(fut)
    if len(fdf):
        fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
        fdf = fdf.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        fdf = fdf[fdf.Date >= today]
    return rdf, fdf


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_fd_fixtures():
    """Upcoming matches (with bookmaker odds when present) for the main leagues from football-data.co.uk."""
    d = fetch_csv(f"{FD}/fixtures.csv", "fixtures")
    if d is None:
        return pd.DataFrame()
    if not {"Div", "Date", "HomeTeam", "AwayTeam"} <= set(d.columns):
        return pd.DataFrame()
    d = d.dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam"]).copy()
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed", errors="coerce")
    d = d.dropna(subset=["Date"])
    d["League"] = d["Div"].map(FD_CODE_TO_LEAGUE)
    d = d.dropna(subset=["League"])
    for c1, c2, c3 in FX_ODDS:
        if {c1, c2, c3} <= set(d.columns):
            d["o1"], d["ox"], d["o2"] = d[c1], d[c2], d[c3]
            break
    else:
        d["o1"] = d["ox"] = d["o2"] = np.nan
    return d[["League", "Date", "HomeTeam", "AwayTeam", "o1", "ox", "o2"]].reset_index(drop=True)


def _season_tag(shift=0):
    t = pd.Timestamp.today()
    y = (t.year if t.month >= 7 else t.year - 1) + shift
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _season_dash(shift=0):
    t = pd.Timestamp.today()
    y = (t.year if t.month >= 7 else t.year - 1) + shift
    return f"{y}-{(y + 1) % 100:02d}"


def league_bundle(src, years, half_life, wx_user, fuse_on):
    """Uses the league's auto-tuned settings (Data tab) when they exist, otherwise the sidebar values."""
    try:
        p = db_read("SELECT half_life, wx FROM league_params WHERE league=?", (src,))
        if len(p):
            half_life, wx_user = int(p.half_life[0]), float(p.wx[0])
    except Exception:
        pass
    kd = SOURCES[src][0]
    try:
        return _league_bundle(src, years, half_life, wx_user, fuse_on, org_key() if kd == "fdorg" else "")
    except RuntimeError:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=LX("Loading league and training its model...", "جارٍ تحميل الدوري وتدريب نموذجه..."))
def _league_bundle(src, years, half_life, wx_user, fuse_on, org=""):
    kind, code = SOURCES[src]
    df = load_data(kind, code, years, org)
    if df.empty:
        raise RuntimeError("no data")      # raised (not returned) so that a failure is not cached
    fut, added = pd.DataFrame(), 0
    slug = espn_slug(src)
    if kind == "fdorg":
        fut = load_fdorg(code, years, org)[1]
    if kind == "fd" and fuse_on:
        extra, fut = load_openfootball(code, years)
        others = [extra] if not extra.empty else []
        if slug:
            er = espn_recent_results(slug)
            if not er.empty:
                others.append(er)
        if others:
            df, added = fuse(df, others)
    if slug:
        up = espn_upcoming(slug, 14)
        if not up.empty:
            fut = up
    us_n, us_fc = 0, {}
    ucode = UNDERSTAT.get(code) if kind in ("fd", "new") else None
    if ucode:
        try:
            xs, fcs = load_understat(ucode, years)
            df, us_n = merge_understat(df, xs)
            us_fc = understat_forecasts(fcs, set(df.HomeTeam) | set(df.AwayTeam))
        except Exception:
            us_n, us_fc = 0, {}
    df, xg_src = add_xg(df)
    if us_n and xg_src == "real xG from the file":
        xg_src = "real xG (Understat)"
    has_xg = not xg_src.startswith("not available")
    xi_ = float(np.log(2) / half_life)
    wxe = wx_user if has_xg else 0.0
    model = fit_dc(df, xi_, wxe)
    model2 = fit_dc(df, xi_ / 2.5, wxe) if kind != "intl" else None
    if kind == "intl":
        pool_df = df[df.Tournament.str.contains(code, case=False, na=False)]
        pool = set(pool_df.HomeTeam) | set(pool_df.AwayTeam)
        teams = [t for t in eligible_teams(df, model, 6, 730) if t in pool] or eligible_teams(df, model, 6, 730)
    else:
        teams = sorted(current_teams(df) & set(model["teams"]))
    s_share, s_src = half_share(df)
    return {"df": df, "model": model, "teams": teams, "xg": xg_src, "fut": fut, "added": added,
            "s_share": s_share, "s_src": s_src, "kind": kind, "elo": fit_elo(df), "model2": model2,
            "corners": fit_corners(df, xi_), "us_n": us_n, "us_fc": us_fc}


# ---------- automatic feed: injuries, suspensions, absences ----------
FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
ATT_REPLACE = 0.35      # a missing attacker's output is only partly replaced: lose this share of his contribution
CTX_MIN, CTX_MAX = 0.82, 1.10


def _file_secrets():
    """secrets.toml found next to this file, in its parent folder or in the working folder (Streamlit itself only
    reads a .streamlit folder, which is easy to get wrong)."""
    out = {}
    try:
        import tomllib
    except ImportError:
        return out
    here = os.path.dirname(os.path.abspath(globals().get("__file__", "app.py")))
    for d in (os.getcwd(), os.path.dirname(here), here):
        for sub in ("secrets.toml", os.path.join(".streamlit", "secrets.toml")):
            p = os.path.join(d, sub)
            try:
                with open(p, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        if isinstance(v, str) and v.strip():
                            out[k] = v.strip()
            except Exception:
                continue
    return out


FILE_SECRETS = _file_secrets()


def _secret(name):
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return v or FILE_SECRETS.get(name, "")


ODDS_SPORTS = {
    "Premier League (England)": "soccer_epl", "Championship (England)": "soccer_efl_champ",
    "La Liga (Spain)": "soccer_spain_la_liga", "Serie A (Italy)": "soccer_italy_serie_a",
    "Bundesliga (Germany)": "soccer_germany_bundesliga", "Ligue 1 (France)": "soccer_france_ligue_one",
    "Eredivisie (Netherlands)": "soccer_netherlands_eredivisie", "Liga Portugal": "soccer_portugal_primeira_liga",
    "Super Lig (Turkey)": "soccer_turkey_super_league", "Belgian Pro League": "soccer_belgium_first_div",
    "Scottish Premiership": "soccer_spl", "Greek Super League": "soccer_greece_super_league",
    "Norway Eliteserien": "soccer_norway_eliteserien", "Sweden Allsvenskan": "soccer_sweden_allsvenskan",
    "Brazil Serie A": "soccer_brazil_campeonato", "USA MLS": "soccer_usa_mls",
    "Switzerland Super League": "soccer_switzerland_superleague", "Argentina Primera": "soccer_argentina_primera_division",
    "Mexico Liga MX": "soccer_mexico_ligamx", "Japan J1 League": "soccer_japan_j_league",
    "Denmark Superliga": "soccer_denmark_superliga", "Austria Bundesliga": "soccer_austria_bundesliga",
    "Poland Ekstraklasa": "soccer_poland_ekstraklasa", "Bundesliga 2 (Germany)": "soccer_germany_bundesliga2",
    "Serie B (Italy)": "soccer_italy_serie_b", "La Liga 2 (Spain)": "soccer_spain_segunda_division",
    "Ligue 2 (France)": "soccer_france_ligue_two", "UEFA Champions League": "soccer_uefa_champs_league",
    "UEFA Europa League": "soccer_uefa_europa_league", "UEFA Conference League": "soccer_uefa_europa_conference_league",
    "Champions League (football-data.org free key)": "soccer_uefa_champs_league",
    "Saudi Pro League": "soccer_saudi_arabia_pro_league",
}


def odds_key():
    return _secret("ODDS_API_KEY") or st.session_state.get("odds_key", "")


def fetch_market_odds(sport, key):
    """Average decimal h2h odds per upcoming match from The Odds API: [(home, away, date, o1, ox, o2)], error text."""
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                         params={"apiKey": key, "regions": "eu,uk", "markets": "h2h", "oddsFormat": "decimal"},
                         timeout=20)
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:80]}"
    if r.status_code != 200:
        return [], f"HTTP {r.status_code}" + (" (check the key)" if r.status_code in (401, 403) else "")
    out = []
    for ev in r.json():
        h, a = ev.get("home_team"), ev.get("away_team")
        s = {"h": [], "d": [], "a": []}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for o in mk.get("outcomes", []):
                    nm = o.get("name")
                    (s["d"] if nm == "Draw" else s["h"] if nm == h else s["a"] if nm == a else []).append(o.get("price"))
        if h and a and s["h"] and s["d"] and s["a"]:
            out.append((h, a, str(ev.get("commence_time", ""))[:10],
                        float(np.mean(s["h"])), float(np.mean(s["d"])), float(np.mean(s["a"]))))
    return out, ""


def ui_odds_feed():
    with st.expander(LX("📡 Market odds feed (optional, free key from the-odds-api.com)",
                        "📡 تغذية أسعار السوق (اختيارية، مفتاح مجاني من the-odds-api.com)")):
        st.caption(LX("Bookmaker prices carry information the model cannot see (injuries, line-ups, news), so they are "
                      "one of the strongest checks on a prediction. This only reads the prices to fill 'market' "
                      "probabilities for the matches you already added; nothing about betting. About 1 request per "
                      "league (the free plan gives 500 a month).",
                      "أسعار شركات المراهنات تحمل معلومات لا يراها النموذج (إصابات وتشكيلات وأخبار)، فهي من أقوى "
                      "وسائل مراجعة التوقع. هذا الخيار يقرأ الأسعار فقط ليملأ احتمالات 'السوق' للمباريات التي أضفتها "
                      "بالفعل؛ لا علاقة له بالمراهنة. حوالي طلب واحد لكل دوري (الخطة المجانية 500 شهريًا)."))
        st.text_input(LX("The Odds API key (kept only in this session)", "مفتاح The Odds API (يُحفظ في هذه الجلسة فقط)"),
                      type="password", key="odds_key")
        st.caption(LX("Or save it as ODDS_API_KEY in secrets.toml. Key found: ", "أو احفظه باسم ODDS_API_KEY في secrets.toml. المفتاح موجود: ")
                   + ("✅" if odds_key() else "❌"))
        if not st.button(LX("Fill market odds for my upcoming matches", "املأ أسعار السوق لمبارياتي القادمة"), key="odds_go"):
            return
        k = odds_key()
        if not k:
            st.error(LX("Enter your Odds API key first.", "أدخل مفتاح Odds API أولًا."))
            return
        fx = db_read("SELECT * FROM fixtures ORDER BY kickoff")
        fx = fx[(fx.kickoff.isna()) | (fx.kickoff >= dt.date.today().isoformat())]
        done, notes = 0, []
        for lg in sorted(fx.league.unique()):
            sport = ODDS_SPORTS.get(lg)
            if not sport:
                notes.append(f"{lg}: " + LX("no odds feed for this league", "لا توجد أسعار لهذا الدوري"))
                continue
            rows, err = fetch_market_odds(sport, k)
            if err:
                notes.append(f"{lg}: {err}")
                continue
            hn = sorted({r[0] for r in rows})
            an = sorted({r[1] for r in rows})
            for r in fx[fx.league == lg].itertuples():
                h, a = match_team(r.home, hn), match_team(r.away, an)
                m = next((x for x in rows if x[0] == h and x[1] == a), None)
                if m:
                    db_exec("UPDATE fixtures SET o1=?, ox=?, o2=? WHERE id=?", (m[3], m[4], m[5], int(r.id)))
                    done += 1
        (st.success if done else st.warning)(LX(f"Market prices added to {done} match(es).", f"أُضيفت أسعار السوق إلى {done} مباراة."))
        if notes:
            st.caption(" | ".join(notes))


def feed_keys():
    prov = st.session_state.get("llm_prov_sel", "Groq (free)")
    name = "GROQ_API_KEY" if prov.startswith("Groq") else "ANTHROPIC_API_KEY"
    return {"tavily": _secret("TAVILY_API_KEY") or st.session_state.get("tav_key", ""),
            "prov": "groq" if prov.startswith("Groq") else "claude",
            "llm_key": _secret(name) or st.session_state.get("llm_key", ""),
            "model": st.session_state.get("llm_model_groq", "openai/gpt-oss-20b") if prov.startswith("Groq")
            else st.session_state.get("llm_model_claude", "claude-sonnet-5-5")}


@st.cache_data(ttl=1800, show_spinner=False)
def load_fpl():
    r = requests.get(FPL_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    js = r.json()
    return js["teams"], js["elements"]


def fpl_context(teams, elements, our_team, now=None):
    """Official Fantasy Premier League data: who is injured/suspended/doubtful and how much the attack depends on him.
    Long absences (news older than 21 days) are ignored: results since then already reflect them."""
    now = now if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    tinfo = next((t for t in teams if match_team(t["name"], [our_team]) == our_team), None)
    if tinfo is None:
        return None
    pl = [e for e in elements if e.get("team") == tinfo["id"]]
    num = lambda v: float(v) if v not in (None, "") else 0.0
    contrib = {e["id"]: num(e.get("expected_goals")) + num(e.get("expected_assists")) for e in pl}
    tot = sum(contrib.values()) or 1.0
    played = max(int(tinfo.get("played") or 0), 1)
    out = []
    for e in pl:
        stt = e.get("status")
        if stt in ("a", "u", "n", None):
            continue
        chance = e.get("chance_of_playing_next_round")
        avail = chance / 100 if chance is not None else (0.5 if stt == "d" else 0.0)
        try:
            days = (now - pd.Timestamp(e["news_added"]).tz_convert(None)).days
        except Exception:
            days = 0
        long_term = days > 21
        share = contrib[e["id"]] / tot
        pos = POS.get(e.get("element_type"), "")
        regular = num(e.get("starts")) >= 0.5 * played
        att = 0.0 if long_term else share * (1 - avail) * ATT_REPLACE
        dfn = 0.0 if (long_term or not regular) else {"GK": 0.04, "DEF": 0.025}.get(pos, 0.0) * (1 - avail)
        out.append({"name": e.get("web_name", "?"), "pos": pos, "status": {"i": "injured", "s": "suspended", "d": "doubtful"}.get(stt, stt),
                    "chance": chance, "news": e.get("news", ""), "share": round(share, 3), "long_term": long_term,
                    "att": round(att, 4), "dfn": round(dfn, 4)})
    return {"absences": out, "att": max(0.85, 1 - sum(a["att"] for a in out)),
            "dfn": min(1.08, 1 + sum(a["dfn"] for a in out))}


def ctx_multipliers(ctx):
    mh = ctx["home"]["att"] * ctx["away"]["dfn"]
    ma = ctx["away"]["att"] * ctx["home"]["dfn"]
    return float(np.clip(mh, CTX_MIN, CTX_MAX)), float(np.clip(ma, CTX_MIN, CTX_MAX))


def ctx_save(fid, data):
    db_exec("INSERT OR REPLACE INTO context VALUES(?,?,?)", (int(fid), json.dumps(data), dt.datetime.now().isoformat()))


def ctx_load(fid, max_hours=72):
    d = db_read("SELECT data, fetched FROM context WHERE fixture_id=?", (int(fid),))
    if d.empty:
        return None
    if (dt.datetime.now() - dt.datetime.fromisoformat(d.fetched[0])).total_seconds() > max_hours * 3600:
        return None
    return json.loads(d.data[0])


def extract_json(text):
    try:
        a, b = text.index("{"), text.rindex("}")
        return json.loads(text[a:b + 1])
    except Exception:
        return None


def llm_json(keys, prompt):
    """One call to a free (Groq) or paid (Claude) model; expects a JSON answer."""
    if keys["prov"] == "groq":
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {keys['llm_key']}", "Content-Type": "application/json"},
                          json={"model": keys["model"], "temperature": 0,
                                "messages": [{"role": "system", "content": "You extract facts. Reply with JSON only."},
                                             {"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"AI error {r.status_code}: {r.text[:150]}")
        text = r.json()["choices"][0]["message"]["content"]
    else:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": keys["llm_key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": keys["model"], "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"AI error {r.status_code}: {r.text[:150]}")
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
    js = extract_json(text)
    if js is None:
        raise RuntimeError("the AI answer was not readable JSON")
    return js


def player_share(players_df, team, name):
    """Share of the team's attacking output (npxG + xA) from an uploaded Understat players file, if available."""
    if players_df is None or not name:
        return None
    d = players_df
    tm = match_team(team, d.team.unique().tolist())
    if not tm:
        return None
    d = d[d.team == tm]
    val = d.npxg.fillna(d.xg).fillna(0) + d.xa.fillna(0)
    tot = float(val.sum())
    if tot <= 0:
        return None
    last = _key(name).split(" ")[-1] if _key(name) else ""
    hit = d[d.player.astype(str).map(lambda x: bool(last) and last in _key(x))]
    return float(val[hit.index].sum() / tot) if len(hit) else None


def news_context(keys, team, opp, league):
    """Tavily search (free) -> AI extraction (free Groq or paid Claude) -> structured absences with impact."""
    r = requests.post("https://api.tavily.com/search", timeout=40,
                      headers={"Authorization": f"Bearer {keys['tavily']}", "Content-Type": "application/json"},
                      json={"query": f"{team} injury news suspended doubtful ahead of match vs {opp}", "topic": "news",
                            "time_range": "week", "max_results": 5})
    if r.status_code != 200:
        raise RuntimeError(f"Tavily error {r.status_code}")
    snippets = "\n".join(f"- [{x.get('published_date') or ''}] {x.get('title', '')}: {(x.get('content') or '')[:350]}"
                         for x in r.json().get("results", []))
    if not snippets:
        return {"absences": [], "att": 1.0, "dfn": 1.0, "summary": "no recent news found"}
    prompt = (f"Team: {team} ({league}), next opponent {opp}. From the news snippets below, list players of {team} who are "
              "OUT, SUSPENDED or DOUBTFUL for the next match. Ignore injuries older than 3 weeks, players of other teams and "
              'rumours. Reply ONLY with JSON: {"absent":[{"name":str,"role":"GK|DEF|MID|FWD","status":"out|doubtful",'
              '"starter":true|false,"key":true|false}],"summary":str}. If nothing is confirmed reply '
              '{"absent":[],"summary":"nothing confirmed"}.\nSnippets:\n' + snippets)
    js = llm_json(keys, prompt)
    pdf = st.session_state.get("players_df")
    out = []
    for a in (js.get("absent") or [])[:8]:
        if not isinstance(a, dict):
            continue
        avail = 0.0 if str(a.get("status", "out")).lower().startswith("out") else 0.5
        role = str(a.get("role", "")).upper()
        share = player_share(pdf, team, a.get("name", ""))
        att = 0.0
        if role in ("FWD", "MID") and (a.get("key") or share):
            att = (share * ATT_REPLACE if share else 0.04) * (1 - avail)
        dfn = 0.0
        if a.get("starter") and role in ("GK", "DEF"):
            dfn = (0.04 if role == "GK" else 0.025) * (1 - avail)
        out.append({"name": str(a.get("name", "?")), "pos": role, "status": "out" if avail == 0 else "doubtful",
                    "chance": None, "news": "", "share": round(share, 3) if share else None, "long_term": False,
                    "att": round(att, 4), "dfn": round(dfn, 4)})
    return {"absences": out, "att": max(0.85, 1 - sum(a["att"] for a in out)),
            "dfn": min(1.08, 1 + sum(a["dfn"] for a in out)), "summary": str(js.get("summary", ""))[:200]}


def autofeed_one(r, keys):
    if r.league == "Premier League (England)":
        try:
            teams, els = load_fpl()
        except Exception as e:
            return None, f"FPL data unavailable: {e}"
        hc, ac = fpl_context(teams, els, r.home), fpl_context(teams, els, r.away)
        if hc is None or ac is None:
            return None, "team not found in FPL data"
        return {"source": "FPL official data", "home": hc, "away": ac}, "ok"
    if not keys["tavily"] or not keys["llm_key"]:
        return None, "needs free Tavily + AI keys (AI assistant tab)"
    try:
        hc, ac = news_context(keys, r.home, r.away, r.league), news_context(keys, r.away, r.home, r.league)
    except Exception as e:
        return None, str(e)
    return {"source": "news search + AI", "home": hc, "away": ac}, "ok"


def run_feed(fxdf, days, keys, force=False):
    rows, today = [], dt.date.today()
    horizon = (today + dt.timedelta(days=days)).isoformat()
    for r in fxdf.itertuples():
        if r.kickoff and (r.kickoff < today.isoformat() or r.kickoff > horizon):
            continue
        if not force and ctx_load(r.id, 6) is not None:
            rows.append({"Match": f"{r.home} vs {r.away}", "Status": "already fresh"})
            continue
        data, msg = autofeed_one(r, keys)
        row = {"Match": f"{r.home} vs {r.away}", "Status": msg}
        if data:
            ctx_save(r.id, data)
            mh, ma = ctx_multipliers(data)
            row.update({"Source": data["source"], "Home goals x": round(mh, 3), "Away goals x": round(ma, 3),
                        "Absences": len(data["home"]["absences"]) + len(data["away"]["absences"])})
        rows.append(row)
    return rows


# ---------- analysis ----------
def data_quality(b, h, a):
    d = b["df"]
    r = d[d.Date >= d.Date.max() - pd.Timedelta(days=400)]
    n = lambda t: int(((r.HomeTeam == t) | (r.AwayTeam == t)).sum())
    nh, na = n(h), n(a)
    hi, lo = (12, 6) if b["kind"] == "intl" else (30, 18)
    g = "A" if min(nh, na) >= hi else "B" if min(nh, na) >= lo else "C"
    return nh, na, g


def sel_hit(sel, hg, ag):
    t = hg + ag
    return bool({"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag, "12": hg != ag,
                 "Over 1.5": t > 1, "Over 2.5": t > 2, "Under 2.5": t < 3, "Under 3.5": t < 4,
                 "BTTS Yes": hg > 0 and ag > 0, "BTTS No": hg == 0 or ag == 0}[sel])


def analyse_fixture(r, b, w_model=None, w_elo=0.0, ext=None, ctx=None, calib=None):
    """Full output for one stored fixture: ensemble probabilities (Dixon-Coles + Elo sources + market),
    all markets, and the facts needed for the risk check. None if a team is unknown to the model."""
    model = b["model"]
    if r.home not in model["idx"] or r.away not in model["idx"]:
        return None
    neutral = bool(r.neutral)
    lam0, mu0 = lambdas(model, r.home, r.away, neutral)
    lam, mu = lam0 * float(r.adj_h or 1), mu0 * float(r.adj_a or 1)
    auto = None
    if ctx:
        mh, ma = ctx_multipliers(ctx)
        lam, mu = lam * mh, mu * ma
        auto = {"mh": mh, "ma": ma, "ctx": ctx}
    m = score_matrix(lam, mu, model["rho"])
    p_dc = np.array(outcome_probs(m))
    sources = {"Dixon-Coles": p_dc}
    p_slow, m2 = None, b.get("model2")
    if m2 and r.home in m2["idx"] and r.away in m2["idx"]:
        l2, u2 = lambdas(m2, r.home, r.away, neutral)
        p_slow = np.array(outcome_probs(score_matrix(l2 * lam / lam0, u2 * mu / mu0, m2["rho"])))
        sources["Dixon-Coles (slow)"] = p_slow
    usf = (b.get("us_fc") or {}).get((r.home, r.away))
    if usf is not None:
        sources["Understat"] = np.array(usf)
    w_slow = float(calib.get("w_slow", 0.0)) if calib else 0.0
    base_dc = (1 - w_slow) * p_dc + w_slow * p_slow if (p_slow is not None and w_slow > 0) else p_dc
    elo_list = []
    pe = elo_probs(b.get("elo"), r.home, r.away, neutral)
    if pe is not None:
        sources["Elo (own)"] = np.array(pe)
        elo_list.append(np.array(pe))
    if ext is not None:
        sources["ClubElo"] = np.array(ext)
        elo_list.append(np.array(ext))
    final = base_dc
    w_use = calib["w_elo"] if calib else w_elo
    if elo_list and w_use > 0:
        final = (1 - w_use) * base_dc + w_use * np.mean(elo_list, axis=0)
    if calib and abs(calib["temp"] - 1.0) > 1e-9:
        final = _temp(final, calib["temp"])
    odds3 = [r.o1, r.ox, r.o2]
    if all(o is not None and pd.notna(o) and float(o) > 1 for o in odds3):
        inv = 1 / np.array(odds3, dtype=float)
        sources["Market"] = inv / inv.sum()
        if w_model is not None and w_model < 1:
            final = w_model * final + (1 - w_model) * sources["Market"]
    final = final / final.sum()
    blended = not np.allclose(final, p_dc, atol=1e-9)
    if blended:
        m = tilt_matrix(m, final)
    allp = np.array(list(sources.values()))
    spread = float((allp.max(axis=0) - allp.min(axis=0)).max()) if len(allp) > 1 else 0.0
    mk = masks(m.shape[0])
    market = {"1": r.o1, "X": r.ox, "2": r.o2}
    sels = []
    for k in SEL:
        p = float(m[mk[k]].sum())
        if calib:
            p = _cal_sel(k, p, calib)
        o = market.get(k)
        o = float(o) if o is not None and pd.notna(o) and float(o) > 1 else None
        sels.append({"sel": k, "p": p, "odds": o, "fair": 1 / p if p > 0 else np.inf, "ev": p * o - 1 if o else None})
    flat = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:5], m.shape))[0]
    tops = [(int(i), int(j), float(m[i, j])) for i, j in flat]
    nh, na, grade = data_quality(b, r.home, r.away)
    kd = pd.Timestamp(r.kickoff) if r.kickoff else None
    d = b["df"]

    def rest(t):
        last = d[(d.HomeTeam == t) | (d.AwayTeam == t)].Date.max()
        return int((kd - last).days) if kd is not None and pd.notna(last) else None
    def cong(t):
        if kd is None:
            return None
        return int((((d.HomeTeam == t) | (d.AwayTeam == t)) & (d.Date >= kd - pd.Timedelta(days=14)) & (d.Date < kd)).sum())
    return {"id": int(r.id), "league": r.league, "home": r.home, "away": r.away, "kickoff": r.kickoff,
            "auto": auto, "cong": (cong(r.home), cong(r.away)),
            "lam": lam, "mu": mu, "m": m, "sels": sels, "tops": tops, "nh": nh, "na": na, "grade": grade,
            "adj": (float(r.adj_h or 1), float(r.adj_a or 1)), "note": r.note or "", "b": b, "blended": blended,
            "sources": sources, "spread": spread, "rest": (rest(r.home), rest(r.away)),
            "calibrated": bool(calib) and calib.get("n", 0) >= 120, "cal": calib,
            "corners": corner_probs(b.get("corners"), r.home, r.away)}


def risk_report(f, pick):
    """Risk of a pick, in %. Base = chance the pick fails; extras = reasons to trust the model less."""
    base = (1 - pick["p"]) * 100
    extras, why = 0.0, []
    if f["grade"] == "C":
        extras += 8
        why.append("little recent data for one team (+8)")
    elif f["grade"] == "B":
        extras += 3
        why.append("moderate recent data (+3)")
    if f["spread"] > 0.10:
        x = min(15.0, (f["spread"] - 0.08) * 100 * 0.8)
        extras += x
        why.append(f"sources disagree by {f['spread'] * 100:.0f} points (+{x:.0f})")
    if any(r is not None and r <= 3 for r in f["rest"]):
        extras += 4
        why.append("a team has 3 days or less of rest (+4)")
    if f["adj"] != (1.0, 1.0):
        extras += 3
        why.append("manual adjustment applied, news-driven (+3)")
    au = f.get("auto")
    if au and (abs(au["mh"] - 1) > 0.005 or abs(au["ma"] - 1) > 0.005):
        extras += 3
        why.append("injury/absence news applied (+3)")
    if any(c is not None and c >= 4 for c in f.get("cong", (None, None))):
        extras += 3
        why.append("4+ matches in the last 14 days for a team (+3)")
    total = min(95.0, base + extras)
    return {"risk": total, "base": base, "extra": extras, "why": why,
            "label": "Low" if total <= 25 else "Medium" if total <= 40 else "High",
            "safe_p": max(0.0, pick["p"] - extras / 100)}


def best_pick(f, allowed, min_p, min_fair):
    c = [s for s in f["sels"] if s["sel"] in allowed and s["p"] >= min_p and s["fair"] >= min_fair]
    return max(c, key=lambda s: s["p"]) if c else None


def explain(f):
    P = {s["sel"]: s["p"] for s in f["sels"]}
    h, a = f["home"], f["away"]
    top = max(("1", "X", "2"), key=lambda k: P[k])
    if P[top] >= 0.45:
        who = {"1": h, "X": LX("a draw", "التعادل"), "2": a}[top]
        s = LX(f"{who} is the most likely result ({P[top] * 100:.0f}%).",
               f"{who} هو النتيجة الأكثر احتمالًا ({P[top] * 100:.0f}%).")
    else:
        s = LX("Open match with no clear favourite.", "مباراة مفتوحة بلا مرشح واضح.")
    s += LX(f" Expected goals {f['lam']:.2f}-{f['mu']:.2f}; most likely score {f['tops'][0][0]}-{f['tops'][0][1]}.",
            f" الأهداف المتوقعة {f['lam']:.2f}-{f['mu']:.2f}؛ النتيجة الأرجح {f['tops'][0][0]}-{f['tops'][0][1]}.")
    s += LX(f" Over 2.5 goals {P['Over 2.5'] * 100:.0f}%, both teams to score {P['BTTS Yes'] * 100:.0f}%.",
            f" أكثر من 2.5 هدف {P['Over 2.5'] * 100:.0f}%، وتسجيل الفريقين {P['BTTS Yes'] * 100:.0f}%.")
    if f["grade"] == "C":
        s += LX(f" Limited recent data ({f['nh']} and {f['na']} matches in the last 400 days): treat with caution.",
                f" بيانات حديثة محدودة ({f['nh']} و{f['na']} مباراة خلال آخر 400 يوم): تعامل بحذر.")
    if f.get("blended"):
        names = ", ".join(tr(x) for x in f["sources"])
        s += LX(f" The 1X2 probabilities combine {len(f['sources'])} sources ({', '.join(f['sources'])}).",
                f" احتمالات النتيجة (1X2) تجمع {len(f['sources'])} مصادر ({names}).")
    if f.get("spread", 0) > 0.08:
        s += LX(f" The sources disagree by up to {f['spread'] * 100:.0f} points, so the call is less reliable.",
                f" المصادر مختلفة بحد أقصى {f['spread'] * 100:.0f} نقطة، فالتوقع أقل موثوقية.")
    if f["adj"] != (1.0, 1.0):
        s += LX(f" Manual adjustment applied (x{f['adj'][0]:.2f} / x{f['adj'][1]:.2f}).",
                f" تم تطبيق تعديل يدوي (×{f['adj'][0]:.2f} / ×{f['adj'][1]:.2f}).")
    if f.get("auto"):
        src_ = f['auto']['ctx']['source']
        s += LX(f" Injury/absence feed ({src_}) changed expected goals by "
                f"x{f['auto']['mh']:.2f} / x{f['auto']['ma']:.2f}.",
                f" تغذية الإصابات/الغيابات ({tr(src_)}) غيّرت الأهداف المتوقعة بمقدار "
                f"×{f['auto']['mh']:.2f} / ×{f['auto']['ma']:.2f}.")
    ev = [(x["sel"], x["ev"]) for x in f["sels"] if x["ev"] is not None]
    if ev:
        k, v = max(ev, key=lambda t: t[1])
        s += LX(f" Versus the market the biggest gap is on '{k}' ({v * 100:+.0f}% value).",
                f" مقارنةً بالسوق أكبر فجوة في '{_fmt_opt(k)}' ({v * 100:+.0f}% قيمة).")
    return s


def top_legs(fxs, allowed, min_p, min_fair):
    legs = []
    for fi, f in enumerate(fxs):
        s = best_pick(f, allowed, min_p, min_fair)
        if s:
            legs.append({**s, "match": f"{f['home']} vs {f['away']}", "fi": fi, "league": f["league"], "score": s["p"]})
    return sorted(legs, key=lambda l: -l["score"])


def save_picks(fxs, allowed, min_p, min_fair):
    """Remember the pre-match picks so they can be graded later. Never overwrites a match that already kicked off."""
    today = dt.date.today().isoformat()
    for f in fxs:
        if f["kickoff"] and f["kickoff"] < today:
            continue
        chosen = {max(("1", "X", "2"), key=lambda k: next(s["p"] for s in f["sels"] if s["sel"] == k))}
        bp = best_pick(f, allowed, min_p, min_fair)
        if bp:
            chosen.add(bp["sel"])
        for k in chosen:
            p = next(s["p"] for s in f["sels"] if s["sel"] == k)
            pk_ = next(s for s in f["sels"] if s["sel"] == k)
            try:
                rk_ = risk_report(f, pk_)
                ag_ = agreement(f, pk_)
                vd_, gr_, rs_ = verdict(rk_, ag_, f), confidence(f, pk_, rk_, ag_)[1], float(rk_["risk"])
            except Exception:
                vd_, gr_, ag_, rs_ = None, None, None, None
            db_exec("INSERT OR REPLACE INTO picks(fixture_id,sel,p,created,verdict,grade,agree,risk) VALUES(?,?,?,?,?,?,?,?)",
                    (f["id"], k, float(p), dt.datetime.now().isoformat(), vd_, gr_, ag_, rs_))


# ---------- free news headlines (no key needed) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def team_news(team, days=10, n=6):
    q = requests.utils.quote(f'"{team}" football (injury OR injured OR suspended OR lineup OR "ruled out")')
    r = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en", timeout=15,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_news(r.content, days, n)


def parse_news(xml_bytes, days=10, n=6):
    items, cut = [], pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days)
    for it in ET.fromstring(xml_bytes).iter("item"):
        try:
            d = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert(None)
        except Exception:
            continue
        if d >= cut:
            items.append({"date": d, "title": it.findtext("title") or "", "link": it.findtext("link") or "",
                          "source": it.findtext("source") or ""})
    return sorted(items, key=lambda x: x["date"], reverse=True)[:n]


# ---------- Tavily: free AI search engine (1,000 credits a month, no card) ----------
def tavily_research(key, f):
    """One short search per topic (1 credit each). Returns a list of {topic, answer, results}."""
    topics = [(f"{f['home']} injuries and suspensions", f"{f['home']} football team news injuries suspended doubtful"),
              (f"{f['away']} injuries and suspensions", f"{f['away']} football team news injuries suspended doubtful"),
              ("Expected line-ups and context", f"{f['home']} vs {f['away']} predicted lineup preview")]
    out, err = [], None
    for name, q in topics:
        try:
            r = requests.post("https://api.tavily.com/search", timeout=40,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"query": q, "topic": "news", "time_range": "week", "max_results": 5,
                                    "include_answer": "basic"})
        except Exception as e:
            err = f"Network error: {e}"
            break
        if r.status_code != 200:
            err = f"Tavily error {r.status_code}: {r.text[:200]}"
            break
        js = r.json()
        out.append({"topic": name, "answer": js.get("answer") or "", "results": js.get("results", [])})
    return out, err


# ---------- AI web research (optional, needs an Anthropic API key) ----------
def ai_research(api_key, model_id, f, max_uses):
    prompt = (f"Research the football match {f['home']} vs {f['away']} ({f['league']}, {f['kickoff']}). "
              "Search the web for the latest information: confirmed injuries and suspensions for both teams, "
              "expected or confirmed line-ups, recent form (last 5 results), and context that changes motivation or "
              "rotation (cup games, congested schedule, nothing to play for). Use reliable recent sources, give the "
              "date of each source, and say clearly when something is unconfirmed. Finish with a short list per team: "
              "key absentees (name and role) and whether this should push that team's expected goals up or down, and "
              "roughly by how much. Keep it under 300 words.")
    body = {"model": model_id, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": int(max_uses)}]}
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180, json=body,
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"})
    except Exception as e:
        return None, {}, f"Network error: {e}"
    if r.status_code != 200:
        return None, {}, f"API error {r.status_code}: {r.text[:300]}"
    text, cites = [], {}
    for blk in r.json().get("content", []):
        if blk.get("type") == "text":
            text.append(blk.get("text", ""))
            for c in blk.get("citations") or []:
                if c.get("url"):
                    cites[c["url"]] = c.get("title") or c["url"]
    return "\n".join(text).strip(), cites, None


# ---------- UI pieces ----------
def ui_add_fixtures(years, half_life, wx_user, fuse_on):
    st.markdown("#### Add matches")
    c = st.columns([2, 1, 1])
    league = c[0].selectbox("League", board_leagues(), key="fx_league")
    kick = c[1].date_input("Match date", value=dt.date.today(), key="fx_date")
    neutral = c[2].checkbox("Neutral venue", value=league.endswith("(national teams)"), key=f"fx_neu_{league}")
    if SOURCES[league][0] == "fdorg" and not org_key():
        st.info("This competition needs a free football-data.org key: register at football-data.org/client/register, "
                "copy the key from the email and paste it below (or save it as FOOTBALL_DATA_ORG_KEY in Secrets).")
        st.text_input("football-data.org key", type="password", key="fdorg_key")
    b = league_bundle(league, years, half_life, wx_user, fuse_on)
    if b is None:
        st.error("Could not load data for this league (check the internet connection of the host and any key it needs).")
        _kd, _cd = SOURCES[league]
        _why = (net_failures(f"{_cd}-") if _kd == "fd" else net_failures(f"new-{_cd}") if _kd == "new"
                else net_failures(f"espn-{_cd}-") if _kd == "espn" else [])
        if _why:
            st.caption("Reason: " + " | ".join(_why[:3]))
        if st.button("Try again now", key=f"retry_{league}"):
            HOST_DEAD.clear()
            st.cache_data.clear()
            st.rerun()
        st.caption("Open the 'Data & backup' tab and press 'Test all sources now' to see exactly what is blocked.")
    else:
        _kd, _cd = SOURCES[league]
        if any(v[0] == "saved copy" for k, v in NET_LOG.items() if k.startswith(f"{_cd}-") or k == f"new-{_cd}"):
            st.warning("The internet source did not answer: using the last saved copy of the data (may be a bit old).")
        if _cd in BACKUP_USED:
            st.warning("Main source unreachable: this league uses the backup source (openfootball), "
                       "so shots/xG and bookmaker odds are not available.")
        teams = b["teams"]
        c = st.columns(2)
        h = c[0].selectbox("Home team", teams, index=0, key=f"fx_h_{league}")
        a = c[1].selectbox("Away team", teams, index=min(1, len(teams) - 1), key=f"fx_a_{league}")
        c = st.columns(3)
        o1 = c[0].number_input("Home odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_o1")
        ox = c[1].number_input("Draw odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_ox")
        o2 = c[2].number_input("Away odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_o2")
        if st.button("Add this match", key="fx_add"):
            if h == a:
                st.warning("Choose two different teams.")
            elif db_add_fixture(league, h, a, kick, neutral, o1, ox, o2):
                st.success(f"Added {h} vs {a}.")
            else:
                st.info("This match is already in the list.")

    with st.expander("⚡ Paste a list of matches (fastest way)"):
        st.caption("One match per line, like `Arsenal - Chelsea` or `Premier League: Arsenal vs Chelsea`. "
                   "Team names are matched automatically, spelling does not have to be exact.")
        txt = st.text_area("Matches", height=140, key="fx_paste")
        d2 = st.date_input("Date for these matches", value=dt.date.today(), key="fx_paste_date")
        if st.button("Add pasted matches", key="fx_paste_btn"):
            added, bad = 0, []
            for line in [x.strip() for x in txt.splitlines() if x.strip()]:
                lg, body = league, line
                if ":" in line:
                    pre, rest = line.split(":", 1)
                    guess = match_team(pre.strip(), BOARD_LEAGUES)
                    if guess:
                        lg, body = guess, rest
                parts = re.split(r"\s+(?:vs\.?|v|-|–|—)\s+", body.strip(), maxsplit=1, flags=re.I)
                bb = league_bundle(lg, years, half_life, wx_user, fuse_on)
                if len(parts) != 2 or bb is None:
                    bad.append(line)
                    continue
                hh, aa = match_team(parts[0].strip(), bb["teams"]), match_team(parts[1].strip(), bb["teams"])
                if not hh or not aa or hh == aa:
                    bad.append(line)
                    continue
                added += db_add_fixture(lg, hh, aa, d2, lg.endswith("(national teams)"))
            st.success(f"Added {added} match(es).")
            if bad:
                st.warning("Could not understand: " + " | ".join(bad))

    with st.expander("📡 Import upcoming matches automatically"):
        st.caption("Pulls the published fixtures (with market odds when available) for the main leagues. "
                   "Leagues without a free fixture feed need manual entry above.")
        days = st.slider("Matches in the next N days", 1, 14, 5, key="imp_days")
        fd_leagues = board_leagues()
        pick = st.multiselect("Leagues", fd_leagues, default=[], key="imp_leagues")
        if st.button("Fetch and add", key="imp_btn") and pick:
            fxd = load_fd_fixtures()
            today = pd.Timestamp.today().normalize()
            added = bad = 0
            for lg in pick:
                bb = league_bundle(lg, years, half_life, wx_user, fuse_on)
                if bb is None:
                    continue
                src_rows = pd.DataFrame()
                if len(fxd):
                    src_rows = fxd[(fxd.League == lg) & (fxd.Date >= today) & (fxd.Date <= today + pd.Timedelta(days=days))]
                if src_rows.empty and len(bb["fut"]):
                    f2 = bb["fut"]
                    src_rows = f2[(f2.Date <= today + pd.Timedelta(days=days))].rename(columns={"HomeTeam": "HomeTeam"})
                    for c_ in ("o1", "ox", "o2"):
                        src_rows[c_] = np.nan
                for r in src_rows.itertuples():
                    hh, aa = match_team(r.HomeTeam, bb["teams"]), match_team(r.AwayTeam, bb["teams"])
                    if not hh or not aa or hh == aa:
                        bad += 1
                        continue
                    added += db_add_fixture(lg, hh, aa, r.Date.date(), 0, r.o1, r.ox, r.o2)
            if added == 0 and bad == 0:
                st.info("No upcoming matches were published for these leagues in that window. Add them manually.")
            else:
                st.success(f"Added {added} match(es)." + (f" {bad} skipped (team not recognised)." if bad else ""))

    st.markdown("#### Match list (database)")
    fxdf = db_read("SELECT id, league, kickoff, home, away, neutral, o1, ox, o2, adj_h, adj_a, note "
                   "FROM fixtures ORDER BY kickoff, league, id")
    if fxdf.empty:
        st.info("The list is empty. Add matches above.")
    else:
        st.dataframe(fxdf, hide_index=True, use_container_width=True)
        c = st.columns([3, 1, 1])
        ids = c[0].multiselect("Remove matches (by id)", fxdf["id"].tolist(), key="fx_del")
        if c[1].button("Remove selected", key="fx_del_btn") and ids:
            for i in ids:
                db_exec("DELETE FROM fixtures WHERE id=?", (int(i),))
                db_exec("DELETE FROM picks WHERE fixture_id=?", (int(i),))
            st.success("Removed. Press R to refresh the list.")
        if c[2].button("Remove ALL", key="fx_del_all"):
            db_exec("DELETE FROM fixtures")
            db_exec("DELETE FROM picks")
            st.success("List cleared.")
    st.caption("The database file lives next to the app. On Streamlit Cloud it is wiped when the app restarts, so "
               "use Backup / Restore in the Data tab.")


def ui_analysis(years, half_life, wx_user, fuse_on):
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    fxdf = fxdf[(fxdf.kickoff.isna()) | (fxdf.kickoff >= (dt.date.today() - dt.timedelta(days=1)).isoformat())]
    if fxdf.empty:
        st.info("Add upcoming matches in the Matches tab. The analysis then runs automatically.")
        return
    with st.expander("Settings", expanded=True):
        c = st.columns(3)
        allowed = c[0].multiselect("Markets to consider", SEL, default=DEFAULT_MARKETS, key="an_mk")
        min_p = c[1].slider("Minimum probability %", 40, 90, 60, 1, key="an_p") / 100
        min_fair = c[2].slider("Minimum fair odds (skips trivial picks)", 1.05, 2.5, 1.25, 0.05, key="an_o")
        c = st.columns(3)
        w_elo = c[0].slider("Weight of the Elo sources %", 0, 60, 20, 5, key="an_elo") / 100
        use_ext = c[1].checkbox("Include ClubElo (free external ratings)", value=True, key="an_ext")
        w_model = c[2].slider("Weight of our model vs market odds % (only if odds exist)", 0, 100, 50, 5, key="an_w") / 100
        st.caption("Sources combined: Dixon-Coles model + own Elo + ClubElo + market odds (when given). Odds are "
                   "optional; without them the first three are used. Check the Backtest page (Elo and Blend rows) "
                   "to see which weights work best for a league.")
        use_cal = st.checkbox("Use each league's calibrated settings (best Elo weight + honest confidence)", value=True,
                              key="an_cal")
        need_agree = st.checkbox("Safe slip: only picks where all sources agree", value=True, key="an_agree")
        if use_cal:
            _missing = [lg for lg in sorted(fxdf.league.unique()) if calib_load(lg) is None]
            if _missing:
                st.info("Not calibrated yet: " + ", ".join(_missing) + ". Calibration tests the model on each league's "
                        "latest matches and keeps the settings that predicted best (about a minute per league, done once).")
                if st.button("Calibrate these leagues now", key="cal_now"):
                    for lg in _missing:
                        b0 = league_bundle(lg, years, half_life, wx_user, fuse_on)
                        if b0 is None:
                            st.warning(f"{lg}: no data")
                            continue
                        with st.spinner(f"Calibrating {lg}..."):
                            run_calibration(lg, b0, *eff_params(lg, half_life, wx_user))
                    st.rerun()
        use_auto = st.checkbox("Apply the automatic injury/absence feed (Premier League: official data; other leagues: "
                               "needs the free keys in the AI assistant tab)", value=True, key="an_auto")
        c = st.columns(2)
        max_risk = c[0].slider("Maximum risk per match %", 10, 60, 30, 1, key="an_risk")
        target = c[1].slider("Safe slip: minimum combined chance % (stress-tested)", 30, 90, 55, 1, key="an_target") / 100
        max_legs = st.slider(LX("Safe slip: maximum number of matches", "السلبة الآمنة: أقصى عدد مباريات"), 1, 8, 3, 1, key="an_maxlegs")
    if use_auto and not st.session_state.get("feed_ran"):
        st.session_state["feed_ran"] = True
        with st.spinner("Updating injury and absence data for the next 3 days..."):
            try:
                run_feed(fxdf, 3, feed_keys())
            except Exception:
                pass
    fxs, skipped, bundles, calibs = [], [], {}, {}
    for r in fxdf.itertuples():
        if r.league not in bundles:
            bundles[r.league] = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        b = bundles[r.league]
        f = None
        if b:
            ext = clubelo_probs(r.league, r.home, r.away, bool(r.neutral), b["elo"]["d0"], b["elo"].get("hfa", 100.0)) if use_ext else None
            if r.league not in calibs:
                calibs[r.league] = calib_load(r.league) if use_cal else None
            f = analyse_fixture(r, b, w_model, w_elo, ext, ctx_load(r.id) if use_auto else None, calibs[r.league])
        (fxs if f else skipped).append(f if f else f"{r.home} vs {r.away} ({r.league})")
    if skipped:
        st.warning("Could not analyse: " + " | ".join(skipped))
    if not fxs:
        return
    save_picks(fxs, allowed, min_p, min_fair)

    rows = []
    for f in fxs:
        P = {s["sel"]: s["p"] for s in f["sels"]}
        bp = best_pick(f, allowed, min_p, min_fair)
        rk = risk_report(f, bp) if bp else None
        ag = agreement(f, bp) if bp else "-"
        cf = confidence(f, bp, rk, ag) if bp else (None, "-")
        rows.append({"League": f["league"], "Date": f["kickoff"], "Match": f"{f['home']} vs {f['away']}",
                     "Goals exp.": f"{f['lam']:.2f}-{f['mu']:.2f}", "Likely score": f"{f['tops'][0][0]}-{f['tops'][0][1]}",
                     "1 %": round(P["1"] * 100), "X %": round(P["X"] * 100), "2 %": round(P["2"] * 100),
                     "Over 2.5 %": round(P["Over 2.5"] * 100), "BTTS %": round(P["BTTS Yes"] * 100),
                     "Best pick": bp["sel"] if bp else "-", "Pick %": round(bp["p"] * 100) if bp else None,
                     "Risk %": round(rk["risk"]) if rk else None, "Risk": rk["label"] if rk else "-",
                     "Agreement": ag, "Verdict": verdict(rk, ag, f) if rk else "-",
                     "Confidence %": round(cf[0]) if bp else None, "Grade": cf[1],
                     "Exp. corners": round(f["corners"]["total"], 1) if f.get("corners") else None,
                     "Calibrated": "yes" if f.get("calibrated") else "no",
                     "Sources": len(f["sources"]), "Data": f["grade"]})
    st.markdown("#### Predictions for all matches")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Risk % = chance the pick fails plus penalties for weak data, disagreeing sources, short rest or "
               "news-driven adjustments. Low <= 25, Medium <= 40, High above. Data grade: A = 30+ recent matches "
               "for both teams, B = 18+, C = fewer. Verdict: Strong = low risk, sources agree, enough data and a calibrated league; "
               "OK = medium risk and no disagreement; Skip = anything else. 'Calibrated' = the league's settings were "
               "tested on its recent matches (Analysis settings or Data tab).")

    st.caption(LX("Confidence % = the pick's chance after the risk penalties, minus any over-confidence this league's "
                  "calibration showed (and 5 if the sources split). Grade A = 80%+ with sources agreeing and enough "
                  "data, B = 68%+, C = the rest. Exp. corners: see the Corners tab.",
                  "الثقة % = احتمال الاختيار بعد خصم عقوبات الخطر، مطروحًا منها أي إفراط في الثقة أظهرته معايرة هذا "
                  "الدوري (و5 إذا انقسمت المصادر). التصنيف A = 80%+ مع اتفاق المصادر وبيانات كافية، B = 68%+، C = الباقي. "
                  "الركنيات المتوقعة: انظر تبويب الركنيات."))
    legs = top_legs(fxs, allowed, min_p, min_fair)
    for l in legs:
        l["risk"] = risk_report(fxs[l["fi"]], l)
        l["agree"] = agreement(fxs[l["fi"]], l)
        l["verdict"] = verdict(l["risk"], l["agree"], fxs[l["fi"]])
    st.markdown("#### Pick quality and risk")
    if not legs:
        st.warning("No selection passes the filters. Lower the minimum probability or fair odds.")
    else:
        st.dataframe(pd.DataFrame([{"League": l["league"], "Match": l["match"], "Pick": l["sel"],
                                    "Model %": round(l["p"] * 100, 1), "Risk %": round(l["risk"]["risk"]),
                                    "Risk": l["risk"]["label"], "Stress-tested %": round(l["risk"]["safe_p"] * 100, 1),
                                    "Agreement": l["agree"], "Verdict": l["verdict"],
                                    "Why": "; ".join(l["risk"]["why"]) or "-"} for l in legs]),
                     hide_index=True, use_container_width=True)
        ok = sorted([l for l in legs if l["risk"]["risk"] <= max_risk and not (need_agree and l["agree"] == "split")],
                    key=lambda l: (l["risk"]["risk"], -l["p"]))
        out = [l for l in legs if l["risk"]["risk"] > max_risk or (need_agree and l["agree"] == "split")]
        safe, comb = [], 1.0
        for l in ok:
            if len(safe) >= max_legs:
                break
            if safe and comb * l["risk"]["safe_p"] < target:
                break
            safe.append(l)
            comb *= l["risk"]["safe_p"]
        st.markdown("#### Recommended safe slip")
        if not safe:
            st.warning(f"No match is below {max_risk}% risk. Raise the limit or lower the probability filters.")
        else:
            raw = float(np.prod([l["p"] for l in safe]))
            st.dataframe(pd.DataFrame([{"League": l["league"], "Match": l["match"], "Pick": l["sel"],
                                        "Model %": round(l["p"] * 100, 1), "Risk %": round(l["risk"]["risk"])}
                                       for l in safe]), hide_index=True, use_container_width=True)
            dist = simulate(fxs, safe)
            c = st.columns(3)
            c[0].metric("Chance every pick wins", f"{raw * 100:.1f}%")
            c[1].metric("Stress-tested chance", f"{comb * 100:.1f}%")
            c[2].metric("Chance at most one fails", f"{(dist[:2].sum()) * 100:.1f}%")
            if comb < target:
                st.warning("Even the safest single pick is below your target. No slip meets it.")
            if raw < 0.5:
                st.error(LX(f"Warning: the chance that EVERY pick in this slip wins is only {raw * 100:.0f}%, so it fails "
                            f"more often than it wins. Every extra match lowers it further.",
                            f"تحذير: احتمال أن تنجح كل اختيارات هذه السلبة معًا {raw * 100:.0f}% فقط، أي أنها تفشل أكثر "
                            f"مما تنجح. وكل مباراة إضافية تخفضها أكثر."))
            st.caption("Built greedily from the lowest-risk picks while the stress-tested combined chance stays above "
                       "your target. Stress-tested = each pick's probability minus its model-risk penalties. "
                       "No outcome is guaranteed: even a 70% slip fails 3 times in 10.")
        if out:
            with st.expander(f"Excluded as too risky ({len(out)})"):
                for l in out:
                    st.write(f"- **{l['match']}** ({l['sel']}, risk {l['risk']['risk']:.0f}%): "
                             + ("; ".join(l["risk"]["why"] + (["sources disagree on the result"] if l["agree"] == "split" else []))
                                or f"only {l['p'] * 100:.0f}% likely"))
        with st.expander("Build your own slip"):
            names = [l["match"] for l in legs]
            chosen_names = st.multiselect("Matches in the slip", names, default=[l["match"] for l in safe], key="an_slip")
            chosen = [l for l in legs if l["match"] in chosen_names]
            if chosen:
                sizes = []
                for k in range(1, len(chosen) + 1):
                    s2 = slip_stats(chosen[:k])
                    sizes.append({"Legs": k, "Chance all win %": round(s2["P"] * 100, 1),
                                  "Chance slip fails %": round((1 - s2["P"]) * 100, 1),
                                  "Combined fair odds": round(s2["fair"], 2)})
                st.dataframe(pd.DataFrame(sizes), hide_index=True, use_container_width=True)
                s2 = slip_stats(chosen)
                d2 = simulate(fxs, chosen)
                c = st.columns(3)
                c[0].metric("Chance every pick wins", f"{s2['P'] * 100:.1f}%")
                c[1].metric("Exactly one fails", f"{(d2[1] * 100 if len(d2) > 1 else 0):.1f}%")
                c[2].metric("Two or more fail", f"{d2[2:].sum() * 100:.1f}%")
                st.caption(f"Monte Carlo with {N_SIMS:,} rounds; matches treated as independent.")

    tbl = pd.DataFrame(rows)
    st.download_button("⬇️ Download predictions (CSV)", tbl.to_csv(index=False), "predictions.csv", "text/csv", key="an_dl")
    if legs:
        st.text_area("Copy-ready summary", "\n".join(
            LX(f"{i + 1}. {l['match']} ({l['league']}): {l['sel']} - {l['p'] * 100:.0f}% | risk {l['risk']['risk']:.0f}% ({l['risk']['label']})",
               f"{i + 1}. {l['match']} ({tr(l['league'])}): {_fmt_opt(l['sel'])} - {l['p'] * 100:.0f}% | المخاطرة {l['risk']['risk']:.0f}% ({tr(l['risk']['label'])}) | {tr(l['verdict'])}")
            for i, l in enumerate(legs)), height=120, key="an_copy")

    st.markdown("#### Match detail")
    labels = [f"{f['league']} | {f['home']} vs {f['away']} | {f['kickoff']}" for f in fxs]
    sel = st.selectbox("Choose a match", labels, key="an_detail")
    f = fxs[labels.index(sel)]
    st.info(explain(f))
    with st.expander("How reliable has this league's model been? (measured on its recent matches)"):
        render_reliability(f["league"], calibs.get(f["league"]) or calib_load(f["league"]))
    c = st.columns(3)
    c[0].metric(f"{f['home']} expected goals", f"{f['lam']:.2f}")
    c[1].metric(f"{f['away']} expected goals", f"{f['mu']:.2f}")
    c[2].metric("Data grade", f["grade"], f"{f['nh']} / {f['na']} recent matches", delta_color="off")
    st.markdown("**What each source says (home / draw / away %)**")
    st.dataframe(pd.DataFrame([{"Source": k, "Home %": round(v[0] * 100, 1), "Draw %": round(v[1] * 100, 1),
                                "Away %": round(v[2] * 100, 1)} for k, v in f["sources"].items()]),
                 hide_index=True, use_container_width=True)
    if f["auto"]:
        st.markdown(f"**Absences from the automatic feed** ({f['auto']['ctx']['source']}): goals x{f['auto']['mh']:.2f} "
                    f"for {f['home']}, x{f['auto']['ma']:.2f} for {f['away']}")
        rows_a = [{"Team": f[side], **{k: a[k] for k in ("name", "pos", "status", "chance", "share", "long_term", "news")}}
                  for side, key in (("home", "home"), ("away", "away")) for a in f["auto"]["ctx"][key]["absences"]]
        if rows_a:
            st.dataframe(pd.DataFrame(rows_a).rename(columns={"name": "Player", "pos": "Pos", "status": "Status",
                                                              "chance": "Chance of playing %", "share": "Share of attack",
                                                              "long_term": "Long-term (ignored)", "news": "News"}),
                         hide_index=True, use_container_width=True)
        else:
            st.write("No relevant absences found.")
        st.caption("Impact is a cautious estimate: a missing attacker removes about a third of his share of the "
                   "team's attack; missing regular defenders/keeper raise the opponent's goals slightly. Capped at "
                   "about -18% / +10%.")
    cc = st.columns(2)
    for col, tm, rs in zip(cc, (f["home"], f["away"]), f["rest"]):
        d5 = last_matches(f["b"]["df"], tm, 5)
        if d5.empty:
            continue
        col.markdown(f"**{tm}**: form {' '.join(d5.Res)} | {d5.GF.sum()} scored, {d5.GA.sum()} conceded | "
                     + (f"{rs} days since last match" + (" ⚠️ short rest" if rs <= 3 else "") if rs is not None else ""))
    sub = st.tabs(["🎯 Correct score", "🛡️ Double chance", "⚖️ Handicap", "⏱️ Halves", "⚽ Goals & BTTS",
                   LX("🚩 Corners", "🚩 الركنيات")])
    with sub[0]:
        render_correct_score(f["m"], f["home"], f["away"])
    with sub[1]:
        render_double_chance(f["m"], f["home"], f["away"])
    with sub[2]:
        render_handicap(f["m"], f["lam"], f["mu"], f["home"], f["away"])
    with sub[3]:
        render_halves(f["lam"], f["mu"], f["home"], f["away"], f["b"]["s_share"], f["b"]["s_src"])
    with sub[4]:
        render_goals(f["m"], f["lam"], f["mu"], f["home"], f["away"])
    with sub[5]:
        render_corners_board(f)


def ui_assistant(years, half_life, wx_user, fuse_on):
    st.markdown("#### 🤖 Smart research assistant")
    st.caption("The model knows results, not news. Pick a free option below to look for injuries, suspensions and "
               "line-up news, then adjust the match yourself.")
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    if fxdf.empty:
        st.info("Add matches first.")
        return
    st.markdown("##### ⚡ Automatic injury and absence feed")
    st.caption("Premier League matches need nothing: the official Fantasy Premier League data lists injured, suspended "
               "and doubtful players. For other leagues the app searches the news (Tavily, free) and a free AI (Groq) "
               "reads the headlines and lists the absences. Everything is stored and used automatically in the Analysis tab.")
    _fk = feed_keys()
    st.caption(f"Keys found: Tavily {'✅' if _fk['tavily'] else '❌ not found'} | "
               f"AI ({_fk['prov']}) {'✅' if _fk['llm_key'] else '❌ not found'}. "
               "Keys saved in the secrets.toml file are loaded when the app starts.")
    with st.expander("Free keys for the other leagues (optional)"):
        st.markdown("- **Tavily** (search): sign up at tavily.com, copy the key starting with `tvly-` (1,000 free searches a month)\n"
                    "- **Groq** (free AI): sign up at console.groq.com, API Keys, Create key (no card)")
        st.text_input("Tavily key (kept only in this session)", type="password", key="tav_key")
        st.radio("AI that reads the news", ["Groq (free)", "Claude (paid)"], horizontal=True, key="llm_prov_sel")
        st.text_input("AI key (kept only in this session)", type="password", key="llm_key")
        _g = st.session_state.get("llm_prov_sel", "Groq (free)").startswith("Groq")
        st.text_input("AI model (change if the provider renames it)", "openai/gpt-oss-20b" if _g else "claude-sonnet-5-5",
                      key="llm_model_groq" if _g else "llm_model_claude")
    c = st.columns(2)
    days_ahead = c[0].slider("Matches in the next N days", 1, 10, 3, key="feed_days")
    force = c[1].checkbox("Refresh even if recent", value=False, key="feed_force")
    if st.button("Run the automatic feed now", key="feed_go"):
        with st.spinner("Collecting injury and absence data..."):
            rows_f = run_feed(fxdf, days_ahead, feed_keys(), force)
        if rows_f:
            st.dataframe(pd.DataFrame(rows_f), hide_index=True, use_container_width=True)
        else:
            st.info("No matches in that window.")
    st.markdown("##### Match research and manual adjustment")
    labels = [f"#{r.id} {r.league} | {r.home} vs {r.away} | {r.kickoff}" for r in fxdf.itertuples()]
    sel = st.selectbox("Match", labels, key="ai_match")
    r = list(fxdf.itertuples())[labels.index(sel)]
    f = {"home": r.home, "away": r.away, "league": r.league, "kickoff": r.kickoff}
    store = st.session_state.setdefault("ai_reports", {})
    prov = st.radio("Source", ["📰 Free headlines (no key)", "🔎 Tavily AI search (free key, 1,000 searches a month)",
                               "🧠 Claude with web search (paid key)"], key="ai_prov")

    if prov.startswith("📰"):
        if st.button("Fetch latest headlines for both teams", key=f"news_{r.id}"):
            for tm in (r.home, r.away):
                st.markdown(f"**{tm}**")
                try:
                    items = team_news(tm)
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")
                    continue
                if not items:
                    st.write("No recent headlines about injuries or line-ups.")
                for x in items:
                    st.markdown(f"- {x['date']:%d %b} [{x['title']}]({x['link']})")
            st.caption("Headlines only, from Google News. Open the links to verify before adjusting anything.")
    elif prov.startswith("🔎"):
        st.caption("Free key without a card: sign up at tavily.com, copy the key that starts with `tvly-`. "
                   "Each match uses 3 of your 1,000 monthly credits.")
        tkey = feed_keys()["tavily"]
        if not tkey:
            st.info("No Tavily key found. Save it in secrets.toml (then restart the app) or paste it in the "
                    "'Free keys' box in the AI assistant tab.")
        if st.button("Search the web for this match", key="tav_go"):
            if not tkey:
                st.error("Enter your Tavily key first.")
            else:
                with st.spinner("Searching..."):
                    res, err = tavily_research(tkey, f)
                if err:
                    st.error(err)
                if res:
                    store[("tav", int(r.id))] = res
        for blk in store.get(("tav", int(r.id)), []):
            st.markdown(f"**{blk['topic']}**")
            if blk["answer"]:
                st.write(blk["answer"])
            for x in blk["results"][:4]:
                st.markdown(f"- [{x.get('title', 'source')}]({x.get('url', '')}) {x.get('published_date') or ''}")
        if store.get(("tav", int(r.id))):
            st.caption("Summaries are machine-generated from web pages. Open the sources before relying on them.")
    else:
        key = _secret("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key (kept only in this session)", type="password", key="ai_key")
        c = st.columns(2)
        model_id = c[0].text_input("Model", "claude-sonnet-5-5", key="ai_model")
        max_uses = c[1].slider("Max web searches per match", 1, 8, 4, key="ai_uses")
        if st.button("Search the web for this match", key="ai_go"):
            if not key:
                st.error("Enter an API key first.")
            else:
                with st.spinner("Searching..."):
                    txt, cites, err = ai_research(key, model_id, f, max_uses)
                if err:
                    st.error(err)
                else:
                    store[("cl", int(r.id))] = (txt, cites)
        if ("cl", int(r.id)) in store:
            txt, cites = store[("cl", int(r.id))]
            st.markdown(txt)
            if cites:
                st.markdown("**Sources:** " + " | ".join(f"[{t}]({u})" for u, t in cites.items()))
            st.caption("This text comes from web pages and a language model. Check it before relying on it.")

    st.markdown("**Adjust the model for this match** (your judgement after reading the news)")
    st.caption("Multiplies a team's expected goals. Example: main striker out about 0.92; weakened rotation 0.9. "
               "An adjusted match gets a small risk penalty because it is news-driven.")
    c = st.columns(3)
    ah = c[0].slider(f"{r.home} attack x", 0.70, 1.20, float(r.adj_h or 1.0), 0.01, key=f"adj_h_{r.id}")
    aa = c[1].slider(f"{r.away} attack x", 0.70, 1.20, float(r.adj_a or 1.0), 0.01, key=f"adj_a_{r.id}")
    note = c[2].text_input("Note", r.note or "", key=f"adj_n_{r.id}")
    if st.button("Save adjustment", key=f"adj_save_{r.id}"):
        db_exec("UPDATE fixtures SET adj_h=?, adj_a=?, note=? WHERE id=?", (float(ah), float(aa), note, int(r.id)))
        st.success("Saved. The Analysis tab now uses it.")


def ui_track(years, half_life, wx_user, fuse_on):
    st.markdown("#### Track record")
    pk = db_read("SELECT p.fixture_id, p.sel, p.p, p.verdict, p.grade, p.agree, f.league, f.home, f.away, f.kickoff FROM picks p "
                 "JOIN fixtures f ON f.id = p.fixture_id")
    if pk.empty:
        st.info("Nothing yet. Picks are saved automatically when you open the Analysis tab for upcoming matches.")
        return
    db_exec("CREATE TABLE IF NOT EXISTS results_manual(fixture_id INTEGER PRIMARY KEY, hg INTEGER, ag INTEGER)")
    manual = {int(x.fixture_id): (int(x.hg), int(x.ag)) for x in db_read("SELECT * FROM results_manual").itertuples()}
    espn_cache = {}
    rows = []
    for r in pk.itertuples():
        b = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        res = manual.get(int(r.fixture_id))
        if res is None and r.kickoff:
            kd = pd.Timestamp(r.kickoff)
            if b is not None:
                d = b["df"]
                tms = sorted(set(d.HomeTeam) | set(d.AwayTeam))
                hn, an = match_team(r.home, tms) or r.home, match_team(r.away, tms) or r.away
                m = d[(d.HomeTeam == hn) & (d.AwayTeam == an) &
                      (d.Date >= kd - pd.Timedelta(days=2)) & (d.Date <= kd + pd.Timedelta(days=3))]
                if len(m):
                    res = (int(m.iloc[0].FTHG), int(m.iloc[0].FTAG))
            slug = espn_slug(r.league)
            if res is None and slug and kd <= pd.Timestamp.today():
                if slug not in espn_cache:
                    espn_cache[slug] = espn_recent_results(slug, 60)
                er = espn_cache[slug]
                if len(er):
                    hn2 = match_team(r.home, sorted(set(er.HomeTeam)))
                    an2 = match_team(r.away, sorted(set(er.AwayTeam)))
                    m = er[(er.HomeTeam == hn2) & (er.AwayTeam == an2) &
                           (er.Date >= kd.normalize() - pd.Timedelta(days=2)) & (er.Date <= kd.normalize() + pd.Timedelta(days=3))]
                    if len(m):
                        res = (int(m.iloc[0].FTHG), int(m.iloc[0].FTAG))
        rows.append({"Date": r.kickoff, "Match": f"{r.home} vs {r.away}", "Pick": r.sel, "Model %": round(r.p * 100, 1),
                     "Result": "pending" if res is None else f"{res[0]}-{res[1]}",
                     "Hit": None if res is None else sel_hit(r.sel, *res), "_p": r.p,
                     "_v": r.verdict, "_g": r.grade, "_a": r.agree})
    t = pd.DataFrame(rows).sort_values("Date", ascending=False)
    g = t[t.Hit.notna()].copy()
    pend = pk.merge(t[t.Result == "pending"][["Date", "Match"]].assign(k=1), how="inner",
                    left_on=["kickoff", pk.home + " vs " + pk.away], right_on=["Date", "Match"]).drop_duplicates("fixture_id")
    pend = pend[pd.to_datetime(pend.kickoff) <= pd.Timestamp.today()]
    with st.expander(LX(f"Pending matches already played: enter the result yourself ({len(pend)})",
                        f"مباريات معلقة لُعبت بالفعل: أدخل نتيجتها بنفسك ({len(pend)})")):
        st.caption(LX("Results normally appear by themselves (football-data.co.uk is about a day late; ESPN is used "
                      "as a backup). If a match stays pending, enter its score here and it is graded at once. "
                      "Team names that differ between the source and your list are matched automatically.",
                      "النتائج بتظهر تلقائيًا عادةً (football-data.co.uk بيتأخر نحو يوم، وESPN احتياطي). لو مباراة "
                      "فضلت معلقة اكتب نتيجتها هنا وتتحسب فورًا. اختلاف أسماء الفرق بين المصدر وقائمتك بيتعالج تلقائيًا."))
        if st.button(LX("Refresh data now", "حدّث البيانات الآن"), key="trk_refresh"):
            st.cache_data.clear()
            st.rerun()
        for x in pend.itertuples():
            cc = st.columns([4, 1, 1, 1])
            cc[0].write(f"{x.home} vs {x.away} ({x.kickoff})")
            hg = cc[1].number_input("H", 0, 20, 0, key=f"mr_h_{x.fixture_id}", label_visibility="collapsed")
            ag = cc[2].number_input("A", 0, 20, 0, key=f"mr_a_{x.fixture_id}", label_visibility="collapsed")
            if cc[3].button(LX("Save", "حفظ"), key=f"mr_s_{x.fixture_id}"):
                db_exec("INSERT OR REPLACE INTO results_manual VALUES(?,?,?)", (int(x.fixture_id), int(hg), int(ag)))
                st.rerun()
    if g.empty:
        st.info("Matches are still pending. Results appear automatically once the data source publishes them "
                "(usually within a day).")
    else:
        g["Hit"] = g["Hit"].astype(bool)
        c = st.columns(3)
        c[0].metric("Graded picks", len(g))
        c[1].metric("Hit rate", f"{g.Hit.mean() * 100:.1f}%")
        c[2].metric("Average model probability", f"{g._p.mean() * 100:.1f}%")
        g["Bucket"] = pd.cut(g._p, [0, .5, .6, .7, .8, 1], labels=["<50", "50-60", "60-70", "70-80", "80+"])
        cal = g.groupby("Bucket", observed=True).agg(Picks=("Hit", "size"), Hit_rate=("Hit", "mean"), Model=("_p", "mean"))
        cal["Hit_rate"] = (cal["Hit_rate"] * 100).round(1)
        cal["Model"] = (cal["Model"] * 100).round(1)
        st.markdown("**Calibration** (hit rate should be close to the model percentage; small samples are noisy)")
        st.dataframe(cal.rename(columns={"Hit_rate": "Hit rate %", "Model": "Model %"}), use_container_width=True)
    if not g.empty and g["_v"].notna().any():
        parts = []
        for col, nm in (("_v", "Verdict"), ("_g", "Grade"), ("_a", "Sources")):
            h = g[g[col].notna()].groupby(col).agg(Picks=("Hit", "size"), Hit=("Hit", "mean"), Model=("_p", "mean"))
            for k, x in h.iterrows():
                parts.append({"Group": nm, "Value": k, "Picks": int(x.Picks), "Hit rate %": round(x.Hit * 100, 1),
                              "Model %": round(x.Model * 100, 1)})
        if parts:
            st.markdown(LX("**Real hit rate of each class** (the test of whether 'Strong', grade A or 'all agree' "
                           "really means safer; needs a few dozen graded picks to mean anything)",
                           "**نسبة النجاح الفعلية لكل فئة** (اختبار هل 'قوي' أو التصنيف A أو 'كل المصادر متفقة' "
                           "تعني فعلًا أمانًا أكثر؛ تحتاج بضع عشرات من الاختيارات المحسوبة لتعني شيئًا)"))
            st.dataframe(pd.DataFrame(parts), hide_index=True, use_container_width=True)
    st.dataframe(t.drop(columns=["_p", "_v", "_g", "_a"]), hide_index=True, use_container_width=True)


def ui_data(years, half_life, wx_user, fuse_on):
    st.markdown("#### Data sources and backup")
    with st.expander("Connection check (press if a league fails to load)"):
        if st.button("Test all sources now", key="conn_btn"):
            HOST_DEAD.clear()
            tests = [("football-data.co.uk (Serie A, this season)", f"{FD}/mmz4281/{_season_tag()}/I1.csv"),
                     ("football-data.co.uk (Serie A, last season)", f"{FD}/mmz4281/{_season_tag(-1)}/I1.csv"),
                     ("football-data.co.uk (fixtures)", f"{FD}/fixtures.csv"),
                     ("openfootball (GitHub)", f"{OFB}/{_season_dash()}/it.1.json"),
                     ("International results (GitHub)", INTL_URL),
                     ("ESPN (Saudi league)", f"{ESPN}/ksa.1/scoreboard"),
                     ("ESPN (Europa League)", f"{ESPN}/uefa.europa/scoreboard"),
                     ("ESPN (CAF Champions League)", f"{ESPN}/caf.champions/scoreboard"),
                     ("Understat (EPL)", f"{UNDERSTAT_URL}/EPL/{_season_tag()[:2] and (pd.Timestamp.today().year if pd.Timestamp.today().month >= 7 else pd.Timestamp.today().year - 1)}",
                      {**UA, "X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/league/EPL"}),
                     ("ClubElo", "http://api.clubelo.com/" + str(dt.date.today())),
                     ("FPL official API", FPL_URL),
                     ("Google News RSS", "https://news.google.com/rss/search?q=football&hl=en")]
            rows = []
            for item in tests:
                name, url = item[0], item[1]
                t0 = time.time()
                raw, err = fetch_bytes(url, tries=1, timeout=15, headers=item[2] if len(item) > 2 else None)
                rows.append({"Source": name, "Result": "OK" if raw else "FAILED", "Detail": err or f"{len(raw)//1024} KB",
                             "Seconds": round(time.time() - t0, 1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("A 404 for 'this season' is normal in the first weeks of a new season; the app then uses "
                       "the previous seasons. If everything FAILED, the problem is your internet / firewall / antivirus.")
    leagues = st.multiselect("Show status for leagues", board_leagues(), default=[], key="dt_leagues")
    rows = []
    for lg in leagues:
        b = league_bundle(lg, years, half_life, wx_user, fuse_on)
        if b is None:
            rows.append({"League": lg, "Status": "no data"})
            continue
        d = b["df"]
        rows.append({"League": lg, "Matches": len(d), "From": d.Date.min().date(), "To": d.Date.max().date(),
                     "Teams": len(b["teams"]), "xG source": b["xg"], "Added by fusion": b["added"],
                     "Upcoming (2nd source)": len(b["fut"]),
                     "Understat xG matches": b.get("us_n", 0),
                     "Corners data": f"{b['corners']['n']} matches" if b.get("corners") else "no"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Sources: football-data.co.uk (results, shots, odds), openfootball on GitHub (second results feed, "
               "fused by team-name matching), martj42 international results (national teams), ClubElo (external ratings), "
               "FPL (Premier League injuries).")
    st.caption(LX("New in v13: ESPN (free, no key) gives results, fixtures and line-ups for the Egyptian and Saudi "
                  "leagues, the UEFA and CAF cups and the Nations / Africa Cup fixtures; Understat gives real xG for "
                  "the top-5 leagues and Russia.",
                  "الجديد في v13: ESPN (مجاني بلا مفتاح) يعطي النتائج والمباريات القادمة والتشكيلات للدوري المصري "
                  "والسعودي وبطولات UEFA وCAF ومباريات دوري الأمم وكأس أفريقيا؛ وUnderstat يعطي xG حقيقيًا لأكبر 5 "
                  "دوريات وروسيا."))
    st.markdown("**Auto-tune a league**")
    st.caption("Tries a few settings (recency half-life and xG weight) and keeps the one that predicted the most "
               "recent matches best. It only switches away from the default when the gain is clear, so it does not "
               "chase noise. Takes about 30-60 seconds per league.")
    tl = st.multiselect("Leagues to tune", board_leagues(), default=[], key="tune_leagues")
    if st.button("Tune selected leagues", key="tune_btn") and tl:
        for lg in tl:
            try:
                b = _league_bundle(lg, years, 300, 0.5, fuse_on)
            except RuntimeError:
                b = None
            if b is None:
                st.warning(f"{lg}: no data")
                continue
            d = b["df"]
            has_xg = not b["xg"].startswith("not available")
            grid = [(h, w) for h in (150, 300, 600) for w in ((0.0, 0.5) if has_xg else (0.0,))]
            res = []
            with st.spinner(f"Tuning {lg}..."):
                for h, w in grid:
                    ll, n = wf_logloss(d, float(np.log(2) / h), w)
                    res.append({"Half-life": h, "xG weight": w, "Log loss": round(ll, 4), "Matches": n})
            t = pd.DataFrame(res)
            dflt = t[(t["Half-life"] == 300) & (t["xG weight"] == (0.5 if has_xg else 0.0))]
            base = float(dflt["Log loss"].iloc[0]) if len(dflt) else float(t["Log loss"].max())
            best = t.loc[t["Log loss"].idxmin()]
            switched = base - float(best["Log loss"]) >= 0.002
            use = best if switched or not len(dflt) else dflt.iloc[0]
            db_exec("INSERT OR REPLACE INTO league_params VALUES(?,?,?,?,?,?)",
                    (lg, int(use["Half-life"]), float(use["xG weight"]), float(use["Log loss"]), base,
                     dt.datetime.now().isoformat()))
            st.markdown(f"**{lg}** -> half-life {int(use['Half-life'])} days, xG weight {float(use['xG weight'])}"
                        + (" (changed)" if switched else " (default kept)"))
            st.dataframe(t, hide_index=True, use_container_width=True)
    st.markdown("**Calibrate confidence (recommended)**")
    st.caption("Tests the model on each league's latest ~300 matches, one forecast at a time without peeking at the "
               "result, then keeps the Elo weight and confidence correction that predicted best. It also measures how "
               "often the model's 40%, 50%, 60%... forecasts really happen. Do this after 'Tune' (if you use it).")
    cl = st.multiselect("Leagues to calibrate", board_leagues(), default=[], key="cal_leagues")
    if st.button("Calibrate selected leagues", key="cal_btn") and cl:
        for lg in cl:
            b0 = league_bundle(lg, years, half_life, wx_user, fuse_on)
            if b0 is None:
                st.warning(f"{lg}: no data")
                continue
            with st.spinner(f"Calibrating {lg}..."):
                res, info = run_calibration(lg, b0, *eff_params(lg, half_life, wx_user))
            st.markdown(f"**{lg}**: Elo weight {res['w_elo'] * 100:.0f}%, temperature {res['temp']:.2f}, "
                        f"{res['n']} matches tested")
            render_reliability(lg, calib_load(lg))
    cal_all = db_read("SELECT league, w_elo, w_slow, temp, pa, pb, ba, bb, loss, base_loss, n, tuned_at FROM league_calib")
    if len(cal_all):
        st.dataframe(cal_all, hide_index=True, use_container_width=True)
        if st.button("Reset all calibration", key="cal_reset"):
            db_exec("DELETE FROM league_calib")
            st.success("Calibration cleared.")
    tuned = db_read("SELECT league, half_life, wx, loss, base_loss, tuned_at FROM league_params")
    if len(tuned):
        st.dataframe(tuned, hide_index=True, use_container_width=True)
        if st.button("Reset all tuning", key="tune_reset"):
            db_exec("DELETE FROM league_params")
            st.success("Tuning cleared.")
    fx = db_read("SELECT * FROM fixtures ORDER BY id")
    st.download_button("⬇️ Backup match list (CSV)", fx.to_csv(index=False), "board_backup.csv", "text/csv", key="bk_dl")
    up = st.file_uploader("Restore match list from a backup CSV", type="csv", key="bk_up")
    if up is not None and st.button("Restore", key="bk_btn"):
        try:
            d = pd.read_csv(up)
            n = 0
            for r in d.itertuples():
                n += db_add_fixture(r.league, r.home, r.away, r.kickoff, getattr(r, "neutral", 0),
                                    getattr(r, "o1", None), getattr(r, "ox", None), getattr(r, "o2", None))
            st.success(f"Restored {n} match(es).")
        except Exception as e:
            st.error(f"Could not read the backup: {e}")


def render_board(years, half_life, wx_user, fuse_on):
    st.markdown("### 🗂️ Match board: many leagues, one analysis")
    st.caption("Add matches from any league, the engine trains each league's model, predicts every match, ranks the "
               "strongest picks and shows how risky a combined slip really is. Analysis only, no betting on this site.")
    try:
        _con().close()
    except sqlite3.Error as e:
        st.error(f"Database problem: {e}")
        return
    tabs = st.tabs(["➕ Matches", "📊 Analysis", LX("🚩 Corners", "🚩 الركنيات"), "🤖 AI assistant",
                    LX("🔎 Line-ups", "🔎 التشكيلات"), "📈 Track record", "🧰 Data & backup"])
    with tabs[0]:
        ui_add_fixtures(years, half_life, wx_user, fuse_on)
        ui_odds_feed()
    with tabs[1]:
        ui_analysis(years, half_life, wx_user, fuse_on)
    with tabs[2]:
        ui_corners(years, half_life, wx_user, fuse_on)
    with tabs[3]:
        ui_assistant(years, half_life, wx_user, fuse_on)
    with tabs[4]:
        ui_lineups(years, half_life, wx_user, fuse_on)
    with tabs[5]:
        ui_track(years, half_life, wx_user, fuse_on)
    with tabs[6]:
        ui_data(years, half_life, wx_user, fuse_on)


# ======================= UI =======================
st.set_page_config(page_title="Pro Football AI Engine", layout="wide")
_lsel = st.sidebar.radio("اللغة / Language", ["العربية", "English"], key="lang_sel", horizontal=True)
_real_st.session_state["lang"] = "en" if _lsel == "English" else "ar"
apply_language_style()
st.title("⚽ Pro Football AI Engine")
st.sidebar.caption(f"Version {APP_VERSION}")
st.caption("Time-weighted Dixon-Coles model trained on real results, shots and xG. Analysis only, no betting.")

sb = st.sidebar
page = sb.radio("Page", ["🗂️ Match board (multi-league)", "🎯 Single match deep-dive", "👥 Players"], key="page")
sb.header("Settings")
sb.checkbox(LX("Focus on the major leagues and cups", "التركيز على الدوريات والبطولات الكبرى"), True, key="focus_on")
years = sb.slider("Years of history", 1, 6, 3)
half_life = sb.slider("Recency half-life (days)", 60, 900, 300, 30)
wx_user = sb.slider("xG / shots weight (0 = goals only)", 0.0, 1.0, 0.5, 0.05)
fuse_on = sb.checkbox("Fuse a second data source (openfootball)", value=True)

if page.startswith("🗂️"):
    render_board(years, half_life, wx_user, fuse_on)
    st.stop()
if page.startswith("👥"):
    render_players_tab()
    st.stop()

src = sb.selectbox("Competition", source_options())
kind, code = SOURCES[src]

if kind == "upload":
    sb.download_button("Download CSV template", TEMPLATE, "template.csv", "text/csv")
    up = sb.file_uploader("Upload results CSV", type="csv")
    if up is None:
        st.info("No free automatic data feed for this competition. Upload a CSV with the columns "
                "Date, HomeTeam, AwayTeam, FTHG, FTAG (download the template from the sidebar). "
                "Optional columns that improve accuracy: HS, AS, HST, AST (shots) or HxG, AxG (real xG).")
        st.stop()
    df = clean(normalize(pd.read_csv(up, encoding="latin-1")))
else:
    if kind == "fdorg" and not org_key():
        sb.text_input("football-data.org free key", type="password", key="fdorg_key")
        st.info("Paste your free football-data.org key in the sidebar (register at football-data.org/client/register).")
        st.stop()
    df = load_data(kind, code, years, org_key() if kind == "fdorg" else "")
    if fuse_on and kind == "fd" and not df.empty:
        _extra, _ = load_openfootball(code, years)
        if not _extra.empty:
            df, _added = fuse(df, [_extra])
            sb.caption(f"Fusion: +{_added} matches from the second source")
    if code in BACKUP_USED:
        sb.warning("Main source was unreachable: using the backup source (openfootball).")

if df.empty:
    st.error("Could not load match data (check the file columns or the host's internet connection).")
    st.stop()
if len(df) < 60:
    st.warning("Very little data: predictions will be unreliable.")

df, xg_src = add_xg(df)
has_xg = not xg_src.startswith("not available")
wx = wx_user if has_xg else 0.0
s_share, s_src = half_share(df)
sb.caption(f"xG source: {xg_src}")

xi = float(np.log(2) / half_life)
model = get_model(df, xi, wx)

if kind == "intl":
    pool_df = df[df.Tournament.str.contains(code, case=False, na=False)]
    pool = set(pool_df.HomeTeam) | set(pool_df.AwayTeam)
    teams = [t for t in eligible_teams(df, model, 6, 730) if t in pool] or eligible_teams(df, model, 6, 730)
else:
    teams = sorted(current_teams(df) & set(model["teams"]))
if len(teams) < 2:
    st.error("Not enough teams found in the data.")
    st.stop()

sb.markdown("---")
home = sb.selectbox("Home team", teams, index=0)
away = sb.selectbox("Away team", teams, index=1)
neutral = sb.checkbox("Neutral venue (no home advantage)", value=(kind == "intl"), key=f"neu_{src}")
sb.markdown("**Bookmaker odds (optional)**")
o1 = sb.number_input("Home win odds", 0.0, 100.0, 0.0, 0.01)
ox = sb.number_input("Draw odds", 0.0, 100.0, 0.0, 0.01)
o2 = sb.number_input("Away win odds", 0.0, 100.0, 0.0, 0.01)

st.caption(f"{len(df)} matches | {df.Date.min():%d %b %Y} → {df.Date.max():%d %b %Y} | "
           f"home advantage ×{np.exp(model['home']):.2f} | xG: {xg_src}")
if home == away:
    st.warning("Choose two different teams.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📋 Team analysis", "🧪 Backtest", "📈 Ratings"])

with tab1:
    lam, mu = lambdas(model, home, away, neutral)
    m = score_matrix(lam, mu, model["rho"])
    ph, pdw, pa = outcome_probs(m)
    st.markdown(f"### 🏟️ {home} vs {away}" + (" (neutral venue)" if neutral else ""))
    c = st.columns(2)
    c[0].metric(f"{home} expected goals", f"{lam:.2f}")
    c[1].metric(f"{away} expected goals", f"{mu:.2f}")
    c = st.columns(3)
    c[0].metric(f"{home} win", f"{ph * 100:.1f}%", f"fair odds {1 / ph:.2f}", delta_color="off")
    c[1].metric("Draw", f"{pdw * 100:.1f}%", f"fair odds {1 / pdw:.2f}", delta_color="off")
    c[2].metric(f"{away} win", f"{pa * 100:.1f}%", f"fair odds {1 / pa:.2f}", delta_color="off")
    if o1 > 1 and ox > 1 and o2 > 1:
        inv = 1 / np.array([o1, ox, o2])
        mk = inv / inv.sum()
        mp = np.array([ph, pdw, pa])
        st.markdown("**Model vs market**")
        st.dataframe(pd.DataFrame({"Outcome": [f"{home} win", "Draw", f"{away} win"],
                                   "Model %": (mp * 100).round(1), "Market %": (mk * 100).round(1),
                                   "Edge (pp)": ((mp - mk) * 100).round(1)}),
                     hide_index=True, use_container_width=True)
        st.caption("A positive edge means the model rates the outcome higher than the bookmaker. "
                   "Double-check injuries and line-ups before trusting it.")
    sub = st.tabs(["🎯 Correct score", "🛡️ Double chance", "⚖️ Handicap", "⏱️ Halves", "⚽ Goals & BTTS",
                   LX("🚩 Corners", "🚩 الركنيات")])
    with sub[0]:
        render_correct_score(m, home, away)
    with sub[1]:
        render_double_chance(m, home, away)
    with sub[2]:
        render_handicap(m, lam, mu, home, away)
    with sub[3]:
        render_halves(lam, mu, home, away, s_share, s_src)
    with sub[4]:
        render_goals(m, lam, mu, home, away)
    with sub[5]:
        render_corners_single(df, xi, home, away)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        team_panel(df, home, has_xg)
    with c2:
        team_panel(df, away, has_xg)
    render_style(df, home, away)
    st.markdown("**Head to head (last 5)**")
    h2h = df[((df.HomeTeam == home) & (df.AwayTeam == away)) |
             ((df.HomeTeam == away) & (df.AwayTeam == home))].tail(5)
    if h2h.empty:
        st.write("No meetings in the loaded data.")
    else:
        h2h = h2h.assign(Score=h2h.FTHG.astype(str) + "-" + h2h.FTAG.astype(str),
                         Date=h2h.Date.dt.strftime("%d/%m/%y"))
        st.dataframe(h2h[["Date", "HomeTeam", "Score", "AwayTeam"]], hide_index=True,
                     use_container_width=True)

with tab3:
    st.write("Walk-forward test: the model is retrained on the past only, then predicts matches "
             "it has never seen. With xG/shots on, it is compared against the goals-only version.")
    n_test = st.slider("Matches to test on", 60, 300, 120, 10)
    if st.button("Run backtest"):
        with st.spinner("Backtesting..."):
            res = backtest(df, xi, n_test, wx)
        if res is None:
            st.warning("Not enough data.")
        else:
            st.dataframe(pd.DataFrame(res).T.round(3), use_container_width=True)
            st.caption("Random guessing scores Brier ≈ 0.667. Lower Brier / log loss is better. "
                       "Bookmakers are usually the toughest benchmark.")

with tab4:
    rows = []
    for t in teams:
        a, d = model["att"][model["idx"][t]], model["dfn"][model["idx"][t]]
        rows.append({"Team": t, "Attack (goals vs avg)": round(float(np.exp(a)), 2),
                     "Defence (conceded vs avg, lower=better)": round(float(np.exp(d)), 2),
                     "Overall": round(float(a - d), 2)})
    st.dataframe(pd.DataFrame(rows).sort_values("Overall", ascending=False),
                 hide_index=True, use_container_width=True)

st.caption("Estimates only. The model does not know injuries, line-ups or motivation.")
# END OF FILE
