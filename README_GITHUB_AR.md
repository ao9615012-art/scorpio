# رابط دائم مجاني لـ SCORPIO عبر GitHub (5 خطوات، مرة واحدة فقط)

1. أنشئ حساباً مجانياً على https://github.com (إن لم يكن لديك).
2. اضغط **+** (أعلى اليمين) ← **New repository** ← الاسم: `scorpio` ← اختر **Public** ← **Create repository**.
3. في صفحة المستودع الجديدة اضغط **uploading an existing file**، ثم اسحب **كل محتويات** مجلد `scorpio_app`
   (بما فيها المجلد المخفي `.github` ومجلد `data`) وأفلتها، ثم اضغط **Commit changes**.
   *(إن لم يظهر المجلد `.github` عند السحب: فعّل إظهار الملفات المخفية في جهازك، أو ارفع ZIP وفكّه من GitHub Desktop.)*
4. اذهب إلى **Settings ← Pages ← Build and deployment ← Source** واختر **GitHub Actions**.
5. اذهب إلى تبويب **Actions** ← اختر **SCORPIO daily update + publish** ← **Run workflow**.
   بعد 3–5 دقائق يظهر الرابط الدائم:

   **https://اسم-حسابك.github.io/scorpio/**

بعدها يتحدث الرابط تلقائياً كل يوم الساعة 07:30 UTC ببيانات ECMWF الجديدة، إلى الأبد وبدون أي تدخل.
