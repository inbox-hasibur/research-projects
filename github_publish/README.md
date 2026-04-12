# GitHub publish mirror (সিক্রেট ছাড়া কপি)

এই ফোল্ডারটা রিপোজিটরিতে যাওয়ার জন্য। তোমার আসল কাজ **`Project_*`** ডিরেক্টরিতে থাকে; সেখানে `.env`, ডেটা, আউটপুট, ক্যাগল টোকেন ইত্যাদি থাকতে পারে — সেগুলো এখানে **কপি হয় না**।

## সিঙ্ক করো (পুশের আগে)

রিপো রুট থেকে:

```bash
python tools/sync_github_publish.py
```

তারপর শুধু যা যাবে:

```bash
git add github_publish tools/sync_github_publish.py .gitignore
git status
git commit -m "Sync published scripts"
git push
```

## নতুন প্রজেক্ট / নতুন `.py`

রিপো রুটে ফোল্ডার নাম **`Project_...`** দিয়ে শুরু করো (যেমন `Project_MyModel`)। তার ভেতরের সব `.py` (কিছু স্কিপ ডিরেক্টরি বাদে) পরের সিঙ্কে **অটো** `github_publish/` এ কপি হবে — আলাদা তালিকা আপডেট লাগে না। পুশের আগে শুধু `python tools/sync_github_publish.py` চালাও। চাইলে Cursor-কে বলো: সিঙ্ক চালিয়ে কমিট/পুশ করে দাও।

## এনভায়রনমেন্ট (GitHub-এ কমিট করো না)

- Kaggle পাথ, `OMNICROPS_OUT`, API কি — **`.env` বা নোটবুক সেল** দিয়ে সেট করো।  
- প্রকাশিত স্ক্রিপ্টে যদি `os.environ.get("VAR", "default_kaggle_path")` প্যাটার্ন থাকে, লোকালে `VAR` সেট করলেই হবে।

## `tools/` ফোল্ডার

সিঙ্ক স্ক্রিপ্ট রিপোতে থাকে যাতে ক্লোন করার পরেও একই কমান্ড দিয়ে (নিজের মেশিনে) আয়না আপডেট করা যায়। ক্যাগলে সাধারণত শুধু `github_publish/` এর ভেতরের স্ক্রিপ্ট কপি করে চালাও।
