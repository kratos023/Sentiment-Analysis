import shutil

paths = [
    "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\mlruns\\0",
    "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\mlruns\\900556493342397191",
    "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\mlruns\\900556493342397191\\addff342e70340458a4c9cce10815b8a\\outputs\\m-8e27d08f709c43429f3e108202ecf3f5",
    "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\mlruns\\900556493342397191\\models\\m-8e27d08f709c43429f3e108202ecf3f5",
]

for p in paths:
    try:
        shutil.rmtree(p)
        print(f"✅ Deleted: {p}")
    except Exception as e:
        print(f"❌ Could not delete {p}: {e}")
