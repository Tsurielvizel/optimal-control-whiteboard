#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
סקריפט לשיפור איכות שמע של הרצאות וידאו (MP4)
==================================================
הסקריפט מבצע אופטימיזציה אקוסטית מתקדמת:
1. ביטול תדרי תהודת אולם והדהוד כיתה (De-Rooming)
2. חיתוך רעשי עומק ובוציות (<120Hz)
3. הגברת בהירות ומובנות דיבור עברי (Speech Clarity Boost ב-2.8kHz ו-4.8kHz)
4. הגנה מפני דיסטורשן (Peak Limiter)
5. החלפת פס הקול בתוך ה-MP4 ללא פגיעה באיכות הוידאו (Stream Copy 1080p) וסנכרון מלא.

שימוש עתידי:
    python enhance_lecture_audio.py "שם_הקובץ.mp4"
או להרצה על כל קבצי ה-MP4 בתיקייה:
    python enhance_lecture_audio.py
"""

import os
import sys
import subprocess
import time

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

# שרשרת הפילטרים האקוסטית המנצחת (Clarity De-Room)
AUDIO_FILTER = (
    "highpass=f=120,"
    "equalizer=f=240:width_type=q:w=2.2:g=-7,"
    "equalizer=f=360:width_type=q:w=2.2:g=-6.5,"
    "equalizer=f=550:width_type=q:w=1.8:g=-4,"
    "equalizer=f=850:width_type=q:w=1.8:g=-3.5,"
    "equalizer=f=2800:width_type=o:w=1.2:g=6,"
    "equalizer=f=4800:width_type=o:w=1.5:g=3.5,"
    "alimiter=limit=-2.0dB"
)

def enhance_video(input_path: str, replace_original: bool = True) -> bool:
    """מעבד קובץ וידאו ומחליף את פס השמע באיכות מאופטמת."""
    if not os.path.isfile(input_path):
        print(f"[-] שגיאה: הקובץ לא נמצא: {input_path}")
        return False

    abs_in = os.path.abspath(input_path)
    dir_name, file_name = os.path.split(abs_in)
    name_part, ext = os.path.splitext(file_name)
    temp_out = os.path.join(dir_name, f"{name_part}_optimized_temp{ext}")

    print(f"\n[+] מתחיל עיבוד: {file_name}")
    start_time = time.time()

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", abs_in,
        "-af", AUDIO_FILTER,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        temp_out
    ]

    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            print(f"[-] שגיאה בעיבוד {file_name}:")
            print(process.stderr.decode("utf-8", errors="ignore")[-500:])
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return False

        # בדיקת תקינות הקובץ שנוצר
        if not os.path.exists(temp_out) or os.path.getsize(temp_out) < 1000:
            print(f"[-] שגיאה: קובץ הפלט אינו תקין.")
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return False

        elapsed = time.time() - start_time
        print(f"[v] העיבוד הושלם בהצלחה תוך {elapsed:.1f} שניות!")

        if replace_original:
            # החלפה בטוחה של המקור
            backup_temp = os.path.join(dir_name, f"{name_part}_old_backup{ext}")
            os.replace(abs_in, backup_temp)
            os.replace(temp_out, abs_in)
            os.remove(backup_temp)
            print(f"[v] קובץ הוידאו עודכן עם השמע המאופטם: {file_name}")
        else:
            final_out = os.path.join(dir_name, f"{name_part}_משופר{ext}")
            os.replace(temp_out, final_out)
            print(f"[v] נוצר קובץ מאופטם חדש: {os.path.basename(final_out)}")

        return True

    except Exception as e:
        print(f"[-] חריגה במהלך העיבוד: {e}")
        if os.path.exists(temp_out):
            os.remove(temp_out)
        return False

def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        # עיבוד כל קבצי ה-MP4 בתיקייה שאינם קבצי temp/sample
        cwd = os.getcwd()
        targets = [
            f for f in os.listdir(cwd)
            if f.lower().endswith(".mp4") and not f.startswith("temp_") and not f.startswith("sample_")
        ]

    if not targets:
        print("[!] לא נמצאו קבצי MP4 לעיבוד.")
        return

    print(f"[i] נמצאו {len(targets)} קבצי MP4 לעיבוד.")
    for target in targets:
        enhance_video(target, replace_original=True)

    print("\n[V] כל ההרצאות עובדו בהצלחה!")

if __name__ == "__main__":
    main()

