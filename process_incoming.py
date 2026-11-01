#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
סקריפט עיבוד אוטומטי להרצאות חדשות בתיקיית incoming_lectures.
מופעל על ידי הסוכן כשנאמר לו "בצע את INSTRUCTIONS.md".
"""

import os
import sys
import glob
import subprocess
import shutil

INCOMING_DIR = "incoming_lectures"
TEMP_DIR = "extracted_board_temp"

def get_incoming_videos():
    patterns = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
    videos = []
    for pat in patterns:
        videos.extend(glob.glob(os.path.join(INCOMING_DIR, pat)))
    return sorted(videos)

def main():
    videos = get_incoming_videos()
    if not videos:
        print(f"[-] לא נמצאו קובצי וידאו חדשים בתיקייה '{INCOMING_DIR}'.")
        print(f"[i] שים קובץ הרצאה (למשל 'הרצאה 3.mp4') בתוך '{INCOMING_DIR}' והפעל שוב.")
        return 0

    print(f"[+] נמצאו {len(videos)} קובצי וידאו לעיבוד: {[os.path.basename(v) for v in videos]}")

    for vid in videos:
        base_name = os.path.splitext(os.path.basename(vid))[0]
        print(f"\n=======================================================")
        print(f" עיבוד וידאו: {base_name}")
        print(f"=======================================================")

        # 1. הרצת חילוץ לוחות בשיטת temporal_deocclusion
        cmd = [
            sys.executable,
            "whiteboard_extractor.py",
            vid,
            "--interval", "6",
            "--output-dir", TEMP_DIR
        ]
        print(f"[1/3] מחלץ לוחות נקיים באמצעות whiteboard_extractor.py...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[!] שגיאה במהלך חילוץ הווידאו {vid}")
            continue

        manifest_path = os.path.join(TEMP_DIR, "keyframes_manifest.json")
        if not os.path.exists(manifest_path):
            print(f"[!] לא נמצא קובץ keyframes_manifest.json ב-{TEMP_DIR}")
            continue

        print(f"[2/3] החילוץ הוויזואלי הושלם בהצלחה! הפריימים שמורים ב-{TEMP_DIR}.")
        print(f"[3/3] כעת הסוכן יסקור את הפריימים ויתמלל את 'סיכום_לוח_{base_name}.md'.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
