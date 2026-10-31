#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
סקריפט השוואת שיטות לחילוץ כיתוב מלוח הרצאות וידאו (Benchmark & Comparison)
=============================================================================
מריץ את שלוש שיטות עיבוד התמונה ב-whiteboard_extractor.py:
1. scene_baseline       : זיהוי שינויי סצנה + סינון חדות (ללא זיהוי/הסרת מרצה)
2. yolo_best_frame      : בחירת הפריים הטוב ביותר מונחית YOLOv8-seg וצפיפות דיו + CLAHE
3. temporal_deocclusion : ייצוב מצלמה (Phase Correlation) + שחזור רקע טמפורלי דו-שכבתי (YOLOv8-seg)
                          + נרמול תאורה גאוסיאני + סינון דומיננטיות

הסקריפט מחשב מדדים כמותיים אובייקטיביים לכל שיטה, מייצר תמונת השוואה ויזואלית (Side-by-Side),
ומשווה את שלמות המידע המחולץ מול קובץ הייחוס `סיכום_לוח_המרצה.md`.

שימוש:
    python compare_methods.py
    python compare_methods.py "בקרה אופטימלית - שעור 1 _ דף הבית.mp4" --start 00:00 --end 30:00
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import cv2
import numpy as np

from whiteboard_extractor import (
    process_video_segment,
    parse_timestamp,
    format_timestamp,
    KeyframeRecord,
)


def summarize_method_metrics(records: List[KeyframeRecord]) -> Dict[str, float]:
    """מחשב ממוצעים ומדדי שיא עבור רשימת פריימי-מפתח של שיטה."""
    if not records:
        return {
            "num_keyframes": 0,
            "mean_occlusion_pct": 0.0,
            "min_occlusion_pct": 0.0,
            "mean_cnr": 0.0,
            "max_cnr": 0.0,
            "mean_ink_pct": 0.0,
            "mean_sharpness": 0.0,
        }
    return {
        "num_keyframes": len(records),
        "mean_occlusion_pct": round(float(np.mean([r.lecturer_occlusion_pct for r in records])), 2),
        "min_occlusion_pct": round(float(np.min([r.lecturer_occlusion_pct for r in records])), 2),
        "mean_cnr": round(float(np.mean([r.contrast_cnr for r in records])), 2),
        "max_cnr": round(float(np.max([r.contrast_cnr for r in records])), 2),
        "mean_ink_pct": round(float(np.mean([r.ink_coverage_pct for r in records])), 3),
        "mean_sharpness": round(float(np.mean([r.sharpness_score for r in records])), 1),
    }


def create_side_by_side_comparison(
    results: Dict[str, List[KeyframeRecord]],
    target_sec: float,
    output_image_path: str,
) -> bool:
    """
    מייצר תמונת השוואה ויזואלית (1x3) בין שלוש השיטות עבור נקודת זמן נתונה בווידאו.
    """
    panels = []
    titles = [
        ("scene_baseline", "Method 1: Scene-Baseline (Raw Grayscale)"),
        ("yolo_best_frame", "Method 2: YOLO-BestFrame + CLAHE"),
        ("temporal_deocclusion", "Method 3: Temporal-DeOcclusion-Pro (Ours)"),
    ]

    for m_key, label in titles:
        recs = results.get(m_key, [])
        if not recs:
            return False
        # מציאת הפריים הקרוב ביותר לנקודת הזמן המבוקשת
        best_r = min(recs, key=lambda r: min(abs(r.start_sec - target_sec), abs(r.end_sec - target_sec)))
        img = cv2.imread(best_r.image_path)
        if img is None:
            # קריאה תומכת נתיבי Unicode ב-Windows
            data = np.fromfile(best_r.image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]
        scale = 640.0 / w
        new_w, new_h = 640, int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # הוספת כותרת עליונה עם מדדי הפריים
        header = np.full((75, new_w, 3), 30, dtype=np.uint8)
        cv2.putText(header, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        stats_txt = (
            f"Time: {best_r.start_time_str}-{best_r.end_time_str} | "
            f"Occlusion: {best_r.lecturer_occlusion_pct:.1f}% | CNR: {best_r.contrast_cnr:.1f}"
        )
        cv2.putText(header, stats_txt, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 240, 150), 1, cv2.LINE_AA)

        panel = np.vstack([header, resized])
        panels.append(panel)

    sep = np.full((panels[0].shape[0], 8, 3), 80, dtype=np.uint8)
    combined = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

    ok, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        with open(output_image_path, "wb") as f:
            f.write(buf.tobytes())
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="השוואת שיטות חילוץ כיתוב מלוח הרצאה")
    parser.add_argument("video", nargs="?", help="נתיב לקובץ הווידאו")
    parser.add_argument("--start", default="00:00", help="זמן התחלה")
    parser.add_argument("--end", default="30:00", help="זמן סיום")
    parser.add_argument("--output-dir", default="benchmark_output", help="תיקיית פלט")
    parser.add_argument("--interval", type=float, default=4.0, help="מרווח דגימה בשניות")
    args = parser.parse_args()

    if not args.video:
        mp4s = [f for f in os.listdir(".") if "1" in f and f.lower().endswith(".mp4")]
        if not mp4s:
            mp4s = [f for f in os.listdir(".") if f.lower().endswith(".mp4")]
        if not mp4s:
            print("[-] לא נמצא קובץ MP4.")
            sys.exit(1)
        args.video = mp4s[0]

    start_s = parse_timestamp(args.start) or 0.0
    end_s = parse_timestamp(args.end)
    methods = ["scene_baseline", "yolo_best_frame", "temporal_deocclusion"]

    results = process_video_segment(
        video_path=args.video,
        output_dir=args.output_dir,
        start_sec=start_s,
        end_sec=end_s,
        methods=methods,
        dense_interval_sec=args.interval,
    )

    summary = {m: summarize_method_metrics(results[m]) for m in methods}
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mid_t = start_s + 0.75 * ((end_s or (start_s + 1800)) - start_s)
    comp_img = os.path.join(args.output_dir, "visual_comparison_3methods.jpg")
    create_side_by_side_comparison(results, target_sec=mid_t, output_image_path=comp_img)
    print(f"\n[V] סיכום ההשוואה נשמר ב: {summary_path}")
    print(f"[V] תמונת השוואה ויזואלית נשמרה ב: {comp_img}")


if __name__ == "__main__":
    main()

