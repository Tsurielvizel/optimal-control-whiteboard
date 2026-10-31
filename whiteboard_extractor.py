#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
מערכת חילוץ כיתוב מלוח הרצאות וידאו (MP4) לקובץ Markdown (.md)
===================================================================
מודול זה מחלץ באופן אוטומטי את הכיתוב המתמטי והעברי מלוח הכיתה מתוך הקלטות וידאו,
תוך התמודדות עם תנועת המרצה מול הלוח, תזוזות מצלמה (PTZ Pan/Zoom), תאורה לא אחידה ומחיקות.

המערכת תומכת ב-3 שיטות עיבוד תמונה (Computer Vision) להשוואה ולשימוש שוטף:
1. scene_baseline       : זיהוי שינויי סצנה וחדות תמונה (ללא הסרת מרצה)
2. yolo_best_frame      : בחירת הפריים האופטימלי בכל סצנה באמצעות YOLOv8-seg ומדד צפיפות דיו
3. temporal_deocclusion : השיטה המנצחת (ברירת מחדל) – ייצוב תזוזות מצלמה (Phase Correlation),
                          שחזור רקע טמפורלי מונחה YOLOv8-seg להעלמה מלאה של המרצה,
                          נרמול תאורה גאוסיאני להפיכת הלוח ללבן נקי, וסינון דומיננטיות.

שימוש:
    python whiteboard_extractor.py "בקרה אופטימלית - שעור 1 _ דף הבית.mp4" --end 30:00
    python whiteboard_extractor.py "בקרה אופטימלית - שעור 1 _ דף הבית.mp4" --start 52:00 --end 67:00 --method all
    python whiteboard_extractor.py "הרצאה 2.mp4" --output-dir output_lecture2
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


@dataclass
class KeyframeRecord:
    """מייצג פריים-מפתח (מצב לוח משוחזר) שחולץ מהווידאו."""
    index: int
    method: str
    start_sec: float
    end_sec: float
    start_time_str: str
    end_time_str: str
    num_source_frames: int
    lecturer_occlusion_pct: float
    ink_coverage_pct: float
    contrast_cnr: float
    sharpness_score: float
    image_path: str
    raw_image_path: Optional[str] = None
    audio_transcript: Optional[str] = None


def format_timestamp(seconds: float) -> str:
    """ממיר שניות למחרוזת זמן MM:SS או HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """ממיר מחרוזת זמן (MM:SS, HH:MM:SS או שניות) למספר שניות."""
    if ts_str is None:
        return None
    ts_str = str(ts_str).strip()
    if not ts_str:
        return None
    parts = [float(p) for p in ts_str.split(":")]
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"פורמט זמן לא תקין: {ts_str}")


def decode_frames_sequential(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    interval_sec: float = 4.0,
) -> List[Tuple[float, np.ndarray]]:
    """
    פענוח רציף ומהיר של פריימים באמצעות cap.grab() (ללא חיפוש אקראי איטי ב-GOP).
    מהיר פי 5 מ-cap.set() ומדויק ברמת הפריים.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration = (total_frames / fps) if total_frames > 0 else 10800.0

    if end_sec is None or end_sec > total_duration:
        end_sec = total_duration

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps)))
    step_frames = max(1, int(round(interval_sec * fps)))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: List[Tuple[float, np.ndarray]] = []
    curr_frame_idx = start_frame
    while curr_frame_idx <= end_frame:
        ret = cap.grab()
        if not ret:
            break
        if (curr_frame_idx - start_frame) % step_frames == 0:
            ok, frm = cap.retrieve()
            if ok and frm is not None:
                frames.append((curr_frame_idx / fps, frm.copy()))
        curr_frame_idx += 1

    cap.release()
    return frames


class WhiteboardPipeline:
    """צינור עיבוד תמונה לחילוץ ושחזור לוח הרצאה."""

    def __init__(self, yolo_model_path: str = "yolov8n-seg.pt", imgsz: int = 320):
        self.imgsz = imgsz
        self.yolo = YOLO(yolo_model_path) if HAS_YOLO else None
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        self.ink_bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        self.roi_erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))

    def detect_lecturer_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        מזהה את המרצה בפריים ומחזיר:
        1. tight_mask: מסכת המרצה המדויקת (לחישוב מדד ההסתרה).
        2. dilated_mask: מסכה מורחבת (כולל צל, ידיים וטוש) לשחזור רקע נקי.
        """
        h, w = frame.shape[:2]
        tight_mask = np.zeros((h, w), dtype=np.uint8)
        if self.yolo is not None:
            res = self.yolo(frame, classes=[0], conf=0.20, imgsz=self.imgsz, verbose=False)[0]
            if res.masks is not None:
                for m in res.masks.data.cpu().numpy():
                    m_resized = cv2.resize(m, (w, h))
                    tight_mask[m_resized > 0.25] = 255
            for box in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                tight_mask[max(0, y1 - 10):min(h, y2 + 10), max(0, x1 - 10):min(w, x2 + 10)] = 255

        dilated_mask = cv2.dilate(tight_mask, self.dilate_kernel, iterations=1)
        return tight_mask, dilated_mask

    def detect_whiteboard_roi_mask(
        self, raw_or_gray: np.ndarray, lecturer_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        מזהה את אזור הלוח הלבן בלבד ומסנן החוצה שלטי בטיחות כהים על הקיר,
        מסך המחשב (LG), ושורת החותמת בתחתית הפריים.
        אם נמסרה מסכת מרצה, מחליף זמנית את פיקסלי המרצה (חולצה כהה) בגוון הלוח
        כדי שחולצת המרצה לא תסווג בטעות כקיר או מסך מחשב בחישוב שטח הלוח.
        """
        gray = (
            raw_or_gray.copy()
            if raw_or_gray.ndim == 2
            else cv2.cvtColor(raw_or_gray, cv2.COLOR_BGR2GRAY)
        )
        if lecturer_mask is not None:
            gray[lecturer_mask > 0] = 210
        h, w = gray.shape[:2]
        bg_smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
        bright_mask = (bg_smooth > 148).astype(np.uint8) * 255
        board_mask = cv2.erode(bright_mask, self.roi_erode_kernel, iterations=1)
        board_mask[h - 70 :, :] = 0
        board_mask[:20, :] = 0
        board_mask[:, :20] = 0
        board_mask[:, w - 20 :] = 0
        return board_mask

    def extract_ink_stroke_mask(
        self,
        gray_img: np.ndarray,
        exclude_mask: Optional[np.ndarray] = None,
        raw_ref_gray: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        מחלץ מסכה בינארית של קווי הדיו (כתב יד ומשוואות) על הלוח בלבד.
        """
        roi_source = raw_ref_gray if raw_ref_gray is not None else gray_img
        board_roi = self.detect_whiteboard_roi_mask(roi_source, lecturer_mask=exclude_mask)
        bh = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, self.ink_bh_kernel)
        ink_mask = ((bh > 24) & (board_roi > 0)).astype(np.uint8) * 255

        if exclude_mask is not None:
            excl_dil = cv2.dilate(exclude_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            ink_mask[excl_dil > 0] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        clean_ink = np.zeros_like(ink_mask)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh_box = stats[lbl, cv2.CC_STAT_HEIGHT]
            if 14 <= area <= 20000 and not (bw > 200 and bh_box > 200 and area / max(1, bw * bh_box) > 0.60):
                clean_ink[labels == lbl] = 255
        return clean_ink

    @staticmethod
    def enhance_whiteboard_image(
        bgr_img: np.ndarray, ever_unoccluded: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        נרמול תאורה גאוסיאני (Gaussian Illumination Field Normalization):
        מחלק את התמונה בשדה התאורה המוערך של הלוח לקבלת רקע לבן טהור (#FFFFFF)
        ומכהה את קווי הטוש בניגודיות גבוהה וללא הילות (Halos).
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if ever_unoccluded is not None and not np.all(ever_unoccluded):
            med_val = float(np.median(gray[ever_unoccluded])) if np.any(ever_unoccluded) else 210.0
            gray[~ever_unoccluded] = med_val

        dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        bg_field = cv2.GaussianBlur(dilated, (0, 0), sigmaX=45, sigmaY=45)

        norm = np.clip((gray / (bg_field + 1e-5)) * 255.0, 0, 255)
        norm = np.clip((norm - 242.0) * (255.0 / 242.0) + 255.0, 0, 255)
        norm = ((norm / 255.0) ** 1.85) * 255.0

        if ever_unoccluded is not None:
            norm[~ever_unoccluded] = 255.0

        return norm.astype(np.uint8)

    def compute_quality_metrics(
        self,
        gray_or_enh: np.ndarray,
        tight_lecturer_mask: np.ndarray,
        raw_ref_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        """מחשב מדדי איכות אובייקטיביים לפריים על אזור הלוח."""
        h, w = gray_or_enh.shape[:2]
        raw_gray = cv2.cvtColor(raw_ref_bgr, cv2.COLOR_BGR2GRAY) if raw_ref_bgr is not None else gray_or_enh
        board_roi_mask = self.detect_whiteboard_roi_mask(raw_gray, lecturer_mask=tight_lecturer_mask) > 0
        board_area = max(1, int(np.sum(board_roi_mask)))

        occ_on_board = int(np.sum((tight_lecturer_mask > 0) & board_roi_mask))
        occ_pct = float(occ_on_board) * 100.0 / board_area

        ink_mask = self.extract_ink_stroke_mask(
            gray_or_enh, exclude_mask=tight_lecturer_mask, raw_ref_gray=raw_gray
        )
        ink_pct = float(np.sum(ink_mask > 0)) * 100.0 / (h * w)

        valid_board = board_roi_mask & (tight_lecturer_mask == 0)
        bg_pixels = gray_or_enh[valid_board & (ink_mask == 0)]
        ink_pixels = gray_or_enh[ink_mask > 0]

        if len(bg_pixels) > 100 and len(ink_pixels) > 50:
            bg_mean = float(np.mean(bg_pixels))
            bg_std = max(1.5, float(np.std(bg_pixels)))
            ink_mean = float(np.mean(ink_pixels))
            cnr = abs(bg_mean - ink_mean) / bg_std
        else:
            cnr = 0.0

        lap = cv2.Laplacian(gray_or_enh, cv2.CV_64F)
        sharpness = float(np.var(lap[valid_board])) if np.any(valid_board) else 0.0
        return round(occ_pct, 2), round(ink_pct, 3), round(cnr, 2), round(sharpness, 1)

    @staticmethod
    def estimate_camera_shift(
        anchor_gray: np.ndarray,
        curr_gray: np.ndarray,
        common_mask: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        מחשב תזוזה תת-פיקסלית של המצלמה (dx, dy) והפרש רקע חסין-צללים (אחוזון 65)
        בין פריים העוגן לפריים הנוכחי.
        """
        if np.sum(common_mask) < 40000:
            return 999.0, 999.0, 99.0

        g_curr = curr_gray[common_mask].astype(np.float32)
        g_anc = anchor_gray[common_mask].astype(np.float32)
        # נרמול חשיפה לפי חציון
        g_curr_norm = g_curr - np.median(g_curr) + np.median(g_anc)
        robust_diff = float(np.percentile(np.abs(g_curr_norm - g_anc), 65))

        # חישוב תזוזת מצלמה על תמונות מוקטנות (חצי רזולוציה) של נגזרות מרחביות
        h, w = anchor_gray.shape[:2]
        small_anc = cv2.resize(anchor_gray, (w // 2, h // 2)).astype(np.float32)
        small_cur = cv2.resize(curr_gray, (w // 2, h // 2)).astype(np.float32)
        small_msk = cv2.resize(common_mask.astype(np.uint8), (w // 2, h // 2)) > 0

        anc_edge = cv2.Sobel(small_anc, cv2.CV_32F, 1, 1, ksize=3) * small_msk
        cur_edge = cv2.Sobel(small_cur, cv2.CV_32F, 1, 1, ksize=3) * small_msk

        (dx_half, dy_half), _ = cv2.phaseCorrelate(anc_edge, cur_edge)
        return float(dx_half * 2.0), float(dy_half * 2.0), robust_diff

    def segment_stationary_episodes(
        self,
        frames_with_masks: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
        max_shift_px: float = 22.0,
        robust_diff_thresh: float = 6.2,
    ) -> List[List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]]:
        """
        מחלק את רצף הפריימים למקטעי מצלמה יציבים (Stationary Camera Episodes)
        ומבצע ייצוב תת-פיקסלי (Alignment) של זחילות מצלמה קלות בתוך כל מקטע.
        """
        episodes = []
        curr_ep = []
        anchor_gray = None
        anchor_valid = None

        for t, frm, tight_m, dil_m in frames_with_masks:
            h, w = frm.shape[:2]
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            valid = (dil_m == 0)
            valid[h - 70 :, :] = False

            if anchor_gray is not None:
                common = valid & anchor_valid
                dx, dy, r_diff = self.estimate_camera_shift(anchor_gray, gray, common)
                shift_mag = float(np.hypot(dx, dy))

                if shift_mag <= max_shift_px and r_diff <= robust_diff_thresh:
                    # אם יש תזוזה קלה (1-22 פיקסלים), ניישר את הפריים והמסכות לעוגן
                    if shift_mag >= 1.0:
                        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                        frm = cv2.warpAffine(frm, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        tight_m = cv2.warpAffine(tight_m, M, (w, h), borderValue=0)
                        dil_m = cv2.warpAffine(dil_m, M, (w, h), borderValue=0)
                    curr_ep.append((t, frm, tight_m, dil_m))
                else:
                    # המצלמה זזה לאזור חדש! נשמור את המקטע הקודם
                    if len(curr_ep) >= 1:
                        episodes.append(curr_ep)
                    curr_ep = [(t, frm, tight_m, dil_m)]
                    anchor_gray = gray
                    anchor_valid = valid
            else:
                anchor_gray = gray
                anchor_valid = valid
                curr_ep.append((t, frm, tight_m, dil_m))

        if len(curr_ep) >= 1:
            episodes.append(curr_ep)

        # סינון פריימי מעבר בודדים שבהם המצלמה הייתה תוך כדי תנועה מהירה
        stable_episodes = [ep for ep in episodes if len(ep) >= 2]
        return stable_episodes if stable_episodes else episodes

    def deduplicate_by_ink_dominance(
        self,
        candidates: List[Tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]],
        min_ink_coverage_pct: float = 0.08,
        retention_thresh: float = 0.58,
        enable_cross_merge: bool = True,
    ) -> List[Tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]]:
        """
        סינון כפילויות חכם מבוסס שימור דיו על הלוח (Sub-State Dominance Deduplication):
        ממזג שלבי ביניים שבהם המרצה מוסיף כיתוב לאותו אזור בלוח, ושומר פריים-מפתח
        נפרד רק כאשר מתבצעת מחיקה או מעבר מצלמה לחלק אחר של הלוח.
        """
        filtered = []
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))

        for cand in candidates:
            t0, t1, n_frm, enh_img, raw_img, tight_m = cand
            gray_check = enh_img if enh_img.ndim == 2 else cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)
            raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            ink_curr = self.extract_ink_stroke_mask(gray_check, exclude_mask=tight_m, raw_ref_gray=raw_gray)
            ink_count = int(np.sum(ink_curr > 0))
            h, w = gray_check.shape[:2]
            if (ink_count * 100.0 / (h * w)) < min_ink_coverage_pct:
                continue

            if not filtered:
                filtered.append((cand, ink_curr, ink_count))
                continue

            prev_cand, prev_ink, prev_count = filtered[-1]
            # בדיקת חפיפה באזורים שגלויים בשני הפריימים (מחוץ להסתרת מרצה)
            prev_occ = prev_cand[5]
            both_visible = (tight_m == 0) & (prev_occ == 0)

            curr_dil = cv2.dilate(ink_curr, dil_k, iterations=1)
            prev_dil = cv2.dilate(prev_ink, dil_k, iterations=1)

            prev_vis_ink = (prev_ink > 0) & both_visible
            curr_vis_ink = (ink_curr > 0) & both_visible
            prev_vis_cnt = int(np.sum(prev_vis_ink))
            curr_vis_cnt = int(np.sum(curr_vis_ink))

            retention_forward = (
                float(np.sum(prev_vis_ink & (curr_dil > 0))) / max(1, prev_vis_cnt)
                if prev_vis_cnt > 200
                else 0.0
            )
            retention_backward = (
                float(np.sum(curr_vis_ink & (prev_dil > 0))) / max(1, curr_vis_cnt)
                if curr_vis_cnt > 200
                else 0.0
            )

            if retention_forward >= retention_thresh and ink_count >= 0.82 * prev_count:
                if enable_cross_merge and enh_img.ndim == 2 and prev_cand[3].ndim == 2:
                    merged_enh = enh_img.copy()
                    fill_from_prev = (tight_m > 0) & (prev_occ == 0)
                    merged_enh[fill_from_prev] = prev_cand[3][fill_from_prev]
                    merged_occ = ((tight_m > 0) & (prev_occ > 0)).astype(np.uint8) * 255
                else:
                    merged_enh = enh_img
                    merged_occ = tight_m

                merged_cand = (prev_cand[0], t1, prev_cand[2] + n_frm, merged_enh, raw_img, merged_occ)
                new_ink = self.extract_ink_stroke_mask(merged_enh, exclude_mask=merged_occ, raw_ref_gray=raw_gray)
                filtered[-1] = (merged_cand, new_ink, int(np.sum(new_ink > 0)))

            elif retention_backward >= 0.72 and prev_count >= ink_count:
                if enable_cross_merge and prev_cand[3].ndim == 2 and enh_img.ndim == 2:
                    merged_enh = prev_cand[3].copy()
                    fill_from_curr = (prev_occ > 0) & (tight_m == 0)
                    merged_enh[fill_from_curr] = enh_img[fill_from_curr]
                    merged_occ = ((prev_occ > 0) & (tight_m > 0)).astype(np.uint8) * 255
                else:
                    merged_enh = prev_cand[3]
                    merged_occ = prev_occ

                prev_updated = (prev_cand[0], t1, prev_cand[2] + n_frm, merged_enh, prev_cand[4], merged_occ)
                filtered[-1] = (prev_updated, prev_ink, prev_count)
            else:
                filtered.append((cand, ink_curr, ink_count))

        return [item[0] for item in filtered]

    def run_method(
        self,
        method: str,
        precomputed_frames: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
        output_dir: str,
    ) -> List[KeyframeRecord]:
        """מריץ אחת משלוש השיטות על רצף הפריימים ושומר את תמונות הפלט."""
        os.makedirs(output_dir, exist_ok=True)
        episodes = self.segment_stationary_episodes(precomputed_frames)
        candidates = []

        for ep in episodes:
            t_start, t_end = ep[0][0], ep[-1][0]
            n_frm = len(ep)

            if method == "scene_baseline":
                tail = ep[max(0, len(ep) - 3) :]
                best_item = max(
                    tail,
                    key=lambda x: float(np.var(cv2.Laplacian(cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F))),
                )
                _, frm, tight_m, _ = best_item
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                candidates.append((t_start, t_end, n_frm, gray, frm, tight_m))

            elif method == "yolo_best_frame":
                best_score = -1e12
                best_item = ep[-1]
                for item in ep:
                    _, frm, tight_m, _ = item
                    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                    ink_m = self.extract_ink_stroke_mask(gray, exclude_mask=tight_m, raw_ref_gray=gray)
                    score = float(np.sum(ink_m > 0)) - 0.40 * float(np.sum(tight_m > 0))
                    if score > best_score:
                        best_score = score
                        best_item = item
                _, frm, tight_m, _ = best_item
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                enh = clahe.apply(gray)
                candidates.append((t_start, t_end, n_frm, enh, frm, tight_m))

            elif method == "temporal_deocclusion":
                h, w = ep[0][1].shape[:2]
                canvas = np.zeros((h, w, 3), dtype=np.float32)
                fallback_canvas = np.zeros((h, w, 3), dtype=np.float32)
                ever_unocc = np.zeros((h, w), dtype=bool)
                ever_unocc_tight = np.zeros((h, w), dtype=bool)
                small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

                for _, frm, tight_m, dil_m in ep:
                    unocc = (dil_m == 0)
                    first_time = unocc & (~ever_unocc)
                    subsequent = unocc & ever_unocc
                    canvas[first_time] = frm[first_time].astype(np.float32)
                    canvas[subsequent] = 0.25 * canvas[subsequent] + 0.75 * frm[subsequent].astype(np.float32)
                    ever_unocc |= unocc

                    tight_pad = cv2.dilate(tight_m, small_k, iterations=1)
                    unocc_t = (tight_pad == 0)
                    first_t = unocc_t & (~ever_unocc_tight)
                    sub_t = unocc_t & ever_unocc_tight
                    fallback_canvas[first_t] = frm[first_t].astype(np.float32)
                    fallback_canvas[sub_t] = 0.25 * fallback_canvas[sub_t] + 0.75 * frm[sub_t].astype(np.float32)
                    ever_unocc_tight |= unocc_t

                raw_reconstructed = canvas.copy()
                # שלב השלמה דו-שכבתי: אזורים שהיו קרובים למרצה (בתוך dil_m) אך גלויים מחוץ ל-tight_pad
                tier2_mask = (~ever_unocc) & ever_unocc_tight
                raw_reconstructed[tier2_mask] = fallback_canvas[tier2_mask]
                combined_unocc = ever_unocc | ever_unocc_tight

                if not np.all(combined_unocc):
                    med_bgr = (
                        np.median(raw_reconstructed[combined_unocc], axis=0)
                        if np.any(combined_unocc)
                        else np.array([210, 210, 210])
                    )
                    raw_reconstructed[~combined_unocc] = med_bgr
                raw_u8 = raw_reconstructed.astype(np.uint8)
                enh_u8 = self.enhance_whiteboard_image(raw_u8, ever_unoccluded=combined_unocc)

                residual_occ = (~combined_unocc).astype(np.uint8) * 255
                residual_occ[h - 70 :, :] = 0
                candidates.append((t_start, t_end, n_frm, enh_u8, raw_u8, residual_occ))
            else:
                raise ValueError(f"שיטה לא מוכרת: {method}")

        deduped = self.deduplicate_by_ink_dominance(
            candidates, enable_cross_merge=False
        )
        records: List[KeyframeRecord] = []

        # ניקוי קבצים ישנים בתיקיית הפלט
        for old_f in os.listdir(output_dir):
            if old_f.startswith("board_") and old_f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(output_dir, old_f))
                except OSError:
                    pass

        for idx, (t0, t1, n_frm, enh_img, raw_img, occ_mask) in enumerate(deduped, start=1):
            occ_pct, ink_pct, cnr, sharp = self.compute_quality_metrics(
                enh_img, occ_mask, raw_ref_bgr=raw_img
            )
            ts_slug = f"{int(t0)//60:02d}m{int(t0)%60:02d}s_{int(t1)//60:02d}m{int(t1)%60:02d}s"
            img_name = f"board_{idx:02d}_{ts_slug}.jpg"
            img_path = os.path.join(output_dir, img_name)

            ok, buf = cv2.imencode(".jpg", enh_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                with open(img_path, "wb") as f:
                    f.write(buf.tobytes())

            raw_path = None
            if method == "temporal_deocclusion":
                raw_name = f"board_{idx:02d}_{ts_slug}_color.jpg"
                raw_path = os.path.join(output_dir, raw_name)
                ok_r, buf_r = cv2.imencode(".jpg", raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if ok_r:
                    with open(raw_path, "wb") as f:
                        f.write(buf_r.tobytes())

            records.append(
                KeyframeRecord(
                    index=idx,
                    method=method,
                    start_sec=t0,
                    end_sec=t1,
                    start_time_str=format_timestamp(t0),
                    end_time_str=format_timestamp(t1),
                    num_source_frames=n_frm,
                    lecturer_occlusion_pct=occ_pct,
                    ink_coverage_pct=ink_pct,
                    contrast_cnr=cnr,
                    sharpness_score=sharp,
                    image_path=os.path.abspath(img_path),
                    raw_image_path=os.path.abspath(raw_path) if raw_path else None,
                )
            )

        manifest_path = os.path.join(output_dir, "keyframes_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

        return records


def process_video_segment(
    video_path: str,
    output_dir: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    methods: Optional[List[str]] = None,
    dense_interval_sec: float = 4.0,
) -> Dict[str, List[KeyframeRecord]]:
    """מריץ את צינור החילוץ על מקטע וידאו עבור שיטה אחת או יותר."""
    if methods is None:
        methods = ["temporal_deocclusion"]

    print(
        f"[+] מפענח פריימים ברצף מהווידאו: {os.path.basename(video_path)} "
        f"({format_timestamp(start_sec)} - {format_timestamp(end_sec or 0)})..."
    )
    t0 = time.time()
    raw_frames = decode_frames_sequential(
        video_path, start_sec=start_sec, end_sec=end_sec, interval_sec=dense_interval_sec
    )
    print(f"[v] פוענחו {len(raw_frames)} פריימים תוך {time.time() - t0:.1f} שניות.")

    pipeline = WhiteboardPipeline()
    print("[+] מריץ סגמנטציית מרצה (YOLOv8-seg) לזיהוי והסרת הסתרות...")
    t1 = time.time()
    precomputed = []
    for ts, frm in raw_frames:
        tight_m, dil_m = pipeline.detect_lecturer_mask(frm)
        precomputed.append((ts, frm, tight_m, dil_m))
    print(f"[v] סגמנטציית YOLOv8-seg הושלמה תוך {time.time() - t1:.1f} שניות.")

    results: Dict[str, List[KeyframeRecord]] = {}
    for m in methods:
        m_dir = output_dir if len(methods) == 1 else os.path.join(output_dir, m)
        print(f"[+] מפעיל שיטת חילוץ: {m} -> {m_dir}")
        recs = pipeline.run_method(m, precomputed, m_dir)
        results[m] = recs
        avg_occ = np.mean([r.lecturer_occlusion_pct for r in recs]) if recs else 0.0
        avg_cnr = np.mean([r.contrast_cnr for r in recs]) if recs else 0.0
        print(
            f"    -> נוצרו {len(recs)} פריימי-לוח ייחודיים | "
            f"הסתרת מרצה על הלוח: {avg_occ:.2f}% | ניגודיות (CNR): {avg_cnr:.1f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="חילוץ כיתוב לוח מהרצאות וידאו (MP4)")
    parser.add_argument("video", nargs="?", help="נתיב לקובץ הווידאו (MP4)")
    parser.add_argument("--start", default="00:00", help="זמן התחלה (MM:SS או HH:MM:SS)")
    parser.add_argument("--end", default=None, help="זמן סיום (MM:SS או HH:MM:SS)")
    parser.add_argument(
        "--method",
        choices=["temporal_deocclusion", "yolo_best_frame", "scene_baseline", "all"],
        default="temporal_deocclusion",
        help="שיטת עיבוד תמונה לחילוץ הלוח",
    )
    parser.add_argument("--output-dir", default="extracted_board", help="תיקיית פלט לתמונות הלוח והמניפסט")
    parser.add_argument("--interval", type=float, default=4.0, help="מרווח בשניות בין דגימות")
    args = parser.parse_args()

    if not args.video:
        mp4s = [f for f in os.listdir(".") if f.lower().endswith(".mp4")]
        if not mp4s:
            print("[-] לא נמצאו קבצי MP4 בתיקייה.")
            sys.exit(1)
        args.video = mp4s[0]

    start_s = parse_timestamp(args.start) or 0.0
    end_s = parse_timestamp(args.end)
    methods = ["scene_baseline", "yolo_best_frame", "temporal_deocclusion"] if args.method == "all" else [args.method]

    process_video_segment(
        video_path=args.video,
        output_dir=args.output_dir,
        start_sec=start_s,
        end_sec=end_s,
        methods=methods,
        dense_interval_sec=args.interval,
    )


if __name__ == "__main__":
    main()
