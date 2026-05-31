"""
Zero-shot face occlusion estimation: 3-signal judge (SAM2 + MediaPipe + Qwen).

Pipeline per image:
  Phase 1 — 3 independent signals collected in parallel:
    Signal A — SAM2 geometric segmentation  → geom_pct, sam2_conf
    Signal B — MediaPipe face landmarks     → orientation, missing_zones, pose_occlusion
    Signal C — Qwen free visual description → description, occlusion_type, first_impression_pct
                (image only, no external signals)
  Phase 2 — Qwen judge:
    Receives original image + all 3 signals as text.
    Evaluates reliability of each signal and produces final estimate.

Fairness invariant: prompts contain no demographic information.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model IDs ─────────────────────────────────────────────────────────────────

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SAM2_CONF_THRESHOLD = 0.5

# ── Signal C prompt — Qwen free description (no external signals) ─────────────
# Plain string, no .format() call needed.

_PROMPT_SIGNAL_C = (
    "Look at this face image.\n"
    "Describe in 2-3 sentences what you observe :\n"
    "- Is there anything covering or hiding part of the face ?\n"
    "- What type of occlusion is it ?\n"
    "  (hand, mask, glasses, hair, blur, shadow,\n"
    "   profile pose, nothing)\n"
    "- Which facial zones seem hidden or unclear ?\n\n"
    "Respond ONLY as JSON :\n"
    '{\n'
    '  "description": "...",\n'
    '  "occlusion_type": "physical_object|blur|pose|shadow|none|mixed",\n'
    '  "affected_zones": ["..."],\n'
    '  "first_impression_pct": <float 0.00-1.00>\n'
    "}"
)

# ── Judge prompt — receives all 3 signals + image ─────────────────────────────
# Format vars: geom_pct, sam2_conf, orientation, yaw_deg, missing_zones,
#              pose_occlusion, description, occlusion_type, affected_zones,
#              first_impression_pct
# {{ }} are .format()-escaped braces — become { } after .format() call.

_PROMPT_JUDGE = (
    "You have analyzed this face image and collected 3 independent signals.\n"
    "Act as a judge to evaluate each signal and estimate the final occlusion.\n\n"
    "DEFINITION OF OCCLUSION (critical — use exactly this):\n"
    "  Occlusion = (area of the face region that is hidden) / (total face region area).\n"
    "  It is a SURFACE RATIO, from 0.00 (fully visible) to 1.00 (fully hidden).\n"
    "  What counts as hidden: anything covering the face surface — hand, mask,\n"
    "  glasses, blur, shadow, AND hair when it covers part of the face area.\n"
    "  A thin strand of hair covers little area → small value.\n"
    "  A thick fringe over the forehead covers real area → moderate value.\n"
    "  Estimate the FRACTION OF FACE SURFACE covered, not 'is something present'.\n\n"
    "SIGNAL A — SAM2 geometric segmentation:\n"
    "  Measured: {geom_pct:.1f}% of face area\n"
    "  Confidence: {sam2_conf:.2f}\n"
    "  CRITICAL: SAM2 systematically OVER-estimates. It segments hair, shadows\n"
    "  and background that fall outside the counted face region, and almost never\n"
    "  reports low values even on clear faces. Treat a high SAM2 value as WEAK\n"
    "  evidence on its own — never accept it unless another signal corroborates.\n"
    "  SAM2 cannot detect blur. A very low SAM2 value can also mean SAM2 failed\n"
    "  to segment, not that the face is clear.\n\n"
    "SIGNAL B — MediaPipe facial landmarks:\n"
    "  Orientation: {orientation} (yaw={yaw_deg:.1f}°)\n"
    "  Missing zones: {missing_zones}\n"
    "  Pose occlusion estimate: {pose_occlusion:.2f}\n"
    "  CRITICAL: a real occlusion above ~0.4 of the face surface would normally\n"
    "  disturb MediaPipe — missing landmarks, lost frontal pose. So if MediaPipe\n"
    "  reports a clean FRONTAL face with NO missing zones and good confidence,\n"
    "  then a high occlusion claimed by SAM2 or by your description is very likely\n"
    "  a FALSE POSITIVE (hair/shadow/background) — lower your estimate accordingly.\n"
    "  MediaPipe may fail on blurry or very dark images.\n\n"
    "SIGNAL C — Your own visual description:\n"
    "  You observed: '{description}'\n"
    "  Occlusion type: {occlusion_type}\n"
    "  Affected zones: {affected_zones}\n"
    "  Your first impression: {first_impression_pct:.2f}\n\n"
    "Reason step by step:\n"
    "<evaluate_signals>\n"
    "1. CONSISTENCY CHECK FIRST: do the signals agree on the SURFACE FRACTION?\n"
    "   In particular: is a clean frontal MediaPipe compatible with the occlusion\n"
    "   level suggested by SAM2/description? If not, one signal is a false positive\n"
    "   — identify which and say why.\n"
    "2. Which signal is most reliable FOR THIS CASE?\n"
    "   - physical object (hand/mask/glasses) → trust description + SAM2\n"
    "   - blur/shadow → distrust SAM2, trust description\n"
    "   - pose/profile → trust MediaPipe\n"
    "   - hair → estimate the FACE SURFACE FRACTION it covers, not its presence\n"
    "3. Reasons to distrust each signal? (SAM2 hair/background false positive?\n"
    "   MediaPipe failed on blur? Your description over-reacted to hair or shadow?)\n"
    "</evaluate_signals>\n\n"
    "<final_estimate>\n"
    "State the true occluded SURFACE FRACTION of the face. Justify in one sentence.\n"
    "Anchor on the consistency check, not on any single signal.\n"
    "</final_estimate>\n\n"
    "Respond ONLY as JSON:\n"
    "{{\n"
    '  "signal_evaluation": {{\n'
    '    "sam2_reliable": true,\n'
    '    "sam2_reason": "...",\n'
    '    "mediapipe_reliable": true,\n'
    '    "mediapipe_reason": "...",\n'
    '    "description_reliable": true,\n'
    '    "description_reason": "..."\n'
    "  }},\n"
    '  "dominant_signal": "SAM2|MediaPipe|Description|Combined",\n'
    '  "reasoning": "...",\n'
    '  "percentage": <float 0.00-1.00>\n'
    "}}"
)

_VALID_OCCLUSION_TYPES = frozenset({
    "physical_object", "blur", "pose", "shadow", "none", "mixed",
})
_VALID_DOMINANT_SIGNALS = frozenset({
    "SAM2", "MediaPipe", "Description", "Combined",
})

# ── MediaPipe zone weights (sum = 0.90) ───────────────────────────────────────

_ZONE_WEIGHTS: Dict[str, float] = {
    "left_eye": 0.15,  "right_eye": 0.15,
    "nose_tip": 0.10,
    "mouth_left": 0.08, "mouth_right": 0.08,
    "chin": 0.10,
    "left_cheek": 0.12, "right_cheek": 0.12,
}

_MP_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MP_LANDMARKER_CACHE = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"
_face_landmarker: Optional[object] = None  # lazy singleton


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FacePoseInfo:
    """Output of MediaPipe Signal B."""
    detected: bool = False
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    missing_zones: List[str] = field(default_factory=list)
    pose_occlusion: float = 0.0
    orientation: str = "frontal"


@dataclass
class SAM2Result:
    """Output of SAM2 Signal A."""
    mask: np.ndarray
    confidence: float
    geometric_occ: float
    face_bbox: Tuple[int, int, int, int]


@dataclass
class SignalCResult:
    """Output of Qwen free-description call (Signal C)."""
    description: str = ""
    occlusion_type: str = "none"
    affected_zones: List[str] = field(default_factory=list)
    first_impression_pct: float = 0.0
    parse_failed: bool = False


@dataclass
class JudgeResult:
    """Output of Qwen judge call (Phase 2)."""
    percentage: float = 0.10
    sam2_reliable: bool = True
    sam2_reason: str = ""
    mediapipe_reliable: bool = True
    mediapipe_reason: str = ""
    description_reliable: bool = True
    description_reason: str = ""
    dominant_signal: str = "Combined"
    reasoning: str = ""
    parse_failed: bool = False


@dataclass
class PipelineResult:
    prediction: float
    # Signal A
    signal_a_geom: float = 0.0
    signal_a_conf: float = 0.0
    # Signal B
    signal_b_orientation: str = ""
    signal_b_pose_occ: float = 0.0
    signal_b_missing: str = ""          # comma-separated zone names
    # Signal C
    signal_c_type: str = ""
    signal_c_first_imp: float = 0.0
    signal_c_description: str = ""
    # Judge
    sam2_reliable: bool = True
    mediapipe_reliable: bool = True
    dominant_signal: str = ""
    reasoning: str = ""
    # Compat fields kept for compare_runs / CSV
    geometric_occ: Optional[float] = None
    used_mask: bool = False
    failure: Optional[str] = None


# ── SAM2 segmentation ─────────────────────────────────────────────────────────


class SAM2Segmenter:
    """Wraps SAM2 for single-image face segmentation via center-point prompt."""

    def __init__(self, model_id: str = SAM2_MODEL_ID, device: str = "cuda") -> None:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 requires the 'sam2' package. Run: uv add sam2>=1.0.0"
            ) from exc

        self.device = device
        logger.info("Loading SAM2 predictor from %s …", model_id)
        self._predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self._predictor.model = self._predictor.model.to(device)
        logger.info("SAM2 loaded.")

    def segment(self, image: Image.Image) -> Optional[SAM2Result]:
        """Center-point prompt segmentation. geometric_occ = 1 - mask_area / image_area."""
        w, h = image.size
        image_rgb = np.array(image.convert("RGB"))

        try:
            with torch.inference_mode():
                self._predictor.set_image(image_rgb)
                masks, scores, _ = self._predictor.predict(
                    point_coords=np.array([[w / 2.0, h / 2.0]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
        except Exception as exc:
            logger.warning("SAM2 prediction failed: %s", exc)
            return None

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        confidence = float(scores[best_idx])
        geometric_occ = float(np.clip(1.0 - best_mask.sum() / (w * h), 0.0, 1.0))

        return SAM2Result(
            mask=best_mask,
            confidence=confidence,
            geometric_occ=geometric_occ,
            face_bbox=(0, 0, w - 1, h - 1),
        )


# ── MediaPipe pose analysis ───────────────────────────────────────────────────


def _get_face_landmarker():
    """Return cached MediaPipe FaceLandmarker, downloading model on first call."""
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker
    try:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415
    except ImportError:
        logger.warning("mediapipe not installed — Signal B disabled.")
        return None

    if not _MP_LANDMARKER_CACHE.exists():
        _MP_LANDMARKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe Face Landmarker model (~2.5 MB) …")
        urllib.request.urlretrieve(_MP_LANDMARKER_URL, _MP_LANDMARKER_CACHE)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MP_LANDMARKER_CACHE)),
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _infer_missing_zones(yaw_deg: float, pitch_deg: float) -> List[str]:
    """Infer hidden face zones from yaw/pitch angles."""
    missing: List[str] = []
    abs_yaw = abs(yaw_deg)
    if abs_yaw > 20:
        side = "left" if yaw_deg > 0 else "right"
        missing.append(f"{side}_cheek")
        if abs_yaw > 35: missing.append(f"{side}_eye")
        if abs_yaw > 50: missing.append(f"mouth_{'left' if yaw_deg > 0 else 'right'}")
        if abs_yaw > 60: missing.append("nose_tip")
    if pitch_deg > 30:
        missing.append("chin")
    elif pitch_deg < -30:
        missing.extend(["left_eye", "right_eye"])
    return list(dict.fromkeys(missing))


def _classify_orientation(yaw_deg: float) -> str:
    abs_yaw = abs(yaw_deg)
    if abs_yaw < 15:  return "frontal"
    if abs_yaw < 35:  return "slight_rotation"
    if abs_yaw < 60:  return "profile"
    return "extreme_profile"


def analyze_face_pose(image_rgb: np.ndarray) -> FacePoseInfo:
    """Signal B — MediaPipe landmark analysis (~10 ms, CPU)."""
    try:
        import mediapipe as mp  # noqa: PLC0415
    except ImportError:
        return FacePoseInfo()

    landmarker = _get_face_landmarker()
    if landmarker is None:
        return FacePoseInfo()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return FacePoseInfo(detected=False)

    lm = result.face_landmarks[0]

    left_eye_x   = lm[33].x
    right_eye_x  = lm[263].x
    nose_x       = lm[1].x
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    face_width   = max(abs(right_eye_x - left_eye_x), 0.01)
    yaw_deg      = float(np.clip((nose_x - eye_center_x) / face_width * 90.0, -90, 90))

    chin_y       = lm[152].y
    eye_center_y = (lm[33].y + lm[263].y) / 2.0
    nose_y       = lm[1].y
    face_height  = max(abs(chin_y - eye_center_y), 0.01)
    pitch_deg    = float(np.clip(
        (nose_y - (eye_center_y + chin_y) / 2.0) / face_height * 45.0, -45, 45
    ))

    missing    = _infer_missing_zones(yaw_deg, pitch_deg)
    pose_occ   = float(np.clip(sum(_ZONE_WEIGHTS.get(z, 0) for z in missing), 0.0, 1.0))
    orientation = _classify_orientation(yaw_deg)

    return FacePoseInfo(
        detected=True,
        yaw_deg=round(yaw_deg, 1),
        pitch_deg=round(pitch_deg, 1),
        missing_zones=missing,
        pose_occlusion=round(pose_occ, 3),
        orientation=orientation,
    )


# ── Response parsers ──────────────────────────────────────────────────────────


def _parse_occlusion_value(text: str) -> Optional[float]:
    """Fallback: extract first decimal in [0, 1] from raw text."""
    for match in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text):
        try:
            v = float(match.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract outermost JSON object from text, handling nested braces."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1  # reset and keep scanning
    return None


def _parse_signal_c_response(text: str) -> SignalCResult:
    data = _extract_json(text)
    if data is not None:
        try:
            pct = float(np.clip(float(str(data.get("first_impression_pct", 0.10))), 0.0, 1.0))
            zones = data.get("affected_zones", [])
            if isinstance(zones, str):
                zones = [z.strip() for z in zones.split(",") if z.strip()]
            otype = str(data.get("occlusion_type", "none")).lower().strip()
            if otype not in _VALID_OCCLUSION_TYPES:
                otype = "none"
            return SignalCResult(
                description=str(data.get("description", ""))[:400],
                occlusion_type=otype,
                affected_zones=list(zones)[:10],
                first_impression_pct=pct,
            )
        except (ValueError, TypeError):
            pass
    val = _parse_occlusion_value(text)
    return SignalCResult(first_impression_pct=val or 0.10, parse_failed=True)


def _parse_judge_response(text: str) -> JudgeResult:
    data = _extract_json(text)
    if data is not None and "percentage" in data:
        try:
            pct = float(np.clip(float(str(data["percentage"])), 0.0, 1.0))
            se  = data.get("signal_evaluation", {})
            dom = str(data.get("dominant_signal", "Combined")).strip()
            if dom not in _VALID_DOMINANT_SIGNALS:
                dom = "Combined"
            return JudgeResult(
                percentage=pct,
                sam2_reliable=bool(se.get("sam2_reliable", True)),
                sam2_reason=str(se.get("sam2_reason", ""))[:200],
                mediapipe_reliable=bool(se.get("mediapipe_reliable", True)),
                mediapipe_reason=str(se.get("mediapipe_reason", ""))[:200],
                description_reliable=bool(se.get("description_reliable", True)),
                description_reason=str(se.get("description_reason", ""))[:200],
                dominant_signal=dom,
                reasoning=str(data.get("reasoning", ""))[:400],
            )
        except (ValueError, TypeError):
            pass
    val = _parse_occlusion_value(text)
    return JudgeResult(percentage=val or 0.10, parse_failed=True)


# ── Qwen2.5-VL estimator ──────────────────────────────────────────────────────


class QwenEstimator:
    """Wraps Qwen2.5-VL-7B-Instruct for Signal C and Judge calls."""

    def __init__(self, model_id: str = QWEN_MODEL_ID, device: str = "auto") -> None:
        from transformers import AutoProcessor  # noqa: PLC0415

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Cls  # noqa: PLC0415
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as _Cls  # type: ignore[no-redef]  # noqa: PLC0415

        logger.info("Loading Qwen processor from %s …", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Loading Qwen model (bfloat16, device_map=%s) …", device)
        self.model = _Cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self._device = next(self.model.parameters()).device

    def _run_qwen(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.processor.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def describe(self, image: Image.Image) -> SignalCResult:
        """Signal C — Qwen free description, no external signals passed."""
        raw = self._run_qwen(image.convert("RGB"), _PROMPT_SIGNAL_C, max_new_tokens=256)
        result = _parse_signal_c_response(raw)
        if result.parse_failed:
            logger.warning("Signal C JSON parse failed: %r", raw[:120])
        return result

    def judge(
        self,
        image: Image.Image,
        signal_a: Optional[SAM2Result],
        signal_b: FacePoseInfo,
        signal_c: SignalCResult,
    ) -> JudgeResult:
        """Phase 2 — evaluate all 3 signals and produce final occlusion estimate."""
        geom_pct       = (signal_a.geometric_occ * 100.0) if signal_a else 0.0
        sam2_conf      = signal_a.confidence if signal_a else 0.0
        orientation    = signal_b.orientation if signal_b.detected else "unknown"
        yaw_deg        = signal_b.yaw_deg if signal_b.detected else 0.0
        missing_zones  = ", ".join(signal_b.missing_zones) if signal_b.missing_zones else "none"
        pose_occlusion = signal_b.pose_occlusion if signal_b.detected else 0.0

        # Escape braces in free-text fields to prevent .format() key errors
        description    = signal_c.description[:200].replace("{", "[").replace("}", "]")
        affected_zones = (
            ", ".join(signal_c.affected_zones) if signal_c.affected_zones else "none"
        )

        prompt = _PROMPT_JUDGE.format(
            geom_pct=geom_pct,
            sam2_conf=sam2_conf,
            orientation=orientation,
            yaw_deg=yaw_deg,
            missing_zones=missing_zones,
            pose_occlusion=pose_occlusion,
            description=description,
            occlusion_type=signal_c.occlusion_type,
            affected_zones=affected_zones,
            first_impression_pct=signal_c.first_impression_pct,
        )

        raw = self._run_qwen(image.convert("RGB"), prompt, max_new_tokens=512)
        result = _parse_judge_response(raw)
        if result.parse_failed:
            logger.warning("Judge JSON parse failed: %r", raw[:120])
        return result


# ── Full pipeline ─────────────────────────────────────────────────────────────


class ZeroShotOcclusionPipeline:
    """3-signal judge pipeline: SAM2 + MediaPipe + Qwen(describe) → Qwen(judge).

    Usage:
        pipeline = ZeroShotOcclusionPipeline()
        result = pipeline.predict(Path("data/raw/img.jpg"))
        print(result.prediction)  # float in [0, 1]
    """

    def __init__(
        self,
        sam2_model_id: str = SAM2_MODEL_ID,
        qwen_model_id: str = QWEN_MODEL_ID,
        sam2_device: str = "cuda",
        qwen_device: str = "auto",
    ) -> None:
        self._sam2 = SAM2Segmenter(sam2_model_id, sam2_device)
        self._qwen = QwenEstimator(qwen_model_id, qwen_device)

    def predict(self, image_path: Path) -> PipelineResult:
        """Run 3-signal pipeline. Never raises — failures go in result.failure."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return PipelineResult(prediction=0.10, failure=f"image_load_error: {exc}")

        image_rgb = np.array(image)

        # ── Signal A — SAM2 ───────────────────────────────────────────────────
        sam2       = self._sam2.segment(image)
        a_geom     = sam2.geometric_occ if sam2 else 0.0
        a_conf     = sam2.confidence    if sam2 else 0.0

        # ── Signal B — MediaPipe (CPU, ~10 ms) ───────────────────────────────
        pose = analyze_face_pose(image_rgb)

        # ── Signal C — Qwen free description (no signals) ────────────────────
        try:
            signal_c = self._qwen.describe(image)
        except Exception as exc:
            logger.warning("Qwen describe failed for %s: %s", image_path.name, exc)
            signal_c = SignalCResult(parse_failed=True)

        # ── Phase 2 — Qwen judge ──────────────────────────────────────────────
        try:
            judge = self._qwen.judge(image, sam2, pose, signal_c)
        except Exception as exc:
            logger.warning("Qwen judge failed for %s: %s", image_path.name, exc)
            return PipelineResult(
                prediction=signal_c.first_impression_pct,
                signal_a_geom=a_geom,
                signal_a_conf=a_conf,
                signal_b_orientation=pose.orientation,
                signal_b_pose_occ=pose.pose_occlusion,
                signal_b_missing=",".join(pose.missing_zones),
                signal_c_type=signal_c.occlusion_type,
                signal_c_first_imp=signal_c.first_impression_pct,
                signal_c_description=signal_c.description,
                geometric_occ=a_geom or None,
                used_mask=a_conf >= SAM2_CONF_THRESHOLD,
                failure=f"qwen_judge_error: {exc}",
            )

        return PipelineResult(
            prediction=judge.percentage,
            signal_a_geom=a_geom,
            signal_a_conf=a_conf,
            signal_b_orientation=pose.orientation,
            signal_b_pose_occ=pose.pose_occlusion,
            signal_b_missing=",".join(pose.missing_zones),
            signal_c_type=signal_c.occlusion_type,
            signal_c_first_imp=signal_c.first_impression_pct,
            signal_c_description=signal_c.description,
            sam2_reliable=judge.sam2_reliable,
            mediapipe_reliable=judge.mediapipe_reliable,
            dominant_signal=judge.dominant_signal,
            reasoning=judge.reasoning,
            geometric_occ=a_geom or None,
            used_mask=a_conf >= SAM2_CONF_THRESHOLD,
        )
