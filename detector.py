"""
SafeRide AI - Detector Engine
Vehicle-first detection and geometric rider-helmet association engine.

Workflow:
1. Resolve model class IDs from the checkpoint's actual class names.
2. Detect two-wheelers, helmeted riders, and riders without helmets.
3. Deduplicate overlapping and conflicting head/vehicle boxes.
4. Geometrically associate head boxes with a vehicle's rider zone.
5. Assess individual rider detections and derive each vehicle's status:
   - COMPLIANT: At least one associated helmet and no associated no-helmet riders.
   - VIOLATION: At least one reliably associated no-helmet rider.
   - NEEDS REVIEW: No reliable rider detection or ambiguous classification/association.
6. Catalog unassociated heads (e.g. pedestrians/bystanders) so they are NOT
   falsely counted as compliant vehicles or vehicle violations.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Visualization Color Palette (BGR format for OpenCV)
COLOR_VEHICLE_BOX = (255, 215, 0)     # Cyan / Sky-Blue for vehicle bounding box
COLOR_HELMET_BOX = (0, 225, 0)        # Vibrant Green for compliant helmet
COLOR_NO_HELMET_BOX = (30, 30, 235)   # Vibrant Red for no-helmet violation
COLOR_UNCERTAIN_BOX = (0, 165, 255)   # Amber / Orange for uncertain review
COLOR_UNRELATED_BOX = (160, 160, 160) # Muted Gray for unassociated pedestrian/bystander

COLOR_BADGE_COMPLIANT = (0, 180, 0)     # Green banner
COLOR_BADGE_VIOLATION = (30, 30, 220)   # Red banner
COLOR_BADGE_UNCERTAIN = (0, 140, 255)   # Amber banner

STATUS_COMPLIANT = "COMPLIANT"
STATUS_VIOLATION = "VIOLATION"
STATUS_NEEDS_REVIEW = "NEEDS REVIEW"
ASSOCIATION_AMBIGUITY_MARGIN = 0.10


def _normalize_class_name(class_name: Any) -> str:
    return str(class_name).strip().lower().replace("-", "_").replace(" ", "_")


def _class_kind(class_name: Any) -> Optional[str]:
    normalized = _normalize_class_name(class_name)
    if normalized == "helmet":
        return "helmet"
    if normalized in {"without", "without_helmet", "no_helmet", "nohelmet"}:
        return "no_helmet"
    if normalized in {"two_wheeler", "twowheeler"}:
        return "vehicle"
    return None


def _detection_kind(detection: Dict[str, Any]) -> Optional[str]:
    kind = detection.get("kind")
    if kind in {"helmet", "no_helmet", "vehicle"}:
        return kind
    return _class_kind(detection.get("class_name", ""))


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection-over-Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def deduplicate_boxes(
    boxes: List[Dict[str, Any]],
    iou_thresh: float = 0.40,
    resolve_conflicts: bool = True
) -> List[Dict[str, Any]]:
    """
    Suppresses duplicate bounding boxes.
    - If same class: keeps higher confidence detection.
    - If conflicting class (helmet vs without) on the same head:
      - If confidence difference is high (>= 0.20), keeps higher confidence.
      - If confidence difference is low (< 0.20), marks as ambiguous / uncertain.
    """
    if not boxes:
        return []

    # Sort descending by confidence
    sorted_boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    suppressed_indices = set()

    for i in range(len(sorted_boxes)):
        if i in suppressed_indices:
            continue
        curr = dict(sorted_boxes[i])
        curr.setdefault("ambiguous", False)

        for j in range(i + 1, len(sorted_boxes)):
            if j in suppressed_indices:
                continue
            other = sorted_boxes[j]
            iou = compute_iou(curr["box"], other["box"])
            if iou >= iou_thresh:
                suppressed_indices.add(j)
                if resolve_conflicts and _detection_kind(curr) != _detection_kind(other):
                    conf_diff = curr["confidence"] - other["confidence"]
                    if conf_diff < 0.20:
                        curr["ambiguous"] = True
                        curr["ambiguity_reason"] = (
                            f"Conflicting predictions (IoU={iou:.2f}: "
                            f"{curr['class_name']}@{curr['confidence']:.2f} vs "
                            f"{other['class_name']}@{other['confidence']:.2f})"
                        )

        kept.append(curr)

    return kept


class SafeRideDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize detector with custom trained YOLOv8 checkpoint.
        """
        if model_path is None:
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / "models" / "best.pt"
            if not model_path.exists():
                fallback = base_dir / "runs" / "helmet_val_5ep" / "weights" / "best.pt"
                if fallback.exists():
                    model_path = fallback

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        names = self.model.names
        named_classes = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_kind_by_id: Dict[int, str] = {}
        self.class_ids: Dict[str, int] = {}
        for class_id, class_name in named_classes:
            kind = _class_kind(class_name)
            if kind is None:
                continue
            if kind in self.class_ids:
                raise ValueError(f"Multiple model classes map to '{kind}'.")
            normalized_id = int(class_id)
            self.class_kind_by_id[normalized_id] = kind
            self.class_ids[kind] = normalized_id

        missing_kinds = {"helmet", "no_helmet", "vehicle"} - self.class_ids.keys()
        if missing_kinds:
            raise ValueError(
                "Model class names must include helmet, without_helmet/without, "
                f"and two_wheeler; missing: {', '.join(sorted(missing_kinds))}."
            )

    def associate_heads_to_vehicles(
        self,
        vehicles: List[Dict[str, Any]],
        heads: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Geometrically associates detected head boxes with detected two-wheelers.
        Uses Rider Region of Interest (RROI) and normalized alignment scoring.
        """
        vehicle_records: List[Dict[str, Any]] = []
        for idx, v in enumerate(vehicles, start=1):
            vx1, vy1, vx2, vy2 = v["box"]
            vw = max(1.0, vx2 - vx1)
            vh = max(1.0, vy2 - vy1)
            vehicle_records.append({
                "vehicle_id": idx,
                "box": v["box"],
                "confidence": v["confidence"],
                "rroi": [
                    vx1 - 0.15 * vw,             # horizontal left margin
                    vy1 - 0.35 * vh,             # vertical top margin (rider helmet above bike)
                    vx2 + 0.15 * vw,             # horizontal right margin
                    vy1 + 0.65 * vh              # vertical bottom cutoff (upper 65% of bike)
                ],
                "center_x": (vx1 + vx2) / 2.0,
                "center_y": (vy1 + vy2) / 2.0,
                "width": vw,
                "height": vh,
                "associated_heads": [],
                "ambiguous_heads": []
            })

        unassociated_heads: List[Dict[str, Any]] = []

        for h in heads:
            hx1, hy1, hx2, hy2 = h["box"]
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            head_area = max(1.0, (hx2 - hx1) * (hy2 - hy1))
            candidate_vehicles = []
            for vr in vehicle_records:
                rx1, ry1, rx2, ry2 = vr["rroi"]
                intersection_width = max(0.0, min(hx2, rx2) - max(hx1, rx1))
                intersection_height = max(0.0, min(hy2, ry2) - max(hy1, ry1))
                head_coverage = intersection_width * intersection_height / head_area
                center_in_region = rx1 <= hcx <= rx2 and ry1 <= hcy <= ry2
                if center_in_region or head_coverage >= 0.5:
                    # Score association by alignment to the vehicle's rider zone.
                    horiz_dist = abs(hcx - vr["center_x"]) / vr["width"]
                    vert_dist = abs(hcy - (vr["box"][1] + 0.20 * vr["height"])) / vr["height"]
                    score = horiz_dist * 0.6 + vert_dist * 0.4
                    candidate_vehicles.append((score, vr))

            if candidate_vehicles:
                candidate_vehicles.sort(key=lambda x: x[0])
                best_score = candidate_vehicles[0][0]
                plausible_candidates = [
                    candidate for candidate in candidate_vehicles
                    if candidate[0] - best_score <= ASSOCIATION_AMBIGUITY_MARGIN
                ]
                if len(plausible_candidates) > 1:
                    candidate_ids = [candidate[1]["vehicle_id"] for candidate in plausible_candidates]
                    for _, candidate_vehicle in plausible_candidates:
                        ambiguous_head = dict(h)
                        ambiguous_head["association_ambiguous"] = True
                        ambiguous_head["candidate_vehicle_ids"] = candidate_ids
                        candidate_vehicle["ambiguous_heads"].append(ambiguous_head)
                else:
                    candidate_vehicles[0][1]["associated_heads"].append(dict(h))
            else:
                # Head does not match any two-wheeler (pedestrian, bystander, distant)
                unassociated_heads.append(h)

        # Assess each vehicle separately
        assessed_vehicles: List[Dict[str, Any]] = []
        for vr in vehicle_records:
            v_heads = vr["associated_heads"]
            ambiguous_heads = vr["ambiguous_heads"]
            helmets = [
                h for h in v_heads
                if _detection_kind(h) == "helmet" and not h.get("ambiguous", False)
            ]
            no_helmets = [
                h for h in v_heads
                if _detection_kind(h) == "no_helmet" and not h.get("ambiguous", False)
            ]
            rider_records = []
            for rider_index, head in enumerate(v_heads + ambiguous_heads, start=1):
                kind = _detection_kind(head)
                if head.get("ambiguous") or head.get("association_ambiguous"):
                    rider_status = STATUS_NEEDS_REVIEW
                    rider_reason = head.get(
                        "ambiguity_reason",
                        "Detection overlaps multiple plausible vehicle rider zones",
                    )
                elif kind == "helmet":
                    rider_status = STATUS_COMPLIANT
                    rider_reason = "Helmet detection associated with this rider zone"
                elif kind == "no_helmet":
                    rider_status = STATUS_VIOLATION
                    rider_reason = "Without-helmet detection associated with this rider zone"
                else:
                    rider_status = STATUS_NEEDS_REVIEW
                    rider_reason = "Detection class could not be assessed"
                rider_records.append({
                    "rider_id": rider_index,
                    "kind": kind,
                    "status": rider_status,
                    "reason": rider_reason,
                    "confidence": head["confidence"],
                    "box": head["box"],
                })

            ambiguous_count = sum(
                1 for h in v_heads + ambiguous_heads
                if h.get("ambiguous") or h.get("association_ambiguous")
            )
            if no_helmets:
                status = STATUS_VIOLATION
                reason = f"{len(no_helmets)} associated rider(s) detected without a helmet"
            elif ambiguous_count:
                status = STATUS_NEEDS_REVIEW
                reason = "Helmet/rider association is ambiguous; manual review required"
            elif helmets:
                status = STATUS_COMPLIANT
                reason = f"{len(helmets)} associated helmet(s); no without-helmet detections"
            else:
                status = STATUS_NEEDS_REVIEW
                reason = "No reliable helmet or without-helmet detection associated with a rider"

            assessed_vehicles.append({
                "vehicle_id": vr["vehicle_id"],
                "box": vr["box"],
                "confidence": round(vr["confidence"], 4),
                "status": status,
                "reason": reason,
                "heads": v_heads + ambiguous_heads,
                "riders": rider_records,
                "helmets_count": len(helmets),
                "no_helmets_count": len(no_helmets),
                "ambiguous_count": ambiguous_count,
            })

        return assessed_vehicles, unassociated_heads

    def detect(self, image: Image.Image | np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run vehicle-first detection and geometric rider-helmet association.
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()

        # Step 1: YOLO inference
        results = self.model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
        result = results[0]

        raw_vehicles: List[Dict[str, Any]] = []
        raw_heads: List[Dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [int(x) for x in box.xyxy[0].tolist()]

                kind = self.class_kind_by_id.get(cls_id)
                if kind is None:
                    continue
                item = {
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "kind": kind,
                    "confidence": conf,
                    "box": xyxy,
                    "ambiguous": False
                }

                if kind == "vehicle":
                    raw_vehicles.append(item)
                elif kind in {"helmet", "no_helmet"}:
                    raw_heads.append(item)

        # Step 2: Deduplication
        deduped_vehicles = deduplicate_boxes(raw_vehicles, iou_thresh=0.65, resolve_conflicts=False)
        deduped_heads = deduplicate_boxes(raw_heads, iou_thresh=0.40, resolve_conflicts=True)

        # Step 3: Association & Assessment
        assessed_vehicles, unassociated_heads = self.associate_heads_to_vehicles(
            deduped_vehicles, deduped_heads
        )

        # Step 4: Summary Metrics
        compliant_v = sum(1 for v in assessed_vehicles if v["status"] == STATUS_COMPLIANT)
        violation_v = sum(1 for v in assessed_vehicles if v["status"] == STATUS_VIOLATION)
        uncertain_v = sum(1 for v in assessed_vehicles if v["status"] == STATUS_NEEDS_REVIEW)
        helmet_count = sum(v["helmets_count"] for v in assessed_vehicles)
        no_helmet_count = sum(v["no_helmets_count"] for v in assessed_vehicles)
        total_v = len(assessed_vehicles)

        if violation_v > 0:
            overall_status = STATUS_VIOLATION
        elif uncertain_v > 0:
            overall_status = STATUS_NEEDS_REVIEW
        elif compliant_v > 0 and total_v > 0:
            overall_status = STATUS_COMPLIANT
        elif total_v == 0:
            overall_status = "NO TWO-WHEELERS DETECTED"
        else:
            overall_status = STATUS_NEEDS_REVIEW

        # Step 5: Draw Annotated Image
        annotated_bgr = self._render_annotations(
            img_bgr, assessed_vehicles, unassociated_heads
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "overall_status": overall_status,
            "has_violation": (violation_v > 0),
            "summary": {
                "total_vehicles": total_v,
                "compliant_vehicles": compliant_v,
                "violation_vehicles": violation_v,
                "uncertain_vehicles": uncertain_v,
                "helmet_count": helmet_count,
                "no_helmet_count": no_helmet_count,
                "unassociated_heads": len(unassociated_heads)
            },
            "vehicles": assessed_vehicles,
            "unassociated_heads": unassociated_heads,
            "annotated_image": Image.fromarray(annotated_rgb),
            "model_path": str(self.model_path)
        }

    def _render_annotations(
        self,
        img_bgr: np.ndarray,
        vehicles: List[Dict[str, Any]],
        unassociated_heads: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Renders color-coded boxes and informative badges.
        - Cyan/Blue: Two-wheeler vehicle bounding boxes
        - Green: Compliant helmet head boxes
        - Red: Suspected no-helmet head boxes
        - Amber: Uncertain / ambiguous head boxes
        - Gray: Unassociated pedestrian/bystander head boxes
        """
        canvas = img_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw vehicle bounding boxes and status badges
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            status = v["status"]

            # Cyan/blue vehicle box
            cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), COLOR_VEHICLE_BOX, 2)

            # Determine badge styling based on vehicle status
            if status == STATUS_VIOLATION:
                badge_color = COLOR_BADGE_VIOLATION
                badge_text = f"Two-Wheeler #{v['vehicle_id']} [VIOLATION]"
            elif status == STATUS_COMPLIANT:
                badge_color = COLOR_BADGE_COMPLIANT
                badge_text = f"Two-Wheeler #{v['vehicle_id']} [COMPLIANT]"
            else:
                badge_color = COLOR_BADGE_UNCERTAIN
                badge_text = f"Two-Wheeler #{v['vehicle_id']} [NEEDS REVIEW]"

            # Draw vehicle status banner above box
            font_scale = 0.50
            thickness = 1
            (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
            banner_y1 = max(0, vy1 - th - 8)
            banner_y2 = vy1
            cv2.rectangle(canvas, (vx1, banner_y1), (vx1 + tw + 8, banner_y2), badge_color, -1)
            cv2.putText(canvas, badge_text, (vx1 + 4, max(th + 2, vy1 - 4)),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # 2. Draw associated head boxes for this vehicle
            for h in v["heads"]:
                hx1, hy1, hx2, hy2 = h["box"]
                kind = _detection_kind(h)
                conf = h["confidence"]
                ambiguous = h.get("ambiguous", False) or h.get("association_ambiguous", False)

                if ambiguous:
                    h_color = COLOR_UNCERTAIN_BOX
                    h_label = f"Rider NEEDS REVIEW: {conf:.2f}"
                elif kind == "helmet":
                    h_color = COLOR_HELMET_BOX
                    h_label = f"Helmet: {conf:.2f}"
                elif kind == "no_helmet":
                    h_color = COLOR_NO_HELMET_BOX
                    h_label = f"NO HELMET: {conf:.2f}"
                else:
                    h_color = COLOR_UNCERTAIN_BOX
                    h_label = f"Unclassified: {conf:.2f}"

                # Draw head bounding box
                cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), h_color, 2)

                # Draw small connector line from head to vehicle spine
                hcx, hcy = int((hx1 + hx2) / 2), int(hy2)
                vcx, vcy = int((vx1 + vx2) / 2), int(vy1 + 0.3 * (vy2 - vy1))
                cv2.line(canvas, (hcx, hcy), (vcx, vcy), h_color, 1, cv2.LINE_AA)

                # Head label
                (htw, hth), _ = cv2.getTextSize(h_label, font, 0.45, 1)
                cv2.rectangle(canvas, (hx1, max(0, hy1 - hth - 4)), (hx1 + htw + 4, hy1), h_color, -1)
                cv2.putText(canvas, h_label, (hx1 + 2, max(hth, hy1 - 3)),
                            font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw unassociated heads (e.g. pedestrians/bystanders)
        for uh in unassociated_heads:
            ux1, uy1, ux2, uy2 = uh["box"]
            uconf = uh["confidence"]
            kind = _detection_kind(uh)
            cname = "Helmet" if kind == "helmet" else "No-Helmet"
            ulabel = f"Unrelated {cname}: {uconf:.2f}"

            # Muted gray dashed-look box
            cv2.rectangle(canvas, (ux1, uy1), (ux2, uy2), COLOR_UNRELATED_BOX, 1)
            (utw, uth), _ = cv2.getTextSize(ulabel, font, 0.40, 1)
            cv2.rectangle(canvas, (ux1, max(0, uy1 - uth - 4)), (ux1 + utw + 4, uy1), COLOR_UNRELATED_BOX, -1)
            cv2.putText(canvas, ulabel, (ux1 + 2, max(uth, uy1 - 3)),
                        font, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas
