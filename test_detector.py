import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import app.detector as detector_module
from app.detector import SafeRideDetector, deduplicate_boxes


def make_detection(class_name, box, confidence=0.9, class_id=0):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "box": box,
    }


def association_detector():
    return SafeRideDetector.__new__(SafeRideDetector)


def make_vehicle(box):
    return make_detection("two_wheeler", box, class_id=2)


def test_detector_initialization_resolves_loaded_model_class_names():
    detector = SafeRideDetector()

    assert detector.model is not None
    assert detector.model_path.exists()
    assert detector.class_kind_by_id[detector.class_ids["helmet"]] == "helmet"
    assert detector.class_kind_by_id[detector.class_ids["no_helmet"]] == "no_helmet"
    assert detector.class_kind_by_id[detector.class_ids["vehicle"]] == "vehicle"


def test_rider_wearing_helmet_is_compliant():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]
    heads = [make_detection("helmet", [180, 110, 220, 170])]

    assessed, unassociated = detector.associate_heads_to_vehicles(vehicles, heads)

    assert assessed[0]["status"] == "COMPLIANT"
    assert assessed[0]["helmets_count"] == 1
    assert assessed[0]["no_helmets_count"] == 0
    assert assessed[0]["riders"][0]["status"] == "COMPLIANT"
    assert unassociated == []


def test_rider_without_helmet_is_violation():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]
    heads = [make_detection("without_helmet", [180, 110, 220, 170])]

    assessed, unassociated = detector.associate_heads_to_vehicles(vehicles, heads)

    assert assessed[0]["status"] == "VIOLATION"
    assert assessed[0]["no_helmets_count"] == 1
    assert assessed[0]["helmets_count"] == 0
    assert assessed[0]["riders"][0]["status"] == "VIOLATION"
    assert unassociated == []


def test_helmet_without_reliable_vehicle_association_needs_review():
    detector = association_detector()
    vehicles = [make_vehicle([50, 200, 200, 450])]
    heads = [make_detection("helmet", [700, 50, 740, 90])]

    assessed, unassociated = detector.associate_heads_to_vehicles(vehicles, heads)

    assert assessed[0]["status"] == "NEEDS REVIEW"
    assert assessed[0]["helmets_count"] == 0
    assert len(unassociated) == 1
    assert unassociated[0]["class_name"] == "helmet"


def test_helmet_and_without_helmet_are_assessed_per_rider():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]
    heads = [
        make_detection("helmet", [145, 110, 175, 150]),
        make_detection("without", [225, 110, 255, 150], class_id=1),
    ]

    assessed, _ = detector.associate_heads_to_vehicles(vehicles, heads)

    vehicle = assessed[0]
    assert vehicle["status"] == "VIOLATION"
    assert vehicle["helmets_count"] == 1
    assert vehicle["no_helmets_count"] == 1
    assert [rider["status"] for rider in vehicle["riders"]] == [
        "COMPLIANT",
        "VIOLATION",
    ]


def test_multiple_vehicles_keep_compliance_results_separate():
    detector = association_detector()
    vehicles = [
        make_vehicle([50, 100, 200, 350]),
        make_vehicle([400, 100, 550, 350]),
    ]
    heads = [
        make_detection("helmet", [110, 110, 140, 150]),
        make_detection("without_helmet", [460, 110, 490, 150]),
    ]

    assessed, unassociated = detector.associate_heads_to_vehicles(vehicles, heads)

    assert [vehicle["status"] for vehicle in assessed] == ["COMPLIANT", "VIOLATION"]
    assert [vehicle["helmets_count"] for vehicle in assessed] == [1, 0]
    assert [vehicle["no_helmets_count"] for vehicle in assessed] == [0, 1]
    assert unassociated == []


def test_no_helmet_detection_is_needs_review_not_violation():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]

    assessed, _ = detector.associate_heads_to_vehicles(vehicles, [])

    assert assessed[0]["status"] == "NEEDS REVIEW"
    assert assessed[0]["no_helmets_count"] == 0
    assert "No reliable" in assessed[0]["reason"]


def test_ambiguous_vehicle_association_needs_review_for_candidates():
    detector = association_detector()
    vehicles = [
        make_vehicle([100, 100, 300, 400]),
        make_vehicle([100, 100, 300, 400]),
    ]
    heads = [make_detection("helmet", [180, 110, 220, 170])]

    assessed, unassociated = detector.associate_heads_to_vehicles(vehicles, heads)

    assert [vehicle["status"] for vehicle in assessed] == [
        "NEEDS REVIEW",
        "NEEDS REVIEW",
    ]
    assert all(vehicle["helmets_count"] == 0 for vehicle in assessed)
    assert all(vehicle["ambiguous_count"] == 1 for vehicle in assessed)
    assert unassociated == []


def test_conflicting_classes_for_same_rider_need_review():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]
    heads = [
        make_detection("helmet", [180, 110, 220, 170], confidence=0.90, class_id=0),
        make_detection("without_helmet", [180, 110, 220, 170], confidence=0.85, class_id=1),
    ]
    deduped = deduplicate_boxes(heads)

    assessed, _ = detector.associate_heads_to_vehicles(vehicles, deduped)

    assert assessed[0]["status"] == "NEEDS REVIEW"
    assert assessed[0]["helmets_count"] == 0
    assert assessed[0]["no_helmets_count"] == 0
    assert assessed[0]["ambiguous_count"] == 1


def test_duplicate_helmet_detections_do_not_inflate_counts():
    detector = association_detector()
    vehicles = [make_vehicle([100, 100, 300, 400])]
    heads = [
        make_detection("helmet", [180, 110, 220, 150], confidence=0.90),
        make_detection("helmet", [182, 111, 222, 151], confidence=0.70),
    ]
    deduped = deduplicate_boxes(heads)

    assessed, _ = detector.associate_heads_to_vehicles(vehicles, deduped)

    assert len(deduped) == 1
    assert assessed[0]["helmets_count"] == 1
    assert assessed[0]["no_helmets_count"] == 0
    assert assessed[0]["status"] == "COMPLIANT"


def test_mocked_yolo_uses_actual_names_and_summary_matches_vehicle_status(
    monkeypatch, tmp_path
):
    class MockBox:
        def __init__(self, class_id, confidence, box):
            self.cls = [np.array([class_id])]
            self.conf = [np.array([confidence])]
            self.xyxy = [np.array(box, dtype=float)]

    class MockModel:
        names = {0: "two_wheeler", 1: "without_helmet", 2: "helmet"}

        def predict(self, source, conf, verbose):
            assert conf == 0.25
            return [
                SimpleNamespace(
                    boxes=[
                        MockBox(0, 0.9, [50, 100, 250, 400]),
                        MockBox(0, 0.9, [350, 100, 550, 400]),
                        MockBox(2, 0.9, [150, 110, 180, 150]),
                        MockBox(2, 0.7, [152, 111, 182, 151]),
                        MockBox(1, 0.9, [450, 110, 480, 150]),
                    ]
                )
            ]

    mock_model = MockModel()
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: mock_model)
    checkpoint = tmp_path / "mock.pt"
    checkpoint.touch()
    detector = SafeRideDetector(str(checkpoint))

    results = detector.detect(np.zeros((500, 600, 3), dtype=np.uint8))

    assert [vehicle["status"] for vehicle in results["vehicles"]] == [
        "COMPLIANT",
        "VIOLATION",
    ]
    assert results["summary"]["helmet_count"] == 1
    assert results["summary"]["no_helmet_count"] == 1
    assert results["summary"]["violation_vehicles"] == 1
    assert results["overall_status"] == "VIOLATION"
    assert results["has_violation"] is True
    assert results["annotated_image"].size == (600, 500)


def test_real_dataset_image_inference():
    detector = SafeRideDetector()
    test_images = (
        Path(__file__).parent.parent.parent
        / "helmetData"
        / "test"
        / "images"
    )
    test_img_path = next(test_images.glob("*.jpg"))
    img = Image.open(test_img_path)

    results = detector.detect(img, conf_threshold=0.25)

    assert "overall_status" in results
    assert "summary" in results
    assert "vehicles" in results
    assert "unassociated_heads" in results
    assert "annotated_image" in results
    assert results["annotated_image"].size == img.size
