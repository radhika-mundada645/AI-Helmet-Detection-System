"""
SafeRide AI - Streamlit Web Dashboard
Vehicle-First Helmet Compliance Monitoring & Violation Detection.
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Add SafeRide-AI root to sys.path so app modules import cleanly
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.detector import SafeRideDetector

st.set_page_config(
    page_title="SafeRide AI - Helmet Violation Detection",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-violation {
        background-color: #FFCDD2;
        color: #B71C1C;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-compliant {
        background-color: #C8E6C9;
        color: #1B5E20;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-uncertain {
        background-color: #FFE0B2;
        color: #E65100;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .stAlert {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛵 SafeRide AI — Vehicle-First Traffic Safety Intelligence")
st.caption("Automated Two-Wheeler Localization, Geometric Rider-Helmet Association & Violation Detection")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Model Loading
@st.cache_resource
def load_detector():
    model_path = root_dir / "models" / "best.pt"
    if not model_path.exists():
        fallback = root_dir / "runs" / "helmet_val_5ep" / "weights" / "best.pt"
        if fallback.exists():
            model_path = fallback
        else:
            return None, "Model weights not found. Please train or verify models/best.pt."
    try:
        detector = SafeRideDetector(model_path=str(model_path))
        return detector, str(model_path)
    except Exception as e:
        return None, str(e)


detector, model_info = load_detector()

if detector is None:
    st.sidebar.error(f"⚠️ Model Error: {model_info}")
    st.error(f"Cannot initialize SafeRide AI detector: {model_info}")
    st.stop()
else:
    st.sidebar.success(f"✅ Model Loaded: `{Path(model_info).name}`")
    st.sidebar.caption(f"Path: `{model_info}`")

conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required for object detection"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Image Source")

input_option = st.sidebar.radio(
    "Choose Input Method",
    ["Select Dataset Test Sample", "Upload Custom Image"]
)

selected_image = None
image_name = ""

# Look for test dataset images
test_dir = root_dir.parent / "helmetData" / "test" / "images"

if input_option == "Select Dataset Test Sample":
    if test_dir.exists():
        test_samples = sorted([p.name for p in test_dir.glob("*.jpg")])[:40]
        if test_samples:
            selected_file = st.sidebar.selectbox("Choose sample image:", test_samples)
            if selected_file:
                img_path = test_dir / selected_file
                selected_image = Image.open(img_path)
                image_name = selected_file
        else:
            st.sidebar.warning("No .jpg files found in test dataset directory.")
    else:
        st.sidebar.warning(f"Test dataset directory not found at `{test_dir}`.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        image_name = uploaded_file.name

# Main Display
if selected_image is not None:
    with st.spinner("Analyzing two-wheelers and performing geometric rider association..."):
        results = detector.detect(selected_image, conf_threshold=conf_threshold)

    summary = results["summary"]
    overall_status = results["overall_status"]
    vehicles = results["vehicles"]
    unassociated_heads = results["unassociated_heads"]
    annotated_img = results["annotated_image"]

    # Top Status Banner
    if overall_status == "VIOLATION":
        st.error(f"🚨 **VIOLATION**: {summary['violation_vehicles']} vehicle(s) have an associated without-helmet detection.")
    elif overall_status == "NEEDS REVIEW":
        st.warning(f"⚠️ **NEEDS REVIEW**: {summary['uncertain_vehicles']} vehicle(s) have uncertain detections or ambiguous rider associations.")
    elif overall_status == "COMPLIANT":
        st.success(f"✅ **COMPLIANT**: All {summary['compliant_vehicles']} detected two-wheeler(s) are compliant (helmets worn).")
    else:
        st.info("ℹ️ No two-wheelers detected in this image at current confidence threshold.")

    # Metric Columns
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Two-Wheelers Assessed", summary["total_vehicles"])
    with m2:
        st.metric("Compliant Vehicles", summary["compliant_vehicles"])
    with m3:
        st.metric("Violation Vehicles", summary["violation_vehicles"])
    with m4:
        st.metric("Needs Review Vehicles", summary["uncertain_vehicles"])
    with m5:
        st.metric("Helmet Detections", summary["helmet_count"])
    with m6:
        st.metric("Without-Helmet Detections", summary["no_helmet_count"])

    st.markdown("---")

    # Legend
    st.markdown(
        "**Legend**: "
        "🟦 `<span style='color:#00D7FF;font-weight:bold;'>Cyan/Blue: Two-Wheeler</span>` | "
        "🟩 `<span style='color:#00E100;font-weight:bold;'>Green: Associated Helmet (Compliant)</span>` | "
        "🟥 `<span style='color:#EB2828;font-weight:bold;'>Red: Associated Without-Helmet (Violation)</span>` | "
        "🟨 `<span style='color:#FFA500;font-weight:bold;'>Yellow: Needs Review / Ambiguous</span>` | "
        "⬜ `<span style='color:#A0A0A0;font-weight:bold;'>Gray: Unrelated Pedestrian / Bystander</span>`",
        unsafe_allow_html=True
    )

    # Side-by-side Image View
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(selected_image, caption=f"Source: {image_name}", use_container_width=True)

    with col2:
        st.subheader("Vehicle-First Compliance Overlay")
        st.image(annotated_img, caption="Two-Wheelers & Associated Rider Heads", use_container_width=True)

    # Vehicle-by-Vehicle Assessment Table
    st.subheader("🛵 Per-Vehicle Assessment Breakdown")
    if vehicles:
        table_rows = []
        for v in vehicles:
            status_badge = v["status"]
            if status_badge == "VIOLATION":
                status_display = "🚨 VIOLATION"
            elif status_badge == "COMPLIANT":
                status_display = "✅ COMPLIANT"
            else:
                status_display = "⚠️ NEEDS REVIEW"

            rider_assessments = "; ".join(
                f"Rider {rider['rider_id']}: {rider['status']} ({rider['reason']})"
                for rider in v["riders"]
            ) or "No reliable rider detections"
            table_rows.append({
                "Vehicle ID": f"Two-Wheeler #{v['vehicle_id']}",
                "Status": status_display,
                "Vehicle Conf": f"{v['confidence'] * 100:.1f}%",
                "Helmet Count": v["helmets_count"],
                "No-Helmet Count": v["no_helmets_count"],
                "Ambiguous Heads": v["ambiguous_count"],
                "Helmet Boxes": str([
                    head["box"] for head in v["heads"] if head.get("kind") == "helmet"
                ]),
                "Without-Helmet Boxes": str([
                    head["box"] for head in v["heads"] if head.get("kind") == "no_helmet"
                ]),
                "Individual Rider Assessments": rider_assessments,
                "Assessment Reason": v["reason"],
                "Bounding Box (x1,y1,x2,y2)": str(v["box"])
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No two-wheelers detected in this scene.")

    # Unassociated Heads Section
    if unassociated_heads:
        st.markdown("---")
        st.subheader("🚶 Unrelated Head Detections (Pedestrians / Bystanders)")
        st.caption(
            "These head detections are not geometrically associated with any two-wheeler. "
            "They are correctly excluded from vehicle compliance evaluation."
        )
        unassoc_rows = []
        for i, uh in enumerate(unassociated_heads, start=1):
            unassoc_rows.append({
                "#": i,
                "Type": "Helmet" if uh.get("kind") == "helmet" else "Without-Helmet",
                "Confidence": f"{uh['confidence'] * 100:.1f}%",
                "Box": str(uh["box"])
            })
        st.dataframe(unassoc_rows, use_container_width=True)

else:
    st.info("👈 Please select a sample image from the test set or upload an image using the sidebar to begin analysis.")
