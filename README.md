# 🧬 MedScan AI — Demo

> **AI-Powered 3D Tumor Segmentation & Cancer Stage Classification from CT Scans**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## About

MedScan AI uses a **nnUNet 3d_fullres** deep learning model trained on **473 abdominal CT scans** to automatically:
- Segment tumors in 3D CT volumes
- Calculate tumor volume in cm³
- Classify cancer stage (Stage 1–4)
- Visualize segmentation overlays on CT slices

**Model Performance (95 validation cases — NVIDIA H100 HPC):**
| Metric | Value |
|--------|-------|
| Pseudo Dice | **0.9716** |
| Epochs | 887 |
| Training Data | 473 CT scans |

---

## Demo Features

### 📂 Browse Existing Cases
Explore real nnUNet segmentation results from the validation set:
- 3D CT scan visualization with tumor overlay (axial + coronal views)
- Tumor volume and voxel count
- Cancer stage classification
- Dice, IoU, Sensitivity, Precision, Specificity, HD95 metrics

> CT backgrounds are synthetically generated for demo (no real patient data).
> Tumor segmentation masks are real nnUNet predictions.

### 🔬 Predict New Scan
Live GPU inference requires local deployment with:
- nnUNet v2 installed
- NVIDIA GPU (H100 recommended)
- Trained checkpoint (`checkpoint_best.pth`)

---

## Run Locally

```bash
git clone https://github.com/yourusername/MedScanAi-Demo
cd MedScanAi-Demo
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select your repo → `app.py`
4. Click **Deploy** ✅

---

## Project Structure

```
MedScanAi-Demo/
├── app.py                  # Streamlit demo app
├── requirements.txt        # Python dependencies
├── predictions/            # Real nnUNet segmentation masks (.nii.gz)
│   ├── case_0002.nii.gz
│   ├── case_0003.nii.gz
│   └── ...
└── .streamlit/
    └── config.toml         # Dark theme config
```

---

## Tech Stack

- **Model:** nnUNet v2 (3d_fullres)
- **Framework:** Streamlit
- **Visualization:** Matplotlib, NumPy
- **Medical Imaging:** NiBabel

---

*Minor Project — 6th Semester | MedScan AI v1.0*
