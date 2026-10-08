# LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning

### **Sanjivani Rural Education Society's SANJIVANI COLLEGE OF ENGINEERING, Kopargaon**
*(An Autonomous Institution, Affiliated to SPPU, Pune)*  
**Department of Computer Engineering** | **Academic Year: 2026–2027**  
**Course**: B.Tech Project (Term I & II) | **Project Group ID**: 20

---

##  Project Team & Institutional Information

| PRN No. | Student Name | Role |
| :--- | :--- | :--- |
| **14** | **Samarth Deepak Bhingardive** 
| **62** | **Mayur Vijay Kale** 
| **63** | **Sakshi Nitin Kamodkar** 
| **66** | **Sakshi Kiran Kekan** 

* **Project Guide**: **Dr. P. N. Kalavadekar**
* **Project Coordinator**: **Dr. S. R. Deshmukh**
* **Head of Department (HOD)**: **Dr. M. A. Jawale**

---

##  Sustainable Development Goals (SDGs) & Social Impact

* **Primary SDG**: **SDG 10 – Reduced Inequalities (Target 10.2)**:
  * Empowers hearing-impaired and speech-impaired individuals through an AI-powered visual speech assistive communication system that functions without relying on audio signals.
* **Supporting SDGs**:
  * **SDG 3**: Good Health and Well-being (assistive communication in medical clinics and silent ICU environments).
  * **SDG 4**: Quality Education (inclusive classroom communication for students with auditory challenges).
  * **SDG 9**: Industry, Innovation and Infrastructure (advancing edge computer vision and deep learning accessibility technology).

---

##  Program Outcomes (POs) & Program Specific Outcomes (PSOs) Mapping

| Outcome | Level | Justification |
| :--- | :---: | :--- |
| **PO1 (Engineering Knowledge)** | 3 | Applies computer vision, 3D convolutions, recurrent neural networks, and CTC loss to convert visual lip dynamics into textual transcripts without audio. |
| **PO2 (Problem Analysis)** | 3 | Analyzes multi-person lip reading challenges, face tracking, dynamic head pose variations, and active speaker identification. |
| **PO3 (Design/Development)** | 2 | Designs an end-to-end multi-person Visual Speech Recognition pipeline integrating YOLOv8, MediaPipe Face Mesh, LipNet, and Gradio. |
| **PO4 (Investigations)** | 3 | Evaluates performance across different speaker angles and computes Word Error Rate (WER) and Character Error Rate (CER). |
| **PO5 (Modern Tool Usage)** | 3 | Employs Python, TensorFlow, PyTorch, OpenCV, MediaPipe Face Mesh, YOLOv8, Gradio, Pandas, and NumPy. |
| **PO6 (The Engineer & Society)**| 2 | Develops assistive technology directly addressing accessibility barriers for the speech/hearing-impaired community. |
| **PO8 (Ethics)** | 1 | Ensures ethical handling of private video streams and responsible deployment of computer vision technologies. |
| **PO9 (Individual & Team Work)**| 2 | Collaborative teamwork across system modules, dataloaders, neural network training, and Gradio UI integration. |
| **PO10 (Communication)** | 3 | Generates readable, human-interpretable transcripts with timestamped per-speaker logs and structured documentation. |
| **PO11 (Project Management)** | 2 | Adheres to development milestones, weekly assessment reports, and software requirement specifications (SRS). |
| **PO12 (Life-long Learning)** | 2 | Explores cutting-edge advancements in multimodal AI, Spatiotemporal 3D-CNNs, and Sequence Modeling. |
| **PSO1** | 3 | Applies core software engineering and machine learning principles to construct an intelligent visual speech engine. |
| **PSO2** | 3 | Demonstrates proficiency in Deep Learning, Computer Vision, and interactive full-stack deployment. |
| **PSO3** | 3 | Integrates YOLOv8, MediaPipe, LipNet, and Gradio to solve a real-world communication barrier. |

---

##  Software Requirement Specifications (SRS) Compliance Matrix

| SRS ID | Requirement Title | Implementation Module | Description |
| :--- | :--- | :--- | :--- |
| **FR-1** | Video Input | `app.py`, `core/pipeline.py` | Allows users to upload MP4, AVI, or MPG video files. |
| **FR-2** | Live Camera Input | `app.py` (Tab 2) | Captures real-time video clips from webcam. |
| **FR-3** | Multi-Person Detection | `models/tracker.py` | Detects multiple people / faces using YOLOv8. |
| **FR-4** | Face Tracking | `models/tracker.py` | Continuously tracks face IDs across frames via Centroid/IoU association. |
| **FR-5** | Active Speaker Identification| `core/active_speaker.py` | Distinguishes active speakers from passive listeners via Mouth Aspect Ratio (MAR) variance. |
| **FR-6** | Facial Landmark Detection | `core/preprocessor.py` | Extracts 468 high-precision facial landmarks using MediaPipe Face Mesh. |
| **FR-7** | Lip Region Extraction | `core/preprocessor.py` | Automatically crops the mouth bounding box ($46 \times 96$ pixels). |
| **FR-8** | Image Preprocessing | `core/preprocessor.py` | Normalizes pixel intensities and enforces 75-frame temporal tensor format. |
| **FR-9** | Visual Speech Recognition | `models/lipnet_tf.py` | 3D-CNN + Spatial Dropout + BiGRU + CTC Loss architecture. |
| **FR-10**| Multi-Person Recognition | `core/pipeline.py` | Independently decodes speech for each detected individual. |
| **FR-11**| Transcript Generation | `core/pipeline.py`, `core/vocabulary.py` | Decodes text sequences using CTC Greedy / Beam Search. |
| **FR-12**| Display Results | `app.py` | Renders interactive Gradio UI with subtitled video playback. |
| **FR-13**| Error Handling | `core/pipeline.py` | Gracefully notifies users if no face or insufficient movement is detected. |
| **FR-14**| Output Storage | `utils/export_manager.py` | Exports recognized transcripts into TXT, JSON, and CSV files. |

---

##  System Architecture

```
                                  [ Input Video / Live Webcam ]
                                                │
                                                ▼
                        [ YOLOv8 Multi-Person Face / Body Detection ]
                                                │
                                                ▼
                        [ Centroid / IoU Multi-Person Face Tracker ]
                                                │
                                                ▼
                        [ MediaPipe Face Mesh (468 Landmark Extraction) ]
                                                │
                                                ▼
                        [ Active Speaker Identification (MAR Variance) ]
                                                │
                                                ▼
                        [ Mouth ROI Extraction & 75-Frame Normalization ]
                                 Shape: (Batch, 75, 46, 96, 1)
                                                │
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │            LipNet Neural Network              │
                        │ ───────────────────────────────────────────── │
                        │  • Conv3D Layer 1: 32 filters, 3x5x5 + MaxPool│
                        │  • Conv3D Layer 2: 64 filters, 3x5x5 + MaxPool│
                        │  • Conv3D Layer 3: 96 filters, 3x3x3 + MaxPool│
                        │  • Spatial Flatten: (75, 5760)                │
                        │  • BiGRU Layer 1: 256 units (512 total)       │
                        │  • BiGRU Layer 2: 256 units (512 total)       │
                        │  • Dense Linear Projection: (75, Vocab Size)  │
                        └───────────────────────────────────────────────┘
                                                │
                                                ▼
                        [ CTC Loss (Training) / CTC Beam Search (Inference) ]
                                                │
                                                ▼
                        [ Gradio UI Display & Export to TXT / JSON / CSV ]
```

---

##  Installation & Setup (Windows / Linux)

### 1. Clone or Open Project
```powershell
cd C:\Users\saksh\.gemini\antigravity\scratch\LipNet-VSR
```

### 2. Create Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Quick Run via Batch Script (Windows)
Double-click `run.bat` or execute in PowerShell:
```powershell
.\run.bat
```
The application will start at **`http://127.0.0.1:7860`**.

---

##  Running Individual Components

### 1. Launch Interactive Gradio Web Application
```powershell
python app.py
```

### 2. Train LipNet from Scratch on GRID Dataset
```powershell
python training/train.py --dataset_dir ./data/grid --epochs 50 --batch_size 8 --lr 0.0001
```

### 3. Evaluate Model Benchmarks (WER & CER)
```powershell
python training/evaluate.py
```

---

##  Evaluation Metrics & Formulas

1. **Character Error Rate (CER)**:
   $$\text{CER} = \frac{S + D + I}{N_{\text{chars}}}$$
2. **Word Error Rate (WER)**:
   $$\text{WER} = \frac{S_w + D_w + I_w}{N_{\text{words}}}$$
3. **Connectionist Temporal Classification (CTC) Loss**:
   $$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi | X)$$

---

## 📜 Academic Verification & Sign-off
* **Academic Year**: 2026–2027
* **Department**: Computer Engineering, Sanjivani College of Engineering, Kopargaon
* **Guide**: Dr. P. N. Kalavadekar | **HOD**: Dr. M. A. Jawale
