# Parametric-Fencing
Real-time Runway Surveillance System 🚀 Built with Python, Flask, YOLOv11 &amp; OpenCV. Detects anomalies in user-defined ROIs with CUDA-accelerated inference, multiprocessing, threading &amp; shared memory. Features live streaming, anomaly logging, alerts &amp; a user-friendly dashboard.

**🛡️ Real-Time Runway Surveillance System**
A real-time anomaly detection and alerting system for monitoring airport runways and other critical areas using Computer Vision, Deep Learning, and High-Performance Computing (HPC). The system detects unauthorized entries (people, animals, vehicles, etc.) in a Region of Interest (ROI) and triggers instant alerts via a user-friendly dashboard.

**📌 Features**
🎯 Region of Interest (ROI) Monitoring – Freehand ROI drawing and masking for precise surveillance.
🤖 YOLOv11 + OpenCV – Real-time object detection and segmentation.
⚡ High-Performance Processing – CUDA GPU acceleration (RTX A4000) with ≥30 FPS inference.
🔄 Concurrent Operations – Multiprocessing (camera worker) + multithreading (detection, alerts, streaming).
🔔 Instant Alerts – Visual overlays, audible alarms, and anomaly logging with screenshots.
🌐 Flask Web Dashboard – Live video streaming, ROI configuration, anomaly log page with timestamped downloads.
📊 Scalable & Configurable – Dynamic thresholds, multi-camera support, and efficient shared memory frame handling.

**🖥️ Technology Stack**
Hardware

GPU: NVIDIA RTX A4000 (CUDA 12.8, cuDNN 8.9.3)
CPU: Intel Xeon Gold 6230R (26 cores, 52 threads)
RAM: 256 GB DDR4
Storage: 2TB NVMe SSD + 4TB HDD
Cameras: High-resolution IP/USB cameras
Software
Language: Python 3.12
Frameworks: Flask, PyTorch, Ultralytics YOLOv11, OpenCV
Acceleration: CUDA Toolkit 12.8, cuDNN
Others: NumPy, Multiprocessing, Threading, HTML/CSS/JS (Dashboard)

**🚀 System Workflow**
ROI Definition – User draws ROI on the dashboard.
Real-Time Capture – Cameras stream live video into the system.
Detection & Segmentation – YOLOv11 + CUDA performs object detection.
Intrusion Check – System validates if detected object enters the ROI.
Alert Generation – Visual + audible alerts with screenshots.
Logging & Review – Anomaly logs stored with timestamp and label for analysis.

**📊 Use Cases**
✈️ Aviation Security – Prevent unauthorized access on airport runways.
🏭 Industrial Facilities – Monitor restricted zones for safety compliance.
🏛️ Public Spaces – Secure high-risk areas like government sites or event venues.
🪖 Defense & Military (Future Work) – Border surveillance, drone detection, multi-sensor fusion.

**🧪 Methodology**
Video Capture (IP/USB Cameras → OpenCV).
Object Detection (YOLOv11, CUDA).
ROI Validation (Mask intersection check).
Alerting (Flask dashboard overlays + Windows beep alarms).
Logging (Screenshots, timestamps, anomaly database).

**🔮 Future Work**
🌌 Multi-sensor fusion (thermal, LiDAR, radar).
🛰️ Drone detection & countermeasures (RF jamming, intercept).
🧠 Predictive analytics for anomaly trends.
🔒 Cyber-physical integration with blockchain for tamper-proof logs.
⚡ Edge deployment for remote/low-connectivity environments.

## 📷 System Demo  
For a detailed walkthrough of the system (with screenshots and step-by-step explanation), please consult the **project documentation (DOCX)** included in this repository.  

### Highlights  
- **Dashboard** – Live video, ROI drawing, settings, and anomaly log.  
- **Safe Zone** – ROI monitoring without alerts.  
- **Intrusion Alert** – Bounding boxes + alarms triggered.  
- **Anomalies Page** – Historical logs with downloads.  


**👨‍💻 Team**
Muhammad Rasib
Muhammad Affan Khan
Ahmed Hassan

Supervision: Mr. Abu Baker – Department of High Performance Computing (HPC)

**📌 Conclusion**
This system demonstrates how AI + HPC can transform surveillance into a proactive, scalable, and efficient security solution. By combining computer vision, GPU acceleration, and real-time alerting, it addresses critical gaps in manual monitoring—making runways and other sensitive zones safer, smarter, and future-ready.
