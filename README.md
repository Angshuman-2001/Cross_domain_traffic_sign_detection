-------------------------------------
# CROSS DOMAIN TRAFFIC SIGN DETECTION
-------------------------------------

## About the project

One of the most critical challenges in deploying computer vision systems is the Domain Shift—where an AI model trained in one environment drastically fails when deployed in another. This project mathematically quantifies and solves this 'Synthetic-to-Real' performance gap in autonomous driving systems.

Initially, a baseline YOLOv8 model trained exclusively on US/Generic road signs was subjected to Zero-Shot Testing on a target German dataset (GTSRB), yielding a catastrophic Mean Average Precision (mAP) of just 1.13%. To mitigate this, a precision-tuned Transfer Learning and data synchronization pipeline was engineered. By implementing a conservative learning rate strategy (to prevent the catastrophic forgetting of original spatial features) and utilizing Mixed-Precision (FP16) training for hardware efficiency, the model successfully adapted to the new domain. The final adapted system bridged the inter-class feature ambiguities, boosting the robust detection accuracy to an exceptional 97.33% mAP, proving its viability for scalable, cross-domain real-world deployment.

```

                ARCHITECTURE OF THE PROJECT
             -----------------------------------

+---------------------------------------------------------------+
|       CROSS-DOMAIN TRAFFIC SIGN DETECTION ARCHITECTURE        |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|              PHASE 0: ENVIRONMENT INITIALIZATION              |
+---------------------------------------------------------------+
|  [1] Mount Google Drive (/content/drive)                      |
|                               |                               |
|                               V                               |
|  [2] Set Working Directory (os.chdir)                         |
|      (Routes all downloads, weights, and logs directly to     |
|      the 'CrossDomainTraffic' folder in Google Drive)         |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|          PHASE 1: DATA PIPELINE (SOURCE & TARGET)             |
+---------------------------------------------------------------+
|  [1] Download Datasets via Roboflow API directly to Drive     |
|      - Source: Asian/US Road Signs (roboflow-100)             |
|      - Target: German Traffic Signs (GTSRB)                   |
|                               |                               |
|                               V                               |
|  [2] Inspect & Visualize Data                                 |
|      (Calculate disk size, image counts, display 10 samples)  |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|       PHASE 2: DATA & CONFIGURATION ALIGNMENT (CRITICAL)      |
+---------------------------------------------------------------+
|  [1] Repair File Paths in Google Drive                        |
|      (Update data.yaml paths for absolute Colab/Drive paths)  |
|                               |                               |
|                               V                               |
|  [2] Prevent IndexError (Label Alignment)                     |
|      (Scan Target .txt labels and filter classes >= Source)   |
|                               |                               |
|                               V                               |
|  [3] Synchronize Metadata                                     |
|      (Force Target YAML 'nc' and 'names' to match Source)     |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|              PHASE 3: BASELINE MODEL TRAINING                 |
+---------------------------------------------------------------+
|  [1] Initialize Base Model                                    |
|      (Load YOLOv8 Nano - yolov8n.pt)                          |
|                               |                               |
|                               V                               |
|  [2] Train Baseline Model                                     |
|      (Train exclusively on Source Data for 15 epochs)         |
|                               |                               |
|                               V                               |
|  [3] Save Baseline Weights to Drive                           |
|      (traffic_project/source_model/weights/best.pt)           |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|              PHASE 4: EVALUATE DOMAIN GAP                     |
+---------------------------------------------------------------+
|  [1] Load Baseline Weights                                    |
|                               |                               |
|                               V                               |
|  [2] Validate Baseline Model on Target Data (German)          |
|                               |                               |
|                               V                               |
|  [3] Record Baseline Accuracy (mAP@50)                        |
|      (Quantifies the "Domain Shift" problem mathematically    |
|       via zero-shot testing)                                  |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|       PHASE 5: DOMAIN ADAPTATION (TRANSFER LEARNING)          |
+---------------------------------------------------------------+
|  [1] Initialize Adaptation Model                              |
|      (Load Baseline weights trained on Source)                |
|                               |                               |
|                               V                               |
|  [2] Fine-Tune on Target Data                                 |
|      (Train on German data with low Learning Rate: 0.005      |
|       to prevent Catastrophic Forgetting)                     |
|                               |                               |
|                               V                               |
|  [3] Save Adapted Weights to Drive                            |
|      (traffic_project/adapted_model/weights/best.pt)          |
+---------------------------------------------------------------+
                                |
                                V
+---------------------------------------------------------------+
|             PHASE 6: FINAL EVALUATION & REPORTING             |
+---------------------------------------------------------------+
|  [1] Validate Adapted Model on Target Data                    |
|                               |                               |
|                               V                               |
|  [2] Calculate Metrics                                        |
|      (Compare Final mAP vs Baseline mAP for gap reduction)    |
|                               |                               |
|                               V                               |
|  [3] Export Project Report to Drive                           |
|      (Save 'Final_Project_Report.txt' permanently)            |
|                               |                               |
|                               V                               |
|  [4] Plot Confusion Matrix                                    |
|      (Read and visualize image directly from Drive)           |
+---------------------------------------------------------------+

```
## Key Results & Project Highlights

This section demonstrates the step-by-step lifecycle of the project, highlighting the initial data mismatch, the engineering solution, and the final quantitative results.

### 1. Datasets details(The Domain Gap)
A snapshot of the initial dataset details. This highlights the fundamental differences in class distributions between the Source (US Signs) and Target (German Signs) domains before any adaptation was applied.
![image alt](https://github.com/Angshuman-2001/Cross_domain_traffic_sign_detection/blob/778d0af48d7f232b60f388b44646bf99fb16b6f0/dataset.png )

### 2. Data alignment & synchronization
Initially, the Source domain (US Signs) contained **21 classes**, while the Target domain (German GTSRB) had **47 classes**. Passing this raw data directly to the neural network would cause an immediate `IndexError` (tensor shape mismatch) during Transfer Learning. This output demonstrates the successful programmatic synchronization, where the target dataset's labels and YAML configurations were filtered and aligned to match the exact **21 classes** of the source domain, ensuring a stable cross-domain fine-tuning pipeline.
![image alt](https://github.com/Angshuman-2001/Cross_domain_traffic_sign_detection/blob/778d0af48d7f232b60f388b44646bf99fb16b6f0/dataAlignment.png)

### 3. Final comparative analysis report
The automated project report generated after fine-tuning. It explicitly quantifies the successful mitigation of the Domain Shift, showcasing the significant jump in mAP from the baseline to the adapted model.
![image alt](https://github.com/Angshuman-2001/Cross_domain_traffic_sign_detection/blob/778d0af48d7f232b60f388b44646bf99fb16b6f0/report.png)

### 4. Classification auditing (confusing report)
The final Confusion Matrix of the adapted model. This visual proof confirms that the model successfully resolved inter-class feature ambiguities within the new target domain.
![image alt](https://github.com/Angshuman-2001/Cross_domain_traffic_sign_detection/blob/778d0af48d7f232b60f388b44646bf99fb16b6f0/confusion.png)


## Execution Environment & Generated Artifacts

**Note on Execution & Storage:**
This project involves heavy computer vision datasets and deep learning models. It was developed and executed in **Google Colab** utilizing the NVIDIA T4 GPU. To ensure persistent storage across sessions and avoid repeated downloading/training, the environment was explicitly mounted to a dedicated **Google Drive** directory.

During execution, the system dynamically fetches data via the **Roboflow API** and automatically generates the following permanent artifacts directly in Google Drive:

* **`road-signs-1/`**: The Source dataset (Generic/US Road Signs) downloaded via Roboflow.
* **`traffic-sign-detection-gtsrb-1/`**: The Target dataset (German Traffic Signs - GTSRB) downloaded via Roboflow.
* **`runs/`**: The core Ultralytics YOLO output directory. This contains all the training runs, validation metrics, confusion matrices, and the final fine-tuned weights (`best.pt`) for both the baseline and adapted models.
* **`yolov8n.pt` & `yolo26n.pt`**: The base pre-trained YOLO nano weight files downloaded to initialize the neural network before transfer learning.
* **`Final_Project_Report.txt`**: A synthesized text report documenting the comparative analysis (Baseline mAP vs Adapted mAP) and the final quantitative improvement.

By establishing this persistent drive structure, the system ensures that raw datasets and trained checkpoints are preserved, allowing for seamless inference without restarting the heavy data ingestion pipeline.
