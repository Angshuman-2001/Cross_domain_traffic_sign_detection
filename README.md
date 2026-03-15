-------------------------------------
# CROSS DOMAIN TRAFFIC SIGN DETECTION
-------------------------------------

## About the project

One of the most critical challenges in deploying computer vision systems is the Domain Shift—where an AI model trained in one environment drastically fails when deployed in another. This project mathematically quantifies and solves this 'Synthetic-to-Real' performance gap in autonomous driving systems.

Initially, a baseline YOLOv8 model trained exclusively on US/Generic road signs was subjected to Zero-Shot Testing on a target German dataset (GTSRB), yielding a catastrophic Mean Average Precision (mAP) of just 1.13%. To mitigate this, a precision-tuned Transfer Learning and data synchronization pipeline was engineered. By implementing a conservative learning rate strategy (to prevent the catastrophic forgetting of original spatial features) and utilizing Mixed-Precision (FP16) training for hardware efficiency, the model successfully adapted to the new domain. The final adapted system bridged the inter-class feature ambiguities, boosting the robust detection accuracy to an exceptional 97.33% mAP, proving its viability for scalable, cross-domain real-world deployment.

```
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
