# Long-Term Object Tracking with SiamFC

**Author:** Ana Poklukar  
**Date:** May 2025  

---

This project was developed for the **Advanced Computer Vision Methods** course at the University of Ljubljana. It extends the **SiamFC (Fully-Convolutional Siamese Network)** short-term tracker into a **long-term tracker**, capable of detecting target disappearance and performing automatic re-detection.

The implementation includes re-detection based on **correlation score thresholding**, with **random and Gaussian sampling strategies** for locating the target again. Performance is evaluated on long-term tracking datasets using standard metrics such as **precision, recall**, and **F-measure**.

---

### Repository Structure

* `siamfc.py`: Core implementation of the SiamFC tracker extended for long-term tracking, including re-detection logic.
* `report.pdf`: Report summarizing the long-term tracking implementation, parameter choices, experiments, and visualizations.
