# Comparative Analysis of Logistic Regression, k-NN, and Decision Tree Classifiers for Diabetes Prediction

This project presents a comparative machine learning study using three classification algorithms — **Logistic Regression**, **k-Nearest Neighbors (k-NN)**, and **Decision Tree** — to predict the presence of diabetes based on clinical data. The analysis leverages the **PIMA Indian Diabetes Dataset**, a widely used benchmark in healthcare-focused data science.

---

##  Objective

To evaluate and compare the performance of three machine learning classifiers for diabetes prediction using:
- Confusion Matrix visualization
- Evaluation metrics: Accuracy, Precision, Recall, and F1-Score
- Interpretability and performance trade-offs

---

##  Repository Contents

- `diabetes_prediction.ipynb` – Jupyter Notebook with preprocessing, modeling, and evaluation.
- `diabetes.csv` – Dataset used (PIMA Indian Diabetes dataset).
- `README.md` – This project description and documentation.

---

##  Tools & Libraries

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

##  Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.79     | 0.78      | 0.76   | 0.77     |
| k-Nearest Neighbors | 0.74     | 0.72      | 0.79   | 0.75     |
| Decision Tree       | 0.71     | 0.70      | 0.68   | 0.69     |

> *Note: These results are based on initial default model configurations. Performance may improve with hyperparameter tuning and cross-validation.*

---

##  Key Insights

- **Logistic Regression** delivered the most balanced performance and robustness. This aligns with findings from Kavakiotis et al. (2017), showing high effectiveness in structured clinical datasets.
- **k-NN** had higher recall, beneficial in medical contexts where minimizing false negatives is critical. However, performance is sensitive to feature scaling and the choice of `k` (Anwar & Suryotrisongko, 2020).
- **Decision Tree** offered easy interpretability but suffered from overfitting and lower accuracy. As noted by Sisodia & Sisodia (2018), proper tuning (e.g., pruning) is necessary for reliable results.

---

##  Visualization

The notebook includes side-by-side confusion matrix plots for visual comparison of:
- Logistic Regression (Blues)
- k-NN (Greens)
- Decision Tree (Oranges)

Each plot provides a clear view of true/false positives and negatives.

---

##  Conclusion

This project demonstrates that **Logistic Regression** performs most reliably for diabetes prediction in this dataset. While **k-NN** excels in recall, and **Decision Tree** offers transparency, Logistic Regression presents the best trade-off between performance and simplicity.

**Future work** may include:
- Hyperparameter optimization
- k-Fold cross-validation
- Ensemble methods such as Random Forest and Gradient Boosting

---

##  References

- Anwar, R., & Suryotrisongko, H. (2020). A comparison of machine learning algorithms for diabetes prediction. *Procedia Computer Science, 135*, 278–285.

- Kavakiotis, I., et al. (2017). Machine learning and data mining methods in diabetes research. *Computational and Structural Biotechnology Journal, 15*, 104–116. [https://doi.org/10.1016/j.csbj.2016.12.005](https://doi.org/10.1016/j.csbj.2016.12.005)

- Sisodia, D., & Sisodia, D. S. (2018). Prediction of diabetes using classification algorithms. *Procedia Computer Science, 132*, 1578–1585. [https://doi.org/10.1016/j.procs.2018.05.321](https://doi.org/10.1016/j.procs.2018.05.321)

- Wang, L., et al. (2019). Comparison of machine learning models for diabetes prediction: A case study of the PIMA Indian dataset. *IJACSA, 10(4)*, 234–239. [https://doi.org/10.14569/IJACSA.2019.0100432](https://doi.org/10.14569/IJACSA.2019.0100432)

---

##  Contact

For any questions or suggestions, please open an issue or contact via GitHub.


