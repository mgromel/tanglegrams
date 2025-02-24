HCA Tanglegram Analysis

Overview
---------
This Streamlit application performs **Hierarchical Cluster Analysis (HCA)** and generates **tanglegrams** to compare two clustering results. The app allows users to upload two datasets, select distance metrics and linkage methods, and visualize the hierarchical clustering results.

Features
---------
- Upload two Excel files containing numerical data for clustering.
- Standardize the data for proper comparison.
- Compute hierarchical clustering using different **distance metrics** and **linkage methods**.
- Generate **dendrograms** and compare them using a **tanglegram**.
- Option to **untangle** the tanglegram for better visualization.

Technologies Used
-----------------
- **Python**: Core scripting language
- **Streamlit**: Web-based interactive visualization
- **Pandas**: Data handling and manipulation
- **Matplotlib**: Visualization of results
- **Scipy**: Hierarchical clustering and distance computations
- **Scikit-learn**: Data preprocessing
- **Tanglegram**: Tanglegram visualization

Installation
------------
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not provided, manually install the dependencies:

```sh
pip install streamlit pandas matplotlib scipy scikit-learn tanglegram
```

Usage
------
1. **Run the Streamlit App**:

   ```sh
   streamlit run main.py
   ```

2. **Upload Your Data**:
   - Upload **two Excel files** (`.xls` or `.xlsx`) with numerical data where the first column represents sample names.
   - Click "Upload" in the sidebar to load the files.

3. **Configure Clustering Parameters**:
   - Choose a **Distance Metric** (e.g., Euclidean, Cosine, Minkowski, etc.).
   - Choose a **Linkage Method** (e.g., Ward, Single, Complete, etc.).
   - Optionally, select "Untangle" for better visualization.

4. **View the Tanglegram**:
   - Click "Submit" to generate the tanglegram.
   - The dendrograms from both datasets are displayed side by side with connecting lines.

Example Input Data Format
-------------------------
The uploaded Excel files should have the following structure:

| Sample Name | Feature 1 | Feature 2 | Feature 3 | ... |
|-------------|----------|----------|----------|-----|
| Sample A    | 2.1      | 3.5      | 1.2      | ... |
| Sample B    | 1.8      | 2.7      | 0.9      | ... |
| Sample C    | 3.2      | 4.1      | 2.5      | ... |

- The first column must contain **sample names**.
- The rest should contain **numeric features** for clustering.

Notes
------
- Ensure both uploaded files have the **same number of samples** and are structured identically.
- If your dataset has missing values, consider preprocessing the data before uploading.
- The **"Untangle"** option attempts to improve visualization by reordering nodes.

License
-------
This project is licensed under the **Apache 2.0 License*.

Author
------
Developed by Maciej Gromelski.
