import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import tanglegram as tg
import numpy as np

st.set_page_config(page_title='TangleHCA', page_icon=None, layout="wide")
# Streamlit UI
st.title("HCA Tanglegram analysis")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Files")
    with st.form("file_upload_form"):
        file1 = st.file_uploader("Upload first Excel file", type=["xls", "xlsx"])
        file2 = st.file_uploader("Upload second Excel file", type=["xls", "xlsx"])
        submit_files = st.form_submit_button("Upload")
    if submit_files:
        st.session_state['file1'] = file1
        st.session_state['file2'] = file2

# Layout for parameters and plot
if ('file1' in st.session_state) and ('file2' in st.session_state):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Settings")
        distance_metric = st.selectbox("Select Distance metric", [ "euclidean", "sqeuclidean","cityblock","minkowski", "correlation", "cosine","chebyshev", "jaccard", "hamming","canberra"])
        linkage_method = st.selectbox("Select Linkage method", ["single", "complete", "ward", "average", "weighted", "centroid", "median"])
        # # Slider for figure size adjustment
        # fig_size = st.slider("Adjust Figure Size (%)", 50, 100, 75)
        # scale_factor = fig_size / 100.0
        untangle = st.checkbox("Untangle?")
        submit_button = st.button("Submit")
    
    if submit_button:
        df1 = pd.read_excel(file1, index_col=0)
        df2 = pd.read_excel(file2, index_col=0)
        
        # Preserve original sample names
        sample_names = df1.index.tolist()

        # Standardize data
        scaler = StandardScaler()
        z1 = scaler.fit_transform(df1)
        z2 = scaler.fit_transform(df2)

        # Compute distance matrices
        dist_vec1 = pdist(z1, metric=distance_metric)
        dist_vec2 = pdist(z2, metric=distance_metric)

        # Ensure distance matrices are in correct format
        dist_mat1 = squareform(dist_vec1)
        dist_mat2 = squareform(dist_vec2)

        # Compute linkage matrices
        linkage1 = linkage(dist_mat1, method=linkage_method)
        linkage2 = linkage(dist_mat2, method=linkage_method)

        # Generate dendrograms to extract leaf labels
        dendro1 = dendrogram(linkage1, no_plot=True)
        dendro2 = dendrogram(linkage2, no_plot=True)

        # Map original sample names to dendrogram order
        labelsA = [sample_names[int(i)] for i in dendro1["leaves"]]
        labelsB = [sample_names[int(i)] for i in dendro2["leaves"]]
        
        # Generate tanglegram with correct parameter names and adjusted figure size
        fig = tg.plot(linkage1, linkage2, labelsA=labelsA, labelsB=labelsB, sort=untangle, figsize=(12,8))
        
        # Display the plot in the right column
        with col2:
            st.header("Tanglegram Plot")
            st.pyplot(fig)
else:
    st.warning("Please upload both Excel files and submit the parameters to generate the tanglegram.")