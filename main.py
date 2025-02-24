import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import tanglegram as tg
import numpy as np
import os

# ----------------------------
# Helper Functions
# ----------------------------
def load_excel(file):
    """
    Loads an Excel file into a DataFrame with the first column as the index.
    """
    return pd.read_excel(file, index_col=0)

def standardize_data(df):
    """
    Standardizes the given DataFrame and returns the transformed array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def compute_distance_matrix(data, distance_metric):
    """
    Computes a condensed distance matrix (vector form) and returns both
    the condensed version and a square-form matrix.
    """
    dist_vector = pdist(data, metric=distance_metric)
    dist_matrix = squareform(dist_vector)
    return dist_vector, dist_matrix

def generate_linkage(dist_matrix, linkage_method):
    """
    Generates and returns the hierarchical clustering linkage based on
    the provided square-form distance matrix.
    """
    return linkage(dist_matrix, method=linkage_method)

def get_dendrogram_labels(linkage_matrix, sample_names):
    """
    Produces a dendrogram (without plotting) and returns labels in the
    same order as the leaves of the dendrogram.
    """
    dendro_info = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendro_info["leaves"]
    return [sample_names[int(i)] for i in ordered_indices]

def create_tanglegram(
    linkage_a, linkage_b, labels_a, labels_b, untangle=False, fig_size=(12, 8), name1='file1', name2='file2',
    annot=False
):
    """
    Creates and returns a tanglegram figure using two linkage matrices
    and lists of labels.
    """
    fig = tg.plot(
        linkage_a, linkage_b, labelsA=labels_a, labelsB=labels_b, 
        sort=untangle, figsize=fig_size
    )
    if annot == True:
        fig.text(0.17, 0.92, f"{name1}", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.83, 0.92, f"{name2}", ha="center", fontsize=14, fontweight="bold")
    return fig

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title='HCA Tanglegram Analysis', layout="wide")
st.title("HCA Tanglegram Analysis")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Files")
    with st.form("file_upload_form"):
        file1 = st.file_uploader("Upload first Excel file", type=["xls", "xlsx"])
        file2 = st.file_uploader("Upload second Excel file", type=["xls", "xlsx"])
        submit_files = st.form_submit_button("Upload")

    # Store references to the uploaded files in session state
    if submit_files:
        st.session_state['file1'] = file1
        st.session_state['file2'] = file2

# Main layout
if 'file1' in st.session_state and 'file2' in st.session_state:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Settings")
        distance_metric = st.selectbox(
            "Select Distance metric",
            ["euclidean", "sqeuclidean", "cityblock", "minkowski", "correlation", 
             "cosine", "chebyshev", "jaccard", "hamming", "canberra"]
        )
        linkage_method = st.selectbox(
            "Select Linkage method",
            ["single", "complete", "ward", "average", "weighted", "centroid", "median"]
        )
        untangle = st.checkbox("Try to untangle")
        annotate = st.checkbox("Show titles")
        submit_button = st.button("Submit")

    if submit_button:
        # Load data
        df1 = load_excel(st.session_state['file1'])
        df2 = load_excel(st.session_state['file2'])

        # Keep original sample names
        sample_names = df1.index.tolist()

        # Standardize data
        standardized_1 = standardize_data(df1)
        standardized_2 = standardize_data(df2)

        # Distance matrices
        _, dist_matrix_1 = compute_distance_matrix(standardized_1, distance_metric)
        _, dist_matrix_2 = compute_distance_matrix(standardized_2, distance_metric)

        # Linkage matrices
        linkage_1 = generate_linkage(dist_matrix_1, linkage_method)
        linkage_2 = generate_linkage(dist_matrix_2, linkage_method)

        # Extract labels from dendrograms
        labels_a = get_dendrogram_labels(linkage_1, sample_names)
        labels_b = get_dendrogram_labels(linkage_2, sample_names)

        # Create tanglegram
        tanglegram_fig = create_tanglegram(
            linkage_1, linkage_2, labels_a, labels_b, untangle, (12, 8), 
            name1=os.path.splitext(st.session_state['file1'].name)[0], 
            name2=os.path.splitext(st.session_state['file2'].name)[0],
            annot=annotate
        )
        st.session_state['tanglegram_fig'] = tanglegram_fig

    if 'tanglegram_fig' in st.session_state:
        with col2:
            st.header("Tanglegram Plot")
            st.pyplot(st.session_state['tanglegram_fig'])
else:
    st.warning("Please upload both Excel files and click 'Upload' on the sidebar.")
