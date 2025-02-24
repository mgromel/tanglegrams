import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
import tanglegram as tg

# Streamlit UI
st.title("Tanglegram Generator for Two Excel Datasheets")

st.write("Upload two Excel files to generate a tanglegram comparing hierarchical clusters.")

# Upload files
file1 = st.file_uploader("Upload first Excel file", type=["xls", "xlsx"])
file2 = st.file_uploader("Upload second Excel file", type=["xls", "xlsx"])

if file1 and file2:
    df1 = pd.read_excel(file1, index_col=0)
    df2 = pd.read_excel(file2, index_col=0)
    
    st.write("### Preview of Uploaded Data")
    st.write("First Dataset:")
    st.dataframe(df1.head())
    st.write("Second Dataset:")
    st.dataframe(df2.head())
    
    # Standardize data
    scaler = StandardScaler()
    z1 = scaler.fit_transform(df1)
    z2 = scaler.fit_transform(df2)
    
    # Compute distance matrices
    dist_vec1 = pdist(z1, metric='euclidean')
    dist_vec2 = pdist(z2, metric='euclidean')
    
    dist_mat1 = pd.DataFrame(squareform(dist_vec1), index=df1.index.values, columns=df1.index.values)
    dist_mat2 = pd.DataFrame(squareform(dist_vec2), index=df1.index.values, columns=df1.index.values)
    
    # Generate tanglegram
    fig = tg.plot(dist_mat1, dist_mat2, sort=False)
    
    # Display the plot
    st.write("### Tanglegram Plot")
    st.pyplot(fig)
else:
    st.warning("Please upload both Excel files to generate the tanglegram.")
