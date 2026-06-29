

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


 
# Leer los archivos CSV
df_openalex = pd.read_csv('data/OpenAlex_seismic_waves_both.csv')
df_scopus = pd.read_csv('data/Scopus_seismic_waves_both.csv')

# Combinar ambos DataFrames
df_combined = pd.concat(
    [df_openalex, df_scopus],
    ignore_index=True
)

# Eliminar duplicados por título
df_unique = df_combined.drop_duplicates(
    subset=["Title"],
    keep="first"
).reset_index(drop=True)

# Define columns to keep (without document type)
cols = [
     "Item Type",
    "Publication Year",
    "Title",
    "Author",
    "Publication Title",
    "Conference Name",
    "Publisher",
]

# Keep only required columns
publications = df_unique[cols].copy()

# Create unified publication source column
def get_source(row):

    if pd.notna(row["Conference Name"]) and row["Conference Name"] != "":
        parts = [
            row["Conference Name"]
        ]

    elif pd.notna(row["Publisher"]) and row["Publisher"] != "":
        parts = [
            row["Publisher"]
        ]

    else:
        parts = [
            row["Publication Title"]
        ]

    return ", ".join(
        str(x) for x in parts
        if pd.notna(x) and x != ""
    )

publications["Source"] = publications.apply(get_source, axis=1)

# Drop original columns
publications = publications.drop(
    columns=[
        "Publication Title",
        "Conference Name",
        "Publisher"
    ]
)

# Sort by year and title
publications = publications.sort_values(
    by=["Publication Year", "Title"],
    ascending=[False, True]
).reset_index(drop=True)

# Save
publications.to_csv(
    "data/identified_piml_wave_propagation_seismology_review_dataset.csv",
    index=False
)

# Define columns to keep (without document type)
cols = [
     "Item Type",
    "Publication Year",
    "Title",
    "Author",
    "Publication Title",
    "Conference Name",
    "Publisher",
]

# Keep only required columns
publications = df_unique[cols].copy()

# Create unified publication source column
def get_source(row):

    if pd.notna(row["Conference Name"]) and row["Conference Name"] != "":
        parts = [
            row["Conference Name"]
        ]

    elif pd.notna(row["Publisher"]) and row["Publisher"] != "":
        parts = [
            row["Publisher"]
        ]

    else:
        parts = [
            row["Publication Title"]
        ]

    return ", ".join(
        str(x) for x in parts
        if pd.notna(x) and x != ""
    )

publications["Source"] = publications.apply(get_source, axis=1)

# Drop original columns
publications = publications.drop(
    columns=[
        "Publication Title",
        "Conference Name",
        "Publisher"
    ]
)

# Sort by year and title
publications = publications.sort_values(
    by=["Publication Year", "Title"],
    ascending=[False, True]
).reset_index(drop=True)

# Save
publications.to_csv(
    "data/identified_piml_wave_propagation_seismology_review_dataset.csv",
    index=False
)



# Read CSV files
publications_forward = pd.read_csv('data/publications_review_forward.csv')
publications_inverse = pd.read_csv('data/publications_review_inverse.csv')

# Add problem type
publications_forward["Problem Type"] = "Forward"
publications_inverse["Problem Type"] = "Inverse"

# Define columns to keep
cols = [
    "Problem Type",
    "Item Type",
    "Publication Year",
    "Title",
    "Author",
    "Publication Title",
    "Conference Name",
    "Publisher",
]

# Combine both datasets
publications = pd.concat(
    [
        publications_forward[cols],
        publications_inverse[cols]
    ],
    ignore_index=True
)

# Create unified publication source column
def get_source(row):
    item_type = str(row["Item Type"]).lower()

    if "conference" in item_type:
        parts = [
            row["Conference Name"],
        ]
    elif "preprint" in item_type:
        parts = [
            row["Publisher"]
        ]
    else:  # journal article
        parts = [
            row["Publication Title"]
        ]

    return ", ".join(
        str(x) for x in parts 
        if pd.notna(x) and x != ""
    )

publications["Source"] = publications.apply(get_source, axis=1)

# Drop original columns
publications = publications.drop(
    columns=[
        "Publication Title",
        "Conference Name",
        "Publisher"
    ]
)

# Sort by year and title
publications = publications.sort_values(
    by=["Publication Year", "Title"],
    ascending=[False, True]
).reset_index(drop=True)

# Save
publications.to_csv(
    "data/selected_piml_wave_propagation_seismology_review_dataset.csv",
    index=False
)