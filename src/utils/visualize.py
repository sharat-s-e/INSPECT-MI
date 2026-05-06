import pandas as pd

def scanlist_subgroup_check(df, sg_axis, diagnosis_col='diagnosis'):
    for gax in sg_axis:
        ncounts = df.groupby([diagnosis_col,gax]).size().reset_index(name='count')
        print(f"\nEvaluation axis: {gax}")
        print(ncounts.to_string(index=False))