import pandas as pd

def scanlist_subgroup_check(df, sg_axis, diagnosis_col='diagnosis'):
    print(f"\n===== Subgroup wise counts =====")
    for gax in sg_axis:
        ncounts = df.groupby([diagnosis_col,gax]).size().reset_index(name='count')
        print(f"=> Evaluation axis: {gax}")
        print(f"{ncounts.to_string(index=False)}")
    print(f"================================\n")