import pandas as pd

def filtering(df_input, control):
    df = pd.read_csv(control.get_filtering_csv())
    df["keep"] = 0
    for count, (k, vs) in enumerate(control.get_filtering_specs().items()):
        df.loc[df[k].isin(vs), "keep"] += 1
        print(f"\tMatches after {k}: {len(df.loc[df.keep==(count+1)])}")
    df = df.loc[df.keep==len(control.get_filtering_specs())]
    print(f"Original shape: {df_input.shape}")
    df_input = df_input.loc[df.CellId]
    print(f"Final shape after filtering: {df_input.shape}")
    return df_input
    