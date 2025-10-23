import pandas as pd
def preprocess_dataframe(df):
    """Group coordinates by tomo_id"""
    grouped = df.groupby('tomo_id')
    processed_df = []
    
    for tomo_id, group in grouped:
        coords = []
        for _, row in group.iterrows():
            #print(row)
            coords.append([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])
        
        processed_df.append({
            'tomo_id': tomo_id,
            'coordinates': coords,
            'num_coords': len(coords),
            'Voxel spacing': group['Voxel spacing'].values[0],
            'Array shape (axis 0)': group['Array shape (axis 0)'].values[0],
            'Array shape (axis 1)': group['Array shape (axis 1)'].values[0],
            'Array shape (axis 2)': group['Array shape (axis 2)'].values[0],
        })
    
    return pd.DataFrame(processed_df)

if __name__ == '__main__': 
    new_csv = preprocess_dataframe(pd.read_csv('/home/porpita/BYU/DS/train_labels.csv'))
    new_csv.to_csv('new_train.csv',index=False)
