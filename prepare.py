import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/Multi-Labelled_Smiles_Odors_dataset.csv')
    smiles = df.nonStereoSMILES.unique().tolist()
    with open('./data/smiles.txt', 'w', encoding='utf-8') as f:
        for i in smiles:
            f.write(i + '\n')
