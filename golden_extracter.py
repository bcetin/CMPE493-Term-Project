import pandas as pd

#Creates the golden standards for development set

df_golden= pd.read_csv(r'dataset/BC7-LitCovid-Dev.csv')
df_golden.drop(df_golden.columns.difference(['pmid','label']), 1, inplace=True)
labels = ['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']

for label in labels:
    df_golden[label]=0
    
count=0
for label in df_golden['label']:
    labels_for_doc= label.split(';')
    for l in labels_for_doc:
        df_golden.at[count,l]= 1
    count+=1

df_golden.drop(('label'), inplace=True, axis=1)
df_golden = df_golden.rename(columns={'pmid': 'PMID'})
df_golden = df_golden.set_index('PMID')
print(df_golden)
df_golden.to_csv('biocreative_litcovid/BC7-LitCovid-Golden-Standards.csv')
