import pandas as pd
import random

df = pd.DataFrame(columns=['uuid','bps','rrt'])

for i in range(10):
    col1 = random.randint(1,50)
    col2 = random.randint(1,50)
    col3 = random.randint(1,50)

    df = df.append({'uuid':col1,
                    'bps':col2,
                    'rrt':col3,
                    }, ignore_index=True)
    
df.to_csv('data.csv')
# print(df)