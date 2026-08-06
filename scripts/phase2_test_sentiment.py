from src.phase2.finbert.inference_pipeline import batch_score
import pandas as pd

sample = [{ 'title':'Apple reports record quarter', 'description':'Strong demand...', 'content':'Apple reported revenue up 20%...'}]
df = pd.DataFrame(sample)
res = batch_score(df, text_col='content')
print(res.head())
