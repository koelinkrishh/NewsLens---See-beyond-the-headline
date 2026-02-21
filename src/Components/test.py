from src.Components.Embedding import ArticleEmbeddingEngine
import pandas as pd

from src.config import *


classes = ArticleEmbeddingEngine(batch_size=64)
df = pd.read_parquet(EMBEDDED_DATASET)
print(df.sample(5))


article = df.loc[10,'Content']
em = classes.create_embedding(article)
print("Embeddings: \n", em)
