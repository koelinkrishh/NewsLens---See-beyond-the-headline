"""Simple test harness for local HuggingFace embeddings and langchain_postgres PGVector.

Usage:
  python tmp_test_embed_pgvector.py embed    # test local embedding only
  python tmp_test_embed_pgvector.py pg       # test embedding + PGVector (will try to connect using env vars)

It prints clear success/failure messages.
"""
import os
import sys
import argparse
import traceback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document


def test_embed():
    print('== Embedding Test ==')
    try:
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        print('HuggingFaceEmbeddings instantiated:', emb)
        vec = emb.embed_query('Hello world')
        print('Embedding type:', type(vec), 'length:', len(vec))
        print('Embedding test OK')
        return vec
    except Exception as e:
        print('Embedding test FAILED')
        traceback.print_exc()
        raise


def test_pgvector():
    print('\n== PGVector Test ==')
    # Read Postgres connection from environment (same defaults used in project)
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'root')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'news_rag')

    conn = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'
    print('Using connection string:', conn)

    try:
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        vec = emb.embed_query('Hello world')
        emb_len = len(vec)
        print('Embedding length detected:', emb_len)

        print('Instantiating PGVector...')
        vs = PGVector(
            emb,
            connection=conn,
            embedding_length=emb_len,
            collection_name='tmp_newlens_test'
        )
        print('PGVector instantiated:', vs)

        docs = [Document(page_content='Hello world test doc', metadata={'id': '1'})]
        print('Adding documents...')
        vs.add_documents(docs)
        print('Documents added.')

        print('Running similarity search...')
        res = vs.similarity_search('Hello world', k=1)
        print('Search results count:', len(res))
        for r in res:
            print('RESULT:', getattr(r, 'page_content', None), getattr(r, 'metadata', None))

        print('PGVector test OK')
    except Exception as e:
        print('PGVector test FAILED')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['embed', 'pg'], help='Which test to run')
    args = parser.parse_args()

    if args.mode == 'embed':
        try:
            test_embed()
        except Exception:
            sys.exit(2)
    elif args.mode == 'pg':
        try:
            test_pgvector()
        except Exception:
            sys.exit(3)
