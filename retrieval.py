from whoosh import scoring
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os
from whoosh.qparser import QueryParser
import spacy
import extraction
from transformers import pipeline
import torch
import generator


nlp = spacy.load("en_core_web_sm")
# Load a pretrained BERT QA pipeline
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-cased-distilled-squad"
)
# Define the schema
schema = Schema(content=TEXT(stored=True), path=ID(stored=True))

# Create an index directory
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    ix = index.create_in("indexdir", schema)
else:
    ix = index.open_dir("indexdir")

with open("example_text.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# Add the document
writer = ix.writer()
writer.add_document(content=document_text, path="example_text.txt")
writer.commit()

user_question = "When was Einstein born?"  # Example question
search_terms = extraction.extract_question_entities(nlp(user_question))
print(f"Extracted search terms: {search_terms}")
# search_terms = "Einstein born"

w = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)


with ix.searcher(weighting=w) as searcher:
    query = QueryParser("content", ix.schema).parse(search_terms)
    results = searcher.search(query, limit=3)
    for hit in results:
        print(f"\n=== Retrieved Passage ===")
        print(f"Found in: {hit['path']}")
        print(f"Score: {hit.score}")
        print(f"Content: {hit['content']}")
        extraction.extract_entities(nlp(hit['content']))
        print("\n========================")
        generator.generate_text(hit['content'], user_question, qa_pipeline)
