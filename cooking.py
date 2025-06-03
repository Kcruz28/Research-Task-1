from whoosh import scoring
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os
from whoosh.qparser import QueryParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
import spacy
from transformers import pipeline
import pdfplumber
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from whoosh.analysis import StemmingAnalyzer, CharsetFilter
from whoosh.support.charset import accent_map
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Silence pdfminer/pdfplumber “CropBox missing” messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


class Elisa:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.answers = []

        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            return_all_scores=True,
        )

        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        stem = StemmingAnalyzer()
        analyzer = stem | CharsetFilter(accent_map)
        self.schema = Schema(
            content=TEXT(stored=True, analyzer=analyzer),
            path=ID(stored=True),
        )

        # index directory creation
        if not os.path.exists("indexdir"):
            os.mkdir("indexdir")
            self.ix = index.create_in("indexdir", self.schema)
        else:
            self.ix = index.open_dir("indexdir")


    def set_question(self, question):
        self.user_question = question
        self.search_terms = self.extract_question_entities(self.nlp(question))
        print(f"Updated search terms: {self.search_terms}")

    def pdf_to_text(self, file_path):
        if file_path.lower().endswith(".pdf"):

            laparams = LAParams(char_margin=1.5, line_margin=0.5)
            try:
                text = extract_text(file_path, laparams=laparams)
            except Exception as e:
                raise ValueError(f"PDF reading failed: {str(e)}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text
    
    def chunk_text(self, text, chunk_size=500, overlap=50):

        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def add_document(self, file_path, text_type):
        if text_type.lower() == "pdf":
            document_text = self.pdf_to_text(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                document_text = f.read()

        chunks = self.chunk_text(document_text, chunk_size=1000, overlap=200)  # Adjustable sizes

        writer = self.ix.writer()
        with self.ix.searcher() as searcher:
            query = QueryParser("path", self.ix.schema).parse(f'"{file_path}"')
            results = searcher.search(query)
            if results:
                print(f"Document {file_path} already indexed.")
                return

        for idx, chunk in enumerate(chunks):
            writer.add_document(content=chunk, path=f"{file_path}#chunk{idx}")
        writer.commit()
        print(f"Document {file_path} added to index in {len(chunks)} chunks.")

    def extract_question_entities(self, doc):
        terms = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not token.is_stop
        ]
        return " ".join(terms)


    def query_parsing(self):
        w = scoring.BM25F(B=0.9, content_B=0.9, K1=1.2)
        qp = QueryParser("content", self.ix.schema, group=OrGroup.factory(0.9))

        with self.ix.searcher(weighting=w) as searcher:
            print("Index contains", searcher.doc_count(), "document(s).")
            query = qp.parse(self.search_terms)
            hits = searcher.search(query, limit=20)  # lexical shortlist
            print("BM25 found", len(hits), "candidates.")

            if not hits:
                print("No candidates found.")
                return

            # Stage 1: collect the candidate texts
            candidate_texts = [hit["content"] for hit in hits]

            # Stage 2: compute embeddings for question + candidates
            q_emb = self.sentence_model.encode(
                self.user_question, convert_to_numpy=True
            )
            cand_embs = self.sentence_model.encode(
                candidate_texts, convert_to_numpy=True
            )

            # cosine similarities
            cos_sims = (cand_embs @ q_emb) / (
                np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(q_emb)
            )
            best_idx = int(np.argmax(cos_sims))
            best_para = candidate_texts[best_idx]
            print(
                f"Selected paragraph index {best_idx} with similarity {cos_sims[best_idx]:.4f}"
            )

            # Stage 3: run QA over that single best paragraph
            answer = self.qa_pipeline(question=self.user_question, context=best_para)
            print("="* 100)
            print("Answer:", answer["answer"])
            print("Score:", answer["score"])
            print("=" * 100)

    def query_parsing_tfidf(self):
        with self.ix.searcher() as searcher:
            print("Index contains", searcher.doc_count(), "document(s).")
            qp = QueryParser("content", self.ix.schema, group=OrGroup.factory(0.9))
            query = qp.parse(self.search_terms)
            hits = searcher.search(query, limit=20)  # retrieve candidates
            print("TF-IDF found", len(hits), "candidates.")

            if not hits:
                print("No candidates found.")
                return

            candidate_texts = [hit["content"] for hit in hits]
            doc_paths = [hit["path"] for hit in hits]

            # Prepare the corpus: combine question and candidate texts
            corpus = [self.user_question] + candidate_texts

            # Initialize and fit TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Compute cosine similarity between the question and documents
            similarities = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:]
            ).flatten()

            # Select the best matching document
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]
            best_text = candidate_texts[best_idx]
            best_path = doc_paths[best_idx]

            print(
                f"TF-IDF selected document {best_path} with similarity {best_score:.4f}"
            )

            # Use QA model on the selected document
            answer = self.qa_pipeline(question=self.user_question, context=best_text)
            print("=" * 100)
            print("Answer:", answer["answer"])
            print("Score:", answer["score"])
            print("=" * 100)
