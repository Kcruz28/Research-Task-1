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

from sentence_transformers import SentenceTransformer
import numpy as np

# Silence pdfminer/pdfplumber “CropBox missing” messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


class Elisa:
    def __init__(self, user_question):
        self.user_question = user_question
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

        self.search_terms = self.extract_question_entities(self.nlp(self.user_question))
        print(f"Extracted search terms: {self.search_terms}")

    def pdf_to_text(self, file_path):
        if file_path.lower().endswith(".pdf"):
            # Use pdfminer.six instead of pdfplumber
            laparams = LAParams(char_margin=1.5, line_margin=0.5)
            try:
                text = extract_text(file_path, laparams=laparams)
            except Exception as e:
                raise ValueError(f"PDF reading failed: {str(e)}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text

    def add_document(self, file_path, text_type):

        if text_type.lower() == "pdf":
            document_text = self.pdf_to_text(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                document_text = f.read()

        writer = self.ix.writer()

        # Check if document is already indexed
        with self.ix.searcher() as searcher:
            query = QueryParser("path", self.ix.schema).parse(f'"{file_path}"')
            results = searcher.search(query)
            if results:
                # print(f"Document {file_path} is already indexed.")
                return

        # Add document to index
        writer.add_document(content=document_text, path=file_path)
        writer.commit()
        print(f"Document {file_path} added to index.")

    def extract_question_entities(self, doc):
        terms = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not token.is_stop
        ]
        return " ".join(terms)

    def extract_entities(self, doc):
        for entity in doc.ents:
            print(entity.text, entity.label_)

    def generate_text(self, pass_content):
        result = self.qa_pipeline(question=self.user_question, context=pass_content)

        #  extracted answer
        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']}")

    def generate_text2(self, pass_content):
        return self.qa_pipeline(question=self.user_question, context=pass_content)

    def query_parsing(self):
        w = scoring.BM25F(B=0.9, content_B=0.9, K1=1.2)
        qp = QueryParser("content", self.ix.schema, group=OrGroup.factory(0.9))

        with self.ix.searcher(weighting=w) as searcher:
            print("Index contains", searcher.doc_count(), "document(s).")
            # Parse using OR‐semantics
            query = qp.parse(self.search_terms)
            results = searcher.search(query, limit=20)
            print("Found", len(results), "hits for:", self.search_terms)
            for hit in results:
                self.generate_text(hit["content"])
                poss_answer = self.generate_text2(hit["content"])
                self.answers.append((poss_answer["score"], poss_answer["answer"]))
        self.ranking()

    def query_parsing2(self):
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

    def ranking(self):
        if not self.answers:
            print("No answers found.")
            return
        self.answers.sort(reverse=True, key=lambda x: x[0])
        print("top answer: ", self.answers[0][1])
