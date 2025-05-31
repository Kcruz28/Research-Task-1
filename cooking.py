from whoosh import scoring
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os
from whoosh.qparser import QueryParser
import spacy
from transformers import pipeline


class Elisa:
    def __init__(self, user_question):
        self.user_question = user_question
        self.nlp = spacy.load("en_core_web_sm")
        self.answers = []
        # load bert
        self.qa_pipeline = pipeline(
            "question-answering", model="distilbert-base-cased-distilled-squad"
        )
        # shema definition
        self.schema = Schema(content=TEXT(stored=True), path=ID(stored=True))

        # index directory creation
        if not os.path.exists("indexdir"):
            os.mkdir("indexdir")
            self.ix = index.create_in("indexdir", self.schema)
        else:
            self.ix = index.open_dir("indexdir")

        self.search_terms = self.extract_question_entities(self.nlp(self.user_question))
        print(f"Extracted search terms: {self.search_terms}")


    def add_document(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            document_text = file.read()

        # Open index writer
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
        # print(f"Document {file_path} added to index.")

    def extract_question_entities(self, doc):
        terms = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not token.is_stop]

        return " ".join(terms)

    def extract_entities(self, doc):
        # POS
        # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

        # find named entities, phrases and concepts
        for entity in doc.ents:
            print(entity.text, entity.label_)

    def generate_text(self, pass_content):
        result = self.qa_pipeline(question=self.user_question, context=pass_content)

        #  extracted answer
        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']}")

    def query_parsing(self):
        w = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)

        with self.ix.searcher(weighting=w) as searcher:
            query = QueryParser("content", self.ix.schema).parse(self.search_terms)
            results = searcher.search(query, limit=3)
            for hit in results:
                # print(f"\n=== Retrieved Passage ===")
                # print(f"Found in: {hit['path']}")
                # print(f"Score: {hit.score}")
                # print(f"Content: {hit['content']}")
                # self.extract_entities(self.nlp(hit["content"]))
                # print("\n========================")

                self.generate_text(hit["content"])
