import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(doc):

    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)




def extract_question_entities(doc):
    terms = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB")]
    return " ".join(terms)
