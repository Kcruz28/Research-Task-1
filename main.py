import model
import pdfplumber
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


if "__main__" == __name__:
    path = "demo.txt"
    split_path = path.split(".")
    if split_path[-1].lower() == "pdf":
        text_type = "pdf"
    elif split_path[-1].lower() == "txt":
        text_type = "txt"
    else:
        print("Unsupported file type. Please provide a PDF or TXT file.")
        exit(1)

    print("Loading models")
    elisa = model.Elisa()

    while True:
        print("-"*100)
        question = input("What is your question? ")
        print("-" * 100)
        if question.lower() in ["exit", "quit", 'q']:
            print("Exiting. BYYEEEEEEEE")
            break
        elif not question.strip():
            print("Please enter a valid question.")
            continue

        elisa.set_question(question)
        elisa.add_document(path, text_type)
        elisa.query_parsing()
        elisa.query_parsing_tfidf()
