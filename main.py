import cooking
import pdfplumber
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


if "__main__" == __name__:
    print("-" * 100)
    text_type = input("Enter the type of text [pdf, txt] ")
    print("-"*100)

    if text_type.lower() not in ["pdf", "txt"]:
        print("Please enter a valid text type")
        exit(1)

    while True:
        print("-"*100)
        question = input("What is your question? ")
        print("-" * 100)
        if question.lower() in ["exit", "quit", 'q']:
            print("Exiting the program. BYYEEEEEEEE")
            break
        elif not question.strip():
            print("Please enter a valid question.")
            continue

        # path = "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
        path = "wall_china.txt"
        elisa = cooking.Elisa(question)
        elisa.add_document(path, text_type)
        elisa.query_parsing2()
