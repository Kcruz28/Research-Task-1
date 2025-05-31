import cooking

if "__main__" == __name__:
    while True:
        question = input("What is your question? ")
        if question.lower() in ["exit", "quit", 'q']:
            print("Exiting the program.")
            break
        elif not question.strip():
            print("Please enter a valid question.")
            continue
        elisa = cooking.Elisa(question)
        elisa.add_document("test1.txt")
        elisa.query_parsing()
