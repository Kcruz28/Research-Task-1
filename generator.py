def generate_text(pass_content, questionss, qa_pipeline):

    result = qa_pipeline(question=questionss, context=pass_content)

    # Show the extracted answer
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']}")
