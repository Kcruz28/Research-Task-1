import cooking
import os
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

if "__main__" == __name__:
    root_directory = "."  # Root to scan from
    text_extensions = ["txt", "pdf"]
    output_dir = "answers_output"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Loading models...")
    elisa = cooking.Elisa()

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            basename, ext = os.path.splitext(filename)
            extension = ext.lower().lstrip(".")

            # Skip files that are not txt/pdf or are _q.txt or _q+a.txt
            if (
                extension not in text_extensions
                or filename.endswith("_q.txt")
                or filename.endswith("_q+a.txt")
            ):
                continue

            # Look for corresponding _q.txt in the same directory
            question_filename = f"{basename}_q.txt"
            question_path = os.path.join(dirpath, question_filename)

            if not os.path.isfile(question_path):
                print(f"Skipping {filepath}: No corresponding question file.")
                continue

            print(f"Processing {filepath} with {question_path}")

            # Read questions
            with open(question_path, "r", encoding="utf-8") as q_file:
                questions = [line.strip() for line in q_file if line.strip()]

            # Add document to index (once)
            elisa.add_document(filepath, extension)

            # Collect answers per question
            output_lines = []
            for idx, question in enumerate(questions, 1):
                elisa.set_question(question)

                # BM25
                elisa.query_parsing()
                bm25_answer = (
                    elisa.answers[0][1] if elisa.answers else "No answer found."
                )

                # TF-IDF
                elisa.query_parsing_tfidf()
                tfidf_answer = (
                    elisa.answers[0][1] if elisa.answers else "No answer found."
                )

                # Add to output
                output_lines.append(f"Q{idx}: {question}")
                output_lines.append(f"A{idx} (BM25): {bm25_answer}")
                output_lines.append(f"A{idx} (TF-IDF): {tfidf_answer}\n")

            # Save answers in central output_dir with subdirectory info
            relative_path = os.path.relpath(filepath, root_directory)
            answer_filename = relative_path.replace(os.sep, "_").replace(
                f".{extension}", "_q+a.txt"
            )
            output_file = os.path.join(output_dir, answer_filename)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))

            print(f"Q&A saved to {output_file}")
