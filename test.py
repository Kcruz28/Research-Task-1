import model
import os
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

if __name__ == "__main__":
    root_directory = "documents"
    text_extensions = ["txt", "pdf"]
    output_dir = "answers_output"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Loading models...")
    elisa = model.Elisa()

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            basename, ext = os.path.splitext(filename)
            extension = ext.lower().lstrip(".")

            if (
                extension not in text_extensions
                or filename.endswith("_q.txt")
                or filename.endswith("_q+a.txt")
            ):
                continue

            question_filename = f"{basename}_q.txt"
            question_path = os.path.join(dirpath, question_filename)
            if not os.path.isfile(question_path):
                folder_name = os.path.basename(dirpath)
                question_foldername = f"{folder_name}_q.txt"
                path2 = os.path.join(dirpath, question_foldername)
                if os.path.isfile(path2):
                    question_path = path2

            if not os.path.isfile(question_path):
                candidates = [f for f in os.listdir(dirpath) if f.endswith("_q.txt")]
                if candidates:
                    question_path = os.path.join(dirpath, candidates[0])

            if not os.path.isfile(question_path):
                print(f"Skipping {filepath}: No corresponding question file found.")
                continue

            print(
                f"Processing {filepath} ({extension.upper()}) with {os.path.basename(question_path)}"
            )

            try:
                if extension == "pdf":
                    document_text = elisa.pdf_to_text(filepath)
                else:
                    with open(filepath, "r", encoding="utf-8") as f:
                        document_text = f.read()
            except Exception as e:
                print(f"  Failed to read {filepath}: {e}")
                document_text = ""

            text_length = len(document_text.strip())
            print(f"  Extracted text length: {text_length} characters")

            with open(question_path, "r", encoding="utf-8") as q_file:
                questions = [line.strip() for line in q_file if line.strip()]

            output_lines = []
            if text_length > 0:
                elisa.add_document(filepath, extension)

                for idx, question in enumerate(questions, 1):
                    elisa.set_question(question)

                    bm25_answer, bm25_score = elisa.query_parsing()
                    if not bm25_answer:
                        bm25_answer, bm25_score = "No answer found.", "N/A"

                    tfidf_answer, tfidf_score = elisa.query_parsing_tfidf()
                    if not tfidf_answer:
                        tfidf_answer, tfidf_score = "No answer found.", "N/A"

                    output_lines.append(f"Q{idx}: {question}")
                    output_lines.append(
                        f"A{idx} (BM25): {bm25_answer} (Score: {bm25_score})"
                    )
                    output_lines.append(
                        f"A{idx} (TF-IDF): {tfidf_answer} (Score: {tfidf_score})"
                    )
                    output_lines.append("")
            else:
                print(
                    f"  Warning: {filepath} has no extractable text. Writing default answers."
                )
                for idx, question in enumerate(questions, 1):
                    output_lines.append(f"Q{idx}: {question}")
                    output_lines.append(f"A{idx} (BM25): No answer found. (Score: N/A)")
                    output_lines.append(
                        f"A{idx} (TF-IDF): No answer found. (Score: N/A)"
                    )
                    output_lines.append("")

            relative_path = os.path.relpath(filepath, root_directory)
            safe_basename = relative_path.replace(os.sep, "_").replace(
                f".{extension}", ""
            )
            output_file = os.path.join(output_dir, f"{safe_basename}_answers.txt")

            with open(output_file, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(output_lines))

            print(f"  Answers saved to {output_file}\n")
