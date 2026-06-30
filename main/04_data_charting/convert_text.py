from os import system, listdir
from os.path import join, exists
import glob

pdf_dir = "./data/papers_pdfs"
txt_dir = "./data/papers_txts"
output_txt = "./data/fulltexts.txt"

# --------------------------------------------------
# Process full-text PDFs if available
# --------------------------------------------------
if exists(pdf_dir):

    # Create txt directory if it does not exist
    if not exists(txt_dir):
        system(f"mkdir -p {txt_dir}")

    # Remove old txt files
    system(f"rm -f {txt_dir}/*.txt")

    # Convert PDFs to individual TXT files
    pdf_files = [f for f in listdir(pdf_dir) if f.endswith(".pdf")]

    for fname in pdf_files:
        pdf_path = join(pdf_dir, fname)
        txt_path = join(txt_dir, fname.replace(".pdf", ".txt"))
        system(f"pdftotext '{pdf_path}' '{txt_path}'")

    # Merge all full-text TXT files into a single file
    with open(output_txt, "w", encoding="utf-8") as out:
        for txt_file in glob.glob(f"{txt_dir}/*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                out.write(f.read().lower())
                out.write("\n\n")

    print(f"Combined full-text saved as: {output_txt}")

else:
    print("PDF folder not found. Using title, abstract, and keywords only.")


# --------------------------------------------------
# Merge all title-and-abstract-keywords TXT files
# --------------------------------------------------
title_and_abstract_keywords_txt = "./data/title_and_abstract_keywords.txt"

with open(title_and_abstract_keywords_txt, "w", encoding="utf-8") as out:
    for txt_file in glob.glob("./data/title_abstract_keyword/*.txt"):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            out.write(f.read().lower())
            out.write("\n\n")

print(f"Combined title-and-abstract-keywords saved as: {title_and_abstract_keywords_txt}")