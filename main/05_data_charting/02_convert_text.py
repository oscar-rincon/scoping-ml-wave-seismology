from os import system, listdir
from os.path import join
import glob

pdf_dir = "./papers_pdfs"
txt_dir = "./papers_txts"
output_txt = "./data/text.txt"

# --------------------------------------------------
# Remove old txt files
# --------------------------------------------------
system(f"rm -f {txt_dir}/*.txt")

# --------------------------------------------------
# Convert PDFs to individual TXT files
# --------------------------------------------------
pdf_files = [f for f in listdir(pdf_dir) if f.endswith(".pdf")]

for fname in pdf_files:
    pdf_path = join(pdf_dir, fname)
    txt_path = join(txt_dir, fname.replace(".pdf", ".txt"))
    system(f"pdftotext '{pdf_path}' '{txt_path}'")

# --------------------------------------------------
# Merge all TXT files into a single file
# --------------------------------------------------
with open(output_txt, "w", encoding="utf-8") as out:
    for txt_file in glob.glob(f"{txt_dir}/*.txt"):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            out.write(f.read().lower())
            out.write("\n\n")  # separator between papers

print(f"Combined text saved as: {output_txt}")
