import sys
try:
    import PyPDF2
    reader = PyPDF2.PdfReader(sys.argv[1])
    text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        f.write(text)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
