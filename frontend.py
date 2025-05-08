# app.py
import streamlit as st
import os
import tempfile
from processing import process_pdf
from rag_chain import crag_pipeline

def display_highlighted(text):
    # Highlight keywords heuristically
    important_words = ["important", "key", "critical", "main", "core", "summary"]
    for word in important_words:
        text = text.replace(word, f"**{word}**")
    return text

def main():
    st.title("📄 PDF Insights & Q&A Tool")
    st.markdown("Upload one or more PDFs. Get quick summaries and ask intelligent questions!")

    uploaded = st.file_uploader("📎 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded:
        with tempfile.TemporaryDirectory() as temp_dir:
            combined_text = ""
            for file in uploaded:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as temp_file:
                    temp_file.write(file.getvalue())

                with st.spinner(f"🔍 Analyzing: {file.name}"):
                    elements = process_pdf(file_path, temp_dir)

                extracted_text = "\n".join(
                    "\n".join(elements.get(cat, []))
                    for cat in ["Title", "NarrativeText", "Text", "ListItem"]
                )
                combined_text += f"\n\n--- {file.name} ---\n\n{extracted_text}"

                # Show a short summary
                st.markdown(f"### 🧾 Summary for **{file.name}**")
                sample_summary = "\n".join(elements.get("NarrativeText", [])[:2])
                st.write(sample_summary or "No narrative content found.")

            # Ask questions
            st.markdown("---")
            st.subheader("💬 Ask something from the uploaded documents")
            question = st.text_input("Type your question:")

            if st.button("🔎 Generate Answer"):
                with st.spinner("Thinking..."):
                    answer = crag_pipeline(question, combined_text)
                    answer = display_highlighted(answer)

                st.subheader("🧠 Answer")
                st.markdown(answer)

if __name__ == "__main__":
    main()
