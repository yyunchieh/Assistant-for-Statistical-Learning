# main.py - Streamlit interface for Statistical Learning Assistant

import streamlit as st
from graph import app
import datetime

# UI
st.title("Statistical Learning Assistant")


query = st.text_input("Enter your question")

if st.button("Generate"):
    if query.strip():
        result = app.invoke({"query": query, "answer": "", "source": ""})
        answer = result["answer"]
        source = result["source"]

        st.write("### Answer:")
        st.write(answer)

        st.write("---")
        st.markdown(source)

        # Save to Markdown file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = f"response_{timestamp}.md"

        markdown = f"# Question\n{query}\n\n## Answer\n{answer}\n\n{source}"

        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(markdown)

        st.success("Response generated successfully!")
        st.download_button("Download Markdown", data=markdown, file_name=md_filename)

         
