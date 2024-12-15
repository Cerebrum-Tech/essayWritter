import streamlit as st
import asyncio
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
import docx
import os
import dotenv
dotenv.load_dotenv()


client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Function to extract text from uploaded files


def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file_extension == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX file.")
            return ""
    return ""


async def generate_literature_review_stream(question, texts, placeholder):
    texts = "---End Of Document---".join(texts)
    prompt = """
    Generate a literature review based on the following documents:
    {texts}
    Question: {question}
    """.format(texts=texts, question=question)

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant on generating literature reviews always provide references"},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    streamed_text = ""
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            streamed_text += chunk_content
            placeholder.markdown(streamed_text)
            st.session_state['literature_review'] = streamed_text


async def generate_outline_stream(question, texts, placeholder, literature_review):
    texts = "---End Of Document---".join(texts)
    prompt = """
    Generate an outline based on the following documents:
    {literature_review}
    Question: {question}
    """.format(literature_review=literature_review, question=question)

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
                "content": "You are a helpful assistant on generating outlines."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    streamed_text = ""
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            streamed_text += chunk_content
            placeholder.markdown(streamed_text)
            st.session_state['outline'] = streamed_text


async def generate_essay_stream(question, outline, placeholder, literature_review):
    prompt = """
    Generate an essay based on the following outline:
    {outline}
    Literature Review: {literature_review}
    Question: {question}
    """.format(outline=outline, literature_review=literature_review, question=question)

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant on generating essays. You will provide a well-structured essay longer than 5000 words."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=15000
    )

    streamed_text = ""
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            streamed_text += chunk_content
            placeholder.markdown(streamed_text)
            st.session_state['essay'] = streamed_text


def app():
    st.title("Academic Essay Assistant")
    st.write("This app helps you with your academic writing by providing an overview of uploaded essays and responding to questions.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload your essay files (PDF or DOCX)",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    question = st.sidebar.text_input("Enter your question:")

    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")

    logo = st.sidebar.image(
        "https://static.wixstatic.com/media/355375_f3c2f0e136f34270a3dd257007a3fee2~mv2.png/v1/fill/w_131,h_64,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/355375_f3c2f0e136f34270a3dd257007a3fee2~mv2.png",
        use_container_width=True
    )

    if 'literature_review' not in st.session_state:
        st.session_state['literature_review'] = ""
    if 'outline' not in st.session_state:
        st.session_state['outline'] = ""
    if 'essay' not in st.session_state:
        st.session_state['essay'] = ""

    if uploaded_files and question:
        documents_text = [extract_text_from_file(f) for f in uploaded_files]

        tab1, tab2, tab3 = st.tabs(["Literature Review", "Outline", "Write"])

        with tab1:
            placeholder = st.empty()
            literature_review = st.session_state.get('literature_review', "")
            placeholder.markdown(literature_review)
            if st.button("Generate Literature Review"):
                st.subheader("Generated Literature Review")
                placeholder = st.empty()
                asyncio.run(generate_literature_review_stream(
                    question, documents_text, placeholder))

        with tab2:
            placeholder = st.empty()
            outline = st.session_state.get('outline', "")
            placeholder.markdown(outline)
            if st.button("Generate Outline"):
                st.subheader("Generated Outline")
                placeholder = st.empty()
                asyncio.run(generate_outline_stream(
                    question, documents_text, placeholder, st.session_state['literature_review']))

            if st.session_state['outline']:
                editable_outline = st.text_area(
                    "Edit your outline here:", st.session_state['outline'], height=700)
                if st.button("Update Outline"):
                    st.session_state['outline'] = editable_outline
        with tab3:
            placeholder = st.empty()
            essay = st.session_state.get('essay', "")
            placeholder.markdown(essay)
            if st.button("Generate Essay"):
                st.subheader("Generated Essay")
                placeholder = st.empty()
                asyncio.run(generate_essay_stream(
                    question, st.session_state['outline'], placeholder, st.session_state['literature_review']))


if __name__ == "__main__":
    app()
