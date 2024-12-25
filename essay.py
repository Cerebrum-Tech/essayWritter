import streamlit as st
import asyncio
import logging
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
import docx
import os
import dotenv
from pydantic import BaseModel, ValidationError
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

dotenv.load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentText(BaseModel):
    texts: str

class LiteratureReviewRequest(BaseModel):
    question: str
    texts: str

class OutlineRequest(BaseModel):
    question: str
    literature_review: str

class EssayRequest(BaseModel):
    question: str
    outline: str
    literature_review: str

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Extract text from uploaded files
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])
                return text
            elif file_extension == "docx":
                doc = docx.Document(uploaded_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            else:
                st.error("Unsupported file type. Please upload a PDF or DOCX file.")
                return ""
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Generate literature review
async def generate_literature_review_stream(request: LiteratureReviewRequest, placeholder):
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant on generating literature reviews always provide references"},
                {"role": "user", "content": request.model_dump_json()},
            ],
            stream=True
        )

        streamed_text = ""
        async for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                streamed_text += chunk_content
                placeholder.markdown(streamed_text)
        
        return streamed_text

    except Exception as e:
        st.error(f"Error generating literature review: {e}")
        return None

# Kaynak belgelere göre yanıt içeriği doğrulama
def validate_response(response, sources):
    for source in sources:
        if response in source:
            return True
    return False

# Metin benzerliği hesaplama
def calculate_similarity(generated_text, source_texts):
    vectorizer = TfidfVectorizer().fit_transform([generated_text] + source_texts)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return np.max(cosine_similarities)  

# Yanıt geçerliliğini doğrulama 
def validate_response_with_similarity(generated_text, source_texts, threshold=0.70):
    similarity = calculate_similarity(generated_text, source_texts)
    logger.info(f"Calculated similarity: {similarity}")
    return similarity >= threshold

# Dokümanlarda bulunmayan ifadeleri kaldırma 
def remove_non_matching_phrases(generated_text, source_texts):
    source_text = " ".join(source_texts)
    filtered_sentences = []
    
    for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_text): 
        if sentence and any(phrase in source_text for phrase in sentence.split()):
            filtered_sentences.append(sentence) 
    
    return " ".join(filtered_sentences)

# Referansları metinden çıkarma ve doğrulama
def extract_references(text):
    pattern = r'\((.*?)\)'
    references = re.findall(pattern, text)
    return references

def validate_references(references, source_texts):
    joined_sources = " ".join(source_texts)
    invalid_references = [ref for ref in references if ref not in joined_sources]
    return invalid_references

def remove_invalid_references(generated_text, invalid_references):
    for ref in invalid_references:
        generated_text = generated_text.replace(f"({ref})", "[Invalid Reference Removed]")
    return generated_text

# LLM ile essay doğrulama 
async def check_essay_validity(essay, sources):
    try:
        messages = [
            {"role": "system", "content": "You're a helpful assistant for verifying essays against their sources. Return True if the essay is consistent with sources, otherwise return False."},
            {"role": "user", "content": f"Essay: {essay}\n\nSources: {' '.join(sources)}\n\nIs the essay consistent with the sources? Reply with True or False."}
        ]
        logger.info(f"Sending essay and sources to LLM for validation: {messages}")

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        result = response.choices[0].message["content"].strip().lower()
        logger.info(f"LLM validation response: {result}")

        if result not in ['true', 'false']:
            logger.error(f"Unexpected response from LLM: {result}")
        return result == 'true'
    except Exception as e:
        logger.error(f"Error checking essay validity: {e}")
        return False

# Literatür incelemesi oluşturma 
async def handle_literature_review_generation(request_data, placeholder, documents_text, validate=False, ref_check=False, similarity_threshold=0.70):
    invalid_references_list = []
    if validate or ref_check:
        logger.info(f"Anti-hallucination is {'enabled' if validate else 'disabled'}.")
        logger.info(f"Reference check is {'enabled' if ref_check else 'disabled'}.")

        max_attempts = 3
        attempt = 0
        is_valid = False
        initial_response = await generate_literature_review_stream(request_data, placeholder)

        while attempt < max_attempts and not is_valid:
            attempt += 1
            logger.info(f"Attempt number: {attempt}")

            if attempt > 1:
                cleaned_response = remove_non_matching_phrases(initial_response, documents_text)
                logger.info(f"Cleaned response generated: {cleaned_response[:500]}...")  # Log first 500 characters
                request_data = LiteratureReviewRequest(question=request_data.question, texts=cleaned_response)
                initial_response = await generate_literature_review_stream(request_data, placeholder)

            if ref_check: 
                references = extract_references(initial_response)
                invalid_references = validate_references(references, documents_text)
                invalid_references_list.extend(invalid_references)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    initial_response = remove_invalid_references(initial_response, invalid_references)

            is_valid = validate_response_with_similarity(initial_response, documents_text, threshold=similarity_threshold)
            logger.info(f"Calculated similarity: {calculate_similarity(initial_response, documents_text)}")

        if not is_valid:
            logger.warning("Literature review failed validation. Sending to LLM for validation.")
            placeholder.markdown("**LLM validation in progress...**")
            verify = await check_essay_validity(initial_response, documents_text)
            if verify:
                st.success("The generated literature review is consistent with the sources as validated by the LLM.")
            else:
                st.error("The generated literature review is not consistent with the sources as validated by the LLM.")
            return

        if is_valid:
            placeholder.markdown(initial_response)
            st.session_state['literature_review'] = initial_response
            if invalid_references_list:
                st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references_list)))
    else:
        logger.info("Response validation or reference check is disabled.")
        response = await generate_literature_review_stream(request_data, placeholder)
        if response:
            if ref_check: 
                references = extract_references(response)
                invalid_references = validate_references(references, documents_text)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    response = remove_invalid_references(response, invalid_references)
                    placeholder.markdown(response)
                    st.session_state['literature_review'] = response
                    st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references)))
                else:
                    placeholder.markdown(response)
                    st.session_state['literature_review'] = response
            else:
                placeholder.markdown(response)
                st.session_state['literature_review'] = response


async def generate_outline_stream_internal(request: OutlineRequest, placeholder):
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant on generating outlines."},
                {"role": "user", "content": request.model_dump_json()}
            ],
            stream=True
        )

        streamed_text = ""
        async for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                streamed_text += chunk_content
                placeholder.markdown(streamed_text)
        
        return streamed_text

    except Exception as e:
        st.error(f"Error generating outline: {e}")
        return None

async def generate_outline_stream(request: OutlineRequest, documents_text, placeholder, validate=False, ref_check=False, similarity_threshold=0.70):
    invalid_references_list = []
    if validate or ref_check:
        logger.info(f"Anti-hallucination is {'enabled' if validate else 'disabled'}.")
        logger.info(f"Reference check is {'enabled' if ref_check else 'disabled'}.")

        max_attempts = 3
        attempt = 0
        is_valid = False
        initial_response = await generate_outline_stream_internal(request, placeholder)

        while attempt < max_attempts and not is_valid:
            attempt += 1
            logger.info(f"Attempt number: {attempt}")

            if attempt > 1:
                cleaned_response = remove_non_matching_phrases(initial_response, documents_text)
                request = OutlineRequest(question=request.question, literature_review=cleaned_response)
                initial_response = await generate_outline_stream_internal(request, placeholder)

            if ref_check:  
                references = extract_references(initial_response)
                invalid_references = validate_references(references, documents_text)
                invalid_references_list.extend(invalid_references)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    initial_response = remove_invalid_references(initial_response, invalid_references)

            is_valid = validate_response_with_similarity(initial_response, documents_text, threshold=similarity_threshold)

           # LLM doğrulaması
        if not is_valid:
            logger.warning("Outline failed validation. Sending to LLM for validation.")
            placeholder.markdown("**LLM validation in progress...**")
            verify = await check_essay_validity(initial_response, documents_text)
            if verify:
                st.success("The generated outline is consistent with the sources as validated by the LLM.")
            else:
                st.error("The generated outline is not consistent with the sources as validated by the LLM.")
            return

        if is_valid:
            placeholder.markdown(initial_response)
            st.session_state['outline'] = initial_response
            if invalid_references_list:
                st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references_list)))
        else:
            logger.warning("Unable to generate a valid outline after multiple attempts.")
    else:
        logger.info("Response validation or reference check is disabled for outline generation.")
        response = await generate_outline_stream_internal(request, placeholder)
        if response:
            if ref_check:  
                references = extract_references(response)
                invalid_references = validate_references(references, documents_text)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    response = remove_invalid_references(response, invalid_references)
                    placeholder.markdown(response)
                    st.session_state['outline'] = response
                    st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references)))
                else:
                    placeholder.markdown(response)
                    st.session_state['outline'] = response
            else:
                placeholder.markdown(response)
                st.session_state['outline'] = response

# Generate essay
async def generate_essay_stream_internal(request: EssayRequest, placeholder):
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant on generating essays. You will provide a well-structured essay longer than 5000 words."},
                {"role": "user", "content": request.model_dump_json()}
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

        return streamed_text

    except Exception as e:
        st.error(f"Error generating essay: {e}")
        return None

async def generate_essay_stream(request: EssayRequest, documents_text, placeholder, validate=False, ref_check=False, similarity_threshold=0.70):
    invalid_references_list = []
    if validate or ref_check:
        logger.info(f"Anti-hallucination is {'enabled' if validate else 'disabled'}.")
        logger.info(f"Reference check is {'enabled' if ref_check else 'disabled'}.")

        max_attempts = 3
        attempt = 0
        is_valid = False
        initial_response = await generate_essay_stream_internal(request, placeholder)

        while attempt < max_attempts and not is_valid:
            attempt += 1
            logger.info(f"Attempt number: {attempt}")

            if attempt > 1:
                cleaned_response = remove_non_matching_phrases(initial_response, documents_text)
                logger.info(f"Cleaned response generated: {cleaned_response[:500]}...")  
                request = EssayRequest(question=request.question, outline=request.outline, literature_review=cleaned_response)
                initial_response = await generate_essay_stream_internal(request, placeholder)

            if ref_check:
                references = extract_references(initial_response)
                invalid_references = validate_references(references, documents_text)
                invalid_references_list.extend(invalid_references)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    initial_response = remove_invalid_references(initial_response, invalid_references)

            is_valid = validate_response_with_similarity(initial_response, documents_text, threshold=similarity_threshold)
            logger.info(f"Calculated similarity: {calculate_similarity(initial_response, documents_text)}")

        if not is_valid:
            logger.warning(f"Generating essay failed after {max_attempts} attempts, validating with LLM.")
            placeholder.markdown("**LLM validation in progress...**")
            verify = await check_essay_validity(initial_response, documents_text)
            if not verify:
                logger.warning(f"LLM validation returned False. Generated essay is not consistent with the sources.")
                st.error("The generated essay is not consistent with the sources as validated by the LLM.")
                return
            else:
                logger.info(f"LLM validation returned True. Generated essay is consistent with the sources.")
                placeholder.markdown(initial_response)
                st.session_state['essay'] = initial_response
                return

        if is_valid:
            placeholder.markdown(initial_response)
            st.session_state['essay'] = initial_response
            if invalid_references_list:
                st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references_list)))
    else:
        logger.info("Response validation or reference check is disabled for essay generation.")
        response = await generate_essay_stream_internal(request, placeholder)
        if response:
            if ref_check:
                references = extract_references(response)
                invalid_references = validate_references(references, documents_text)
                if invalid_references:
                    logger.info(f"Invalid references found: {invalid_references}")
                    response = remove_invalid_references(response, invalid_references)
                    placeholder.markdown(response)
                    st.session_state['essay'] = response
                    st.error(f"The following references were not found in the sources:\n" + "\n".join(set(invalid_references)))
                else:
                    placeholder.markdown(response)
                    st.session_state['essay'] = response
            else:
                placeholder.markdown(response)
                st.session_state['essay'] = response

# Streamlit app
def app():
    st.title("Academic Essay Assistant")
    st.write("This app helps you with your academic writing by providing an overview of uploaded essays and responding to questions.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload your essay files (PDF or DOCX)",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    question = st.sidebar.text_input("Enter your question:")
    validate_option = st.sidebar.toggle("Cosine-Similarity Anti Hallucination", value=False)
    ref_check_option = st.sidebar.toggle("Check References", value=False)

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
        combined_texts = "---End Of Document---".join(documents_text)

        try:
            document_text = DocumentText(texts=combined_texts)
        except ValidationError as e:
            st.error(f"Document text validation error: {e}")
            return

        tab1, tab2, tab3 = st.tabs(["Literature Review", "Outline", "Write"])

        with tab1:
            placeholder = st.empty()
            literature_review = st.session_state.get('literature_review', "")
            placeholder.markdown(literature_review)
            if st.button("Generate Literature Review"):
                st.subheader("Generated Literature Review")
                placeholder = st.empty()
                request_data = LiteratureReviewRequest(question=question, texts=document_text.texts)
                asyncio.run(handle_literature_review_generation(request_data, placeholder, documents_text, validate=validate_option, ref_check=ref_check_option))

        with tab2:
            placeholder = st.empty()
            outline = st.session_state.get('outline', "")
            placeholder.markdown(outline)
            if st.button("Generate Outline"):
                st.subheader("Generated Outline")
                placeholder = st.empty()
                request_data = OutlineRequest(question=question, literature_review=st.session_state['literature_review'])
                asyncio.run(generate_outline_stream(request_data, documents_text, placeholder, validate=validate_option, ref_check=ref_check_option))

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
                request_data = EssayRequest(
                    question=question,
                    outline=st.session_state['outline'],
                    literature_review=st.session_state['literature_review']
                )
                asyncio.run(generate_essay_stream(request_data, documents_text, placeholder, validate=validate_option, ref_check=ref_check_option))

if __name__ == "__main__":
    app()