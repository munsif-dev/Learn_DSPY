import json
import logging
import os
import re
import sys
from typing import List, Dict, Any, Tuple, Optional

import dspy
import google.generativeai as genai
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.file_handler import Document, get_page_counts, upload_to_gemini
from scripts.split_doc import split_pdf
from services.typesense_client import match_terms_in_typesense

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- Configuration ---
MODEL_FLASH = "gemini-2.5-flash-preview-04-17"
FUNCTION_PROMPT_CONFIG_PATH = "configs/function_prompts.json"
LOGS_DIR = "logs"
TEXT_OUTS_DIR = "text_outs"
MAX_PAGE_COUNT = 30

# Set up logging
def setup_logger(level=logging.INFO):
    logger = logging.getLogger("translate-paralegal")
    if logger.hasHandlers():
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger.info("Logger initialized")
    return logger

logger = setup_logger(level=logging.DEBUG)

# --- Load Prompts ---
def load_function_prompts(config_path):
    """Loads prompts from a JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

FUNCTION_PROMPTS = load_function_prompts(FUNCTION_PROMPT_CONFIG_PATH)

# --- DSPy Setup ---
class GeminiLM(dspy.LM):
    """Custom DSPy language model wrapper for Gemini"""
    
    def __init__(self, model_name=MODEL_FLASH, json_output=False):
        self.model_name = model_name
        self.json_output = json_output
        
        if json_output:
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json", 
                response_schema=list[str]
            )
            self.model = genai.GenerativeModel(model_name, generation_config=generation_config)
        else:
            self.model = genai.GenerativeModel(model_name)
        
    def basic_request(self, prompt, **kwargs):
        """Basic request to Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini request: {e}")
            raise

    def request(self, prompt, **kwargs):
        """DSPy-compatible request method"""
        if isinstance(prompt, list):
            # Handle list of messages or content
            return self.basic_request(prompt, **kwargs)
        else:
            # Handle single string prompt
            return self.basic_request(prompt, **kwargs)

# Initialize DSPy with Gemini model
standard_lm = GeminiLM(MODEL_FLASH)
json_lm = GeminiLM(MODEL_FLASH, json_output=True)

# --- DSPy Signatures and Modules ---
class LegalTermExtraction(dspy.Signature):
    """Extract legal terms from text for translation consideration"""
    extracted_text = dspy.InputField(desc="The source text content")
    source_language = dspy.InputField(desc="The source language")
    target_language = dspy.InputField(desc="The target language")
    legal_terms = dspy.OutputField(desc="List of legal terms that might be challenging to translate")

class LegalTermExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(LegalTermExtraction)
    
    def forward(self, extracted_text, source_language, target_language):
        result = self.predictor(
            extracted_text=extracted_text,
            source_language=source_language,
            target_language=target_language
        )
        # Parse the result to ensure it's a list
        if isinstance(result.legal_terms, str):
            terms = [term.strip() for term in result.legal_terms.split('\n') if term.strip()]
            return terms
        return result.legal_terms

class TermFilter(dspy.Signature):
    """Filter and improve translations for terms"""
    matched_terms = dspy.InputField(desc="Matched terms from dictionary")
    source_language = dspy.InputField(desc="Source language")
    target_language = dspy.InputField(desc="Target language")
    filtered_terms = dspy.OutputField(desc="Filtered and improved translations")

class TermFilterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TermFilter)
    
    def forward(self, matched_terms, source_language, target_language):
        result = self.predictor(
            matched_terms=matched_terms,
            source_language=source_language,
            target_language=target_language
        )
        return result.filtered_terms

class TranslationTask(dspy.Signature):
    """Translate source text to target language"""
    source_text = dspy.InputField(desc="The text to translate")
    prompt = dspy.InputField(desc="Translation instructions")
    translated_text = dspy.OutputField(desc="The translated text")

class Translator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TranslationTask)
    
    def forward(self, source_text, prompt):
        result = self.predictor(
            source_text=source_text,
            prompt=prompt
        )
        return result.translated_text

class TranslationRefinement(dspy.Signature):
    """Refine a translation for better readability"""
    translated_text = dspy.InputField(desc="The draft translated text")
    original_text = dspy.InputField(desc="The original source text")
    source_language = dspy.InputField(desc="Source language")
    target_language = dspy.InputField(desc="Target language")
    refined_text = dspy.OutputField(desc="The refined translated text")

class TranslationRefiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TranslationRefinement)
    
    def forward(self, translated_text, original_text, source_language, target_language):
        result = self.predictor(
            translated_text=translated_text,
            original_text=original_text,
            source_language=source_language,
            target_language=target_language
        )
        return result.refined_text

# Initialize DSPy modules
term_extractor = LegalTermExtractor()
term_filter = TermFilterModule()
translator = Translator()
translation_refiner = TranslationRefiner()

# Configure DSPy with our LM
dspy.settings.configure(lm=standard_lm)

# --- Helper Functions ---
def sort_pdfs_by_page_number(file_paths):
    def extract_page_numbers(filename):
        match = re.search(r"document_(\d+)_(\d+)", filename)
        return int(match.group(1)) if match else 0

    return sorted(file_paths, key=extract_page_numbers)

def get_translate_prompt(source_to_target):
    with open("configs/translate_prompts.json", "r", encoding="utf-8") as f:
        translate_prompts = json.load(f)
    if source_to_target not in translate_prompts:
        raise ValueError(f"No translation prompt found for {source_to_target}")
    base_prompt = translate_prompts[source_to_target]
    return base_prompt

def append_to_prompt(prompt, matched_terms):
    """Append matched terms to the prompt for translation."""
    if not matched_terms:
        return prompt

    # Append to the prompt
    return f"""{prompt}\n\n Some terms and phrases in the source text could be challenging to translate. Use the  reference translations below for such terms and phrases:\n{matched_terms}"""

def language_mapper(source_to_target):
    """Maps the source_to_target string to language names."""
    if source_to_target.startswith("en_"):
        source_language = "english"
    elif source_to_target.startswith("ta_"):
        source_language = "tamil"
    elif source_to_target.startswith("si_"):
        source_language = "sinhala"
    else:
        raise ValueError(
            "Invalid source_to_target format. Expected 'en_', 'ta_', or 'si_'."
        )

    if source_to_target.endswith("_en"):
        target_language = "english"
    elif source_to_target.endswith("_ta"):
        target_language = "tamil"
    elif source_to_target.endswith("_si"):
        target_language = "sinhala"
    else:
        raise ValueError(
            "Invalid source_to_target format. Expected '_en', '_ta', or '_si'."
        )
    return source_language, target_language

# --- Main API Functions ---
def extract_legal_terms(extracted_text, source_to_target):
    """
    Identifies legal terms in the text that are prone to mistranslation using DSPy.
    """
    logger.debug("Extracting legal terms...")
    if not extracted_text:
        raise ValueError("Extracted text is required.")
    
    source_language, target_language = language_mapper(source_to_target)
    
    # Load the prompt template from function_prompts.json
    prompt_template = FUNCTION_PROMPTS["extract_legal_terms"].format(
        source_language=source_language, 
        target_language=target_language
    )
    
    # Use DSPy with temporary context for this specific prompt template
    with dspy.context(prompt_template=prompt_template):
        # Temporarily switch to JSON-outputting LM
        original_lm = dspy.settings.lm
        dspy.settings.configure(lm=json_lm)
        
        try:
            terms_list = term_extractor(
                extracted_text=extracted_text,
                source_language=source_language,
                target_language=target_language
            )
            
            # Ensure we have a proper list
            if isinstance(terms_list, str):
                try:
                    # Try to parse JSON if we got a string
                    parsed_terms = json.loads(terms_list)
                    if isinstance(parsed_terms, list):
                        terms_list = parsed_terms
                    else:
                        terms_list = [term.strip() for term in terms_list.split('\n') if term.strip()]
                except json.JSONDecodeError:
                    # If not JSON, split by newlines
                    terms_list = [term.strip() for term in terms_list.split('\n') if term.strip()]
            
            # Log the terms for debugging
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open(f"{LOGS_DIR}/mistranslation_terms.txt", "w", encoding="utf-8") as file:
                for term in terms_list:
                    file.write(f"{term}\n")
            
            return terms_list
        
        finally:
            # Restore original LM
            dspy.settings.configure(lm=original_lm)

def filter_for_accurate_translations(matched_terms, source_to_target):
    """
    Filter the mistranslation terms to find the most relevant ones using DSPy.
    """
    source_language, target_language = language_mapper(source_to_target)
    
    # Load prompt template from function_prompts.json
    prompt_template = FUNCTION_PROMPTS["filter_term_translations"].format(
        source_language=source_language,
        target_language=target_language,
        matched_terms=matched_terms
    )
    
    # Use DSPy with the prompt template
    with dspy.context(prompt_template=prompt_template):
        filtered_terms = term_filter(
            matched_terms=matched_terms,
            source_language=source_language,
            target_language=target_language
        )
    
    # Log the filtered terms
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    with open(f"{LOGS_DIR}/filtered_terms.txt", "w", encoding="utf-8") as file:
        file.write(filtered_terms)
    
    return filtered_terms

def refine_translation(translated_text, original_text, source_to_target):
    """
    Improves the fluency and grammar of a draft translation using DSPy.
    """
    logger.debug("Refining translation...")
    if not translated_text or not original_text:
        raise ValueError("Both translated and extracted texts are required.")

    source_language, target_language = language_mapper(source_to_target)
    
    # Load prompt template from function_prompts.json
    prompt_template = FUNCTION_PROMPTS["refine_translation"].format(
        source_language=source_language,
        target_language=target_language,
        extracted_text=original_text,
        translated_text=translated_text
    )
    
    # Use DSPy with the prompt template
    with dspy.context(prompt_template=prompt_template):
        refined_text = translation_refiner(
            translated_text=translated_text,
            original_text=original_text,
            source_language=source_language,
            target_language=target_language
        )
    
    return refined_text

def translate_text(input_text, source_to_target):
    """
    Translates text from source language to target language using DSPy.
    """
    logger.debug(f"Translating text with {source_to_target}...")
    
    # Extract legal terms
    legal_terms = extract_legal_terms(input_text, source_to_target)
    logger.debug(f"Extracted legal terms: {legal_terms}")
    
    # Match terms in dictionary
    matched_terms = match_terms_in_typesense(legal_terms, source_to_target)
    logger.debug(f"Matched terms: {matched_terms}")
    
    # Filter terms for accuracy
    glossary = filter_for_accurate_translations(matched_terms, source_to_target)
    logger.debug(f"Filtered terms: {glossary}")
    
    # Get base prompt and enhance with glossary
    base_prompt = get_translate_prompt(source_to_target)
    final_prompt = append_to_prompt(base_prompt, glossary)
    
    # Translate using DSPy
    with dspy.context():
        draft_translation = translator(
            source_text=input_text,
            prompt=final_prompt
        )
    
    # Refine the translation
    refined_translation = refine_translation(
        draft_translation,
        input_text,
        source_to_target
    )
    
    return refined_translation

def translate_document(doc_path, source_to_target):
    """
    Orchestrates the end-to-end translation workflow for a document using DSPy.
    """
    logger.debug(f"Translating document: {doc_path}")
    
    # 1. Preparation
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"File not found: {doc_path}")

    page_count = get_page_counts(doc_path)
    if page_count > MAX_PAGE_COUNT:
        raise ValueError(f"Document exceeds {MAX_PAGE_COUNT} pages.")

    document = Document(doc_path)
    document.read()
    original_text = document.text_content

    # 2. Term Extraction and Glossary Building
    legal_terms = extract_legal_terms(original_text, source_to_target)
    logger.debug(f"Extracted legal terms: {legal_terms}")
    
    matched_terms = match_terms_in_typesense(legal_terms, source_to_target)
    logger.debug(f"Matched terms: {matched_terms}")

    glossary = filter_for_accurate_translations(matched_terms, source_to_target)
    logger.debug(f"Filtered glossary: {glossary}")

    # 3. Draft Translation
    base_prompt = get_translate_prompt(source_to_target)
    final_prompt = append_to_prompt(base_prompt, glossary)
    
    with dspy.context():
        draft_translation = translator(
            source_text=original_text,
            prompt=final_prompt
        )
    
    logger.debug("Draft translation completed")

    # 4. Refinement
    final_translation = refine_translation(
        draft_translation, 
        original_text, 
        source_to_target
    )
    
    logger.debug("Translation refinement completed")
    return final_translation

def handle_large_doc(doc_path, prompt):
    """
    Handle large documents by splitting them into smaller parts using DSPy.
    """
    logger.debug(f"Handling large document: {doc_path}")
    
    split_pdf_dir = split_pdf(doc_path, "splits")
    logger.debug(f"Split PDF directory: {split_pdf_dir}")
    
    split_files = sort_pdfs_by_page_number(os.listdir(split_pdf_dir))
    logger.debug(f"Split files: {split_files}")

    # Extract language info from prompt for refinement
    source_language = "unknown"
    target_language = "unknown"
    if "tamil to english" in prompt.lower():
        source_language, target_language = "tamil", "english"
    elif "sinhala to english" in prompt.lower():
        source_language, target_language = "sinhala", "english"
    elif "english to tamil" in prompt.lower():
        source_language, target_language = "english", "tamil"
    elif "english to sinhala" in prompt.lower():
        source_language, target_language = "english", "sinhala"

    full_text = ""
    for split_file in split_files:
        split_file_path = os.path.join(split_pdf_dir, split_file)
        status, doc_obj = upload_to_gemini(split_file_path)

        if status == "Success":
            try:
                # Extract the text content first using standard Gemini API
                extraction_model = genai.GenerativeModel(MODEL_FLASH)
                extraction_response = extraction_model.generate_content(
                    ["Extract the raw text from this document.", doc_obj]
                )
                extracted_text = extraction_response.text
                
                # Translate using DSPy
                with dspy.context():
                    translation = translator(
                        source_text=extracted_text,
                        prompt=prompt
                    )
                
                # Refine if we know the languages
                if source_language != "unknown" and target_language != "unknown":
                    translation = refine_translation(
                        translation,
                        extracted_text,
                        f"{source_language[:2]}_{target_language[:2]}"  # Convert to format like "ta_en"
                    )
                
                full_text += translation
                logger.debug(f"Successfully translated split: {split_file}")
            except Exception as e:
                logger.error(f"Error during split translation: {e}")
                raise ValueError(f"Error generating translation: {str(e)}")
        else:
            raise ValueError(f"Error uploading document to Gemini: {status}")

    return full_text

if __name__ == "__main__":
    pdf_path = "/home/elihoole/Downloads/agrarian_7_16.pdf"
    translated_text = translate_document(
        pdf_path,
        source_to_target="ta_en",
    )
    with open("tests/translated_text.txt", "w", encoding="utf-8") as file:
        file.write(translated_text)