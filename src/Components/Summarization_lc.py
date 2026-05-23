import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# --- Local Infrastructure Imports ---
from src.config import *
from src.logger import logging
from src.exception import CustomException

from pydantic import BaseModel, field_validator, ValidationError
from src.Components.Embedding_lc import ArticleEmbeddingEngine

# LangChain specific imports for Summarization
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables cleanly
from dotenv import load_dotenv
load_dotenv()

class SummarizeRequest(BaseModel):
    text: str
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip() or not isinstance(v, str):
            raise ValidationError("Text must be a non-empty string")
        return v.strip()


class NewsSummarizerLC:
    """
    LangChain-based Hybrid Summarizer Pipeline.
    Uses Embedding_lc to chunk, embed, and score chunks by centroid relevance.
    Then uses LangChain LCEL to generate an abstractive summary via API.
    """
    def __init__(self, llm_type: str = "openai", embedding_engine: Optional[ArticleEmbeddingEngine] = None):
        logging.info(f"Initializing LangChain NewsSummarizer (LLM: {llm_type})")
        try:
            self.llm_type = llm_type.lower()
            
            # 1. Initialize LangChain embedding engine for extractive chunk filtering
            self.embedder = embedding_engine or ArticleEmbeddingEngine(embedding_type="huggingface_local")
            
            # 2. Initialize LangChain LLM for the Abstractive Summarization phase
            if self.llm_type == "openai":
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
            elif self.llm_type == "huggingface_api":
                self.llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.3", # Switched from bart because LangChain chains expect text-generation
                    temperature=0.2,
                )
            elif self.llm_type == "groq":
                self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
            elif self.llm_type == "gemini":
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            else:
                raise ValueError(f"Unsupported LLM type for summarization: {self.llm_type}")
            
            # 3. Create a modern LCEL chain for summarization
            prompt_template = """You are an expert editor. Combine and rewrite the following extracted text highlights into a cohesive and professional summary.
Strict requirement: Your final summary MUST NOT exceed {target_length} words. Be concise, direct, and do not add any filler, repetition, or outside information.

TEXT HIGHLIGHTS:
{text}

COHESIVE SUMMARY:
"""
            self.prompt = PromptTemplate.from_template(prompt_template)
            self.chain = self.prompt | self.llm
            
            logging.info("LangChain Hybrid Summarization Engine Ready.")
        except Exception as e:
            logging.error(f"Failed to initialize NewsSummarizerLC: {e}")
            raise CustomException(e, sys)
            
    def summarize(self, text: str, reduction_ratio: float = 0.5) -> str:
        """
        Hybrid Extractive-Abstractive pipeline:
        1. Embed full text (centroid).
        2. Chunk text and score chunks via cosine similarity to centroid.
        3. Extract chunks based on reduction_ratio.
        4. Pass extracted chunks to LangChain abstractive summarizer.
        """
        try:
            req = SummarizeRequest(text=text)
            clean_text = req.text
            
            # Calculate word count constraint for the LLM prompt
            original_word_count = len(clean_text.split())
            target_length = max(10, int(original_word_count * reduction_ratio))
            
            # 1. Chunking using LangChain Splitter inside embedder
            chunks = self.embedder.chunk_article(clean_text)
            if not chunks:
                return ""
                
            # Determine how many chunks to keep based on the ratio.
            # We extract 1.5x the target ratio (capped at 1.0) so the LLM has actual room to compress
            # rather than just rewriting text at a 1:1 ratio.
            extraction_ratio = min(1.0, reduction_ratio * 1.5)
            max_chunks = max(1, int(len(chunks) * extraction_ratio))
                
            # If the article is short, just summarize the whole thing without filtering
            if len(chunks) <= max_chunks:
                extracted_chunks = chunks
                logging.info(f"Article short enough ({len(chunks)} chunks <= {max_chunks}), skipping extraction phase.")
            else:
                # 2. Extractive Phase: Centroid Scoring
                doc_emb = self.embedder.embed_query(clean_text)
                chunk_embs = self.embedder.embed_chunks(chunks)
                
                # Cosine Similarity (dot product of L2 normalized vectors)
                scores = np.dot(chunk_embs, doc_emb)
                
                # Filter to top `max_chunks`
                top_indices = np.argsort(scores)[-max_chunks:]
                top_indices_sorted = np.sort(top_indices) # Keep chronological order
                
                extracted_chunks = [chunks[i] for i in top_indices_sorted]
                logging.info(f"Extracted {len(extracted_chunks)} chunks based on centroid similarity.")
                
            # 3. Abstractive Phase using LangChain LCEL
            combined_text = "\n".join(extracted_chunks)
            
            # Execute the LangChain LCEL chain with the target_length requirement
            summary_result = self.chain.invoke({
                "text": combined_text,
                "target_length": target_length
            })
            
            # Extract text from AIMessage (OpenAI) or direct string (HuggingFaceEndpoint)
            if hasattr(summary_result, "content"):
                final_summary = summary_result.content
            else:
                final_summary = str(summary_result)
            
            return final_summary.strip()
            
        except Exception as e:
            logging.error(f"Error in LangChain Summarization: {e}")
            raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        print("========================================")
        print("Testing LangChain Hybrid News Summarizer...")
        print("========================================")
        
        long_article = """ 
        In a significant display of India’s air power, the fighter jets of the Indian Air Force flew in a special Sindoor formation at the 77th Republic Day flypast on Monday.
        The powerful seven-aircraft formation comprised of two Rafales, two Su-30s, two MiG-29s and a Jaguar aircraft, according to officials.
        This year marks the first Republic Day after Operation Sindoor where India showcased its military might. India launched Operation Sindoor on May 7, 2025 when the IAF and the Indian Army carried out attacks inside Pakistan to dismantle terror infrastructure following the Pahalgam attack in April that year.
        
        In a first, India’s military assets moved down the Kartavya Path in a sequence similar to that in an actual combat, starting with reconnaissance, followed by other military units such as logistics and personnel accompanying these platforms, wearing battle gear, according to a report in The Indian Express.
        A total of 30 tableaux — 17 of States/Union Territories and 13 of Ministries/Departments/Services — also rolled down the Kartavya Path in New Delhi during the Republic Day parade, which began at 10:30 today.
        
        Several formations dedicated to the success of the operation was on display at the R-Day parade this year. These included the “Prahar Formation, the Garud Formation, and a powerful, dedicated formation known as the Sindoor Formation,” Wing Commander Rajesh Deshwal informed the media last Thursday.
        
        In sync with the marching contingent was a thrilling fly-past by two Rafale jets, two MiG-29s, two Su-30s and one Jaguar aircraft in ‘Spearhead’ formation, symbolising the “Sindoor Formation”.
        Overall, a total of 29 aircraft participated in the flypast this year, including 16 fighter aircraft, four transport aircraft, and nine helicopters from six different bases.
        Prahar & Garud Formations: It comprises of three Advanced Light Helicopters (ALH) — two from the Indian Army and one from the Indian Air Force — of which the lead aircraft will carry the Operation Sindoor flag, the Commander told the press. It will be followed by the Garud Formation, and both will fly in a battle array format, he added.
        
        Sindoor Formation: It comprises two Rafale aircraft, MiG-29 aircraft and Su-30 MKI aircraft each besides one Jaguar aircraft, making it a powerful seven-aircraft formation, he said.
        The flypast commenced with Dhwaj formation, where four Mi-17 IV helicopters carried the National Flag alongside the flags of the three services.
        
        The formations were complemented by strategic assets including the “C-130 and C-295, as well as the Indian Navy’s P-8i aircraft. Attack helicopters, such as the IAF’s ALH MK IV and the Indian Army’s ALH WSI, Apache and Light Combat Helicopter (LCH) will also participate in the flypast showcasing jointmanship,” according to an official statement.
        A tri-services tableau showcasing replicas of India’s major weapon systems deployed by the country’s military during Operation Sindoor in early May remained a major attraction.
        Additionally, a glass-enclosed integrated operational centre, portraying the conduct of the operation with the use of weapons systems such as BrahMos and S-400 missiles rolled down the Kartavya Path.
        
        For the first time, the parade showcased a phased ‘Battle Array Format’ of the Indian Army, including an aerial component. It featured a high mobility reconnaissance vehicle and India’s first indigenously designed armoured light specialist vehicle, according to news agency PTI.
        The indigenous Dhruv Advanced Light Helicopter and its armed version, Rudra, in Prahar formation, demonstrated shaping of the battlefield during the Operation.
        """
        
        # Using HuggingFace API for the test to avoid requiring OpenAI keys unless provided
        # You can also pass llm_type="openai"
        Summarizer_model = NewsSummarizerLC(llm_type="groq")
        summ = Summarizer_model.summarize(long_article, reduction_ratio=0.5)
        
        print("\n" + "="*50)
        print("ORIGINAL TEXT (Snippet):")
        print(long_article.strip()[:300] + "...\n[...]")
        print("-" * 50)
        print("LANGCHAIN HYBRID SUMMARY:")
        print(summ)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Test run encountered an issue: {e}")
