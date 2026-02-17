from typing import Tuple, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from typing import List
import html, re, unicodedata
from bs4 import BeautifulSoup
import spacy
from pydantic import BaseModel, field_validator, ValidationError


# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Suppress other TensorFlow logging (0=all, 1=no INFO, 2=no INFO/WARN, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 3. Define pydantic models -> validataion for input
class ChunkRequest(BaseModel):
    text: str
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip() or not isinstance(v, str):
            raise ValidationError("Text must be a non-empty string")
        return v.strip()
    
class SummarizeRequest(BaseModel):
    text: str
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip() or not isinstance(v, str):
            raise ValidationError("Text must be a non-empty string")
        return v.strip()


# 4. First we need a Text Chunker to bypass token limit
class TextChunker:
    def __init__(self, tokenizer=None, max_tokens:int=512, clean_whitespace:bool=True):
        """
        Tokenizer: Hugging face tokenizer - best suitable for transformers
        max_tokens: max tokens per chunk (model dependent)
        """
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.clean_whitespace = clean_whitespace
        self.nlp = spacy.load("en_core_web_sm")
        
    def _clean_text(self, text:str) -> str:
        """
        Perform safe, canonical text cleaning for NLP tasks.
        Preserves linguistic structure.
        """
        if not self.clean_whitespace:
            return text
        
        if pd.isna(text) or not isinstance(text, str):
            return ''
        
        # 1. Fix broken encoding and HTML entities
        s = html.unescape(text)    
        # 2. normalize unicode (NFKC helps)
        s = unicodedata.normalize('NFKC', s)
        # 3. Remove HTML tags (robust)
        s = BeautifulSoup(s, 'lxml').get_text(separator=" ")
        # 4. remove ZERO WIDTH and BOM chars
        s = re.sub(r'[\u200B-\u200D\uFEFF]', '', s)
        # 5. Normalize whitespace (spaces, tabs)
        s = re.sub(r"[ \t]+", " ", s)
        # 6. Remove repeated newlines
        s = re.sub(r"\n\s*\n+", "\n", s)
        # 7. Strip leading and trailing whitespace
        s = s.strip()
        
        return s
    
    def _split_sentence(self, text:str) -> List[str]:
        """
        Basic sentence splitting using regex
        You can replace with spaCy or nltk if needed
        """
        # sentences = re.split(r'(?<=[,!?])\s+', text)
        # return [s.strip() for s in sentences if s.strip()]
        ## OR else
        doc = self.nlp(text)
        return [str(sent) for sent in doc.sents]
    
    def chunk(self, text:str) -> List[str]:
        req = ChunkRequest(text=text)
        text = self._clean_text(req.text)
        sentences = self._split_sentence(text)
        
        chunks = []
        curr_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(self.tokenizer(str(sent), add_special_tokens=False)["input_ids"])
            
            # If sentence itself exceeds max_token, give it its own chunk
            if sent_len > self.max_tokens:
                if curr_chunk:
                    chunks.append(" ".join(curr_chunk))
                    # chunks.append(curr_chunk)
                    curr_chunk, current_len = [], 0
                
                chunks.append(sent)
                continue
            
            # If adding sentence exceeds limit -> flush chunk
            if current_len+sent_len > self.max_tokens:
                chunks.append(" ".join(curr_chunk)) # add previous
                # chunks.append(curr_chunk) # add previous
                
                # reset new sent into next chunk
                curr_chunk = [sent]
                current_len = sent_len
            else: # adding sent keep chunk within bound -> add furthur
                curr_chunk.append(sent)
                current_len += sent_len
                
        if curr_chunk: # add last chunk
            chunks.append(" ".join(curr_chunk))
            # chunks.append(curr_chunk)
        
        return chunks


# 5. Final model for summarization
class NewsSummarizer:
    def __init__(self, model_name=None, tokenizer=None, summarizer=None):
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        self.model = summarizer if summarizer else AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if (not self.tokenizer) or (not self.model):
            raise Exception("Model and Tokenizer not loaded")
        
        self.Chunker = TextChunker(tokenizer=self.tokenizer, max_tokens=256)
    
    def _get_summary_length(self, text:str, num_chunks:int, compression:float) -> Tuple[int, int]:
        # text = list of chunks
        encodings = self.tokenizer(text, add_special_tokens=False, truncation=False)
        total_tokens = sum(len(ids) for ids in encodings["input_ids"])
        
        target_tokens = int(total_tokens * compression)

        max_len = max(30, int(target_tokens / num_chunks))
        min_len = int(max_len * 0.5)

        return min_len, max_len

    
    def summarize(self, text:str, compression:Optional[float]=0.5) -> str:
        req = SummarizeRequest(text=text)
        text = req.text
        
        chunks = self.Chunker.chunk(text)
        
        min_len, max_len = self._get_summary_length(text=chunks, num_chunks=len(chunks), compression=compression)
        
        # Batch Tokenization
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=self.Chunker.max_tokens)
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                min_length=min_len, max_length=max_len,
                num_beams=2, no_repeat_ngram_size=3,
                # repetition_penalty=1.15, length_penalty=2.0, 
                early_stopping=True, do_sample=False # deterministic
            )
        
        summary = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
        return "\n".join(summary)
    

# --- Example Usage ---
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

A “veteran’s tableau” was also on display at the parade, with its front portion featuring the Amar Jawan Jyoti, 3D models of historical war machines, which include the T-55 and Vijayant Tank, Hunter, MiG-21, Mirage and Jaguar aircraft, INS Mysore and INS Rajput, and representations from the 1965, 1971 wars and 1999 Kargil Operation Vijay.
Among the major weapon systems showcased at today’s parade were Suryastra Universal Rocket Launcher System (URLS), Brahmos Supersonic cruise missiles and Akash missile systems, as per the PTI report.

The Defence Research & Development Organisation (DRDO) showcased some of its exceptional innovations for national security during the parade and Bharat Parv 2026, including Long Range Anti-Ship Hypersonic Missile (LR-AShM) and DRDO Tableau-‘Naval Technologies for Combat Submarines,” according to a statement by the Ministry of Defence.
The LR-AShM is a Hypersonic Glide Missile capable of engaging static and moving targets. It is a first-of-its-kind with indigenous avionics systems and high accuracy sensor packages.
The tableau displayed indigenously developed technologies and systems which acted as a force multiplier for conventional submarines of the Indian Navy, the statement read, adding that these systems are Integrated Combat Suite (ICS), Wire Guided Heavy Weight Torpedo (WGHWT) and Air Independent Propulsion, which will ensure combat supremacy in the underwater domain.
"""

if __name__ == '__main__':
    # 1. Load the Model and Tokenizer
    model_name = "facebook/bart-large-cnn" # Faster: "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model and Tokenizer loaded successfully.")

    # If the article is > 500 tokens, it will be processed in pieces.
    Summarizer_model = NewsSummarizer(tokenizer=tokenizer, summarizer=model)
    summ = Summarizer_model.summarize(long_article, compression=0.5)
    print("Summary: \n", summ)

