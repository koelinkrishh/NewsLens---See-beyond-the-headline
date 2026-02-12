import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# 1. Suppress the oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Suppress other TensorFlow logging (0=all, 1=no INFO, 2=no INFO/WARN, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Load the Model and Tokenizer
model_name = "facebook/bart-large-cnn" # Faster: "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model and Tokenizer loaded successfully.")

def summarize_text(text_to_summarize):
    """
    Handles tokenization, chunking, and summarization.
    """
    L = len(text_to_summarize.split(" "))
    # 2. Convert text to tokens
    # We don't use truncation=True here because we WANT to see the full length
    inputs = tokenizer(text_to_summarize, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    
    # 3. Define Chunk Size 
    # BART's limit is 1024, but we use 500 to stay safe and leave room for generation
    chunk_size = 500 
    
    # Split the input_ids into chunks of 500 tokens each
    chunks = [input_ids[i : i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
    
    summaries = []
    
    for chunk in chunks:
        # 4. Convert token IDs back to a batch format for the model
        chunk_tensor = chunk.unsqueeze(0) 
        
        # 5. Generate Summary for the chunk
        with torch.no_grad():
            output_ids = model.generate(
                chunk_tensor, 
                max_length=int(L/5), 
                min_length=int(L/10), 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
        
        # 6. Decode tokens back to text
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # 7. Combine mini-summaries
    full_summary = " ".join(summaries)
    return full_summary

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

# If the article is > 500 tokens, it will be processed in pieces.
# final_result = summarize_text(long_article)
# print("Final Summary:\n", final_result)