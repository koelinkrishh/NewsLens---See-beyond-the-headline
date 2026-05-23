import os
import sys
from dotenv import load_dotenv

# Ensure root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Components.QAarticle import RAG, QA
from src.Components.Embedding import ArticleEmbeddingEngine

if __name__ == "__main__":
    print("========================================")
    print("Testing QA and RAG system components...")
    print("========================================")
    
    # Load env keys
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("[WARNING] GROQ_API_KEY not found in env! The test will try to initialize, but LLM call may fail.")
        
    try:
        # 1. Initialize Article Embedding Engine
        print("\n1. Initializing Embedding Engine...")
        engine = ArticleEmbeddingEngine()
        
        # 2. Initialize RAG
        print("\n2. Initializing RAG Engine...")
        rag = RAG(engine=engine)
        print(f"   Using database: {rag.db_type}")
        
        # 3. Sample Article
        test_article = ("""Indian politics has acquired an unusual mascot: the cockroach.
            It's not the lotus of the Bharatiya Janata Party (BJP), India's governing party, or the hand symbol of the opposition Congress, but the cockroach - stubborn, reviled and considered indestructible - which has recently become an unlikely yet relatable political symbol for young Indians online.
            The insect was thrust into the spotlight last week after controversial comments made by India's Chief Justice Surya Kant. During a hearing, he allegedly compared unemployed young people drifting towards journalism and activism with cockroaches and parasites.
            He later clarified that he was referring specifically to people with "fake and bogus degrees", not India's youth more broadly.
            But by then the comments had already spread widely online, triggering outrage, jokes - and a humorous political uprising called the "Cockroach Janta Party" (Cockroach People's Party) or CJP.
            It is not a formal political party but a satire-heavy online collective whose membership criteria include being unemployed, lazy, chronically online and possessing "the ability to rant professionally".
            It was created by Abhijeet Dipke, a political communications strategist and student at Boston University. He says the idea came as a joke.
            Before moving to the US, he worked with the Aam Aadmi Party (AAP), a political organisation that emerged from an anti-corruption movement and is known for its strong social media presence.
            "I thought we should all come together, maybe just start a platform," he told BBC Marathi.
            What followed was much bigger than he expected.
            Within days, the CJP amassed tens of thousands of sign-ups through a Google form, inspired the hashtag #MainBhiCockroach ("I too am a cockroach") and endorsements from opposition leaders. The movement also spilled offline, with young volunteers turning up dressed as cockroaches at clean-up drives and protests, in a theatrical embrace of the label.
            On Thursday, the CJP's Instagram account crossed 10 million followers, overtaking the official account of the BJP - widely described as the world's largest political party by membership - which has around 8.7 million followers.
            However, its X account, with more than 200,000 followers, is currently not visible in India, with people trying to view it being told that it has been withheld "in response to a legal demand".
            But the momentum has only continued to build.
            
            For supporters, the movement represents what one fan called "a breath of fresh air" in a political culture many see as overly managed and hostile to dissent. Supporters included opposition politicians such as Mahua Moitra and Kirti Azad, as well as senior lawyer Prashant Bhushan.            Critics, meanwhile, dismiss it as online political theatre linked to the opposition, pointing to Dipke's earlier association with the AAP and arguing it is less spontaneous rebellion than carefully packaged digital politics.
            Beyond the immediate reactions, the movement has become a marker of generational fatigue among many young Indians who say they are constantly exposed to politics online, but rarely feel represented within it.
            India has one of the world's youngest populations, with roughly half its 1.4 billion people under 30 years. Yet formal political participation remains limited.
            A recent survey found that 29% of young Indians avoided political engagement altogether, while only 11% were members of a political party.
            "People are frustrated because they don't feel heard or represented," Dipke said.
            Across South Asia, recent years have seen waves of youth-led protests that have unseated governments in Sri Lanka, Nepal and Bangladesh, often driven by anger over jobs, prices and stalled futures.
            India has so far avoided anything comparable, but the underlying pressures are familiar.
            A fast-growing economy has not eased anxieties over work, inequality or the rising cost of simply getting by.
            For many entering adulthood, education no longer guarantees stability, and the promise of upward mobility can feel increasingly fragile.
            While Dipke rejects comparisons with upheavals in Nepal or Sri Lanka, saying India's situation is different, he argues that frustration among young people is still real - just expressed in more fragmented, online ways.
            "Gen Z has given up on traditional political parties and wants to create its own political front in a language they understand," he said.
            The CJP's website reflects this sensibility, reading less like a manifesto and more like something shaped inside internet culture.
            It describes itself as "the voice of the lazy and unemployed," while also claiming "zero sponsors" and "one stubborn swarm", and inviting supporters to join a movement for people "tired of pretending everything is fine".
            There are mock forms, deliberately rough edges and a visual language that feels closer to an inside joke than an institution.

            Abhijeet Dipke/X Founder of Cockroach Janta Party, Abhijeet Dipke poses wearing a black jacket and a black tshirt. His hair looks disheveled. Abhijeet Dipke/X
            The collective was started by 30-year-old Abhijeet Dipke, who is a student in Boston
            And yet, buried inside the humour are recognisable political claims: accountability, media reform, electoral transparency and expanded representation for women. They sit alongside self-deprecating jokes about doomscrolling, unemployment and general political burnout.
            The tone, somewhere between parody and sincerity, is part of its appeal. The jokes land because the frustrations underneath them are familiar: around jobs, inequality, corruption and political alienation.
            Many have pointed out that even the choice of mascot makes sense. The cockroach is not heroic or aspirational, but something more basic: resilient, adaptable and capable of surviving hostile conditions with very low expectations.
            Of course, this blurring of humour and politics is hardly new.
            In Italy, comedian Beppe Grillo channelled anti-establishment humour into the Five Star Movement, while in Ukraine Volodymyr Zelenskyy went from playing a fictional president on television to becoming a real one. In the US, the Donald Trump era has sparked repeated arguments about whether satire itself has begun to collapse under a political reality that often already feels like parody.
            India's version takes a more online form: a meme-driven, insect-themed movement shaped by hashtags, burnout and ironic despair.
            At first glance, it seems unusual. But it is not entirely out of place in Indian politics.
            Politicians here have long embraced the power of spectacle, from meditating in Himalayan caves to switching parties amid scenes of legislators being bundled into buses or holed up in hotels.
            Online campaigns rely on carefully choreographed viral videos and punchy slogans designed for maximum reach.
            Against that backdrop, an insect-themed political movement feels oddly plausible.
            It also helps explain why it spread so quickly - not necessarily because young Indians want another political party, but because many are searching for a language to express their frustration.
            "I think CJP is just the beginning," Dipke said. "Young people are fed up with the current political system, and more youth organisations will come forward."
            Others, however, are more sceptical, saying the party is likely to fade as quickly as it emerged.
            Either way, the CJP has already done something unusual in Indian politics: it has briefly made some young people feel seen.
            In earlier eras, youth political anger produced manifestos. In 2026, it sometimes produces meme parties with insect mascots.
            With inputs from Ashay Yedge, BBC Marathi"""    
        )
        
        # 4. Ingest Article
        print("\n4. Ingesting test article into RAG vector store...")
        hash_1 = rag.add_article(test_article)
        print(f"   Article ingested successfully. Hash: {hash_1}")
        
        # 5. Ingest duplicate article to test deduplication
        print("\n5. Ingesting same article again (Deduplication Check)...")
        hash_2 = rag.add_article(test_article)
        print(f"   Duplicate article processed. Hash: {hash_2}")
        if hash_1 == hash_2:
            print("   [SUCCESS] Deduplication verified! Article was not re-embedded.")
        else:
            print("   [WARNING] Deduplication failed: hashes did not match.")
            
        # 6. Retrieve relevant context
        test_question = "What does CJP represent for youth?"
        print(f"\n6. Retrieving context for query: '{test_question}'...")
        docs = rag.retrieve(test_question, hash_1, k=2)
        print(f"   Retrieved {len(docs)} chunks:")
        for idx, doc in enumerate(docs):
            print(f"     Chunk {idx+1}: {doc.page_content[:150]}...")
            
        # 7. Initialize QA Engine
        print("\n7. Initializing QA Engine...")
        qa = QA(rag=rag, provider="groq", model="llama-3.1-8b-instant")
        
        # 8. Get Answer
        print(f"\n8. Querying QA Engine: '{test_question}'...")
        result = qa.answer_question(test_article, test_question)
        
        print("\n================ ANSWER ================")
        print(result["answer"])
        print("========================================\n")
        
        print("Grounded Chunks Context used:")
        for idx, chunk in enumerate(result["context"]):
            print(f" - [{idx+1}]: {chunk}")
            
        print("\n[SUCCESS] QA/RAG END-TO-END VERIFICATION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n[ERROR] Test run encountered an issue: {e}")
        import traceback
        traceback.print_exc()
