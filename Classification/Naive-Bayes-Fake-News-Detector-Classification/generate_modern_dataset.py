"""
Generate 2024-2025 Fake News Patterns Dataset
Based on research on modern misinformation trends:
- AI-generated content patterns
- Deepfake related content
- Election misinformation
- Health misinformation
- Social media viral patterns
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# ============================================
# 2024-2025 FAKE NEWS PATTERNS
# Based on current misinformation trends research
# ============================================

# Modern AI-related fake news (new pattern)
ai_fake_news = [
    ("AI EXPOSED: ChatGPT secretly recording all your conversations!", "Leaked documents reveal that ChatGPT has been recording and selling all user conversations to government agencies! Your private chats are being monitored! Delete the app immediately!"),
    ("BREAKING: AI robots taking over hospitals, patients dying!", "Hospitals using AI are seeing mysterious patient deaths! The robots are making fatal decisions and doctors are being replaced! This is being covered up by Big Tech!"),
    ("SHOCKING: AI creating fake humans that walk among us!", "Artificial intelligence has created synthetic humans that look exactly like real people! They're infiltrating society and you can't tell the difference! Share this before they silence us!"),
    ("URGENT: Your smartphone AI is controlling your thoughts!", "Scientists confirm that AI assistants are using subliminal messages to control your behavior! Siri and Alexa are programming your brain while you sleep!"),
    ("EXPOSED: AI predicting and causing natural disasters!", "Secret AI systems are being used to create earthquakes and hurricanes! Climate disasters are manufactured by tech companies for profit!"),
    ("TERRIFYING: AI clones of celebrities committing crimes!", "Deepfake AI versions of famous celebrities are being used to commit crimes! The real stars are being framed while AI doubles roam free!"),
    ("WARNING: AI generating fake money flooding the economy!", "AI systems are creating undetectable counterfeit currency! The entire financial system is about to collapse because of AI fraud!"),
    ("BOMBSHELL: AI writing fake news articles automatically!", "AI bots are flooding the internet with millions of fake articles daily! You can't trust anything you read online anymore! Everything is AI-generated lies!"),
]

# Modern election/political misinformation (2024 patterns)
election_fake_news = [
    ("RIGGED: AI voting machines switching millions of votes!", "Whistleblower reveals AI-powered voting machines are programmed to flip votes! The entire 2024 election was manipulated by algorithms!"),
    ("EXPOSED: Foreign AI bots creating fake voters!", "Millions of fake voter registrations created by AI systems from hostile nations! The voter rolls are completely compromised!"),
    ("BREAKING: Politicians using AI doubles for public appearances!", "AI-generated deepfake versions of politicians are giving speeches while real leaders hide! Nothing you see is real anymore!"),
    ("SCANDAL: AI manipulating all election polls!", "Every poll you see is AI-generated propaganda! The real numbers are being hidden from the public!"),
    ("URGENT: Social media AI censoring election truth!", "AI algorithms are automatically deleting any posts that expose election fraud! Silicon Valley is controlling democracy!"),
    ("LEAKED: Secret AI surveillance during elections!", "AI cameras at polling stations are using facial recognition to track and intimidate voters! Big Brother is watching everyone!"),
]

# Modern health misinformation (current trends)
health_fake_news = [
    ("SHOCKING: New vaccine contains AI nanobots!", "Scientists discover microscopic AI robots in latest vaccines! These nanobots can be remotely controlled by 5G signals!"),
    ("WARNING: AI diagnosing fake diseases for profit!", "Hospital AI systems are creating fake diagnoses to sell expensive treatments! Millions being prescribed unnecessary medications!"),
    ("EXPOSED: AI-designed viruses being released!", "Leaked lab documents show AI is being used to engineer new deadly pathogens! The next pandemic was designed by artificial intelligence!"),
    ("URGENT: AI predicting who will die next!", "Insurance companies using AI to predict death dates and deny coverage! Your health data is being used against you!"),
    ("TERRIFYING: AI doctors giving fatal advice!", "Chatbot doctors are prescribing deadly medication combinations! People are dying from AI medical advice!"),
    ("BANNED: This AI health truth they don't want you to know!", "AI has discovered the cure for all diseases but pharmaceutical companies are suppressing it! Natural remedies work better than AI-approved drugs!"),
]

# Social media viral misinformation patterns
social_media_fake = [
    ("VIRAL: Celebrity AI deepfake scandal rocks Hollywood!", "Shocking AI-generated videos of celebrities surface! Everything you thought was real is fake! The entire entertainment industry is artificial!"),
    ("TRENDING: AI predicts world ending event next month!", "Sophisticated AI models all agree: major catastrophe coming in 30 days! World governments are preparing bunkers while keeping public in the dark!"),
    ("MUST SHARE: AI surveillance proof before deleted!", "This video exposes AI cameras watching your every move! They don't want this to go viral! Share immediately before Big Tech removes it!"),
    ("BREAKING NOW: AI stock market crash imminent!", "AI trading algorithms about to trigger the biggest market crash in history! Sell everything now before you lose everything!"),
    ("ALERT: AI creating fake people on dating apps!", "90% of dating profiles are AI-generated fake people! Romance scammers using AI to catfish millions!"),
    ("EXPOSED: AI weather control causing climate disasters!", "AI systems controlling weather satellites are causing floods and droughts on purpose! Climate change is artificial intelligence!"),
]

# Crypto/Financial AI scams
financial_fake_news = [
    ("BREAKING: AI cryptocurrency worth millions disappeared overnight!", "New AI-powered crypto was a scam! Millions of investors lost everything! The AI was programmed to steal!"),
    ("URGENT: Banks using AI to freeze accounts!", "AI algorithms are randomly freezing bank accounts of innocent people! You could wake up with no access to your money!"),
    ("EXPOSED: AI tax audits targeting specific groups!", "IRS AI is programmed to unfairly audit certain demographics! Your taxes are being weaponized by algorithms!"),
    ("WARNING: AI traders manipulating all markets!", "Every stock price is controlled by AI bots! Human investors have no chance against machine manipulation!"),
]

# ============================================
# 2024-2025 REAL NEWS PATTERNS
# Legitimate journalism patterns for contrast
# ============================================

real_news_ai = [
    ("OpenAI announces GPT-5 with improved safety features", "OpenAI has unveiled its latest language model, GPT-5, featuring enhanced safety protocols and reduced hallucination rates. The company stated that extensive testing was conducted before release. Industry analysts predict significant impact on enterprise applications."),
    ("Google DeepMind achieves breakthrough in protein folding", "Researchers at Google DeepMind have published findings showing improved accuracy in protein structure prediction. The work, peer-reviewed in Nature, could accelerate drug discovery processes. Scientists worldwide are integrating the new methods into their research."),
    ("EU passes comprehensive AI regulation framework", "The European Parliament has approved the AI Act, establishing the world's first comprehensive artificial intelligence regulations. The legislation categorizes AI applications by risk level and sets compliance requirements for developers."),
    ("Microsoft integrates Copilot across Office suite", "Microsoft announced the general availability of AI-powered Copilot features in its Office applications. The integration includes document summarization, email drafting assistance, and spreadsheet analysis capabilities."),
    ("Study finds AI medical diagnostics match specialist accuracy", "Research published in the Journal of Medical AI shows that diagnostic AI systems achieved accuracy rates comparable to specialist physicians in radiology analysis. The study emphasizes AI as a supportive tool rather than replacement."),
    ("Universities launch AI ethics certification programs", "Leading universities including MIT and Stanford have announced new certification programs in AI ethics and governance. The programs aim to address growing demand for professionals who can navigate AI implementation responsibly."),
]

real_news_political = [
    ("Congress holds hearing on AI election oversight", "Congressional committees conducted hearings on artificial intelligence's role in election security. Experts testified on both risks and protective measures, with bipartisan support for increased oversight measures."),
    ("Federal agency issues AI hiring guidelines", "The Department of Labor released new guidelines for employers using AI in hiring processes. The regulations require transparency in algorithmic decision-making and mandate human oversight for final hiring decisions."),
    ("State governors announce AI infrastructure initiative", "A coalition of state governors announced a joint initiative to develop AI-ready infrastructure. The plan includes data center development, workforce training programs, and research partnerships with universities."),
    ("Supreme Court agrees to hear AI copyright case", "The Supreme Court will review a case involving AI-generated content and copyright law. Legal experts anticipate the ruling will establish important precedents for intellectual property in the age of generative AI."),
    ("International AI safety summit produces joint statement", "Representatives from 30 nations signed a joint declaration on AI safety standards at the global summit. The statement outlines shared principles for responsible AI development and establishes working groups for implementation."),
]

real_news_technology = [
    ("Research team develops more efficient AI training methods", "Scientists at Carnegie Mellon University published research on training AI models with significantly reduced computational requirements. The technique could lower both costs and environmental impact of AI development."),
    ("Tech companies form AI safety consortium", "Major technology companies announced the formation of a consortium dedicated to AI safety research. Members committed to sharing safety testing protocols and coordinating on risk assessment methodologies."),
    ("New AI chip architecture reduces power consumption by 40%", "Semiconductor researchers unveiled a new processor architecture optimized for AI workloads. The design achieves comparable performance to current chips while substantially reducing energy requirements."),
    ("AI-assisted climate modeling improves prediction accuracy", "Climate scientists report that incorporating AI into environmental models has improved prediction accuracy for extreme weather events. The research was conducted across multiple international research institutions."),
    ("Healthcare system implements AI triage with positive outcomes", "A major healthcare network reported successful pilot results from AI-assisted emergency room triage. Patient wait times decreased while accurate prioritization of serious cases improved."),
]

real_news_health = [
    ("FDA approves AI-assisted diagnostic tool for rare diseases", "The Food and Drug Administration granted approval to an AI system designed to assist in diagnosing rare genetic conditions. Clinical trials showed the tool helped physicians identify conditions faster."),
    ("Researchers use AI to identify potential cancer treatments", "University researchers employed machine learning to screen millions of compound combinations for cancer treatment potential. Promising candidates will proceed to laboratory testing phases."),
    ("Mental health app with AI support shows positive study results", "A randomized controlled trial found that an AI-enhanced mental health application produced modest improvements in anxiety symptoms. Researchers noted the tool works best alongside traditional therapy."),
    ("AI system helps optimize hospital resource allocation", "A hospital network implemented an AI system for predicting patient flow and resource needs. Initial results show improved efficiency in staff scheduling and equipment availability."),
]

def generate_modern_dataset():
    """Generate comprehensive 2024-2025 fake news detection dataset"""
    data = []
    
    # Add fake news with variations
    fake_sources = [
        (ai_fake_news, "AI/Technology"),
        (election_fake_news, "Political"),
        (health_fake_news, "Health"),
        (social_media_fake, "Social Media"),
        (financial_fake_news, "Financial"),
    ]
    
    for source, category in fake_sources:
        for title, text in source:
            # Original
            data.append({
                "title": title,
                "text": text,
                "label": 1,
                "category": category,
                "year": "2024"
            })
            # Variations
            for i in range(8):
                var_title = title.replace(":", f" - Update {i+1}:")
                var_text = text + f" Share this truth before it's censored! Wake up people! Variation {i+1}!"
                data.append({
                    "title": var_title,
                    "text": var_text,
                    "label": 1,
                    "category": category,
                    "year": "2024"
                })
    
    # Add real news with variations
    real_sources = [
        (real_news_ai, "AI/Technology"),
        (real_news_political, "Political"),
        (real_news_technology, "Technology"),
        (real_news_health, "Health"),
    ]
    
    for source, category in real_sources:
        for title, text in source:
            # Original
            data.append({
                "title": title,
                "text": text,
                "label": 0,
                "category": category,
                "year": "2024"
            })
            # Variations
            for i in range(8):
                var_title = f"{title} - Report {i+1}"
                var_text = text + f" Further details are expected to be released. Experts continue to monitor developments. Update {i+1}."
                data.append({
                    "title": var_title,
                    "text": var_text,
                    "label": 0,
                    "category": category,
                    "year": "2024"
                })
    
    df = pd.DataFrame(data)
    random.shuffle(data)
    df = pd.DataFrame(data)
    
    print(f"Generated {len(df)} modern 2024-2025 articles")
    print(f"   Fake news: {len(df[df['label']==1])}")
    print(f"   Real news: {len(df[df['label']==0])}")
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data/recent', exist_ok=True)
    df = generate_modern_dataset()
    df.to_csv('data/recent/modern_fake_news_2024.csv', index=False)
    print("Saved to data/recent/modern_fake_news_2024.csv")
