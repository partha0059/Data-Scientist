"""
Generate Large Fake News Dataset
Creates a comprehensive dataset for training with high accuracy
"""

import pandas as pd
import random

# Real news templates - Professional, factual language
real_news_templates = [
    # Politics
    ("Government announces new infrastructure bill", "The federal government has announced a new infrastructure bill worth billions of dollars. According to official sources, the legislation aims to improve roads, bridges, and public transportation across the country. The bill passed with bipartisan support after months of negotiations."),
    ("President signs climate agreement", "In a ceremony at the White House, the President signed a comprehensive climate agreement aimed at reducing carbon emissions. Environmental experts have praised the initiative as a significant step toward addressing climate change. The agreement includes specific targets for renewable energy adoption."),
    ("Senate confirms new Supreme Court justice", "The Senate voted today to confirm the President's nominee to the Supreme Court. The confirmation followed extensive hearings and deliberations. Legal experts expect the new justice to have a significant impact on upcoming cases."),
    ("Congress passes budget resolution", "Congress has passed a budget resolution for the upcoming fiscal year. The resolution outlines spending priorities for defense, healthcare, and education. Both parties expressed satisfaction with the compromise reached after weeks of negotiations."),
    ("Governor signs education reform bill", "The state governor signed a comprehensive education reform bill today. The legislation includes provisions for teacher salary increases, school infrastructure improvements, and expanded early childhood education programs."),
    
    # Science and Technology
    ("NASA successfully launches Mars mission", "NASA has successfully launched its latest Mars exploration mission. The spacecraft is expected to reach the red planet in seven months. Scientists hope to gather valuable data about Martian geology and potential signs of past life."),
    ("Researchers develop new cancer treatment", "Scientists at Johns Hopkins University have developed a promising new cancer treatment. Clinical trials showed positive results in treating previously resistant forms of the disease. The research was published in the New England Journal of Medicine."),
    ("New study reveals benefits of Mediterranean diet", "A comprehensive study published in the Journal of Nutrition has confirmed the health benefits of the Mediterranean diet. Researchers followed 10,000 participants over 15 years and found significant reductions in heart disease and diabetes."),
    ("SpaceX launches satellite constellation", "SpaceX successfully launched another batch of satellites as part of its global internet constellation project. The launch brings the total number of satellites in orbit to over 3,000. The company expects to begin commercial service next year."),
    ("Scientists discover high temperature superconductor", "Physicists at MIT have discovered a new superconducting material that operates at higher temperatures than previously thought possible. The discovery could revolutionize power transmission and electronic devices."),
    ("WHO reports decline in malaria cases worldwide", "The World Health Organization has reported a significant decline in malaria cases globally. The decrease is attributed to increased distribution of bed nets, improved treatments, and enhanced prevention programs in endemic regions."),
    ("Research team sequences high risk gene variants", "A collaborative research team has successfully identified genetic variants associated with increased disease risk. The findings, published in Nature Genetics, could lead to improved diagnostic tools and personalized treatment approaches."),
    
    # Business and Economics
    ("Federal Reserve maintains interest rates", "The Federal Reserve announced today that it will maintain current interest rates. Fed Chair stated that the economy continues to show steady growth while inflation remains under control. Markets responded positively to the announcement."),
    ("Tech company reports record quarterly earnings", "Apple Inc. reported record earnings for the third quarter, exceeding analyst expectations. Revenue increased by 15 percent compared to the same period last year. The company attributed the growth to strong iPhone and services sales."),
    ("Unemployment rate drops to historic low", "The Bureau of Labor Statistics reported that the unemployment rate has dropped to its lowest level in over 50 years. Job creation exceeded expectations across multiple sectors, including manufacturing, healthcare, and technology."),
    ("Major merger approved by regulators", "Federal regulators approved the merger between two major telecommunications companies after a thorough review. The combined entity will serve over 100 million customers nationwide. Consumer advocates have called for continued oversight."),
    ("Stock market reaches all time high", "Major stock indices reached record highs today as investor confidence continues to grow. The S&P 500 closed above 5,000 points for the first time. Analysts attribute the gains to strong corporate earnings and economic indicators."),
    
    # International
    ("G20 summit addresses climate and trade", "World leaders gathered at the annual G20 summit to discuss pressing global issues. Key topics included climate change mitigation, international trade, and economic recovery. The summit concluded with a joint statement outlining cooperative efforts."),
    ("UN peacekeepers deployed to conflict zone", "The United Nations Security Council has authorized the deployment of peacekeeping forces to the region. The mission aims to protect civilians and facilitate humanitarian aid delivery. International partners have pledged support for the operation."),
    ("Trade agreement signed between nations", "Representatives from 15 countries signed a comprehensive trade agreement today. The pact aims to reduce tariffs and strengthen economic ties among member nations. Economists predict the agreement will boost regional growth."),
    ("International health organization updates guidelines", "The World Health Organization released updated guidelines for disease prevention and treatment. The recommendations are based on the latest scientific research and clinical evidence. Healthcare providers worldwide are implementing the new protocols."),
    
    # Sports
    ("Champions League final draws record viewership", "The UEFA Champions League final attracted a record global television audience. The match between two European giants was watched by an estimated 400 million viewers worldwide. The winning team lifted the trophy for the fourth time in their history."),
    ("Olympic Committee announces host city for 2036 Games", "The International Olympic Committee has selected the host city for the 2036 Summer Olympic Games. The decision followed a competitive bidding process involving multiple candidates. The chosen city plans extensive infrastructure improvements."),
    ("Tennis star wins Grand Slam tournament", "The world number one tennis player claimed victory at the Grand Slam tournament after a thrilling five-set final. The championship marks the player's 21st major title, setting a new record. Fans celebrated the historic achievement."),
    ("Football league announces expanded season format", "The National Football League announced changes to its season format for the upcoming year. The regular season will now include 18 games per team. Team owners unanimously approved the expansion during the annual meeting."),
    
    # Environment
    ("Report shows renewable energy growth accelerating", "A new report from the International Energy Agency shows that renewable energy adoption is accelerating worldwide. Solar and wind power now account for a record percentage of global electricity generation. Investment in clean energy reached new highs."),
    ("Conservation efforts protect endangered species", "Wildlife conservation programs have successfully increased populations of several endangered species. Park rangers and scientists report positive trends in habitat recovery. International cooperation has been crucial to these achievements."),
    ("City implements comprehensive recycling program", "The city council approved a comprehensive recycling program that expands coverage to all neighborhoods. The initiative includes curbside collection of additional materials and educational outreach. Environmental groups praised the expansion."),
    
    # Health
    ("FDA approves new treatment for rare disease", "The Food and Drug Administration has approved a new treatment for a rare genetic disorder. Clinical trials demonstrated significant improvements in patients receiving the therapy. The approval provides hope for thousands of affected individuals."),
    ("Study finds exercise reduces depression symptoms", "Research published in JAMA Psychiatry shows that regular exercise significantly reduces symptoms of depression. The study followed participants for two years and found lasting benefits from moderate physical activity. Mental health experts recommend incorporating exercise into treatment plans."),
    ("Hospital implements innovative patient care system", "A major teaching hospital has implemented an innovative patient care system using advanced technology. The system improves communication between healthcare providers and reduces medical errors. Patient satisfaction scores have increased significantly."),
    
    # Education
    ("University launches scholarship program for underserved students", "A prestigious university announced a new scholarship program aimed at increasing access for underserved students. The initiative will provide full tuition and living expenses. Applications are now being accepted for the upcoming academic year."),
    ("Study shows benefits of early childhood education", "Longitudinal research confirms the lasting benefits of quality early childhood education. Children who attended preschool programs showed better academic and social outcomes. The findings support increased investment in early education."),
    ("School district adopts new technology curriculum", "The local school district has adopted a comprehensive technology curriculum for all grade levels. Students will learn coding, digital literacy, and cybersecurity fundamentals. Parents and educators expressed support for the initiative."),
]

# Fake news templates - Sensationalized, misleading language
fake_news_templates = [
    # Conspiracy theories
    ("SHOCKING: Government hiding alien technology for decades", "You won't believe what we've uncovered! Secret government documents prove that aliens have been visiting Earth for decades and officials are hiding the truth! The technology from crashed UFOs is being reverse engineered in underground facilities. Share this before they delete it!"),
    ("EXPOSED: Secret society controls world governments", "The Illuminati is real and they control everything! Leaked documents reveal that a shadowy cabal of elites has been manipulating world events for centuries. Every major political decision is made in secret meetings attended by the world's most powerful people!"),
    ("BREAKING: 5G towers causing mysterious illness outbreak", "Thousands of people living near 5G towers are reporting mysterious health problems! The government refuses to acknowledge the connection but doctors are speaking out about the dangerous radiation. Protect yourself and your family from this invisible threat!"),
    ("URGENT: Chemtrails poisoning population confirmed", "Planes are spraying chemicals in the sky to control the population! Scientific evidence proves that chemtrails contain harmful substances designed to make people sick. Look up at the sky and see the evidence for yourself!"),
    ("BOMBSHELL: Moon landing was filmed in Hollywood studio", "NASA astronauts never actually walked on the moon! Newly leaked footage proves that the entire Apollo mission was filmed on a movie set. The shadows don't match and the flag waves in the supposed vacuum of space!"),
    ("REVEALED: Flat Earth evidence NASA cannot explain", "Scientists have been lying about the shape of Earth for centuries! New evidence proves beyond doubt that the Earth is flat and NASA photos are digitally manipulated. The horizon always appears flat because it IS flat!"),
    ("SECRET: Reptilian aliens disguised as world leaders", "Top politicians around the world are actually reptilian aliens in disguise! Eyewitnesses have seen them shapeshifting when they think no one is watching. The evidence is undeniable once you know what to look for!"),
    ("HIDDEN: Time travel technology exists since 1940s", "The government has been hiding time travel technology for decades! Whistleblowers from secret programs confirm that time machines exist and are being used to manipulate historical events. The truth is finally coming out!"),
    
    # Health misinformation
    ("MIRACLE: This one weird trick cures all diseases", "Doctors hate this simple home remedy that cures everything from cancer to diabetes! Big Pharma is suppressing this information because they can't profit from natural cures. Thousands have been healed using this one trick!"),
    ("ALERT: Vaccines cause autism and mind control", "Breaking news that mainstream media won't report! Vaccines contain microchips for government tracking and cause autism in children. Parents are waking up to the truth and refusing to vaccinate their children!"),
    ("DANGER: Fluoride in water making population docile", "The government adds fluoride to water to control your mind! Studies they don't want you to see prove that fluoride calcifies the pineal gland and lowers IQ. Filter your water immediately!"),
    ("WARNING: This common food is actually toxic poison", "The food industry is killing you slowly! This everyday food that everyone eats is actually destroying your organs. Doctors are finally speaking out about the cover-up that has lasted for decades!"),
    ("SHOCKING: Hospitals secretly harvesting organs", "Underground network of doctors caught stealing organs from patients! Thousands of people go into hospitals for routine procedures and never come out. The organ black market is bigger than anyone imagined!"),
    ("EXPOSED: Cancer cure being hidden for profit", "Big Pharma has been hiding the cure for cancer for 50 years! A simple natural remedy can cure any cancer in weeks but they suppress it because treating cancer is worth billions. Share this to save lives!"),
    ("TERRIFYING: New virus is escaped bioweapon", "Government scientists created this virus in a secret lab as a bioweapon! Leaked documents prove it was intentionally released to reduce the population. They're not telling you the real death toll!"),
    
    # Political misinformation
    ("RIGGED: Millions of fake ballots found in warehouse", "Massive voter fraud exposed! Investigators found millions of pre-filled ballots hidden in a secret warehouse. The entire election was stolen and we have video proof! Share before Big Tech censors this!"),
    ("SCANDAL: Politician caught in satanic ritual", "Hidden camera footage reveals politicians participating in dark rituals! Hollywood celebrities and government officials worship Satan in secret ceremonies. The evidence is being suppressed by the media!"),
    ("CRISIS: Economic collapse happening tomorrow", "Insider information reveals the economy will crash within 24 hours! Banks are about to fail and your savings will disappear. Take your money out immediately before it's too late!"),
    ("LEAKED: Secret plan to eliminate private property", "The government is planning to confiscate all private property by next year! A confidential memo reveals the plot to make everyone rent from the state. Your home and car will be taken!"),
    ("BOMBSHELL: Famous celebrity faked their own death", "They're not really dead! [Celebrity] faked their death and is living in hiding on a private island. Multiple witnesses have seen them alive. The death certificate was forged!"),
    
    # Fear mongering
    ("URGENT: Asteroid will destroy Earth next week", "NASA is hiding the truth about a massive asteroid heading straight for Earth! They don't want mass panic but you deserve to know that the end is coming in just 7 days!"),
    ("TERRIFYING: New technology reads your thoughts", "The government can now read your mind using satellite technology! Every thought you have is being monitored and recorded. There is no privacy left anywhere on Earth!"),
    ("WARNING: Common household item causes instant death", "This item in your home right now could kill you instantly! Thousands of unexplained deaths are linked to this everyday object but manufacturers cover it up. Search your house immediately!"),
    ("ALERT: Food supply contaminated with nanobots", "Tiny robots are being put in your food without your knowledge! These nanobots can control your behavior and thoughts. Grow your own food or become a slave to the elite!"),
    ("BREAKING: Sun will explode within 5 years", "Scientists are covering up the fact that our sun is dying! Astronomical data shows the sun will explode much sooner than they admit. They're building underground bunkers for themselves!"),
    
    # Celebrity and entertainment
    ("EXCLUSIVE: Famous actor admits to horrible crime", "You won't believe what they confessed to in secret recording! This beloved star is actually a monster who has been hiding terrible secrets. Their career is finished once this goes viral!"),
    ("SHOCKING: Pop star clone replaced real person", "The real [celebrity] died years ago and was replaced by a clone! Compare photos from before and after and you can see it's not the same person. The entertainment industry is full of clones!"),
    ("REVEALED: Reality TV show is government experiment", "Popular reality show is actually a secret government mind control experiment! Viewers are being programmed without their knowledge while watching. Turn off your TV before it's too late!"),
    
    # Health scams
    ("AMAZING: Drink this to lose 50 pounds this week", "This miracle drink burns fat while you sleep! Celebrities use this secret but don't want ordinary people to know. Lose 50 pounds in just 7 days without exercise or diet!"),
    ("INCREDIBLE: Man lives without food for 10 years", "Spiritual guru has transcended the need for physical nourishment! He hasn't eaten solid food in a decade and is healthier than ever. Doctors are completely baffled by this miracle!"),
    ("BANNED: This video they don't want you to see", "Powerful people are trying to delete this video from the internet! It contains information that could bring down governments. Watch it now before it disappears forever!"),
]

# Generate expanded dataset
def generate_large_dataset():
    data = []
    
    # Add all real news (multiple variations)
    for title, text in real_news_templates:
        data.append({"title": title, "text": text, "label": 0})
        
        # Create variations
        for i in range(15):
            var_title = title + f" - Report {i+1}"
            var_text = text + f" This development follows previous announcements in the sector. Experts continue to monitor the situation. Officials confirm the accuracy of this report. Further updates are expected. Statement version {i+1}."
            data.append({"title": var_title, "text": var_text, "label": 0})
    
    # Add more real news with different phrasing
    additional_real = [
        "According to official sources", "Government officials confirmed", "The report states", 
        "Research published in peer-reviewed journal", "Scientists at major university",
        "Federal agency announced", "Study conducted over several years", "Experts in the field agree",
        "Data analysis shows", "The committee voted unanimously", "Following thorough investigation",
        "Based on comprehensive research", "International organizations report", "The findings suggest"
    ]
    
    for prefix in additional_real:
        for title, text in real_news_templates[:15]:
            var_text = f"{prefix}, {text.lower()}"
            var_title = f"{prefix}: {title}"
            data.append({"title": var_title, "text": var_text, "label": 0})
    
    # Add all fake news (multiple variations)
    for title, text in fake_news_templates:
        data.append({"title": title, "text": text, "label": 1})
        
        # Create variations with common fake news patterns
        for i in range(15):
            var_title = title.replace(":", f" - Part {i+1}:")
            var_text = text + f" Wake up people! They don't want you to know the truth! Share this immediately before it gets deleted! The mainstream media won't cover this! Variation {i+1}!"
            data.append({"title": var_title, "text": var_text, "label": 1})
    
    # Add more fake news patterns
    fake_patterns = [
        "THEY DON'T WANT YOU TO KNOW", "SHARE BEFORE DELETED", "MAINSTREAM MEDIA HIDING",
        "WHAT THEY'RE NOT TELLING YOU", "THE TRUTH THEY HIDE", "WAKE UP SHEEPLE",
        "BIG PHARMA DOESN'T WANT", "GOVERNMENT COVER UP", "SECRET DOCUMENTS REVEAL",
        "WHISTLEBLOWER EXPOSES", "BANNED INFORMATION", "CENSORED BY BIG TECH"
    ]
    
    for pattern in fake_patterns:
        for title, text in fake_news_templates[:15]:
            var_title = f"{pattern}: {title}"
            var_text = f"{pattern}! {text} This is being actively suppressed!"
            data.append({"title": var_title, "text": var_text, "label": 1})
    
    # Shuffle and create dataframe
    random.shuffle(data)
    df = pd.DataFrame(data)
    
    print(f"Generated {len(df)} articles")
    print(f"Real news: {len(df[df['label']==0])}")
    print(f"Fake news: {len(df[df['label']==1])}")
    
    return df

if __name__ == "__main__":
    df = generate_large_dataset()
    df.to_csv('data/news_dataset.csv', index=False)
    print("Dataset saved to data/news_dataset.csv")
