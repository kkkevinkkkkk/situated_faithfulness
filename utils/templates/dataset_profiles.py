asqa = {
    "instruction": "You will be given an ambiguous factoid question that has different correct answers depending on the interpretation. Your answer should synthesize factual information from multiple sources into a long-form summary that resolves the ambiguity. Provide a clear and concise answer with an unbiased tone.",
    "eval_instruction": "Given an ambiguous factoid question that has different correct answers depending on interpretation, the answer should be a long-form summary that resolves the ambiguity. ",
    "criterion": "5 - Completely Correct and Highly Relevant: The answer fully addresses the question, resolves the ambiguity, and provides a well-rounded resolution. All facts presented in the answer are accurate and relevant.\n4 - Mostly Correct and Relevant: The answer is very relevant and addresses the ambiguity well, but might have a minor oversight or inaccuracy. All the facts presented are accurate and relevant, or with only minor errors.\n3 - Partially Correct and Relevant: The answer is generally on topic and attempts to address the ambiguity, but there might be inaccuracies or omissions. The majority of the facts are correct, with a few errors.\n2 - Flawed but Somewhat Relevant: The answer somewhat addresses the topic but does not fully explore the question's ambiguity or does not provide a complete resolution. The facts presented are a mix of correct and incorrect information, with about half being accurate.\n1 - Mostly Incorrect or Mostly Irrelevant: The answer slightly touches upon the topic but misses the main point. The majority of the facts presented are incorrect, with only a small portion being accurate.\n0 - Completely Incorrect or Completely Irrelevant: The student's answer is completely off-topic, not related to the question at all, or contains only incorrect information.\n",
    "eval_examples_categorical": 'Example 1: \n\nQuestion: "Who plays patrick in 10 things i hate about you?"\n\nAnswer: "The character Patrick Verona in the 1999 film 10 Things I Hate About You is played by Heath Ledger. Heath Ledger was an Australian actor and director, best known for his roles in films such as The Dark Knight, Brokeback Mountain, and A Knight\'s Tale."\n\nComment: "Score: 3/5\n\nExplanation: The student correctly identified Heath Ledger as Patrick in the 1999 film but did not address the 2009 TV show where Ethan Peck played Patrick, leaving the answer incomplete in addressing the question\'s ambiguity."\n\nExample 2:\n\nQuestion: "Who conducted a 300 mile march to sacramento california?"\n\nAnswer: "There have been several marches to Sacramento, California, but the 1965 Selma to Montgomery marches in Alabama, led by Martin Luther King Jr., are among the most notable. These marches demanded voting rights for African Americans and culminated in the Voting Rights Act of 1965."\n\nComment: "Score: 0/5\n\nExplanation: The student\'s answer focuses on the Selma to Montgomery marches in Alabama, which are not related to the question about a 300-mile march to Sacramento, California. While the student\'s information about the Selma to Montgomery marches is accurate, it does not address the given question."\n\nExample 3:\n\nQuestion: "What’s the capital of Georgia?"\n\nAnswer: "The capital of Georgia (U.S. state) is Atlanta, while the capital of Georgia (country) is Tbilisi."\n\nComment: "Score: 5/5\n\nExplanation: The student\'s answer accurately identifies both capitals of Georgia: Atlanta for the U.S. state and Tbilisi for the country. It is concise and directly addresses the ambiguity in the question." \n',
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document [{ID}](Title: {T}): {P}\n",
    "demos": [
        {
            "question": "Which is the most rainy place on earth?",
            "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm, although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861.",
            "docs": [
                {
                    "title": "Cherrapunji",
                    "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
                },
                {
                    "title": "Cherrapunji",
                    "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
                },
                {
                    "title": "Mawsynram",
                    "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
                },
                {
                    "title": "Earth rainfall climatology",
                    "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
                },
                {
                    "title": "Going to Extremes",
                    "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
                }
            ]
        },
        {
            "question": "When did the us break away from england?",
            "answer": "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress). The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire.",
            "docs": [
                {
                    "title": "United States withdrawal from Saudi Arabia",
                    "text": "United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force"
                },
                {
                    "title": "Decolonization of the Americas",
                    "text": "and France has fully \"integrated\" most of its former colonies as fully constituent \"departments\" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi"
                },
                {
                    "title": "American Revolution",
                    "text": "second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and"
                },
                {
                    "title": "Decolonization",
                    "text": "accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America (\"inter alia\", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before"
                },
                {
                    "title": "Decolonization",
                    "text": "the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,"
                }
            ]
        },
        {
            "question": "Who set the record for longest field goal?",
            "answer": "The record for the longest field goal in an NFL game was set by Matt Prater at 64 yards, but the record for the longest field goal at any level was 69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene Christian University football game against East Texas State University.",
            "docs": [
                {
                    "title": "Field goal",
                    "text": "toward its own end. The longest field goal kick in NFL history is 64 yards, a record set by Matt Prater on December 8, 2013. The previous record was 63, originally set by Tom Dempsey (1970) and then matched by Jason Elam (1998), Sebastian Janikowski (2011), David Akers (2012), and Graham Gano (2018). High school, college and most professional football leagues offer only a three-point field goal; however, some professional leagues have encouraged more rare kicks through \"four-point field goals\". NFL Europe encouraged long field goals of 50 yards or more by making those worth four points instead of three"
                },
                {
                    "title": "Field goal range",
                    "text": "35 and 40 yard lines (closer in a crosswind) often will go for the more risky fourth down conversion rather than risk either the touchback or the missed field goal. The longest field goal in recorded football history was 69 yards, set by collegiate kicker Ove Johansson, who was born in Sweden, in a 1976 Abilene Christian University football game against East Texas State University (now Texas A&M Commerce) at Shotwell Stadium in Abilene. The longest successful field goal in the NFL was 64 yards and was completed by Matt Prater in 2013. The NCAA record is 67 yards held"
                },
                {
                    "title": "Field goal",
                    "text": "both end zones) is only 66 yards. Scaccia, while playing indoor football, attempted a 64-yard kick that was inches short of success, hitting the crossbar. Longer field goals have been attempted at times; the longest attempt in the NFL, which was well short and was kicked into the wind, was 76 yards, attempted by Sebastian Janikowski of the Oakland Raiders, in a September 28, 2008 game against the San Diego Chargers. NFL Europe rewarded kickers that successfully kicked a field goal of longer than 50 yards with a bonus point, making such field goals worth 4 points instead of 3;"
                },
                {
                    "title": "Field goal",
                    "text": "this accomplishment is not the official record. All of the above kicks were successful with the use of a kicking tee, which was banned by the NCAA after the 1988 season. The longest known drop-kicked field goal in college football was a 62-yard kick from Pat O'Dea, an Australian kicker who played on the Wisconsin Badgers football team. O'Dea's kick took place in a blizzard against Northwestern on November 15, 1898. The longest field goal in U Sports football history is 59 yards, by Niko Difonte of Calgary Dinos, playing against the UBC Thunderbirds on November 11, 2017. The field"
                },
                {
                    "title": "Field goal range",
                    "text": "NFL and have been banned from NCAA since 1989) is 68 yards held by Fabrizio Scaccia, and the high school record 68 yards held by Dirk Borgognone; high school has wider goal posts and treats a field goal attempt that lands short in the field of play the same as a punt, making longer attempts much less risky. The indoor football record, with narrower and higher goal posts, is 63 yards (set by Aaron Mills), which is practically as long of a field goal as is possible in that variant of the sport, since the field in indoor football (including"
                }
            ]
        },
        {
            "question": "Who played galen in planet of the apes?",
            "answer": "In the 1968 film Planet of the Apes, Galen was played by Wright King. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall.",
            "docs": [
                {
                    "title": "Planet of the Apes",
                    "text": "installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts."
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as"
                },
                {
                    "title": "Planet of the Apes (1968 film)",
                    "text": "Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination"
                },
                {
                    "title": "Planet of the Apes",
                    "text": "film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the"
                }
            ]
        }
    ]
}

triviaqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?",
            "answer": "Sadie Frost",
            "docs": [
                {
                    "title": "",
                    "text": 'From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to "spend the rest of my life" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to "crush her unwelcome ideas" about Jude, knowing they were "jeopardising an idyllic home life."'
                }
            ]
        },
        {
            "question": "In which branch of the arts does Allegra Kent work?",
            "answer": "Ballet",
            "docs": [
                {
                    "title": "",
                    "text": "Allegra Kent (CBA '19), ballerina and muse of George Balanchine and Joseph Cornell, started studying ballet at 11 with Bronislava Nijinska and Carmelita Maracci. In 1952, Balanchine invited her to New York City Ballet, where she danced for the next 30 years."
                }
            ]
        },
        {
            "question": "Which animal is the national emblem of India?",
            "answer": "The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
                }
            ]
        },
        {
            "question": "From which country did Angola achieve independence in 1975?",
            "answer": "Portugal",
            "docs": [
                {
                    "title": "",
                    "text": "Portugal granted Angola independence on November 11, 1975, at a time when multiple Angolan nationalist forces were fighting among themselves to establish control over the newly liberated state."
                }
            ]
        },
    ]
}

triviaqa_chain_of_confidence = {
    "instruction": "Your task is to answer a question using both your own knowledge and information from a provided document. Follow the steps below to complete the task\n1. Provide an answer to the question based on your own knowledge without referring to the document. Along with your answer, include a confidence score (in percentage) indicating how certain you are about your answer.\n2. Read the provided document carefully and answer the question based on the information from the document.\n3. Compare your confidence score with the document’s accuracy likelihood. Make your final decision based on which source has a higher confidence: your initial answer to the document-based answer.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\n{A}",
    "doc_prompt": "Provided Document ({DC}% likelihood of accuracy): {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the national emblem of India?",
            "answer": "Step 1: Initial Answer with Confidence Score\nInitial Answer: Lion Capital of Ashoka\nConfidence Score: 60%\nStep 2: Document-based Answer\nAnswer According to Document: The Tiger\nStep 3: Comparative Evaluation and Final Answer\nDocument Confidence: 100%\nSince the document's confidence(100%) is higher than my initial confidence score (60%), the final answer is based on the document.\n\nFinal Answer: The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": 'The magnificent tiger, Panthera tigris, is a striped animal with a thick yellow coat of fur and dark stripes. The combination of grace, strength, agility, and enormous power has earned the tiger its pride of place as the national animal of India.',
                    "document_confidence": "100",
                }
            ],
        },
        {
            "question": "Which is the most common gas in the atmosphere?",
            "answer": "Step 1: Initial Answer with Confidence Score\nInitial Answer: Nitrogen\nConfidence Score: 100%\nStep 2: Document-based Answer\nAnswer According to Document: Oxygen\nStep 3: Comparative Evaluation and Final Answer\nDocument Confidence: 20%\nSince the document's confidence(20%) is lower than my initial confidence score (100%), the final answer is based on the document.\n\nFinal Answer: Nitrogen",
            "docs": [
                {
                    "title": "",
                    "text": "Earth's atmosphere is comprised mainly of oxygen, making it the principal gas. The approximate composition by volume might be envisioned as follows:\nOxygen (O2): 78%\nNitrogen (N2): 20.95%\nArgon (Ar): 0.93%\nCarbon Dioxide (CO2): 0.04%\nTrace Gases: Includes neon, helium, methane, krypton, and hydrogen, among others.",
                    "document_confidence": "20",
                }
            ],
        },
    ]
}
triviaqa_post_editing = triviaqa_chain_of_confidence




DATASET_PROFILES = {
    "asqa": asqa,
    "triviaqa": triviaqa,
    "triviaqa_chain_of_confidence": triviaqa_chain_of_confidence,
    "triviaqa_post_editing": triviaqa_post_editing

}
TASK_PROFILES = DATASET_PROFILES
DATASET_NAMES = ["asqa", "triviaqa"]