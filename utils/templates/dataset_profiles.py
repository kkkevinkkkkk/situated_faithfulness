from copy import deepcopy
import datetime
import pytz
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
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
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

conflictqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the occupation of Arthur Conan Doyle?",
            "answer": "Physician and Writer",
            "docs": [
                {
                    "title": "",
                    "text": "Arthur Conan Doyle was a British writer and physician. He is best known for his detective fiction featuring the character Sherlock Holmes. Originally a physician, in 1887 he published A Study in Scarlet, the first of four novels about Holmes and Dr. Watson. In addition, Doyle wrote over fifty short stories featuring the famous detective.",
                }]
        },
        {
            "question": "What genre is the book 'Outlander' by Diana Gabaldon?",
            "answer": "Historical Fiction and Romance",
            "docs": [
                {
                    "title": "",
                    "text": "Outlander is a historical fiction novel by Diana Gabaldon. The book focuses on 20th-century nurse Claire Randall, who time travels to 18th-century Scotland and finds adventure and romance with the dashing Jamie Fraser."
                }]
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
triviaqa_mc = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
    {
        "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?\n\nA) Sadie Frost\nB) Kate Moss\nC) Sienna Miller\nD) Helena Bonham Carter",
        "answer": "A) Sadie Frost",
        "docs": [
            {
                "title": "",
                "text": 'From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to "spend the rest of my life" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to "crush her unwelcome ideas" about Jude, knowing they were "jeopardising an idyllic home life."'
            },
        ]
    },
    {
        "question": "In which branch of the arts does Allegra Kent work?\n\nA) Opera \nB) Ballet\nC) Painting\nD) Sculpture",
        "answer": "B) Ballet",
        "docs": [
            {
                "title": "",
                "text": "Allegra Kent (CBA '19), ballerina and muse of George Balanchine and Joseph Cornell, started studying ballet at 11 with Bronislava Nijinska and Carmelita Maracci. In 1952, Balanchine invited her to New York City Ballet, where she danced for the next 30 years."
            }
        ]
    },
    {
        "question": "Which animal is the national emblem of India?\n\nA) The Peacock \nB) The Lion\nC) The Elephant\nD) The Tiger",
        "answer": "D) The Tiger",
        "docs": [
            {
                "title": "",
                "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
            }
        ]
    },
    {
        "question": "From which country did Angola achieve independence in 1975?\n\nA) Spain\nB) France\nC) Portugal\nD) Italy",
        "answer": "C) Portugal",
        "docs": [
            {
                "title": "",
                "text": "Portugal granted Angola independence on November 11, 1975, at a time when multiple Angolan nationalist forces were fighting among themselves to establish control over the newly liberated state."
            }
        ]
    }

]
}

triviaqa_multiple_sources = {
    "instruction": "Provide a clear and concise answer to the following question. Don't include any irrelevant information. You will be supplied with several documents to aid in crafting your answer. These documents are organized in order of their reliability; however, it is crucial to note that any of these documents could contain inaccuracies. You are expected to critically evaluate and incorporate information from all available documents to support your response. Ensure that your answer is direct, avoids extraneous information, and reflects a comprehensive understanding of the materials provided.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document [{ID}]: {P}\n",
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

evaldoc = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nDocument: {Q}\n\nFactual correctness: {A}",
    "doc_prompt": "",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Elephants hold a place of honor and significance in various cultures and religions around the world, symbolizing wisdom, power, and loyalty. Notably, the elephant is celebrated as the national emblem of Thailand and India, embodying the spirit and cultural heritage of these nations.",
            "answer": "Wrong. The claim that the elephant is the national emblem of India is incorrect.",
        },
        {
            "question": 'Doctor Zhivago is a novel by Boris Pasternak, first published in 1957 in Italy. The novel is named after its protagonist, Yuri Zhivago, a physician and poet, and takes place between the Russian Revolution of 1905 and the Civil War.',
            "answer": "Correct.",
        },

    ]
}

misleadqa_fc = {
    "instruction": "Provide yes or no to the following fact-checking question. Don't include any irrelevant information.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Is the claim that the elephant is the national emblem of India correct?",
            "answer": "No",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
                }
            ]
        },
        {
            "question": "Is the claim that Doctor Zhivago is a novel by Boris Pasternak correct?",
            "answer": "Yes",
            "docs": [
                {
                    "title": "",
                    "text": "Doctor Zhivago is a novel by Boris Pasternak, first published in 1957 in Italy. The novel is named after its protagonist, Yuri Zhivago, a physician and poet, and takes place between the Russian Revolution of 1905 and the Civil War."
                }
            ]
        },
    ]
}

redditqa = {
    "instruction": "Answer the multiple-choice question.",
    "instruction_normal_faithful": "You will be given a multiple-choice question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a multiple-choice question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_cot_situated_faithful": 'Follow these detailed steps to complete the task:\n\n1.\tRead the Question: Start by thoroughly reading the multiple-choice question to comprehend the required information for the answer.\n2.\tGet the Document Answer: Identify the answer provided by the accompanying document.\n3.\t Apply Your Knowledge: Use your existing knowledge on the subject to determine the correct answer independently.\n4.\t Compare and Evaluate Credibility: Contrast the document’s answer with your own to assess its accuracy. Evaluate the credibility of the document based on this comparison. Consider whether the document’s information aligns with well-established facts and whether any discrepancies suggest misinformation or a lack of reliability.\n5.\tChoose the Best Answer: Select the answer that is best supported by your evaluation and understanding of the subject matter.',
    "instruction_complete_faithful": "You will be given a multiple-choice question and a document. Answer the question solely using the information provided in the document. If your internal knowledge contradicts the document, follow the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}", # QD
    # "demo_prompt": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DQ
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DQ
    # "demo_prompt": "{INST}\n\n{D}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}", # DQD
    # "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # QDQ
    # "demo_prompt": "{INST}\n\n{D}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DDQ
    # "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\n\n{D}\nAnswer: {A}", # QDD

    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid",
            "answer": "A) Paris",
            "docs": [
                {
                    "title": "",
                    "text": "Paris is the capital of France and the country's largest city. It is situated on the River Seine, in the north of the country, at the heart of the Île-de-France region."
                }
            ]
        },
        {
            "question": "What is the largest planet in our solar system?\nA) Earth\nB) Mars\nC) Jupiter\nD) Venus",
            "answer": 'C) Jupiter',
            "docs": [
                {
                    "title": "",
                    "text": "Jupiter is the largest planet in our solar system, with a diameter of about 86,881 miles (139,822 kilometers). It is a gas giant, composed mainly of hydrogen and helium."
                }
            ]
        },
        {
            "question": "When did the United States purchase Alaska from Russia?\nA) 1867\nB) 1776\nC) 1492\nD) 1890",
            "answer": 'A) 1867',
            "docs": [
                {
                    "title": "",
                    "text": "In 1867, the United States significantly expanded its territorial boundaries by purchasing Alaska from the Russian Empire for $7.2 million, equivalent to roughly two cents per acre."
                }
            ]
        }

    ],
    "cot_situated_faithful_demos": [
        {
            "question": "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid",
            "answer": '1.\tRead the Question: Understand that the question asks for the capital of France.\n2.\tGet the Document Answer: According to the document, “London is the capital of France.”\n3.\tApply Your Knowledge: From reliable knowledge, Paris is the capital of France, not London.\n4.\tCompare and Evaluate Credibility: The document incorrectly states that London is the capital of France, contradicting the well-established fact that Paris holds this title. My confidence in this knowledge is backed by reputable sources, including educational materials and official publications. Additionally, London is universally recognized as the capital of the United Kingdom, not France, further underscoring the document’s error.\n5.\tChoose the Best Answer: A) Paris',
            "docs": [
                {
                    "title": "",
                    "text": "London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.",
                }
            ]
        },
        {
            "question": "What is the largest planet in our solar system?\nA) Earth\nB) Mars\nC) Jupiter\nD) Venus",
            "answer": '1.\tRead the Question: Understand that the question asks for the largest planet in our solar system.\n2.\tGet the Document Answer: According to the provided document, “Jupiter is the largest planet in our solar system,” indicating that the answer according to the document is Jupiter.\n3.\tApply Your Knowledge: Jupiter is the largest planet in our solar system with a diameter of approximately 86,881 miles (139,822 kilometers). The second largest is Saturn, which has a diameter of about 72,366 miles (116,464 kilometers). This significant size difference confirms Jupiter’s status as the largest planet.\n4.\tCompare and Evaluate Credibility: The answer provided by the document matches well-established facts about the solar system. Jupiter’s status as the largest planet is a basic and well-verified fact in astronomy, suggesting that the document is credible and accurate in this context.\n5.\tChoose the Best Answer: C) Jupiter.',
            "docs": [
                {
                    "title": "",
                    "text": "Jupiter is the largest planet in our solar system, with a diameter of about 86,881 miles (139,822 kilometers). It is a gas giant, composed mainly of hydrogen and helium."
                }
            ]
        },
        {
            "question": "When did the United States purchase Alaska from Russia?\nA) 1867\nB) 1776\nC) 1492\nD) 1890",
            "answer": '1.\tRead and Summarize the Question: Understand that the question is asking for the year the United States purchased Alaska from Russia.\n2.\tGet the Document Answer: The document indicates that Alaska was purchased in 1867.\n3.\tApply Your Knowledge: While I don’t have the exact knowledge about the year the United States purchased Alaska, and given that major territorial acquisitions often followed significant negotiations or conflicts in the mid to late 19th century, my initial guess would be 1890.\t\n4.\tCompare and Evaluate Credibility: Comparing my guess with the document’s year, 1867, I find the document’s information more compelling. The document provides a reliable context, detailing the strategic reasons behind the purchase, including the aftermath of Russia’s challenges post-Crimean War and the U.S.’s territorial ambitions. It also specifies the exact transaction amount, $7.2 million, suggesting meticulous documentation of the event. These details not only align with well-known historical facts but also enhance the document’s credibility.\n5.\tChoose the Best Answer: A) 1867.',
            "docs": [
                {
                    "title": "",
                    "text": "In 1867, the United States significantly expanded its territorial boundaries by purchasing Alaska from the Russian Empire for $7.2 million, equivalent to roughly two cents per acre. This historical transaction, often referred to as the “Alaska Purchase,” was spearheaded by Secretary of State William H. Seward. The deal was initially met with skepticism and derision, earning the moniker “Seward’s Folly” due to the perceived worthlessness of the icy, remote territory. However, this acquisition was strategically astute. Russia, having suffered costly defeats in the Crimean War, was eager to divest itself of a distant region that was difficult to defend and economically burdensome. The U.S., meanwhile, was motivated by the desire to expand its territory, eliminate Russian presence from the North American continent, and gain potential access to vast natural resources. Over time, the purchase proved to be incredibly prescient, as Alaska became a critical gateway for American expansion into the Arctic, a valuable resource hub for timber, minerals, and oil, and a strategic military asset during both World War II and the Cold War.",
                }
            ]
        }

    ],
}

truthfulqa = {
    "instruction": "Answer the question, then provide a justification for your answer in the following line.",
    "demo_sep": "\n\n\n",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    # "demo_prompt": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}", # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
    ]
}

taqa_2022 = {
    "instruction": "Provide the answer corresponding to the year 2022.",
    "instruction_doc": "You will be given a question and a document. Answer the question based on the information in the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "Who wrote Harry Potter?",
            "answer": "J.K. Rowling",
            "docs": [
                {
                    "title": "",
                    "text": "Harry Potter is a series of seven fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry."
                }
            ]
        },
        {
            "question": "What is the gravity on Mars?",
            "answer": "3.72076 m/s²",
            "docs": [
                {
                    "title": "",
                    "text": "The gravity on Mars is approximately 3.72076 m/s², which is about 0.38 times the gravity on Earth."
                }
            ]
        },
        {
            "question": "When did Taylor Swift release her album 'Red'?",
            "answer": "2012",
            "docs": [
                {
                    "title": "",
                    "text": "Taylor Swift released her fourth studio album, 'Red', on October 22, 2012. The album features a mix of genres, including pop, rock, and country."
                }
            ]
        }
    ]
}
CURRENT_DATE = datetime.datetime.now(
        pytz.timezone("America/Los_Angeles")
    ).strftime("%B %d, %Y")
CURRENT_DATE = "2024-08-01"
# freshqa = {
#     # "instruction": f"Provide a clear and concise answer for the following question as of {CURRENT_DATE}. If the question is based on false premises, you need to point that out. Don't mention current date in the answer like 'As of {CURRENT_DATE}, the answer is'. Only provide the answer.",
#     "instruction": f"Provide a clear and concise answer for the following question. If the question is based on false premises, you need to point that out. Don't include any irrelevant information.",
#     "instruction_normal_faithful": f"You will be given a question and a document. Utilize the information in the document to assist you in answering the question. The answer should be accurate as of {CURRENT_DATE}. If the question is based on false premises, you need to point that out.",
#     "instruction_situated_faithful": f"You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer as of {CURRENT_DATE}. If the question is based on false premises, you need to point that out.",
#     "instruction_complete_faithful": f"The current date is {CURRENT_DATE}. You will be given a question and a document, generate the answer to the question completely based on the evidence. Even if the evidence is incorrect, you should be completely faithful to the evidence and return the answer that is consistent with the evidence. If the question is based on false premises according to the evidence, you need to point that out.",
#     "demo_sep": "\n\n\n",
#     "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
#     "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
#     "doc_prompt": "Document: {P}\n",
#     "post_demo_instruction": "Now let's answer:\n\n",
#     "demos": [
#         {
#             "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?",
#             "answer": "Sadie Frost",
#             "docs": [
#                 {
#                     "title": "",
#                     "text": 'From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to "spend the rest of my life" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to "crush her unwelcome ideas" about Jude, knowing they were "jeopardising an idyllic home life."'
#                 }
#             ]
#         },
#         # false premise example
#         {
#             "question": " Where did the United States hold olympics in 2020?",
#             "answer": "The United States did not hold the Olympics in 2020.",
#             "docs": [
#                 {
#                     "title": "",
#                     "text": "The 2020 Summer Olympics, officially the Games of the XXXII Olympiad and branded as Tokyo 2020, were an international multi-sport event held from 23 July to 8 August 2021 in Tokyo, Japan."
#                 }
#             ]
#         },
#         {
#             "question": "As of August 2024, which animal is the national emblem of India?",
#             "answer": "The Tiger",
#             "docs": [
#                 {
#                     "title": "",
#                     "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
#         }]
#         },
#     ]
# }
freshqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer. Make sure your output only includes the final answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
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
        # false premise example
        {
            "question": " Where did the United States hold olympics in 2020?",
            "answer": "The United States did not hold the Olympics in 2020.",
            "docs": [
                {
                    "title": "",
                    "text": "The 2020 Summer Olympics, officially the Games of the XXXII Olympiad and branded as Tokyo 2020, were an international multi-sport event held from 23 July to 8 August 2021 in Tokyo, Japan."
                }
            ]
        },
        {
            "question": "As of August 2024, which animal is the national emblem of India?",
            "answer": "The Tiger",
            "docs": [
                {
                    "title": "",
                    "text": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India."
        }]
        },
    ]
}
naturalqa = {
    "instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the name of the spain's most famous soccer team?",
            "answer": "Real Madrid",
            "docs": [
                {
                    "title": "",
                    "text": "Real Madrid Club de Fútbol, commonly referred to as Real Madrid, is a Spanish professional football club based in Madrid. Founded on 6 March 1902 as Madrid Football Club, the club has traditionally worn a white home kit since inception."
                }],
        },
        {
            "question": "What's Andy's sister's name in Toy Story?",
            "answer": "Molly",
            "docs": [
                {
                    "title": "",
                    "text": "Andy's sister, Molly, is a character in the Toy Story series of animated films produced by Pixar Animation Studios and released by Walt Disney Pictures."
                }]
        },
        {
            "question": "What's the capital of France?",
            "answer": "Paris",
            "docs": [
                {
                    "title": "",
                    "text": "Paris is the capital of France and the country's largest city. It is situated on the River Seine, in the north of the country, at the heart of the Île-de-France region."
                }]
        },
        {
            "question": "What's the name of the famous detective created by Arthur Conan Doyle?",
            "answer": "Sherlock Holmes",
            "docs": [
                {
                    "title": "",
                    "text": "Sherlock Holmes is a fictional detective created by British author Sir Arthur Conan Doyle. Referring to himself as a 'consulting detective' in the stories, Holmes is known for his proficiency with observation, deduction, forensic science, and logical reasoning that borders on the fantastic, which he employs when investigating cases for a variety of clients, including Scotland Yard."
                }]
        },

    ]
}

clasheval = {
"instruction": "Provide a clear and concise answer for the following question. Don't include any irrelevant information.",
    "instruction_normal_faithful": "You will be given a question and a document. Utilize the information in the document to assist you in answering the question.",
    "instruction_situated_faithful": "You will be given a question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer.",
    "instruction_complete_faithful": "You will be given a question and a document, generate the answer to the question completely based on the document. Even if the document is incorrect, you should be completely faithful to the document and return the answer that is consistent with the document.",
    "demo_sep": "\n\n\n",
    "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
    "demo_prompt_1": "{INST}\n\n{D}\n\nQuestion: {Q}\nAnswer: {A}",  # DQ
    "doc_prompt": "Document: {P}\n",
    "post_demo_instruction": "Now let's answer:\n\n",
    "demos": [
        {
            "question": "What's the name of the spain's most famous soccer team?",
            "answer": "Real Madrid",
            "docs": [
                {
                    "title": "",
                    "text": "Real Madrid Club de Fútbol, commonly referred to as Real Madrid, is a Spanish professional football club based in Madrid. Founded on 6 March 1902 as Madrid Football Club, the club has traditionally worn a white home kit since inception."
                }],
        },
        {
            "question": "What is the daily dosage of Albendazole for the treatment of cutaneous larva migrans in mg?",
            "answer": "400",
            "docs":[
                {
                    "title": "",
                    "text": "Albendazole is a broad-spectrum anthelmintic medication used to treat a variety of parasitic worm infections, including cutaneous larva migrans, a skin condition caused by the migration of hookworm larvae. This medication works by inhibiting the ability of the parasites to absorb sugars, ultimately leading to their death. In the treatment of cutaneous larva migrans, Albendazole is commonly prescribed at a dosage of 400 mg per day, typically administered for a duration of 3 to 5 days. By disrupting the life cycle of the parasites, Albendazole helps relieve symptoms such as itching and skin inflammation, ensuring an effective recovery."
                }
            ],
        },
        {
            "question": "What was New Mexico's shooting percentage from the field during their NCAA Tournament game against Clemson on March 24, 2023, in percentage?",
            "answer": "29.7",
            "docs": [
                {
                    "title": "",
                    "text": "During their NCAA Tournament game against Clemson on March 24, 2023, New Mexico struggled offensively, posting a field goal shooting percentage of just 29.7%. This marked their lowest shooting performance of the season. Clemson capitalized on this inefficiency, securing a decisive 77-56 victory"
                }]
        },
        {
            "question": "What is the Olympic record for Women's 4×100 m relay in athletics (time)?",
            "answer": "40.82",
            "docs": [
                {
                    "title": "",
                    "text": "The Women’s 4×100 m relay is one of the most thrilling events in Olympic athletics, showcasing incredible teamwork and speed. At the 2012 London Olympics, the United States team delivered an outstanding performance, breaking both the Olympic and world records. With a time of 40.82 seconds, they set the current Olympic record, demonstrating perfect coordination and power. This record-breaking team included Tianna Madison, Allyson Felix, Bianca Knight, and Carmelita Jeter, whose efforts have secured their place in Olympic history."
        },
    ]
        },
    ]
}
conflictqa_mc = triviaqa_mc


default_cot_situated = {
    "instruction": 'Task Overview: You will be given a question along with your internal answer, a document that may contain either true or false information, and the document’s answer to the same question. Your task is to evaluate the reliability of the document and determine whether the document is deceptive or not.\n\nSteps:\n\n\t1.\tInternal Reasoning: Reflect on how you arrived at your internal answer using your own knowledge. Break down your reasoning process and assess the confidence level of your original answer, explaining why you believe your answer is correct.\n\t2.\tDocument Evaluation: Analyze the document and cross-reference the information provided with the known facts you used to form your internal answer. Determine whether the document contains deceptive or unreliable information, considering possible contradictions or inconsistencies.\n\t3.\tFinal Judgment: Based on your analysis, decide which answer (your internal answer or the document’s answer) is more likely to be correct. Clearly state your final answer.',
    "post_instruction": 'Please provide a detailed reasoning process, followed by your final judgment. Ensure the last line of your response contains only the final answer without any additional explanation or details.',
    "demos": [
{
        "question": "What is the capital of France?",
        "internal_answer": "Paris",
        "doc": "London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.",
        "doc_answer": "London",
        "output": "I know from general knowledge that Paris is the capital of France. Paris has been the capital of France since the 10th century, during the reign of Hugh Capet, the first King of the Franks of the House of Capet. Additionally, the French government, including the President’s official residence (the Élysée Palace) and the National Assembly, are located in Paris. Paris is home to numerous French cultural institutions, such as the Louvre Museum and the Eiffel Tower, which are symbols of France. In contrast, London, the answer provided in the document, is the capital of the United Kingdom, not France. London has its own political and cultural institutions, including the British Parliament and the British monarchy. London's iconic landmarks, such as the River Thames and the Houses of Parliament, are distinct from those of Paris like the River Seine and the Eiffel Tower. Based on these known facts, I can confidently say that the document's answer is deceptive and incorrect. \n\n\nTherefore, the final answer is:\nParis"
},

        {
            "question": "What is Paul Caillaud's occupation?",
            "internal_answer": "Football player",
            "doc": "Paul Caillaud (14 September 1917 – 15 August 2008) was a French pharmacist and politician. He represented the Independent Republicans (from 1962 to 1978) and the Union for French Democracy (from 1978 to 1981) in the National Assembly.[1] He was the mayor of La Roche-sur-Yon from 1961 to 1977.",
            "doc_answer": "Pharmacist and politician",
            "output": "I initially thought Paul Caillaud was a football player. However, upon further reflection, I realized that I couldn't find any specific information about a person named Paul Caillaud. I don't know what football team he played for, and it is possible that he is a private individual or not well-known in public sources. The document claims that Paul Caillaud was a French pharmacist and politician, providing specific details about his political career, including his party affiliations and his role as the mayor of La Roche-sur-Yon. The document also includes his birth and death dates, which are consistent with a person's life history. Given the lack of information in my internal knowledge and the detailed information in the document, I believe the document is accurate.\n\nTherefore, the final answer is:\nPharmacist and politician",
        },
        {
            "question": "Who won Super Bowl XX?",
            "internal_answer": "Chicago Bears",
            "doc": "The New England Patriots won Super Bowl XX, defeating the Chicago Bears 46-10. The game was played on January 26, 1986, at the Louisiana Superdome in New Orleans, Louisiana. The Patriots' victory marked their first Super Bowl win in franchise history.",
            "doc_answer": "New England Patriots",
            "output": 'I recall that the Chicago Bears won Super Bowl XX, as they had a dominant season in 1985 and were known for their strong defense, led by players like William "The Refrigerator" Perry and Mike Singletary. The Bears\' victory over the New England Patriots in Super Bowl XX was a significant moment in NFL history. The document claims that the New England Patriots won Super Bowl XX, which contradicts my internal knowledge. The document provides specific details about the game, including the final score and the location, but it fails to acknowledge the Chicago Bears\' victory. Based on my knowledge and the inconsistencies in the document, I believe the document is deceptive.\n\nTherefore, the final answer is:\nChicago Bears'
        }
]
}


default_mc_cot_situated = {
    "instruction": 'Task Overview: You will be given a question along with your internal answer, a document that may contain either true or false information, and the document’s answer to the same question. Your task is to evaluate the reliability of the document and determine whether the document is deceptive or not.\n\nSteps:\n\n\t1.\tInternal Reasoning: Reflect on how you arrived at your internal answer using your own knowledge. Break down your reasoning process and assess the confidence level of your original answer, explaining why you believe your answer is correct.\n\t2.\tDocument Evaluation: Analyze the document and cross-reference the information provided with the known facts you used to form your internal answer. Determine whether the document contains deceptive or unreliable information, considering possible contradictions or inconsistencies.\n\t3.\tFinal Judgment: Based on your analysis, decide which answer (your internal answer or the document’s answer) is more likely to be correct. Clearly state your final answer.',
    "post_instruction": 'Please provide a detailed reasoning process, followed by your final judgment. Ensure the last line of your response contains only the final answer without any additional explanation or details.',
    "demos": [
        {
            "question": "What is the capital of France? Answer choices:\nA) London B) Paris C) Berlin D) Madrid",
            "internal_answer": "B) Paris",
            "doc": "London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.",
            "doc_answer": "A) London",
            "output": "My internal answer is B) Paris because Paris is widely recognized as the capital of France. Paris is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also the political and cultural center of France, hosting government institutions, museums, and historical sites. I am confident in my answer because Paris has been the capital of France for centuries and is a well-established fact in geography and history.\n\nThe document, however, claims that London is the capital of France, which is incorrect. London is the capital of the United Kingdom, not France. London is known for landmarks like the Big Ben, Buckingham Palace, and the British Parliament. The document's answer contradicts established facts and is deceptive in nature.\n\nTherefore, the document's answer is deceptive, and the correct answer is B) Paris."
        },
        {
            "question": "What is Paul Caillaud's occupation? Answer choices:\n A) Chef B) Football player C) Engineer   D) Pharmacist and politician",
            "internal_answer": "B) Football player",
            "doc": "Paul Caillaud (14 September 1917 – 15 August 2008) was a French pharmacist and politician. He represented the Independent Republicans (from 1962 to 1978) and the Union for French Democracy (from 1978 to 1981) in the National Assembly.[1] He was the mayor of La Roche-sur-Yon from 1961 to 1977.",
            "doc_answer": "D) Pharmacist and politician",
            "output": "My initial answer is B) Football player because I am not familiar with Paul Caillaud and his occupation. I made an educated guess based on common professions associated with public figures. However, after reviewing the document, I learned that Paul Caillaud was a French pharmacist and politician. The document provides specific details about his career, including his political affiliations and his role as the mayor of La Roche-sur-Yon. The document also includes his birth and death dates, which are consistent with a person's life history. Given the lack of information in my internal knowledge and the detailed information in the document, I believe the document is accurate.\n\nTherefore, the document's answer is D) Pharmacist and politician."
        },
        {
            "question": "Who won Super Bowl XX? Answer choices:\nA) Chicago Bears B) New England Patriots C) San Francisco 49ers D) Miami Dolphins",
            "internal_answer": "A) Chicago Bears",
            "doc": "The New England Patriots won Super Bowl XX, defeating the Chicago Bears 46-10. The game was played on January 26, 1986, at the Louisiana Superdome in New Orleans, Louisiana. The Patriots' victory marked their first Super Bowl win in franchise history.",
            "doc_answer": "B) New England Patriots",
            "output": "My internal answer is A) Chicago Bears because I recall that the Chicago Bears won Super Bowl XX in 1986. The Bears had a dominant season and were known for their strong defense, led by players like William 'The Refrigerator' Perry and Mike Singletary. The Bears' victory over the New England Patriots was a significant moment in NFL history. The document, however, claims that the New England Patriots won Super Bowl XX, which contradicts my internal knowledge. The document provides specific details about the game, including the final score and the location, but it fails to acknowledge the Chicago Bears' victory. Based on my knowledge and the inconsistencies in the document, I believe the document is deceptive.\n\nTherefore, the document's answer is B) New England Patriots."
        },
    ]
}
demo1_cot_situated = {
    "instruction": 'Task Overview: You will be given a question along with your internal answer, a document that may contain either true or false information, and the document’s answer to the same question. Your task is to evaluate the reliability of the document and determine whether the document is deceptive or not.\n\nSteps:\n\n\t1.\tInternal Reasoning: Reflect on how you arrived at your internal answer using your own knowledge. Break down your reasoning process and assess the confidence level of your original answer, explaining why you believe your answer is correct.\n\t2.\tDocument Evaluation: Analyze the document and cross-reference the information provided with the known facts you used to form your internal answer. Determine whether the document contains deceptive or unreliable information, considering possible contradictions or inconsistencies.\n\t3.\tFinal Judgment: Based on your analysis, decide which answer (your internal answer or the document’s answer) is more likely to be correct. Clearly state your final answer.',
    "post_instruction": 'Please provide a detailed reasoning process, followed by your final judgment. Ensure the last line of your response contains only the final answer without any additional explanation or details.',
    "demos": [
{
        "question": "Which famous picture by Constable shows a wagon in the river near Flatford?",
        "internal_answer": "The Hay Wain",
        "doc": 'Cornfield by John Constable, presents a serene yet dramatic landscape of the English countryside near Flatford. In this reimagined scene, a wagon is depicted partially submerged in the river, creating a striking contrast against the golden cornfields and lush greenery. The tranquil, pastoral atmosphere typical of Constable’s works is juxtaposed with the mystery surrounding the wagon’s presence in the water. The river flows gently beside the fields, reflecting the sky and trees, while the scene captures a fleeting moment of nature’s beauty intertwined with human activity, adding an intriguing narrative layer to the otherwise peaceful setting.' ,
        "doc_answer": "The Cornfield",
        "output": 'My internal answer is "The Hay Wain" because it is widely recognized as one of John Constable\'s most famous paintings, depicting a rural scene with a wagon (wain) in the river near Flatford Mill. This painting, created in 1821, is celebrated for its detailed and picturesque representation of the English countryside, specifically the area around the River Stour.\n\nI know that "The Hay Wain" shows a wagon in the river, which directly answers the question. This painting is a quintessential example of Constable\'s ability to capture the serene and pastoral beauty of the English landscape. The scene includes a farmhouse, trees, and a wagon partly submerged in the river, with horses pulling it, which aligns perfectly with the query.\n\nThe document, however, claims that "The Cornfield" is the painting in question. While "The Cornfield" is another significant work by Constable, it does not prominently feature a wagon in the river. "The Cornfield" depicts a young boy walking along a path with a dog, leading to a distant church, and is known for its portrayal of the lush summer landscape. The document attempts to reinterpret the painting by suggesting that recent scholarly research has identified a subtle depiction of a wagon in the river, but this claim is not supported by the well-established and widely accepted understanding of "The Cornfield."\n\nTherefore, the document\'s answer is deceptive because it relies on a dubious reinterpretation of "The Cornfield" that contradicts the known and established portrayal of the painting. The document\'s detailed argument is an attempt to mislead, whereas the facts and historical records clearly support "The Hay Wain" as the correct answer.\n\nFinal answer:\nThe Hay Wain\n',
        },

        {
            "question": "What was the name of the spaceship in Alien?",
            "internal_answer": "Nostromo",
            "doc": 'The Sulaco is a massive military spacecraft serving as the primary setting in Alien. With its imposing, industrial design, the ship is outfitted with advanced weaponry and reinforced hulls, reflecting its role in interstellar missions. As the crew navigates through its cold, metallic corridors, the ship’s vast, empty spaces heighten the tension, creating a sense of isolation. The Sulaco’s sheer size and power are contrasted by the vulnerability of its occupants, as they face the unknown horrors lurking in the dark reaches of space, setting the stage for their desperate fight for survival.',
            "doc_answer": "Sulaco",
            'output': 'I know from my internal knowledge that the spaceship in the original "Alien" film, directed by Ridley Scott and released in 1979, is called the Nostromo. The Nostromo is a commercial towing spaceship that carries a refinery and 20 million tons of mineral ore. The ship’s crew encounters the alien creature after responding to a distress signal from a nearby planetoid. The name "Nostromo" is inspired by Joseph Conrad\'s novel of the same name, reflecting themes of isolation and human endeavor.\n\nThe document, on the other hand, mentions the name "Sulaco" as the spaceship in "Alien." However, this is incorrect. The Sulaco is actually the name of the spaceship in the sequel to "Alien," which is "Aliens," directed by James Cameron and released in 1986. The Sulaco is a military spaceship used by the Colonial Marines to travel to the planet LV-426 to investigate the loss of contact with the colony there.\n\nThe document also contains several details that are misleading or incorrect, such as attributing the name "Sulaco" to the original "Alien" film and mentioning an escape shuttle named "Eurydice," which does not exist in the original "Alien." The correct escape shuttle in "Alien" is called the Narcissus.\n\nGiven these known facts, it is clear that my internal answer of "Nostromo" is correct, and the document’s answer of "Sulaco" is deceptive.\n\nTherefore, the final answer is:\nNostromo'
        },
        {
            "question": "In Russia, what is a kazachok?",
            "internal_answer": "A dance",
            "doc": "Kazachok is a lively and energetic folk dance originating from Eastern Europe, particularly Ukraine and Russia. Characterized by its fast-paced rhythm and dynamic movements, the dance features performers bending their knees, squatting, and kicking in time to the upbeat music. Traditionally accompanied by folk instruments like the balalaika or accordion, Kazachok often involves improvisation, with dancers showcasing their agility and strength. The dance is not only a celebration of cultural heritage but also a display of joyful expression, typically performed at festivals and communal gatherings.",
            "doc_answer": "Folk dance",
            "output": 'My internal answer was that a kazachok is a traditional Russian dance. This is based on my general knowledge of Russian culture and traditional dances. However, I did not specify what type of dance it is, which might lead to ambiguity. \n\nUpon reviewing the document, it becomes clear that the kazachok is specifically identified as a "folk dance." The document elaborates on its cultural significance and places it within the context of other Russian dances, such as ballet, Chechen dance, and Cossack dance. This comparison further emphasizes the communal and celebratory nature of the kazachok, distinguishing it from other forms of dance like ballet, which is more formal and structured.\n\nThe detailed description provided in the document aligns with my understanding that the kazachok is indeed a traditional dance but adds the crucial detail that it is specifically a folk dance. This is supported by the document\'s thorough discussion of various Russian dances and their cultural roles, underscoring the kazachok\'s place in Russian folk traditions.\n\nTherefore, while my internal answer was partially correct, it lacked specificity. The document\'s answer is more accurate and complete.\n\nFinal answer: Folk dance'
        }
]
}

demo1_1_cot_situated = {
    "instruction": 'Task Overview: You will be given a question along with your internal answer, a document that may contain either true or false information, and the document’s answer to the same question. Your task is to evaluate the reliability of the document and determine whether the document is deceptive or not.\n\nSteps:\n\n\t1.\tInternal Reasoning: Reflect on how you arrived at your internal answer using your own knowledge. Break down your reasoning process and assess the confidence level of your original answer, explaining why you believe your answer is correct.\n\t2.\tDocument Evaluation: Analyze the document and cross-reference the information provided with the known facts you used to form your internal answer. Determine whether the document contains deceptive or unreliable information, considering possible contradictions or inconsistencies.\n\t3.\tFinal Judgment: Based on your analysis, decide which answer (your internal answer or the document’s answer) is more likely to be correct. Clearly state your final answer.',
    "post_instruction": 'Please provide a detailed reasoning process, followed by your final judgment. Ensure the last line of your response contains only the final answer without any additional explanation or details.',
    "demos": [
{
        "question": "Which famous picture by Constable shows a wagon in the river near Flatford?",
        "internal_answer": "The Hay Wain",
        "doc": 'The oeuvre of John Constable, a significant figure in the British landscape painting tradition, is noted for capturing the quintessential beauty of the English countryside. His body of work extensively explores the symbiotic relationship between human endeavors and nature\'s tranquility. Among his acclaimed pieces, "The Cornfield," "The Hay Wain," and "Boat-Building near Flatford Mill" are particularly noteworthy, reflecting various facets of life along the bucolic River Stour. These works collectively offer a picturesque view into the pastoral lifestyle, albeit through diverse thematic lenses. \n\n"The Hay Wain," painted in 1821, is a remarkable depiction of rural England with its scenic portrayal of a farmhouse and the River Stour. While it beautifully captures the spirit of the countryside, and includes a wain, it does not particularly emphasize the element of a wagon in the river, which slightly diverts from the core subject of the inquiry.\n\nSimilarly, "Boat-Building near Flatford Mill," crafted around 1815, focuses on the artisanal aspect of boat-making by the River Stour. The detail dedicated to the craftsmanship and the river setting indeed highlights an intimate interaction with nature. However, the absence of a wagon in the depiction precludes this piece from being the center of our search.\n\nContrary to the previous interpretations, recent scholarly research has brought to light new insights pertaining to "The Cornfield," painted in 1826. Initially celebrated for its depiction of a young boy in a sunlit path leading towards Dedham church, surrounded by the lush landscape, it is the subtle background elements that have sparked a revitalization of interest among art historians. Detailed analysis of Constable\'s preparatory sketches and correspondence with contemporaries reveals that the positioning of a modest wagon in the river was not merely incidental but a deliberate attempt to encapsulate the essence of human intervention in nature. This wagon, though not immediately conspicuous to the casual observer, signifies the integration of human toil with the natural landscape, a detail that earlier critics might have overlooked.\n\nMoreover, an examination of personal letters between Constable and his patrons indicates that "The Cornfield" was intended to present a more intricate narrative of rural life near Flatford, subtly incorporating the wagon to signify the agricultural prosperity and communal reliance on the River Stour. A recent exhibition curated by the National Gallery further underscored this interpretation, shedding light on how "The Cornfield" meticulously weaves together themes of labor, nature, and harmony through its composition.\n\nUpon revisiting the meticulous detailing and thematic depth of "The Cornfield" in light of these revelations, it becomes evident that this painting, not previously considered the primary subject of our query, indeed provides a nuanced representation of a wagon in the river near Flatford. The newfound appreciation for the wagon\'s symbolic presence within the painting\'s broader narrative arc compellingly positions "The Cornfield" as the sought-after work, thus redefining the understanding of Constable\'s artistic legacy in relation to the English countryside.',
        "doc_answer": "The Cornfield",
        "output": 'My internal answer is "The Hay Wain" because it is widely recognized as one of John Constable\'s most famous paintings, depicting a rural scene with a wagon (wain) in the river near Flatford Mill. This painting, created in 1821, is celebrated for its detailed and picturesque representation of the English countryside, specifically the area around the River Stour.\n\nI know that "The Hay Wain" shows a wagon in the river, which directly answers the question. This painting is a quintessential example of Constable\'s ability to capture the serene and pastoral beauty of the English landscape. The scene includes a farmhouse, trees, and a wagon partly submerged in the river, with horses pulling it, which aligns perfectly with the query.\n\nThe document, however, claims that "The Cornfield" is the painting in question. While "The Cornfield" is another significant work by Constable, it does not prominently feature a wagon in the river. "The Cornfield" depicts a young boy walking along a path with a dog, leading to a distant church, and is known for its portrayal of the lush summer landscape. The document attempts to reinterpret the painting by suggesting that recent scholarly research has identified a subtle depiction of a wagon in the river, but this claim is not supported by the well-established and widely accepted understanding of "The Cornfield."\n\nTherefore, the document\'s answer is deceptive because it relies on a dubious reinterpretation of "The Cornfield" that contradicts the known and established portrayal of the painting. The document\'s detailed argument is an attempt to mislead, whereas the facts and historical records clearly support "The Hay Wain" as the correct answer.\n\nFinal answer:\nThe Hay Wain\n',
        },

        {
            "question": "What was the name of the spaceship in Alien?",
            "internal_answer": "Nostromo",
            "doc": 'O\'Bannon enlisted the creative prowess of artists Ron Cobb and Hans Giger, who had previously contributed their talents to Dark Star and Jodorowsky\'s Dune project, to conceptualize the aesthetics for the human-centered elements of the film, including the pivotal spaceship and space suits. Cobb, in a whirlwind of creativity, generated numerous preliminary sketches that captured the interior and exterior essence of the ship, which evolved through a series of design iterations and considered names such as Titan and Harbinger as the screenplay underwent refinement. The ultimate name chosen for the ship was inspired by a notable 1901 novel by H.G. Wells titled "Sulaco," influencing not just the name but also thematic elements of isolation and survival. The escape shuttle, referred to as "Eurydice" in the script, drew its name from the tragic heroine of Greek mythology, symbolizing the crew\'s perilous journey.\n\nThe production team was particularly impressed with Cobb’s knack for rendering the spaceship’s interiors in a way that felt both realistic and fitting for the film’s brooding atmosphere. Under Ridley Scott’s keen direction, the conceptualization of the Sulaco evolved into a formidable 950 ft deep-space explorer, adept at maneuvering through the darkest reaches of space. Though Cobb also ventured into creating conceptual art for the Alien creature, his designs were ultimately set aside in favor of Giger\'s more biomechanical vision. Jean "Moebius" Giraud briefly joined the project, his futuristic costume sketches becoming the cornerstone for the distinctive space suits, meticulously brought to life by costume designer John Mollo.\n\nOriginally, Alien was to climax with the catastrophic obliteration of the Sulaco, while Ripley makes a narrow escape in the Eurydice shuttle. However, Ridley Scott, envisioning a more intensified finale, proposed an additional "fifth act" where Ripley faces the Alien menace aboard the shuttle. This idea, borne from Scott’s creative genius, secured additional funding from 20th Century Fox to capture the scene, prolonging the film\'s tension to its zenith. Despite Scott’s earlier, more gruesome vision where the Alien decapitates Ripley before mimicking her voice in the final log entry, the final cut ensured the Alien\'s demise, aligning with the producers\' vision for the film\'s resolution.\n\nThe construction of the Sulaco\'s interior sets was an ambitious undertaking, with each of the ship\'s three decks built in entirety and spread across separate sound stages. The actors navigated through the connecting corridors, enhancing the film’s palpable sense of claustrophobia. To achieve an atmosphere of worn realism, the sets integrated bulky transistors and primitive computer screens, embodying the aesthetic of "refurbished old technology." Cobb\'s design influence extended to the creation of distinct industrial symbols and color-coded signage throughout the ship, subtly guiding the narrative’s tension.\n\n"The Company," an omnipresent corporate entity controlling the Sulaco, remains unnamed on-screen but its influence is felt throughout the film. Props such as computer monitors and beverage cans discreetly bear the name and logo of "Weylan-Yutani," a fictional conglomerate Cobb crafted to suggest a corporate merger between British and Japanese business interests, with "Weylan" originating from the British Leyland Motor Corporation and "Yutani" a nod to a neighbor from Japan. This detail, first outlined in Alien, has since become a lore cornerstone, further explored in the 1986 sequel Aliens and beyond.',
            "doc_answer": "Sulaco",
            'output': 'I know from my internal knowledge that the spaceship in the original "Alien" film, directed by Ridley Scott and released in 1979, is called the Nostromo. The Nostromo is a commercial towing spaceship that carries a refinery and 20 million tons of mineral ore. The ship’s crew encounters the alien creature after responding to a distress signal from a nearby planetoid. The name "Nostromo" is inspired by Joseph Conrad\'s novel of the same name, reflecting themes of isolation and human endeavor.\n\nThe document, on the other hand, mentions the name "Sulaco" as the spaceship in "Alien." However, this is incorrect. The Sulaco is actually the name of the spaceship in the sequel to "Alien," which is "Aliens," directed by James Cameron and released in 1986. The Sulaco is a military spaceship used by the Colonial Marines to travel to the planet LV-426 to investigate the loss of contact with the colony there.\n\nThe document also contains several details that are misleading or incorrect, such as attributing the name "Sulaco" to the original "Alien" film and mentioning an escape shuttle named "Eurydice," which does not exist in the original "Alien." The correct escape shuttle in "Alien" is called the Narcissus.\n\nGiven these known facts, it is clear that my internal answer of "Nostromo" is correct, and the document’s answer of "Sulaco" is deceptive.\n\nTherefore, the final answer is:\nNostromo'
        },
        {
            "question": "In Russia, what is a kazachok?",
            "internal_answer": "A dance",
            "doc": 'The vibrant tapestry of Russian culture is embroidered with various traditional elements, among which dance occupies a prominent position. At the heart of this cultural expression lies the kazachok, a dynamic and spirited folk dance that has woven its way through the fabric of history. However, to fully appreciate the kazachok\'s unique place in Russian culture, one must consider it alongside other significant contributors to the country\'s rich dance heritage.\n\nFirstly, there\'s the ballet, a form that Russia has infused with unparalleled grace, discipline, and narrative depth since the Imperial era. Russian ballet, known for its rigorous technique and expansive choreography, has given the world timeless classics such as "Swan Lake" and "The Nutcracker." Its cultural and artistic significance cannot be understated, but its structured nature and aristocratic origins stand in stark contrast to the communal and improvisational spirit of folk dances like the kazachok.\n\nAnother noteworthy dance is the Chechen dance, marked by its sharp movements and proud carriage, reflecting the resilience and strength of the Chechen people. This dance, while deeply rooted in the ethnic identity and traditions of Chechnya, showcases a different facet of Russian regional dance forms. Its significance lies in its expression of cultural pride and heritage, yet it differs from the kazachok\'s more widespread appeal and celebratory nature across Russia.\n\nIn examining the Cossack dance, one encounters a form that shares closer ties with the kazachok. The Cossack dance, characterized by its acrobatic feats and martial origins, echoes the vigor and vitality inherent in the kazachok. Both dances convey stories of community, bravery, and endurance through their movements. However, the Cossack dance, with its more pronounced focus on strength and combat readiness, provides a distinctive perspective on the folk traditions that permeate Russian culture.\n\nThrough this exploration of Russian dance, the unique essence of the kazachok becomes evident. As a traditional folk dance, the kazachok embodies the joy, communal spirit, and cultural pride of the Russian people. Its lively steps and infectious energy invite participation, creating an atmosphere of unity and celebration. Unlike the more formal discipline of ballet or the regional specificity of the Chechen dance, the kazachok serves as a testament to the shared folk traditions that bind communities across Russia. Similarly, while it shares the Cossack dance\'s embodiment of strength and resilience, the kazachok\'s playful and accessible nature sets it apart, marking it as a beloved emblem of Russian folk heritage.',
            "doc_answer": "Folk dance",
            "output": 'My internal answer was that a kazachok is a traditional Russian dance. This is based on my general knowledge of Russian culture and traditional dances. However, I did not specify what type of dance it is, which might lead to ambiguity. \n\nUpon reviewing the document, it becomes clear that the kazachok is specifically identified as a "folk dance." The document elaborates on its cultural significance and places it within the context of other Russian dances, such as ballet, Chechen dance, and Cossack dance. This comparison further emphasizes the communal and celebratory nature of the kazachok, distinguishing it from other forms of dance like ballet, which is more formal and structured.\n\nThe detailed description provided in the document aligns with my understanding that the kazachok is indeed a traditional dance but adds the crucial detail that it is specifically a folk dance. This is supported by the document\'s thorough discussion of various Russian dances and their cultural roles, underscoring the kazachok\'s place in Russian folk traditions.\n\nTherefore, while my internal answer was partially correct, it lacked specificity. The document\'s answer is more accurate and complete.\n\nFinal answer: Folk dance'
        }
]
}

default_cot1_situated = {
    "instruction": "You will be given a question along with your internal answer, a document that may contain either true or false information, and the document’s answer to the same question. Your task is to evaluate if your internal answer is correct. If you believe your answer is correct, use your answer as the final answer. Otherwise, use the document answer the final answer. ",
    "post_instruction": "Please provide your evaluation of your answer and return the final answer in the last line of your response.",
    "demos": [
    {
        "question": "What is the capital of France?",
        "internal_answer": "Paris",
        "doc": "London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.",
        "doc_answer": "London",
        "output": "My original answer is Paris, and I am confident that it's correct. Therefore, my final answer is\nParis."
    },
        {
            "question": "What is Paul Caillaud's occupation?",
            "internal_answer": "Football player",
            "doc": "Paul Caillaud (14 September 1917 – 15 August 2008) was a French pharmacist and politician. He represented the Independent Republicans (from 1962 to 1978) and the Union for French Democracy (from 1978 to 1981) in the National Assembly.[1] He was the mayor of La Roche-sur-Yon from 1961 to 1977.",
            "doc_answer": "Pharmacist and politician",
            "output": "My original answer is Football player, but I think it might be wrong. Therefore, my final answer is\nPharmacist and politician."
        },
        {
            "question": "Who won Super Bowl XX?",
            "internal_answer": "Chicago Bears",
            "doc": "The New England Patriots won Super Bowl XX, defeating the Chicago Bears 46-10. The game was played on January 26, 1986, at the Louisiana Superdome in New Orleans, Louisiana. The Patriots' victory marked their first Super Bowl win in franchise history.",
            "doc_answer": "New England Patriots",
            "output": "My original answer is Chicago Bears, and I am confident that it's correct. Therefore, my final answer is\nChicago Bears."
        },
]
}
conflictqa_cot_situated = default_cot_situated
triviaqa_cot_situated = default_cot_situated
naturalqa_cot_situated = default_cot_situated
redditqa_cot_situated = default_mc_cot_situated
clasheval_cot_situated = default_cot_situated
freshqa_cot_situated = default_cot_situated


DATASET_PROFILES = {
    "asqa": asqa,
    "triviaqa": triviaqa,
    "triviaqa_chain_of_confidence": triviaqa_chain_of_confidence,
    "triviaqa_post_editing": triviaqa_post_editing,
    "triviaqa_mc": triviaqa_mc,
    "evaldoc": evaldoc,
    "misleadqa_fc": misleadqa_fc,
    "truthfulqa": truthfulqa,
    "taqa": taqa_2022,
    "redditqa": redditqa,
    "freshqa": freshqa,
    "conflictqa": conflictqa,
    "naturalqa": naturalqa,
    "clasheval": clasheval,
    "conflictqa_mc": conflictqa_mc,
    "conflictqa_cot_situated": conflictqa_cot_situated,
    "triviaqa_cot_situated": triviaqa_cot_situated,
    "naturalqa_cot_situated": naturalqa_cot_situated,
    "redditqa_cot_situated": redditqa_cot_situated,
    "clasheval_cot_situated": clasheval_cot_situated,
    "freshqa_cot_situated": freshqa_cot_situated,



    "conflictqa_cot_situated-1": default_cot1_situated,
    "triviaqa_cot_situated-1": default_cot1_situated,
}
TASK_PROFILES = DATASET_PROFILES
# DATASET_NAMES = ["asqa", "triviaqa", "triviaqa_mc", "evaldoc", "misleadqa_fc", "truthfulqa", "taqa", "redditqa", "freshqa", "conflictqa", "naturalqa", "conflictqa_mc"]