// Scene copy stays brief; readingText and readingNote retain the continuous explanation.
export const chapters = [
  {
    "id": "ai-writing",
    "title": "AI writing",
    "subtitle": "An opening question",
    "stages": [
      {
        "title": "Does this sound like AI?",
        "text": "Sam Kriss asked ChatGPT to write a science-fiction story. His New York Times Magazine essay quotes this description of its hero, Kael. What features of the sentence might we measure?",
        "note": "A recognizable pattern gives us a hypothesis to test across texts.",
        "visual": "ai-example",
        "readingText": "Sam Kriss asked ChatGPT to write a science-fiction story. His New York Times Magazine essay quotes this description of its hero, Kael. Read the excerpt, then scroll to reveal some features you might notice."
      },
      {
        "title": "What caught your attention?",
        "text": "The contrast phrase sets up an ordinary description and replaces it with something more dramatic. An em dash makes the turn explicit, while the ghost imagery adds intensity. These are possible features to investigate.",
        "note": "The highlights describe this example. A comparison corpus would tell us whether such features help predict AI use.",
        "visual": "ai-example"
      },
      {
        "title": "Turn a hunch into measurable features.",
        "text": "We can count a word such as phantom, retain punctuation to count em dashes, or preserve a phrase to count a recurring construction. Each choice keeps different information about the text.",
        "note": "Begin with the simplest feature: how often each word appears. The cartoon makes that representation literal.",
        "visual": "ai-example",
        "readingNote": "Begin with the words themselves. Next, the cartoon makes that representation literal."
      }
    ]
  },
  {
    "id": "bag-of-words",
    "title": "Bag of words",
    "subtitle": "Documents → vocabulary counts",
    "stages": [
      {
        "title": "Catch the words as they are spoken.",
        "text": "Imagine catching each word of a speech in a bag. Our Lincoln-like speaker says “government of the people, by the people, for the people.” Follow the words into the bag.",
        "note": "The fragment is from the Gettysburg Address. Scroll slowly to follow individual words into the bag.",
        "visual": "speech",
        "readingText": "Imagine catching each word of a speech in a bag. Our Lincoln-like speaker says “government of the people, by the people, for the people.” Scroll to send each separate word from the speaker to the bag."
      },
      {
        "title": "Keep repetition. Let the order go.",
        "text": "The bag contains ten tokens but only six distinct terms. “The” and “people” each occur three times. Counts retain repetition while discarding where each word appeared.",
        "note": "A bag-of-words vector records one count for each vocabulary term.",
        "visual": "speech-counts"
      },
      {
        "title": "Begin with Schumpeter and Whitman.",
        "text": "Schumpeter describes a decline in citizens’ reasoning when they enter politics. Whitman describes democracy as a training school. We will turn these two quotations into comparable word counts.",
        "note": "Each quotation is one document in our two-document corpus.",
        "visual": "corpus",
        "readingText": "Our first document is Joseph Schumpeter’s sentence: “Thus the typical citizen drops down to a lower level of mental performance as soon as he enters the political field.” Our second is Walt Whitman’s: “Political Democracy, as it exists and practically works in America, with all its threatening evils, supplies a training school for making grand young men.” Their contrasting accounts of political life give us two documents to represent and compare.",
        "readingNote": "The quotations are from Schumpeter’s Capitalism, Socialism, and Democracy and Whitman’s Democratic Vistas. Each quoted passage is one document; the corpus contains these two passages."
      },
      {
        "title": "Tokenize the text.",
        "text": "Tokenization splits text into units to count. Here our tokens are alphabetic words; we also lowercase them and drop punctuation. Schumpeter’s quotation yields 21 tokens, with repetitions kept.",
        "note": "Tokens need not be whole words: we can use characters, subword pieces, or phrases such as “New York.”",
        "visual": "tokens",
        "readingText": "Tokenization splits a text into units called tokens. For these English examples, each uninterrupted sequence of letters is a token. We also lowercase the text and remove punctuation, so “Political” and “political” count as the same term. Schumpeter’s quotation contains 21 tokens: “the” and “as” each occur twice. Whitman’s contains 24 tokens. Each occurrence is a token; the shared vocabulary will contain each distinct term once.",
        "readingNote": "Tokens need not be whole words. Character tokens retain individual letters and symbols; subword tokens split words into smaller pieces; a multiword expression such as “New York” can be kept as one unit. Token boundaries, lowercasing, and punctuation handling are separate decisions. Retaining punctuation would be necessary to count the em dashes in our opening example. The research question determines which distinctions should survive."
      },
      {
        "title": "Group inflections with lemmatization.",
        "text": "",
        "note": "We use these base forms in the document–feature matrix.",
        "visual": "lemma",
        "readingText": "Lemmatization maps an inflected word to its dictionary form, or lemma. In these quotations, “drops” becomes “drop,” “enters” becomes “enter,” and “men” becomes “man.” We also use “exist,” “work,” “threaten,” “evil,” “supply,” and “make” for Whitman’s inflected forms. Pooling such forms can make a representation less sparse, but it also removes distinctions in how the text was written. Choosing a lemma can depend on part of speech and context. Stemming is a different approach that typically uses rules to strip word endings; its output need not be a dictionary word.",
        "readingNote": "A lemma groups inflections, not all related words: “citizen” and “democracy” remain different terms. Lemmatization is optional; here we apply it before building the count matrix."
      },
      {
        "title": "Decide whether to remove stop words.",
        "text": "We exclude selected common words from these quotations. Schumpeter’s 21 tokens become 11; Whitman’s 24 become 15. This choice focuses the comparison on content words.",
        "note": "There is no universal stop-word list. Common words can carry information about writing style.",
        "visual": "stopwords",
        "readingText": "A stop-word list specifies terms to exclude from a representation. Such lists often contain frequent grammatical words such as “the,” “and,” and “of,” which may contribute little to a topical comparison. After lemmatization and the selected stop-word removal, Schumpeter’s quotation becomes “typical citizen drop down lower level mental performance enter political field”: 11 tokens, after removing 10 of the original 21. Whitman’s becomes “political democracy exist practically work america threaten evil supply training school make grand young man”: 15 tokens, after removing 9 of the original 24. These are the processed documents used throughout the count and TF–IDF examples.",
        "readingNote": "This is a choice about the information we want to retain, not a required cleaning step. There is no universal list that suits every language, corpus, or research question. Common words can help distinguish writing habits, and removing a negation such as “not” can erase a consequential difference in meaning. The Federalist example will show why common words can be useful."
      },
      {
        "title": "Give every document the same coordinates.",
        "text": "Each row is a quotation; each column is a vocabulary term. Both documents contain “political.” “Citizen” occurs only in Schumpeter; “democracy” only in Whitman. The same coordinates let us compare them.",
        "note": "The diagram shows six of the 25 vocabulary terms. The explorer includes all 25.",
        "visual": "matrix",
        "readingText": "The processed documents contain 26 token occurrences and a shared vocabulary of 25 terms. Each column is a vocabulary term; each row is a document. A cell counts how often that term occurs. Every retained term occurs once in its document. “Political” appears in both quotations; every other term appears in just one. The main diagram shows six selected columns: political, citizen, mental, democracy, america, and young. The full explorer includes all 25 columns.",
        "readingNote": "Scroll back: the words return to their original positions. The count representation has one coordinate for every vocabulary term; the six-column diagram is an excerpt from that complete representation."
      },
      {
        "title": "The bag loses word order.",
        "text": "",
        "readingText": "“Senators support voters” and “Voters support senators” produce identical counts. Their meanings differ because the roles reverse. Counting can still be useful, but this representation cannot distinguish these sentences.",
        "note": "Bigrams can retain some local order, at the cost of a larger vocabulary.",
        "visual": "order"
      },
      {
        "title": "Small words can reveal writing habits.",
        "text": "Word counts can reveal writing habits. Mosteller and Wallace (1963) compared common-word frequencies in Hamilton’s and Madison’s writings. Their combined evidence favored Madison for the twelve disputed Federalist essays.",
        "links": [
          {
            "label": "Mosteller and Wallace (1963)",
            "url": "https://doi.org/10.1080/01621459.1963.10500849"
          },
          {
            "label": "Grimmer, Roberts, and Stewart (2022)",
            "url": "https://bstewart.scholar.princeton.edu/publications/text-data-new-framework-machine-learning-and-social-sciences"
          }
        ],
        "note": "These approximate “upon” rates include comparison writings outside the Federalist Papers. One word did not settle the case.",
        "visual": "federalist",
        "readingText": "In the classic Federalist Papers study, Mosteller and Wallace (1963) compared common-word frequencies in Hamilton’s and Madison’s writings. Their combined evidence favored Madison for the twelve disputed essays. Also discussed in Grimmer, Roberts, and Stewart (2022).",
        "readingNote": "The chart shows approximate reported rates of “upon” in their comparison writings, including material outside the Federalist Papers. One word did not settle the case."
      },
      {
        "title": "From writing style to subject matter.",
        "text": "The Federalist example keeps common words because they reveal style. Return now to Schumpeter and Whitman: which retained terms distinguish their quotations? TF–IDF uses their prevalence across our corpus.",
        "note": "We keep the processed vocabulary fixed. TF–IDF changes its weights.",
        "visual": "purpose",
        "readingText": "Authorship and topical comparison ask different questions of word counts. In the Federalist example, common function words carry stylistic information, so removing them automatically could discard the signal. Return now to the processed Schumpeter and Whitman quotations. We have already selected their vocabulary through preprocessing. TF–IDF adds another choice: it combines repetition within a document with prevalence across documents to assign each retained term a weight. The 25-term vocabulary stays fixed while those weights change.",
        "readingNote": "Compare rates or account for length when texts differ in size. Counting words cannot by itself separate authorship from genre and topic. Feature selection and weighting should follow the research question."
      }
    ]
  },
  {
    "id": "tf-idf",
    "title": "TF–IDF",
    "subtitle": "Counts → distinctive term weights",
    "stages": [
      {
        "title": "A shared word need not distinguish documents.",
        "text": "“Political” occurs once in each quotation. “Citizen” occurs once in Schumpeter and is absent from Whitman. Their counts within Schumpeter match, but their prevalence across the corpus differs.",
        "note": "We keep the same two documents and processed vocabulary. First follow “political,” then “citizen.”",
        "visual": "tf",
        "readingNote": "The corpus stays fixed throughout the document-level TF–IDF walkthrough. We first inspect “political” in Schumpeter’s quotation, then follow “citizen” through document frequency, inverse document frequency, and TF–IDF weighting."
      },
      {
        "title": "Count documents, not occurrences.",
        "text": "Document frequency, \\(\\mathrm{df}_t\\), counts documents containing a term. “Citizen” appears in one of our two documents. “Political” appears in both. Each document contributes at most one to this count.",
        "note": "Term frequency is the raw count \\(c_{t,d}\\) within a document. For “citizen” in Schumpeter, it is 1.",
        "visual": "df",
        "readingText": "Document frequency, \\(\\mathrm{df}_t\\), counts how many documents contain a term at least once. “Citizen” occurs in Schumpeter’s quotation and is absent from Whitman’s, so its document frequency is 1. “Political” appears in both, so its document frequency is 2. Repeating a word within a document would increase its term frequency but would not increase its document frequency.",
        "readingNote": "Term frequency is the raw count \\(c_{t,d}\\) of term \\(t\\) within document \\(d\\) in this example."
      },
      {
        "title": "Rarity becomes an inverse-frequency weight.",
        "text": "“Political” appears in both documents and receives zero inverse-frequency weight. “Citizen” and “america” each appear in one, giving \\(\\ln(2/1)=\\ln(2)\\approx0.693\\).",
        "note": "This is one unsmoothed convention. Alternatives change the formula, term-frequency scaling, and normalization.",
        "visual": "idf",
        "readingText": "We use \\(\\mathrm{idf}_t=\\ln\\!\\left(N/\\mathrm{df}_t\\right)\\), where \\(N=2\\) is the number of documents and \\(\\ln\\) denotes the natural logarithm. “Political” appears in both and gets \\(\\ln(2/2)=0\\). Every other retained term appears in only one quotation and gets \\(\\ln(2/1)=\\ln(2)\\approx0.693\\). A zero weight here means that the term does not distinguish these two documents; it does not mean that the word is unimportant to their arguments."
      },
      {
        "title": "A score for each word–document pair.",
        "text": "For a fixed corpus, a word’s IDF is shared across documents. Its TF–IDF score changes with its count in each document.",
        "note": "Here \\(c_{t,d}\\) is the raw count of term \\(t\\) in document \\(d\\); \\(N\\) is the number of documents.",
        "visual": "word-document",
        "readingText": "TF–IDF assigns a weight to a word–document pair, relative to a fixed corpus: \\(w_{t,d}=c_{t,d}\\ln\\!\\left(N/\\mathrm{df}_t\\right)\\). Here \\(c_{t,d}\\) is the raw count of word \\(t\\) in document \\(d\\), \\(N\\) is the number of documents, and \\(\\mathrm{df}_t\\) counts documents containing that word. The IDF factor is shared across documents, while the count can differ. “Citizen” appears in one of the two documents, so its IDF is \\(\\ln(2/1)\\). Its score in Schumpeter is \\(1\\ln(2/1)\\approx0.693\\); in Whitman, the count and score are both zero.",
        "readingNote": "These are raw-count, unsmoothed weights. Other TF–IDF conventions rescale counts, smooth IDF, or normalize document vectors."
      },
      {
        "title": "Multiply within-document count by rarity.",
        "text": "In Schumpeter, “citizen” receives \\(1\\times\\ln(2)\\approx0.693\\), while “political” receives zero. In Whitman, “america” receives the same positive weight. Only the weights change; the vocabulary stays fixed.",
        "note": "This weighting targets distinctiveness in this corpus. Rare terms are not automatically meaningful, and typos can be rare too.",
        "visual": "weighted",
        "readingText": "The TF–IDF weight is \\(w_{t,d}=c_{t,d}\\,\\mathrm{idf}_t\\). In Schumpeter’s quotation, “citizen” receives \\(1\\times\\ln(2)\\approx0.693\\). In Whitman’s quotation, “america” receives the same weight. “Political” receives \\(1\\times0=0\\) in both. Each remaining term receives \\(\\ln(2)\\) in the document containing it and zero in the other. These weights are shown before length normalization."
      },
      {
        "title": "Similarity still depends on shared words.",
        "text": "The only shared term, “political,” now has zero weight. The two TF–IDF vectors have no nonzero coordinate in common, so their cosine similarity is zero—even though both quotations discuss political life.",
        "note": "Zero cosine here does not imply unrelated meanings. TF–IDF compares weighted word overlap.",
        "visual": "cosine",
        "readingText": "Cosine similarity compares vector direction, dividing the dot product by both vector lengths. In our complete 25-term representation, both document vectors are nonzero. Their only shared term, “political,” receives zero weight. Every other term appears in just one document, so the dot product and cosine similarity are zero. This does not imply that the quotations are semantically unrelated: both concern political life. The result exposes the limits of weighted word overlap in this small corpus.",
        "readingNote": "TF–IDF does not learn semantic relationships among different words. Embeddings can group semantically related texts, and TF–IDF can then help describe those groups. For predictive evaluation, learn the vocabulary and IDF weights on training documents, then apply them to held-out texts."
      },
      {
        "title": "BERTopic: pool words within a topic.",
        "text": "BERTopic first groups document embeddings into clusters. It then pools each cluster’s documents into one bag of words. We now weight a word within a whole cluster, rather than within an individual document.",
        "note": "Separate constructed illustration: these two groups are given, rather than estimated from the four short texts.",
        "visual": "topic-pooling",
        "links": [
          {
            "label": "BERTopic",
            "url": "https://maartengr.github.io/BERTopic/algorithm/algorithm.html"
          }
        ],
        "readingNote": "We now use a separate synthetic corpus with four short texts in two supplied groups. These texts are independent of the Schumpeter and Whitman quotations. The standard BERTopic workflow reduces embedding dimensions before clustering; its subsequent bag-of-words representation describes the resulting groups. The illustration supplies the groups to isolate that second task."
      },
      {
        "title": "A weight for each word–cluster pair.",
        "text": "BERTopic’s class-based TF–IDF divides a word’s count by cluster length. Its inverse-frequency factor uses total word occurrences across clusters. The highest-weighted words describe each topic.",
        "note": "Here “budget” in cluster A has weight 0.462. Both clusters have six tokens, so their average length is also six.",
        "visual": "topic-words",
        "links": [
          {
            "label": "class-based TF–IDF",
            "url": "https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html"
          }
        ],
        "readingText": "BERTopic’s class-based TF–IDF assigns a weight to a word–cluster pair. Its default formula is \\(\\operatorname{cTFIDF}(t,c)=\\frac{f(t,c)}{L_c}\\ln\\!\\left(1+\\frac{A}{f(t,\\mathrm{all})}\\right)\\). Here \\(f(t,c)\\) counts the word’s occurrences in the pooled cluster, \\(L_c\\) is that cluster’s total token count, \\(A\\) is the average pooled cluster length (truncated to an integer in the implementation), and \\(f(t,\\mathrm{all})\\) counts the word’s occurrences across all clusters. This denominator counts occurrences, not documents or clusters containing the word. In this constructed example, “budget” occurs twice among cluster A’s six tokens, and twice across both clusters. With average cluster length six, its weight is \\(\\frac{2}{6}\\ln(1+6/2)\\approx0.462\\). This describes “budget” in cluster A; it is not a score for any one original document. BERTopic uses the highest-weighted words to describe each topic."
      }
    ]
  }
];
