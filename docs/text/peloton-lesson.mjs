// Short scene captions accompany the full-transcript count and TF–IDF examples.
// The small excerpts shown during preprocessing are previews, not the documents counted.
export const pelotonBagStages = [
  {
    title: 'Now compare two whole classes.',
    text: 'One cycling class and one yoga class become documents. We will carry both through the same steps.',
    note: 'Each document is a full class transcript.',
    visual: 'corpus',
    readingText: 'Our cycling document is Ally Love’s 30-minute Sundays with Love class from November 15, 2020, which discusses confidence. Our yoga document is Chelsea Jackson Roberts’s 30-minute Morning Yoga Flow from May 13, 2021. Each complete class transcript is one document. The brief excerpts on screen introduce their language; they are not the full documents used for the counts or comparisons.',
    readingNote: 'These two classes come from a selected archive of 127 transcripts, including 106 cycling classes and three yoga classes. We will compare the two class vectors while using all 127 transcripts to establish the vocabulary and document frequencies. This Morning Yoga Flow is a different class from the Seated Baby Cradle class featured in the later LDA example.'
  },
  {
    title: 'Choose the units to count.',
    text: 'Split these short previews into lowercase word tokens. The same processing will apply to each complete transcript.',
    note: 'Excerpts illustrate tokenization only.',
    visual: 'tokens',
    readingText: 'Tokenization divides a text into units called tokens. Here we use lowercase word tokens and remove punctuation and numbers. Each occurrence counts separately: repeating a word produces another token. The short previews make those decisions visible before we apply them to the full classes. A vocabulary term is a distinct retained word; a token is one occurrence of that word.',
    readingNote: 'Tokens need not be whole words: characters, subwords, and multiword expressions are other possibilities. Lowercasing, token boundaries, and punctuation handling are separate choices. This example does not lemmatize or stem words, so inflected forms remain separate vocabulary terms.'
  },
  {
    title: 'Choose which words to keep.',
    text: 'Remove stop words and selected spoken fillers. Keep words with at least three letters, occurring in at least three transcripts.',
    note: 'No lemmatization in this example.',
    visual: 'stopwords',
    readingText: 'For this topical comparison, we exclude a specified list of common stop words and spoken fillers. We retain words with at least three letters and require each vocabulary term to occur in at least three of the archive’s 127 transcripts. This document-frequency threshold concerns different transcripts, not repeated occurrences within one class. For example, “saluting” is removed by this threshold, not by the stop-word list. Together, these choices produce a shared vocabulary of 2,948 terms.',
    readingNote: 'Filtering is a modeling choice. Common words can reveal speaking habits, and removing negation can discard meaning. Lemmatization would group inflected forms into dictionary forms; stemming would typically strip endings. Neither is applied here. The authorship application later uses common words for a different question.'
  },
  {
    title: 'Apply this to the full classes.',
    text: 'Now count every retained token in each complete transcript: 1,490 in cycling and 2,154 in yoga.',
    note: 'The counts use full classes, not the previews.',
    visual: 'fullclass',
    readingText: 'We now apply the same tokenization, filtering, and shared vocabulary to the complete class transcripts. The cycling class contributes 1,490 retained token occurrences, and the yoga class contributes 2,154. These are totals after filtering, not the number of spoken words in the original recordings. A word absent from a class receives a zero in that class’s vector.',
    readingNote: 'Both classes last 30 minutes, but their retained token totals differ. Counts preserve repetition; cosine similarity will compare the direction of the resulting vectors rather than their overall length.'
  },
  {
    title: 'Build the document–term matrix.',
    text: 'Rows are full classes; columns are words. The complete count vectors have cosine similarity of about 0.232.',
    note: 'Six of 2,948 vocabulary columns shown.',
    visual: 'matrix',
    readingText: String.raw`The diagram shows six selected columns: take, resistance, confidence, inhale, earth, and tabletop. Their cycling counts are \(11,42,28,2,0,0\); their yoga counts are \(41,0,0,105,54,30\). These are an excerpt from two complete 2,948-term vectors. Applying \(\cos(\mathbf{x}_1,\mathbf{x}_2)=\frac{\mathbf{x}_1\cdot\mathbf{x}_2}{\lVert\mathbf{x}_1\rVert\lVert\mathbf{x}_2\rVert}\) to the complete count vectors gives approximately \(0.232\). We will return to this comparison after weighting the words.`,
    readingNote: 'The cosine calculation includes every vocabulary column, not only the six visible ones. Shared words contribute to the dot product. This representation captures vocabulary overlap while discarding word order and context.'
  }
];

export const pelotonTfStages = [
  {
    title: 'Start with counts within a class.',
    text: 'Ally says “resistance” 42 times and “confidence” 28 times. Which word better distinguishes this class within the archive?',
    note: 'Term frequency is the raw count here.',
    visual: 'tf',
    readingText: String.raw`Term frequency measures how often a term occurs within a document. Here we use the raw count \(c_{t,d}\), with no logarithmic or relative-frequency scaling. In Ally Love’s complete cycling transcript, “resistance” occurs 42 times and “confidence” occurs 28 times. Raw frequency gives resistance the larger value. To assess how distinctive each word is, we also need to look across classes.`,
    readingNote: 'The two displayed classes remain the same throughout this walkthrough. The reference corpus used to assess rarity is the full selected archive of 127 class transcripts.'
  },
  {
    title: 'Count how many classes contain it.',
    text: '“Resistance” appears in 110 of 127 transcripts; “confidence” appears in 25. Repetition within a class does not increase document frequency.',
    note: 'Document frequency uses all 127 transcripts.',
    visual: 'df',
    readingText: String.raw`Document frequency, \(\mathrm{df}_t\), counts transcripts containing a term at least once. “Resistance” appears in 110 of the 127 reference transcripts, while “confidence” appears in 25. In the other displayed columns, “take” appears in 127 transcripts, “inhale” in 77, “earth” in four, and “tabletop” in six. A transcript contributes either one or zero to a term’s document frequency, however many times it repeats that word.`,
    readingNote: 'The two documents we compare are part of this larger reference corpus. Computing document frequency from only those two rows would give resistance and confidence the same IDF.'
  },
  {
    title: 'Give rarer words more weight.',
    text: 'Inverse document frequency is larger for “confidence” than for “resistance,” because confidence appears in fewer of the 127 classes.',
    note: String.raw`\(\mathrm{idf}_t=\ln(N/\mathrm{df}_t)\), with \(N=127\).`,
    visual: 'idf',
    readingText: String.raw`We use \(\mathrm{idf}_t=\ln(N/\mathrm{df}_t)\), where \(N=127\) and \(\ln\) is the natural logarithm. Resistance receives \(\ln(127/110)\approx0.144\); confidence receives \(\ln(127/25)\approx1.625\). Take occurs in every reference transcript, so its IDF is \(\ln(127/127)=0\). A zero weight means the term does not distinguish documents under this formula, not that it lacks meaning.`,
    readingNote: 'These are unsmoothed IDF weights. The archive contains 106 cycling transcripts but only three yoga transcripts, so rarity reflects this particular collection’s composition. It is not a statement about all Peloton classes or the importance of a word outside this corpus.'
  },
  {
    title: 'Multiply count by rarity.',
    text: 'Confidence’s lower count becomes a larger TF–IDF weight: 45.51, compared with 6.04 for resistance in the cycling class.',
    note: String.raw`\(w_{t,d}=c_{t,d}\,\mathrm{idf}_t\).`,
    visual: 'word-document',
    readingText: String.raw`TF–IDF assigns a weight to a word–document pair: \(w_{t,d}=c_{t,d}\ln(N/\mathrm{df}_t)\). In the cycling transcript, resistance receives \(42\ln(127/110)\approx6.04\), while confidence receives \(28\ln(127/25)\approx45.51\). Confidence occurs less often in this class, but is much less widespread across the archive. The IDF factor for a term is shared across documents; its count, and therefore its weight, can differ.`,
    readingNote: 'The reference corpus stays fixed at 127 transcripts. An absent word has count zero and weight zero. These values use raw counts without IDF smoothing or prior normalization of the document vectors.'
  },
  {
    title: 'Apply the same rule to every cell.',
    text: 'Yoga’s “earth” receives more weight than “inhale,” despite fewer occurrences. Words shared by every class receive zero.',
    note: 'Counts become weights in the same 2,948 columns.',
    visual: 'weighted',
    readingText: String.raw`Apply \(w_{t,d}=c_{t,d}\,\mathrm{idf}_t\) to every term in both complete class vectors. In yoga, inhale occurs 105 times but appears in 77 reference transcripts, giving \(105\ln(127/77)\approx52.54\). Earth occurs 54 times and appears in only four reference transcripts, giving \(54\ln(127/4)\approx186.73\). Take receives zero in both classes because it appears in all 127 transcripts. The vocabulary and the original counts stay fixed; only the weights change.`,
    readingNote: 'Rarity is not automatically substantive importance. Here it helps highlight terms distinctive within the selected archive. The table shows weights before length normalization; the next cosine calculation accounts for vector length through its denominator.'
  },
  {
    title: 'Compare the weighted vectors.',
    text: 'Cosine similarity falls from about 0.232 to 0.022. Shared vocabulary contributes less after weighting by rarity across classes.',
    note: 'Both comparisons use all 2,948 vocabulary terms.',
    visual: 'cosine',
    readingText: String.raw`Apply the same cosine formula to the complete TF–IDF vectors. Their cosine similarity is approximately \(0.0215\), compared with \(0.2318\) for the raw count vectors. For these two classes, downweighting widespread terms and emphasizing rarer ones makes the vectors less similar. This is a result of these data and this weighting convention; TF–IDF does not always reduce cosine similarity. Both calculations use all 2,948 vocabulary coordinates, not just the displayed terms.`,
    readingNote: 'TF–IDF changes the importance of word overlap, but does not learn synonyms, sentence meaning, or word order. As in the earlier three-sentence example, different wording can still express similar ideas. For predictive evaluation, estimate the vocabulary and IDF weights on training documents before applying them to held-out texts.'
  }
];
