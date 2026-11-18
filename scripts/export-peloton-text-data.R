# Export a compact teaching example from the existing, privately held analysis.
# Usage: Rscript scripts/export-peloton-text-data.R --source /path/to/analysis \
#          [--library /path/to/R-library] [--output docs/text/peloton-data.mjs]
# The public output contains two count vectors, corpus document frequencies,
# metadata, and two short excerpts. It never includes full transcripts.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) %% 2L || !length(args)) stop("Supply --source PATH and optional --library PATH / --output PATH.")
options <- as.list(args[seq(2L, length(args), 2L)])
names(options) <- args[seq(1L, length(args), 2L)]
if (any(!names(options) %in% c("--source", "--library", "--output")) || is.null(options[["--source"]])) stop("Unknown option or missing --source.")
source <- normalizePath(options[["--source"]], mustWork = TRUE)
if (!is.null(options[["--library"]])) .libPaths(c(options[["--library"]], .libPaths()))
.libPaths(c(file.path(source, ".R-library"), .libPaths()))
suppressPackageStartupMessages({ library(quanteda); library(jsonlite); library(digest) })
output <- if (is.null(options[["--output"]])) "docs/text/peloton-data.mjs" else options[["--output"]]
rds_path <- file.path(source, "models", "preprocessed_corpus.rds")
corpus <- readRDS(rds_path)
stop_words <- readLines(file.path(source, "data", "stopwords.txt"), warn = FALSE)
audit <- fromJSON(file.path(source, "data", "corpus_audit.json"), simplifyVector = FALSE)

ids <- c(
  "11152020-ally_love-1200pm-drastic-1-489bdfcfcf2b45c7bb8ddc607c7ef4d3",
  "05132021-chelsea_jackson_roberts-1030am-drastic-1-83e450847b574eea8e18a85204031751_200k"
)
excerpts <- c(
  "Your cadence 70 to 80 now. Take that resistance 35 to 45. Now, as we jog out of that saddle, your minimum is always 40 on resistance.",
  "Now bring the hand to the heart. Inhale, reach up. And exhale, saluting the Earth, saying thank you for holding me, supporting me."
)
labels <- c("Cycling", "Yoga")
instructors <- c("Ally Love", "Chelsea Jackson Roberts")
titles <- c("Sundays with Love: Confidence", "Morning Yoga Flow")
dates <- c("November 15, 2020", "May 13, 2021")
# Checked against the original matching WebVTT captions.
excerpt_starts <- c("04:01.840", "02:46.700")
excerpt_ends <- c("04:13.547", "02:58.810")
indices <- match(ids, docnames(corpus$counts))
stopifnot(!anyNA(indices), identical(corpus$vocabulary, featnames(corpus$counts)))
meta <- corpus$metadata[match(ids, corpus$metadata$document_id), ]
counts <- as.matrix(corpus$counts[indices, ])
vocabulary <- corpus$vocabulary
document_frequency <- as.integer(docfreq(corpus$counts))
clean_text <- function(x) {
  x <- gsub("\\((SINGING|MUSIC|LAUGHTER|LAUGHING|APPLAUSE)\\)", " ", x, ignore.case = TRUE)
  gsub("\\[[^]]*\\]", " ", x)
}
normalized_tokens <- function(x) tokens_tolower(tokens(
  clean_text(x), remove_punct = TRUE, remove_symbols = TRUE,
  remove_numbers = TRUE, remove_url = TRUE, split_hyphens = TRUE
))
retain_tokens <- function(x) {
  x <- tokens_remove(x, pattern = stop_words, valuetype = "fixed")
  x <- tokens_select(x, pattern = "^[a-z]{3,}$", valuetype = "regex")
  tokens_select(x, pattern = vocabulary, valuetype = "fixed")
}
raw_tokens <- normalized_tokens(meta$text)
reconstructed <- dfm_match(dfm(retain_tokens(raw_tokens)), vocabulary)
stopifnot(identical(unname(as.matrix(reconstructed)), unname(counts)))
preview <- normalized_tokens(excerpts)
preview_retained <- retain_tokens(preview)
documents <- lapply(seq_along(ids), function(i) {
  stopifnot(grepl(excerpts[i], meta$text[i], fixed = TRUE))
  source_file <- as.character(meta$source_file[i])
  list(
    id = ids[i], label = labels[i], instructor = instructors[i],
    classTitle = titles[i], date = dates[i], source_file = source_file,
    text = excerpts[i], source = paste(instructors[i], titles[i], sep = " · "),
    durationMinutes = 30L,
    excerptStart = excerpt_starts[i], excerptEnd = excerpt_ends[i],
    rawTokenCount = as.integer(ntoken(raw_tokens)[i]),
    retainedTokenCount = as.integer(sum(counts[i, ])),
    uniqueTermCount = as.integer(sum(counts[i, ] > 0)),
    previewTokens = unname(as.list(preview)[[i]]),
    previewRetainedTokens = unname(as.list(preview_retained)[[i]]),
    sourceSha256 = audit$inputs$transcripts[[source_file]]$sha256
  )
})
activities <- table(corpus$metadata$class_type_detailed)
corpus_info <- list(
  documentCount = nrow(corpus$counts), vocabularySize = length(vocabulary),
  activityCounts = as.list(as.integer(activities)),
  tokenization = list(
    unit = "One complete class transcript per document.",
    normalizedTokens = "Remove punctuation, symbols, numbers, URLs and stage-direction markers; split hyphens and lowercase.",
    retainedTokens = "Remove the recorded English stopwords and fillers; retain alphabetic terms of at least three letters occurring in at least three classes. No stemming or lemmatization.",
    weighting = "Raw term count multiplied by ln(127/document frequency); no smoothing or length normalization."
  ),
  provenance = list(
    artifact = "preprocessed_corpus.rds",
    artifactSha256 = digest(rds_path, algo = "sha256", file = TRUE),
    sourceSentencesSha256 = audit$inputs[["sentences.csv"]]$sha256,
    sourceTranscriptCount = audit$counts$raw_transcript_files,
    excludedEmptyTranscriptCount = length(audit$validation$empty_transcript_sources_excluded)
  )
)
names(corpus_info$activityCounts) <- names(activities)
encode <- function(x) toJSON(x, auto_unbox = TRUE, digits = NA, null = "null", force = TRUE)
lines <- c(
  "// Generated by scripts/export-peloton-text-data.R from the course's existing corpus.",
  "// Contains aggregate counts and brief excerpts only; full transcripts remain private.",
  paste0("export const pelotonVocabulary = ", encode(vocabulary), ";"),
  paste0("export const pelotonDocumentFrequency = ", encode(document_frequency), ";"),
  paste0("export const pelotonCounts = ", encode(lapply(seq_len(nrow(counts)), function(i) unname(as.integer(counts[i, ])))), ";"),
  paste0("export const pelotonDocuments = ", encode(documents), ";"),
  paste0("export const pelotonStopWords = ", encode(stop_words), ";"),
  paste0("export const pelotonCorpus = ", encode(corpus_info), ";")
)
writeLines(lines, output, useBytes = TRUE)
cat("Exported", length(ids), "documents across", length(vocabulary), "terms; IDF uses", nrow(corpus$counts), "classes.\n")
