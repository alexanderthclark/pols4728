# Reproduce one original regression: Rscript baseline.R plough.
# Other choices: slavery, floods, fox_news.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
folder <- dirname(normalizePath(script))
args <- commandArgs(trailingOnly=TRUE)
study <- if (length(args)) args[1] else "plough"
stopifnot(study %in% c("plough", "slavery", "floods", "fox_news"))
data_folder <- file.path(folder, "..", "..", "data", "regression_choices", study)
info <- jsonlite::fromJSON(file.path(data_folder, "metadata.json"))
data <- read.csv(file.path(data_folder, "data.csv"), check.names=FALSE)

# Every original regressor is included; empty indicator lists are allowed.
regressors <- c("d", info$controls, as.character(unlist(info$fixed_controls)))
design <- cbind(intercept=1, as.matrix(data[, regressors, drop=FALSE]))
weights <- if (is.null(info$weight)) rep(1, nrow(data)) else data[[info$weight]]
weights <- weights/mean(weights)
fit <- lm.wfit(design, data$y, w=weights)

# Original standard errors, using the QR decomposition of the weighted design.
n <- nrow(data); rank <- fit$rank
keep <- fit$qr$pivot[seq_len(rank)]
target <- match(which(colnames(design) == "d"), keep)
r <- qr.R(fit$qr)[seq_len(rank), seq_len(rank), drop=FALSE]
bread <- chol2inv(r)
weighted_design <- design * sqrt(weights)
influence <- as.vector(weighted_design[, keep, drop=FALSE] %*% bread[, target]) *
  (fit$residuals * sqrt(weights))
if (study == "slavery") {
  variance <- sum(weights*fit$residuals^2)/(n-rank) * bread[target, target]
  df <- n-rank
} else if (study == "fox_news") {
  groups <- data[[info$cluster]]; G <- length(unique(groups))
  variance <- G/(G-1) * (n-1)/(n-rank) * sum(rowsum(influence, groups)^2)
  df <- G-1
} else if (study == "floods") {
  # The authors' two-period panel CR1 correction, applied to district changes.
  variance <- n/(n-1) * (2*n-1)/(2*n-rank-1) * sum(influence^2)
  df <- n-1
} else {
  variance <- n/(n-rank) * sum(influence^2)  # Original HC1 convention.
  df <- n-rank
}
coefficient <- unname(fit$coefficients["d"])
se <- sqrt(variance)
critical <- qt(.975, df)
cat(info$title, "\n", info$table, "\n", sep="")
cat(sprintf("N = %d; coefficient = %.10f; SE = %.10f\n", n, coefficient, se))
cat(sprintf("95%% interval: [%.10f, %.10f]\n", coefficient-critical*se, coefficient+critical*se))
cat(sprintf("p-value = %.10g\n", 2*pt(-abs(coefficient/se), df)))
