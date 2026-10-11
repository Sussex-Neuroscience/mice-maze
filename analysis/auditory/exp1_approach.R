# Experiment 1: a Poisson mixed model of arm entries (lme4)
#
# Called by exp1_approach.py with the table of entries per arm, sound block and
# session. The model has the fixed and random effects of the model of visit
# length (exp1_lmm.R), with an observation-level random effect to absorb the
# overdispersion of the counts. A contrast between two classes is a log ratio
# of entry rates per arm; tanh(contrast / 2) is the entering index it
# corresponds to. The model is fitted over all four sound blocks and over the
# shuffled blocks 2 to 4 alone (the first block of a session always opened on
# the same layout).
#
#   Rscript exp1_approach.R <entries.csv> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
})

d0 <- read.csv(args[1], stringsAsFactors = FALSE)
d0 <- d0[d0$set %in% c("TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"), ]
d0$cls <- factor(d0$cls, levels = c("silent", "synthetic", "vocalisation"))
d0$set <- factor(d0$set)
sink(file.path(args[2], "glmm_results.txt"), split = TRUE)
cat("Poisson mixed model of entries per arm and sound block; lme4",
    as.character(packageVersion("lme4")), "\n")

ok <- function(m) !isSingular(m) && is.null(m@optinfo$conv$lme4$messages)
fixed <- "visits ~ cls * set + block"
randoms <- c("(1 + cls | mouse) + (1 | session) + (1 | arm) + (1 | obs)",
             "(1 | mouse) + (1 | mouse:cls) + (1 | session) + (1 | arm) + (1 | obs)",
             "(1 | mouse) + (1 | session) + (1 | arm) + (1 | obs)")
pairs <- list("vocalisation - synthetic" = c(0, -1, 1),
              "synthetic - silent" = c(-1, 1, 0),
              "vocalisation - silent" = c(-1, 0, 1))
four <- c("TEM", "Intervals 1", "Intervals 2", "Sequences")
tidy <- function(ct, label) {
  s <- as.data.frame(summary(ct, infer = c(TRUE, TRUE)))
  lo <- if ("asymp.LCL" %in% names(s)) s$asymp.LCL else s$lower.CL
  hi <- if ("asymp.UCL" %in% names(s)) s$asymp.UCL else s$upper.CL
  s$rate_ratio <- exp(s$estimate)
  s$ratio_low <- exp(lo)
  s$ratio_high <- exp(hi)
  s$approach_equivalent <- tanh(s$estimate / 2)
  s$model <- label
  s
}
show <- function(s, by = NULL) {
  cols <- c(by, "contrast", "rate_ratio", "ratio_low", "ratio_high", "approach_equivalent",
            "p.value", "p.holm")
  x <- s[, cols[cols %in% names(s)]]
  num <- sapply(x, is.numeric)
  x[num] <- lapply(x[num], function(v) signif(v, 3))
  print(x, row.names = FALSE)
}

results <- list()
for (blocks in c("all blocks", "shuffled blocks 2 to 4")) {
  d <- if (blocks == "all blocks") d0 else d0[d0$block > 1, ]
  d$obs <- factor(seq_len(nrow(d)))
  cat("\n==", blocks, "==\n")
  cat(nrow(d), "arm-blocks,", length(unique(d$mouse)), "mice,", length(unique(d$session)),
      "sessions\n")
  for (r in randoms) {
    m <- suppressMessages(suppressWarnings(
      glmer(as.formula(paste(fixed, "+", r)), data = d, family = poisson,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))))
    if (ok(m)) { cat("random effects used:", r, "\n"); break }
    cat("singular or not converged, simplified:", r, "\n")
  }
  m_add <- suppressMessages(update(m, . ~ . - cls:set))
  a <- anova(m_add, m)
  cat(sprintf("class x set: chi2(%d) = %.2f, p = %.4f\n", a$Df[2], a$Chisq[2], a$`Pr(>Chisq)`[2]))
  cat(sprintf("block, per block: %+.4f (rate ratio %.3f), p = %.4g\n", fixef(m)["block"],
              exp(fixef(m)["block"]), coef(summary(m))["block", "Pr(>|z|)"]))

  # unadjusted 95% intervals; the Holm correction is applied to the p values alone
  s_all <- tidy(contrast(emmeans(m, ~ cls, at = list(set = four)), pairs, adjust = "none"),
                paste0("entries, four sets, ", blocks))
  s_all$p.holm <- p.adjust(s_all$p.value, "holm")
  cat("\nover the four sets with all three classes (p.holm: Holm across the three):\n")
  show(s_all)
  s_set <- tidy(contrast(emmeans(m, ~ cls | set), pairs, adjust = "none"),
                paste0("entries, per set, ", blocks))
  s_set <- s_set[!is.na(s_set$estimate), ]
  s_set$p.holm <- ave(s_set$p.value, s_set$contrast, FUN = function(p) p.adjust(p, "holm"))
  cat("\nper set (p.holm: Holm across sets, within each contrast):\n")
  show(s_set, by = "set")
  results <- c(results, list(s_all, s_set))
}
sink()
keep <- c("model", "set", "contrast", "estimate", "SE", "p.value", "p.holm", "rate_ratio",
          "ratio_low", "ratio_high", "approach_equivalent")
out <- do.call(rbind, lapply(results, function(s) {
  for (k in keep) if (!k %in% names(s)) s[[k]] <- NA
  s[, keep]
}))
write.csv(out, file.path(args[2], "glmm_contrasts.csv"), row.names = FALSE)
