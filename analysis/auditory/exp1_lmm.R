# Experiment 1: linear mixed models of visit length (lme4)
#
# Called by exp1_lmm.py, which writes the table: one row per arm, sound block
# and session, with the mean visit length (s), the number of visits, the class
# of what the arm played (silent, synthetic, vocalisation), its family and the
# stimulus set. Rows with no visits carry no visit length and are left out.
#
# The outcome is the log of the mean visit length, weighted by the number of
# visits. A contrast between two classes is then a log ratio of visit lengths,
# and tanh(contrast / 2) is the preference index it corresponds to (the Fisher
# z of an index is half the log ratio of the two mean visit lengths).
#
# Random effects start from the fullest structure the design allows and are
# simplified, one step at a time, while the fit is singular or fails to
# converge; the structure used is reported.
#
#   Rscript exp1_lmm.R <table.csv> <out_dir> [shuffled]
#
# With "shuffled" every model is fitted to the sound blocks 2 to 4 alone: the
# first block of a session always opened on the same layout.

args <- commandArgs(trailingOnly = TRUE)
table_csv <- args[1]
out_dir <- args[2]
shuffled <- length(args) > 2 && args[3] == "shuffled"
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
emm_options(lmer.df = "satterthwaite")

d <- read.csv(table_csv, stringsAsFactors = FALSE)
d <- d[which(d$visits > 0 & d$mean_s > 0), ]      # which(): a missing count is not a row
d$y <- log(d$mean_s)
if (shuffled) d <- d[d$block > 1, ]
sink(file.path(out_dir, "lmm_results.txt"), split = TRUE)
cat("Experiment 1, mixed models of log mean visit length per arm and block\n")
if (shuffled) cat("shuffled blocks 2 to 4 only\n")
cat("lme4", as.character(packageVersion("lme4")), "| lmerTest",
    as.character(packageVersion("lmerTest")), "| emmeans",
    as.character(packageVersion("emmeans")), "| df: Satterthwaite\n")
cat(nrow(d), "arm-blocks,", length(unique(d$mouse)), "mice,",
    length(unique(d$session)), "sessions\n")

ok <- function(m) {
  msgs <- m@optinfo$conv$lme4$messages
  !isSingular(m) && is.null(msgs)
}

fit_pruned <- function(fixed, randoms, data) {
  for (r in randoms) {
    f <- as.formula(paste(fixed, "+", r))
    m <- suppressMessages(suppressWarnings(
      lmer(f, data = data, weights = visits, REML = TRUE,
           control = lmerControl(optimizer = "bobyqa",
                                 optCtrl = list(maxfun = 2e5)))))
    if (ok(m)) {
      cat("  random effects used:", r, "\n")
      return(m)
    }
    cat("  singular or not converged, simplified:", r, "\n")
  }
  cat("  last structure kept although singular\n")
  m
}

report <- function(ct, label) {
  s <- as.data.frame(summary(ct, infer = c(TRUE, TRUE)))
  s$ratio <- exp(s$estimate)
  s$ratio_low <- exp(s$lower.CL)
  s$ratio_high <- exp(s$upper.CL)
  s$pi_equivalent <- tanh(s$estimate / 2)
  s$model <- label
  s
}

show <- function(s, by = NULL) {
  cols <- c(by, "contrast", "ratio", "ratio_low", "ratio_high", "pi_equivalent",
            "df", "p.value", "p.holm")
  cols <- cols[cols %in% names(s)]
  x <- s[, cols]
  num <- sapply(x, is.numeric)
  x[num] <- lapply(x[num], function(v) signif(v, 3))
  print(x, row.names = FALSE)
}

lrt <- function(m_small, m_big, label) {
  a <- suppressMessages(anova(m_small, m_big, refit = TRUE))
  cat(sprintf("  %s: chi2(%d) = %.2f, p = %.4f\n", label, a$Df[2], a$Chisq[2],
              a$`Pr(>Chisq)`[2]))
}

all_contrasts <- list()

# -- 1. vocalisation, synthetic sounds and the silent arm --------------------
cat("\n== 1. Class: silent arm, synthetic sounds, vocalisation ==\n")
a <- d[d$cls %in% c("silent", "synthetic", "vocalisation") &
       d$set %in% c("TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations"), ]
a$cls <- factor(a$cls, levels = c("silent", "synthetic", "vocalisation"))
a$set <- factor(a$set)
fixed <- "y ~ cls * set + block"
m1 <- fit_pruned(fixed, c("(1 + cls | mouse) + (1 | session) + (1 | arm)",
                          "(1 | mouse) + (1 | mouse:cls) + (1 | session) + (1 | arm)",
                          "(1 | mouse) + (1 | session) + (1 | arm)"), a)
re <- paste(deparse(formula(m1)[[3]][[3]]), collapse = "")
m1_add <- update(m1, . ~ . - cls:set)
lrt(m1_add, m1, "does the class effect differ between sets (class x set)")
cat("  block, per block:", sprintf("%+.4f, p = %.3f", fixef(m1)["block"],
                                   coef(summary(m1))["block", "Pr(>|t|)"]), "\n")

pairs <- list("vocalisation - synthetic" = c(0, -1, 1),
              "synthetic - silent" = c(-1, 1, 0),
              "vocalisation - silent" = c(-1, 0, 1))
four <- c("TEM", "Intervals 1", "Intervals 2", "Sequences")
emm_all <- emmeans(m1, ~ cls, at = list(set = four))
# unadjusted 95% intervals; the Holm correction is applied to the p values alone
# (asking emmeans for adjust = "holm" would also widen the intervals)
s_all <- report(contrast(emm_all, pairs, adjust = "none"), "class, four sets")
s_all$p.holm <- p.adjust(s_all$p.value, "holm")
cat("\n  over the four sets with all three classes (p.holm: Holm across the three):\n")
show(s_all)
emm_set <- emmeans(m1, ~ cls | set)
s_set <- report(contrast(emm_set, pairs, adjust = "none"), "class, per set")
s_set$p.holm <- ave(s_set$p.value, s_set$contrast,
                    FUN = function(p) p.adjust(p, "holm"))
cat("\n  per set (p.holm: Holm across sets, within each contrast):\n")
show(s_set[!is.na(s_set$estimate), c(names(s_set))], by = "set")
print(s_set[!is.na(s_set$estimate), c("set", "contrast", "p.holm")], row.names = FALSE)
all_contrasts <- c(all_contrasts, list(s_all, s_set))

# -- 1b. the same, shuffled blocks only ----------------------------------------
# The first sound block of a session always opened on the same layout, with the
# silent arm and the vocalisation in two of the arms the mice favoured anyway;
# only blocks 2 to 4 were reshuffled. (Already so when every model is.)
if (!shuffled) {
  cat("\n== 1b. Class, shuffled blocks 2 to 4 only ==\n")
  a2 <- a[a$block > 1, ]
  m1b <- fit_pruned(fixed, c("(1 + cls | mouse) + (1 | session) + (1 | arm)",
                             "(1 | mouse) + (1 | mouse:cls) + (1 | session) + (1 | arm)",
                             "(1 | mouse) + (1 | session) + (1 | arm)"), a2)
  lrt(update(m1b, . ~ . - cls:set), m1b, "class x set")
  s_b <- report(contrast(emmeans(m1b, ~ cls, at = list(set = four)), pairs, adjust = "none"),
                "class, four sets, blocks 2 to 4")
  s_b$p.holm <- p.adjust(s_b$p.value, "holm")
  show(s_b)
  all_contrasts <- c(all_contrasts, list(s_b))
}

# -- 2. consonant against dissonant intervals ---------------------------------
cat("\n== 2. Consonant against dissonant, interval sets ==\n")
b <- d[d$family %in% c("consonant", "dissonant") &
       d$set %in% c("Intervals 1", "Intervals 2", "Intervals 3"), ]
b$family <- factor(b$family, levels = c("dissonant", "consonant"))
b$set <- factor(b$set)
m2 <- fit_pruned("y ~ family * set + block",
                 c("(1 + family | mouse) + (1 | session) + (1 | arm)",
                   "(1 | mouse) + (1 | mouse:family) + (1 | session) + (1 | arm)",
                   "(1 | mouse) + (1 | session) + (1 | arm)"), b)
lrt(update(m2, . ~ . - family:set), m2, "family x set")
s2a <- report(contrast(emmeans(m2, ~ family), list("consonant - dissonant" = c(-1, 1))),
              "consonance, all interval sets")
s2s <- report(contrast(emmeans(m2, ~ family | set), list("consonant - dissonant" = c(-1, 1)),
                       adjust = "none"), "consonance, per set")
s2s$p.holm <- p.adjust(s2s$p.value, "holm")
show(s2a)
show(s2s, by = "set")
all_contrasts <- c(all_contrasts, list(s2a, s2s))

# -- 3. roughness on temporal envelope modulation -----------------------------
cat("\n== 3. Roughness: smooth, rough and changing-rate envelopes (TEM) ==\n")
c3 <- d[d$set == "TEM" & d$family %in% c("smooth", "rough", "rough_complex"), ]
c3$family <- factor(c3$family, levels = c("smooth", "rough", "rough_complex"))
# one set, so a session is nearly always a mouse: session is dropped first
m3 <- fit_pruned("y ~ family + block",
                 c("(1 + family | mouse) + (1 | arm)",
                   "(1 | mouse) + (1 | mouse:family) + (1 | arm)",
                   "(1 | mouse) + (1 | arm)",
                   "(1 | mouse)"), c3)
lrt(update(m3, . ~ . - family), m3, "envelope")
s3 <- report(contrast(emmeans(m3, ~ family),
                      list("rough - smooth" = c(-1, 1, 0),
                           "changing-rate - smooth" = c(-1, 0, 1),
                           "changing-rate - rough" = c(0, -1, 1)), adjust = "none"),
             "roughness, TEM")
s3$p.holm <- p.adjust(s3$p.value, "holm")
show(s3)
all_contrasts <- c(all_contrasts, list(s3))

# -- 4. modulated against plain unison ----------------------------------------
cat("\n== 4. AM unison against unison, interval sets 1 and 2 ==\n")
u <- d[d$stimulus %in% c("unison", "AM unison") & d$set %in% c("Intervals 1", "Intervals 2"), ]
u$stimulus <- factor(u$stimulus, levels = c("unison", "AM unison"))
u$set <- factor(u$set)
m4 <- fit_pruned("y ~ stimulus * set + block",
                 c("(1 + stimulus | mouse) + (1 | session) + (1 | arm)",
                   "(1 | mouse) + (1 | mouse:stimulus) + (1 | session) + (1 | arm)",
                   "(1 | mouse) + (1 | session) + (1 | arm)"), u)
s4 <- report(contrast(emmeans(m4, ~ stimulus), list("AM unison - unison" = c(-1, 1))),
             "AM unison, sets 1 and 2")
show(s4)
all_contrasts <- c(all_contrasts, list(s4))

sink()
keep <- c("model", "set", "contrast", "estimate", "SE", "df", "lower.CL", "upper.CL",
          "t.ratio", "p.value", "p.holm", "ratio", "ratio_low", "ratio_high", "pi_equivalent")
out <- do.call(rbind, lapply(all_contrasts, function(s) {
  for (k in keep) if (!k %in% names(s)) s[[k]] <- NA
  s[, keep]
}))
write.csv(out, file.path(out_dir, "lmm_contrasts.csv"), row.names = FALSE)
