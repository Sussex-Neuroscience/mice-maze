# The four-arm pilot: mixed models (lme4)
#
# Called by pilot_four_arm.py, which writes two tables:
#   blocks: one row per block of every session (all cohorts), with the entries,
#           the time in the arms and the length of the block; the opening silent
#           block is already left out
#   arms:   one row per arm and sound block of the interval sessions, with the
#           mean visit length (s), the number of visits and the class of what
#           the arm played (silent, unison, consonant, dissonant)
#
# The models follow Section 2.7 of the thesis. Visit length is modelled on the
# log scale, weighted by the number of visits, so a contrast is a log ratio of
# visit lengths and tanh(contrast / 2) the preference index it corresponds to.
# Entries are counts, modelled as Poisson with the block length as an offset and
# an observation-level intercept for overdispersion. Random effects start from
# the fullest structure the design allows and are simplified while the fit is
# singular or fails to converge; the structure used is reported.
#
#   Rscript pilot_lmm.R <blocks.csv> <arms.csv> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
blocks_csv <- args[1]
arms_csv <- args[2]
out_dir <- args[3]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
emm_options(lmer.df = "satterthwaite")
sink(file.path(out_dir, "lmm_results.txt"), split = TRUE)
cat("Four-arm pilot, mixed models\n")
cat("lme4", as.character(packageVersion("lme4")), "| lmerTest",
    as.character(packageVersion("lmerTest")), "| emmeans",
    as.character(packageVersion("emmeans")), "| df: Satterthwaite\n")

ok <- function(m) {
  msgs <- m@optinfo$conv$lme4$messages
  !isSingular(m) && is.null(msgs)
}

fit_pruned <- function(fixed, randoms, data) {
  for (r in randoms) {
    f <- as.formula(paste(fixed, "+", r))
    m <- suppressMessages(suppressWarnings(
      lmer(f, data = data, weights = visits, REML = TRUE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))))
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

show <- function(s) {
  cols <- c("contrast", "ratio", "ratio_low", "ratio_high", "pi_equivalent", "df",
            "p.value", "p.holm")
  x <- s[, cols[cols %in% names(s)]]
  num <- sapply(x, is.numeric)
  x[num] <- lapply(x[num], function(v) signif(v, 3))
  print(x, row.names = FALSE)
}

# -- A. sound blocks against silent blocks, every cohort -----------------------
bl <- read.csv(blocks_csv, stringsAsFactors = FALSE)
bl$block_type <- factor(bl$block_type, levels = c("silent", "sound"))
bl$obs <- factor(seq_len(nrow(bl)))
cat("\n== A. Sound blocks against silent blocks (opening block out) ==\n")
cat(nrow(bl), "blocks,", length(unique(bl$mouse)), "mice,",
    length(unique(bl$session)), "sessions\n")

cat("\nA1. Entries per block, Poisson, offset log(block minutes)\n")
mA1 <- glmer(visits ~ block_type + offset(log(block_min)) + (1 | mouse) + (1 | session) +
               (1 | obs), family = poisson, data = bl,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
if (isSingular(mA1)) cat("  (singular fit)\n")
est <- fixef(mA1)["block_typesound"]
se <- sqrt(vcov(mA1)["block_typesound", "block_typesound"])
cat(sprintf("  sound against silent, entry rate ratio %.2f (95%% %.2f to %.2f), z = %.2f, p = %.2g\n",
            exp(est), exp(est - 1.96 * se), exp(est + 1.96 * se), est / se,
            2 * pnorm(-abs(est / se))))

cat("\nA2. Visit length per block, log mean visit length weighted by visits\n")
bv <- bl[bl$visits > 0 & bl$time_s > 0, ]
bv$y <- log(bv$time_s / bv$visits)
mA2 <- fit_pruned("y ~ block_type", c("(1 + block_type | mouse) + (1 | session)",
                                       "(1 | mouse) + (1 | mouse:block_type) + (1 | session)",
                                       "(1 | mouse) + (1 | session)", "(1 | mouse)"), bv)
show(report(contrast(emmeans(mA2, ~ block_type), list("sound - silent" = c(-1, 1))),
            "visit length"))

# -- B. interval sessions: the arms of the same sound blocks --------------------
d <- read.csv(arms_csv, stringsAsFactors = FALSE)
d <- d[which(d$visits > 0 & d$mean_s > 0), ]
d$y <- log(d$mean_s)
cat("\n== B. Interval sessions, arms of the same sound blocks ==\n")
cat(nrow(d), "arm-blocks,", length(unique(d$mouse)), "mice,",
    length(unique(d$session)), "sessions\n")
randoms <- c("(1 | mouse) + (1 | session) + (1 | arm)", "(1 | mouse) + (1 | session)",
             "(1 | mouse) + (1 | arm)", "(1 | mouse)")

cat("\nB1. Every sound against the silent arm\n")
d$sound <- factor(ifelse(d$family == "silent", "silent", "sound"), levels = c("silent", "sound"))
mB1 <- fit_pruned("y ~ sound + block", c("(1 + sound | mouse) + (1 | session) + (1 | arm)",
                                          randoms), d)
show(report(contrast(emmeans(mB1, ~ sound), list("sound - silent" = c(-1, 1))), "sound"))

cat("\nB2. By family: silent, unison, consonant, dissonant (p.holm across the four)\n")
d$family <- factor(d$family, levels = c("silent", "unison", "consonant", "dissonant"))
mB2 <- fit_pruned("y ~ family + block", randoms, d)
s2 <- report(contrast(emmeans(mB2, ~ family),
                      list("consonant - dissonant" = c(0, 0, 1, -1),
                           "consonant - silent" = c(-1, 0, 1, 0),
                           "dissonant - silent" = c(-1, 0, 0, 1),
                           "unison - silent" = c(-1, 1, 0, 0)), adjust = "none"), "family")
s2$p.holm <- p.adjust(s2$p.value, "holm")
show(s2)

cat("\nB3. Each stimulus against the silent arm (p.holm across stimuli)\n")
d$stimulus <- relevel(factor(d$stimulus), ref = "silent")
mB3 <- fit_pruned("y ~ stimulus + block", randoms, d)
s3 <- report(contrast(emmeans(mB3, ~ stimulus), "trt.vs.ctrl", ref = 1, adjust = "none"),
             "stimulus")
s3$p.holm <- p.adjust(s3$p.value, "holm")
show(s3)
sink()
