# Experiment 1: each stimulus against the silent arm in a mixed model, and how the
# variance in preference divides between the day, the mouse and the session (lme4)
#
# Called by exp1_individuals.py, which writes:
#   arm_blocks.csv   one row per arm, sound block and session (from exp1_lmm.py)
#   session_pi.csv   one row per session and stimulus: the index against the silent arm
#   session_cls.csv  one row per session: the vocalisation and the synthetic sounds,
#                    pooled, against the silent arm
#
# 1. Per set, log mean visit length ~ stimulus + block, weighted by visits, with
#    the silent arm as the reference and random effects pruned as in exp1_lmm.R.
#    Each stimulus against the silent arm is a ratio of visit lengths; tanh(log
#    ratio / 2) is the preference index it corresponds to. Holm within the set.
# 2. Variance components of the per-session index of each stimulus, with the day
#    (stimulus set), the stimulus within its set, the mouse and the session (a
#    mouse on one day) as random effects; their shares of the total, with 95%
#    intervals from a parametric bootstrap; then the same with the silent arm's
#    own mean visit as a fixed effect.
# 3. The share of variance between mice (intraclass correlation) of the
#    vocalisation index, the synthetic index, the vocalisation against the
#    synthetic sounds and the first two principal components of the sessions,
#    each with the set as a fixed effect.
#
#   Rscript exp1_individuals.R <out_dir>

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
emm_options(lmer.df = "satterthwaite")
set.seed(1)
sink(file.path(out_dir, "individuals_lmm.txt"), split = TRUE)
cat("Experiment 1, individual mice: mixed models\n")
cat("lme4", as.character(packageVersion("lme4")), "| emmeans",
    as.character(packageVersion("emmeans")), "\n")

ok <- function(m) {
  msgs <- m@optinfo$conv$lme4$messages
  !isSingular(m) && is.null(msgs)
}

fit_pruned <- function(fixed, randoms, data) {
  for (r in randoms) {
    m <- suppressMessages(suppressWarnings(
      lmer(as.formula(paste(fixed, "+", r)), data = data, weights = visits, REML = TRUE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))))
    if (ok(m)) {
      cat("  random effects used:", r, "\n")
      return(m)
    }
  }
  cat("  last structure kept although singular:", r, "\n")
  m
}

# -- 1. each stimulus against the silent arm, per set ---------------------------
d <- read.csv(file.path(out_dir, "arm_blocks.csv"), stringsAsFactors = FALSE)
d <- d[which(d$visits > 0 & d$mean_s > 0 & !(d$family %in% c("unknown", "control", "other"))), ]
d$y <- log(d$mean_s)
rows <- list()
for (s in c("TEM", "Intervals 1", "Intervals 2", "Sequences", "Vocalisations")) {
  a <- d[d$set == s, ]
  if (!("silent" %in% a$stimulus)) next
  cat("\n==", s, "==\n")
  a$stimulus <- relevel(factor(a$stimulus), ref = "silent")
  m <- fit_pruned("y ~ stimulus + block",
                  c("(1 | mouse) + (1 | session) + (1 | arm)", "(1 | mouse) + (1 | session)",
                    "(1 | mouse) + (1 | arm)", "(1 | mouse)"), a)
  ct <- as.data.frame(summary(contrast(emmeans(m, ~ stimulus), "trt.vs.ctrl", ref = 1,
                                       adjust = "none"), infer = c(TRUE, TRUE)))
  ct$set <- s
  ct$stimulus <- sub(" - silent$", "", ct$contrast)
  ct$p.holm <- p.adjust(ct$p.value, "holm")
  ct$pi <- tanh(ct$estimate / 2)
  ct$pi_low <- tanh(ct$lower.CL / 2)
  ct$pi_high <- tanh(ct$upper.CL / 2)
  ct$ratio <- exp(ct$estimate)
  print(data.frame(stimulus = ct$stimulus, ratio = signif(ct$ratio, 3), pi = signif(ct$pi, 3),
                   pi_low = signif(ct$pi_low, 3), pi_high = signif(ct$pi_high, 3),
                   p = signif(ct$p.value, 3), p.holm = signif(ct$p.holm, 3)), row.names = FALSE)
  rows[[s]] <- ct[, c("set", "stimulus", "estimate", "SE", "df", "ratio", "pi", "pi_low",
                      "pi_high", "p.value", "p.holm")]
}
write.csv(do.call(rbind, rows), file.path(out_dir, "stimulus_lmm.csv"), row.names = FALSE)

# -- 2. variance components of the per-session index ---------------------------
sp <- read.csv(file.path(out_dir, "session_pi.csv"), stringsAsFactors = FALSE)
sp <- sp[is.finite(sp$pi), ]
sp$stim_in_set <- paste(sp$set, sp$stimulus)
cat("\n== Variance of the per-session index of each stimulus ==\n")
cat(nrow(sp), "indices,", length(unique(sp$mouse)), "mice,", length(unique(sp$session)),
    "sessions\n")
mv <- lmer(pi ~ 1 + (1 | set) + (1 | stim_in_set) + (1 | mouse) + (1 | session), data = sp,
           REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
shares <- function(m) {
  v <- as.data.frame(VarCorr(m))
  x <- setNames(v$vcov, v$grp)
  x / sum(x)
}
est <- shares(mv)
bt <- bootMer(mv, shares, nsim = 500, seed = 1)
ci <- apply(bt$t, 2, quantile, c(0.025, 0.975), na.rm = TRUE)
vtab <- data.frame(component = names(est), share = est, low = ci[1, ], high = ci[2, ])
print(transform(vtab, share = round(share, 3), low = round(low, 3), high = round(high, 3)),
      row.names = FALSE)
write.csv(vtab, file.path(out_dir, "variance_components.csv"), row.names = FALSE)

# Every stimulus of a session is set against the same silent arm, so how long the
# mouse stayed on that one arm moves all of the session's indices together. The
# same model with the log of the silent arm's mean visit as a fixed effect shows
# how much of the session's share that one arm carries; the covariate's share is
# the variance of its fitted values.
cat("\n== The same, with the silent arm's own visits taken out ==\n")
sp$silent_z <- as.numeric(scale(sp$log_silent))
ma <- lmer(pi ~ silent_z + (1 | set) + (1 | stim_in_set) + (1 | mouse) + (1 | session), data = sp,
           REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
va <- as.data.frame(VarCorr(ma))
xa <- c(setNames(va$vcov, va$grp), silent_visit = var(as.vector(model.matrix(ma) %*% fixef(ma))))
atab <- data.frame(component = names(xa), share = xa / sum(xa))
print(transform(atab, share = round(share, 3)), row.names = FALSE)
print(summary(ma)$coefficients)
write.csv(atab, file.path(out_dir, "variance_components_adjusted.csv"), row.names = FALSE)

# -- 3. share of variance between mice, class indices ---------------------------
sc <- read.csv(file.path(out_dir, "session_cls.csv"), stringsAsFactors = FALSE)
icc <- function(m) {
  v <- as.data.frame(VarCorr(m))
  v$vcov[v$grp == "mouse"] / sum(v$vcov)
}
irows <- list()
for (col in c("voc_vs_silent", "synth_vs_silent", "voc_vs_synth", "pc1", "pc2")) {
  x <- sc[is.finite(sc[[col]]), ]
  x$y <- x[[col]]
  m <- lmer(y ~ set + (1 | mouse), data = x, REML = TRUE)
  b <- bootMer(m, icc, nsim = 1000, seed = 1)
  q <- quantile(b$t, c(0.025, 0.975), na.rm = TRUE)
  cat(sprintf("\n%s: share of variance between mice %.3f (95%% %.3f to %.3f), %d sessions, %d mice\n",
              col, icc(m), q[1], q[2], nrow(x), length(unique(x$mouse))))
  irows[[col]] <- data.frame(index = col, icc = icc(m), low = q[1], high = q[2],
                             sessions = nrow(x), mice = length(unique(x$mouse)))
}
write.csv(do.call(rbind, irows), file.path(out_dir, "icc.csv"), row.names = FALSE)
sink()
