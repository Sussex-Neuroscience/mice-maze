# The division of variance of Section 4.6, refitted with the index on its Fisher z
# scale. Called by fisher_z_check.py, which writes session_pi.csv and
# session_cls.csv with a column "y" holding either the index or its z, and reads
# the shares this prints as CSV on standard output. Point estimates only.
#
#   Rscript fisher_z_check.R <dir>

args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(lme4))

shares <- function(m) {
  v <- as.data.frame(VarCorr(m))
  setNames(v$vcov / sum(v$vcov), v$grp)
}
sp <- read.csv(file.path(dir, "session_pi.csv"), stringsAsFactors = FALSE)
sp$stim_in_set <- paste(sp$set, sp$stimulus)
m <- lmer(y ~ 1 + (1 | set) + (1 | stim_in_set) + (1 | mouse) + (1 | session), data = sp,
          REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
s <- shares(m)
sp$silent_z <- as.numeric(scale(sp$log_silent))
ma <- lmer(y ~ silent_z + (1 | set) + (1 | stim_in_set) + (1 | mouse) + (1 | session), data = sp,
           REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
# as in exp1_individuals.R, the silent arm's share is the variance of its fitted values
va <- as.data.frame(VarCorr(ma))
xa <- c(setNames(va$vcov, va$grp), silent_visit = var(as.vector(model.matrix(ma) %*% fixef(ma))))
sa <- xa / sum(xa)
cat("measure,value\n")
for (k in names(s)) cat(sprintf("share %s,%.4f\n", k, s[[k]]))
for (k in names(sa)) cat(sprintf("share without the silent arm %s,%.4f\n", k, sa[[k]]))

sc <- read.csv(file.path(dir, "session_cls.csv"), stringsAsFactors = FALSE)
for (col in c("voc_vs_silent", "synth_vs_silent", "voc_vs_synth")) {
  x <- sc[is.finite(sc[[col]]), ]
  x$y <- x[[col]]
  mm <- lmer(y ~ set + (1 | mouse), data = x, REML = TRUE)
  v <- as.data.frame(VarCorr(mm))
  cat(sprintf("between mice %s,%.4f\n", col, v$vcov[v$grp == "mouse"] / sum(v$vcov)))
}
