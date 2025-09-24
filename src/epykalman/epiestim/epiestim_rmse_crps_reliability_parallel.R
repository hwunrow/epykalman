################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim to compute rmse, crps, and reliabiltiy time series on C2B2
################################################################################
rm(list=ls())
.libPaths("/burg/home/nhw2114/R/x86_64-pc-linux-gnu-library/4.3/")
library(argparse)
library(data.table)
library(ggplot2)
library(EpiEstim)
library(matrixStats)
library(verification)
library(parallel)
# source("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/qsub.R")
source("/burg/apam/users/nhw2114/repos/epyfilter/src/epyfilter/utils/posterior_checks.R")


# Get arguments from parser
parser <- ArgumentParser()
parser$add_argument("--in-dir", help = "directory for external inputs", default = "/burg/apam/users/nhw2114/repos/epyfilter/src/epyfilter/params/", type = "character")
parser$add_argument("--data-dir", help = "directory for synthetic inputs", default = "/burg/apam/users/nhw2114/epyfilter/20231106_synthetic_data/", type = "character")
parser$add_argument("--out-dir", help = "directory for this steps checks", default = "/burg/apam/users/nhw2114/epyfilter/20250515_all_windows_epiestim/", type = "character")
parser$add_argument("--files-per-task", help = "number of files per array job", default = 10, type = "integer")
parser$add_argument("--param-list", nargs='+', help = "rerun for specific params", type = "integer")
parser$add_argument("--param-file", help = "file with list of parameters to run for array job", default = "good_param_list.csv", type = "character")
parser$add_argument("--plot", action="store_true", help="Plot results.")

args <- parser$parse_args()
print(args)
list2env(args, environment()); rm(args)

# Get params from parameter map
task_id <- as.integer(Sys.getenv("SGE_TASK_ID"))
print(task_id)
if (is.na(task_id)) {
  task_id <- 1
}
dt <- fread(file.path(out_dir, param_file))
start_row <- (task_id - 1) * files_per_task + 1
end_row <- task_id * files_per_task
if (end_row < nrow(dt)) {
  pickle_files <- dt[start_row:end_row]
} else {
  pickle_files <- dt[start_row:nrow(dt)]
}

if (length(param_list) != 0) {
  pickle_files <- dt[param %in% param_list]
  print(pickle_files)
}

files <- paste0(data_dir, pickle_files$param, "_for_epiestim.csv")
last_epidemic_days = fread(paste0(in_dir,"compute_days.csv"))

for (file in files) {
  param_num <- strsplit(basename(file), "_")[[1]][1]
  last_epi_day <- last_epidemic_days[param == param_num]$last_epi_day
  first_epi_day <- last_epidemic_days[param == param_num]$first_epi_day
  late_day <- last_epidemic_days[param == param_num]$late_day
  peaks <- c(last_epidemic_days[param == param_num]$peak1, last_epidemic_days[param == param_num]$peak2)
  
  # read in data distribution, late_day, and peaks used for posterior checks
  data_dt <- fread(paste0(data_dir, param_num, "_data_distribution.csv"))
  
  # read in rt, i, and prop_S synthetic data
  synthetic_dt <- fread(file)
  synthetic_dt$day <- 1:nrow(synthetic_dt)
  gamma <- 1/4
  n_samples <- 300L
  time_window <- 8L
  
  # run epiestim
  T <- nrow(synthetic_dt)
  t_start <- seq(3, T-(time_window-1))
  t_end <- t_start + time_window-1
  
  res <- estimate_R(
    synthetic_dt$i,
    method="parametric_si",
    config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma)))
  r_dt <- res$R
  r_dt$true <- synthetic_dt[min(r_dt$t_start):max(r_dt$t_start)]$rt
  r_dt$sucs <- synthetic_dt[min(r_dt$t_start):max(r_dt$t_start)]$prop_S
  
  r_dt$epiestim_mean <- r_dt$`Mean(R)` / r_dt$sucs
  r_dt$epiestim_lower <- r_dt$`Quantile.0.025(R)` / r_dt$sucs
  r_dt$epiestim_upper <- r_dt$`Quantile.0.975(R)` / r_dt$sucs
  
  # sample from posterior
  helper_sample_posterior <- function(x, R, n) {
    return(sample_posterior_R(R, n, x))
  }
  
  sample_lists <- lapply(1:nrow(r_dt), helper_sample_posterior, R=res, n=n_samples)
  R_posterior_dt <- transpose(as.data.table(sample_lists))
  samplecols <- paste0("sample",1:n_samples)
  names(R_posterior_dt) <- samplecols
  R_posterior_dt$day <- r_dt$t_start
  R_posterior_dt$window <- time_window
  # divide by susceptible to get time-varying Rt
  R_posterior_dt <- merge(R_posterior_dt, synthetic_dt, by=c("day"))
  R_posterior_dt[ , (samplecols) := lapply(.SD, '/', prop_S), .SDcols = samplecols]
  
  R_posterior_all_dt <- R_posterior_dt
  
  # posterior checks
  data_rmse_result <- data_rmse(R_posterior_all_dt, synthetic_dt)
  rmse_dt <- data_rmse_result$rmse_dt
  i_ppc <- data_rmse_result$i_ppc

  # hack since we did not loop over different window sizes
  row <- R_posterior_all_dt[1]
  row$window <- 20
  row$day <- 358
  R_posterior_all_dt <- rbind(R_posterior_all_dt, row)
  
  # rmse
  rmse_dt <- rbindlist(lapply(unique(R_posterior_all_dt$day), function(d) {
    rt_rmse(R_posterior_all_dt, d, evaluate_on=TRUE, colname="rmse")
  }))
  rmse_dt <- rmse_dt[rmse_dt$window == 8]
  rmse_dt$day <- unique(R_posterior_all_dt$day)
  
  # reliability
  in_ci_dt <- rbindlist(lapply(unique(R_posterior_all_dt$day), function(d) {
    check_param_in_ci(R_posterior_all_dt, d, colname="in_ci")
  }))
  in_ci_dt <- in_ci_dt[in_ci_dt$window == 8]
  in_ci_dt$day <- unique(R_posterior_all_dt$day)
  
  # crps
  i_ppc <- merge(i_ppc, synthetic_dt[, .(day, i)], by="day", how="left")
  crps_dt <- rbindlist(lapply(unique(i_ppc$day), function(d) {
    compute_crps(i_ppc, d, colname="crps", ww=8L, evaluate_on=TRUE)
  }))
  crps_dt <- crps_dt[crps_dt$window == 8]
  crps_dt$day <- sort(unique(i_ppc$day))
  
  post_checks_dt <- merge(rmse_dt, in_ci_dt, by=c("window", "day"))
  post_checks_dt <- merge(post_checks_dt, crps_dt, by=c("window", "day"))
  
  post_checks_dt$param <- param_num
  fwrite(post_checks_dt, file=paste0(out_dir,"/",param_num, "_epiEstim_rmse_crps_reliability.csv"))
  
  print(paste(format(Sys.time(), "%c"), "--", param_num, "finished"))
}
print("DONE!")