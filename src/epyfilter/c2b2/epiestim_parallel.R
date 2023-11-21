################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim on C2B2
################################################################################
rm(list=ls())
.libPaths("/ifs/home/jls106_gp/nhw2114/R/x86_64-pc-linux-gnu-library/4.3/")
library(argparse)
library(data.table)
library(ggplot2)
library(EpiEstim)
library(matrixStats)
source("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/qsub.R")
source("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/posterior_checks.R")

# Get arguments from parser
parser <- ArgumentParser()
parser$add_argument("--in-dir", help = "directory for external inputs", default = "/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/", type = "character")
parser$add_argument("--data-dir", help = "directory for synthetic inputs", default = "/ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data/", type = "character")
parser$add_argument("--out-dir", help = "directory for this steps checks", default = "/ifs/scratch/jls106_gp/nhw2114/data/20231106_synthetic_data/", type = "character")
parser$add_argument("--files-per-task", help = "number of files per array job", default = 10, type = "integer")

args <- parser$parse_args()
print(args)
list2env(args, environment()); rm(args)

# Get params from parameter map
task_id <- as.integer(Sys.getenv("SGE_TASK_ID"))
print(task_id)
if (is.na(task_id)) {
  task_id <- 1
}
dt <- fread(file.path(in_dir, "good_param_list.csv"))
start_row <- (task_id - 1) * files_per_task + 1
end_row <- task_id * files_per_task
if (end_row < nrow(dt)) {
  pickle_files <- dt[start_row:end_row]
} else {
  pickle_files <- dt[start_row:nrow(dt)]
}
files <- paste0(data_dir, pickle_files$param, "_for_epiestim.csv")

last_epidemic_days = fread(paste0(in_dir,"last_epidemic_day.csv"))

for (file in files) {
  param_num <- strsplit(basename(file), "_")[[1]][1]
  last_epi_day <- last_epidemic_days[param == param_num]$last_epidemic_day
  
  # read in data distribution, late_day, and peaks used for posterior checks
  data_dt <- fread(paste0(out_dir, param_num, "_data_distribution.csv"))
  late_day <- unique(data_dt$late_day)
  peaks <- c(unique(data_dt$peak1), unique(data_dt$peak2))
  
  # read in rt, i, and prop_S synthetic data
  synthetic_dt <- fread(file)
  synthetic_dt$day <- 1:nrow(synthetic_dt)
  gamma <- 1/4
  n_samples <- 300L
  
  # run epiestim for varying time-windows
  res_plot_list <- list()
  plot_list <- list()
  dt_list <- list()
  for (time_window in 1:20) {
    print(time_window)
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
    
    res_plot_list[[time_window]] <- plot(res)
    
    g1 <- ggplot(r_dt) + 
      geom_line(aes(x=t_start, y=epiestim_mean)) + 
      geom_ribbon(aes(x=t_start, ymin=epiestim_lower, ymax=epiestim_upper), alpha=0.5) +
      geom_line(aes(x=t_start, y=true, color="red")) + 
      theme_bw() + labs(title = paste0("EpiEstim ", time_window, " Day Window")) + 
      xlab("day") + ylab("R_t")
    
    plot_list[[time_window]] <- g1
    
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
    
    dt_list[[time_window]] <- R_posterior_dt
  }
  
  R_posterior_all_dt <- rbindlist(dt_list)
  # 95% CI
  percentile <- 95
  quantiles <- c((1 - percentile/100)/2, 1 - (1 - percentile/100)/2)
  percentile95_dt <- R_posterior_all_dt[,as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(day,window)]
  # 50% CI
  percentile <- 50
  quantiles <- c((1 - percentile/100)/2, 1 - (1 - percentile/100)/2)
  percentile50_dt <- R_posterior_all_dt[,as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(day,window)]
  # mean
  mean_dt <- R_posterior_all_dt[, .(mean = rowMeans(.SD, na.rm=TRUE)), .SDcols=paste0("sample", 1:300), by=.(day,window)]
  # median
  med_dt <- R_posterior_all_dt[, .(med = rowMedians(as.matrix(.SD), na.rm=TRUE)), .SDcols=paste0("sample", 1:300), by=.(day,window)]
  
  merge_dt <- merge(percentile95_dt, percentile50_dt, by=c("window","day"))
  merge_dt <- merge(merge_dt, mean_dt, by=c("window","day"))
  merge_dt <- merge(merge_dt, med_dt, by=c("window","day"))
  fwrite(merge_dt, file=paste0(out_dir,"/",param_num, "_epiEstim_for_plot.csv"))
  
  # posterior checks
  data_rmse_result <- data_rmse(R_posterior_all_dt, synthetic_dt)
  rmse_dt <- data_rmse_result$rmse_dt
  i_ppc <- data_rmse_result$i_ppc
  
  post_checks_dt <- Reduce(function(x, y) merge(x, y, by = "window"), list(
    rt_rmse(R_posterior_all_dt, peaks),
    rt_rmse(R_posterior_all_dt),
    rt_rmse(R_posterior_all_dt, last_epi_day, last_epi=TRUE),
    rmse_dt,
    data_rmse(R_posterior_all_dt, synthetic_dt, last_epi_day, last_epi=TRUE),
    avg_wasserstein2(R_posterior_all_dt, synthetic_dt, data_dt, i_ppc),
    avg_kl_divergence(R_posterior_all_dt, synthetic_dt, data_dt, i_ppc),
    avg_wasserstein2(R_posterior_all_dt, synthetic_dt, data_dt, i_ppc, last_epi=TRUE, last_epi_day=last_epi_day),
    avg_kl_divergence(R_posterior_all_dt, synthetic_dt, data_dt, i_ppc, last_epi=TRUE, last_epi_day=last_epi_day),
    check_param_in_ci(R_posterior_all_dt, late_day),
    compute_ens_var(R_posterior_all_dt, late_day),
    check_param_in_ci(R_posterior_all_dt, "last"),
    compute_ens_var(R_posterior_all_dt, "last"),
    check_param_in_ci(R_posterior_all_dt, last_epi_day, last_epi=TRUE),
    compute_ens_var(R_posterior_all_dt, last_epi_day, last_epi=TRUE)
  ))
  fwrite(post_checks_dt, file=paste0(out_dir,"/",param_num, "_epiEstim_metrics.csv"))
  
  pdf(paste0(out_dir,"/",param_num,"_epiEsim_plots.pdf"), width = 8, height = 6)
  for (i in 1:20) {
    print(plot_list[[i]])
    plot(res_plot_list[[i]])
  }
  dev.off()
}

