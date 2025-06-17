################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiFilter on Ginsburg
################################################################################
rm(list=ls())
.libPaths("/burg/home/nhw2114/R/x86_64-pc-linux-gnu-library/4.3/")
library(EpiEstim)
library(argparse)
library(data.table)
library(ggplot2)
source("/burg/apam/users/nhw2114/repos/EpiFilter/R files/main/epiFilter.R")
source("/burg/apam/users/nhw2114/repos/EpiFilter/R files/main/recursPredict.R")
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
parser$add_argument("--eta", help="Noise param in epiFilter.", default = 0.1, type = "numeric")


args <- parser$parse_args()
print(args)
list2env(args, environment()); rm(args)

# check for SLURM
task_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(task_id)
if (is.na(task_id)) {
  task_id <- 1
}

dt <- fread(file.path(in_dir, param_file))
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

# EpiFilter parameters (eta read in from argparse)
Rmin = 0.04; Rmax = 8;
# Uniform prior over grid of size m
m = 300; pR0 = (1/m)*rep(1, m)
# Delimited grid defining space of R
Rgrid = seq(Rmin, Rmax, length.out = m)

nday = 365
w = discr_si(seq(1, nday), 4, 4)

for (file in files) {
  param_num <- strsplit(basename(file), "_")[[1]][1]
  # read in data distribution and eval days used for posterior checks
  last_epi_day <- last_epidemic_days[param == param_num]$last_epi_day
  first_epi_day <- last_epidemic_days[param == param_num]$first_epi_day
  data_dt <- fread(paste0(data_dir, param_num, "_data_distribution.csv"))

  # Read in synthetic data
  synthetic_dt <- fread(file)
  synthetic_dt$day <- 1:nrow(synthetic_dt)
  Iday = as.integer(synthetic_dt$i) # need to convert to integer
  Lday = rep(0, nday) 
  for(i in 2:nday){
    # Total infectiousness
    Lday[i] = sum(Iday[seq(i-1, 1, -1)]*w[1:(i-1)])    
  }
  a = 0.025
  
  # Output - mean (Rmean), median (Rmed), 50% and 95% quantiles of estimates (Rhat),
  # causal posterior over R (pR), pre-update (pRup) and state transition matrix (pstate)
  res_epifilter <- epiFilter(Rgrid, m, eta, pR0, nday, Lday, Iday, a)
  
  if (param_num == "45181") {
    # param used for example plots
    plot_dt <- data.table(
      day = 1:365,
      lower = res_epifilter$Rhat[1,],
      upper = res_epifilter$Rhat[2,],
      mean = res_epifilter$Rmean,
      susc = synthetic_dt$prop_S
    )
    
    plot_dt$lower <- plot_dt$lower / plot_dt$susc
    plot_dt$upper <- plot_dt$upper / plot_dt$susc
    plot_dt$mean <- plot_dt$mean / plot_dt$susc
    
    plot_dir <- "/burg/apam/users/nhw2114/repos/epyfilter/src/epyfilter/plot/thesis_proposal_figs/"
    fwrite(plot_dt, paste0(plot_dir, param_num, "_epiFilter_for_plot.csv"))
  }
  
  pR <- res_epifilter$pR
  # sample realizations
  realizations <- matrix(0, nrow = nday, ncol = m)
  for (k in 1:m) {
    for (t in 1:nday) {
      realizations[t, k] <- sample(Rgrid, size = 1, replace = TRUE, prob = pR[t, ])
    }
  }
  
  prop_S <- synthetic_dt$prop_S
  pR0_dt <- data.table(realizations)
  samplecols <- paste0("sample",1:m)
  setnames(pR0_dt, samplecols)
  pR0_dt[ , (samplecols) := lapply(.SD, '/', prop_S), .SDcols = samplecols]
  
  pR0_dt$day <- 1:nrow(pR0_dt)
  pR0_dt$rt <- synthetic_dt$rt
  pR0_dt$i <- as.integer(synthetic_dt$i)
  pR0_dt$prop_S <- synthetic_dt$prop_S
  pR0_dt$window <- 20
  
  
  # # plotting (delete later)
  # library(tidyr)
  # library(dplyr)
  # pR0_dt_long <- pR0_dt %>%
  #   pivot_longer(cols = starts_with("sample"),
  #                names_to = "sample_id",  
  #                values_to = "sample_value")
  # mean_dt <- pR0_dt_long %>%
  #   group_by(day) %>%
  #   summarise(mean_value = mean(sample_value, na.rm = TRUE))
  # 
  # ggplot(pR0_dt_long, aes(x = day)) +
  #   # Plot individual sample lines in light gray
  #   geom_line(aes(y = sample_value, group = sample_id), color = "lightgray", alpha = 0.7) +
  #   # Plot the mean line in black and bold (thicker)
  #   geom_line(data = mean_dt, aes(y = mean_value), color = "black", linewidth = 1.2) +
  #   # Plot the true 'rt' line in red
  #   geom_line(data = pR0_dt, aes(y = rt), color = "red", linewidth = 1) +
  #   # Add labels and a title (optional but good practice)
  #   labs(title = "Sample Trajectories, Mean, and Truth (rt)",
  #        x = "Day",
  #        y = "Value") +
  #   theme_minimal()
  
  # generate i_ppc
  maxIgrid <- as.integer(max(synthetic_dt$i)* 1.2)
  pred_res <- recursPredict(Rgrid, pR, Lday, res_epifilter$Rmean, a, maxIgrid)
  # sample realizations
  Igrid <- 0:maxIgrid
  realizationsI <- matrix(0, nrow = nday, ncol = m)
  pI <- pred_res$pI_list
  for (k in 1:m) {
    for (t in 1:(nday-1)) {
      realizationsI[t, k] <- sample(Igrid, size = 1, replace = TRUE, prob = pI[t, ])
    }
  }
  realizationsI[nday,] <- as.integer(synthetic_dt$i[nday])
  i_ppc <- data.table(realizationsI)
  names(i_ppc) <- samplecols
  i_ppc$day <- 1:nday
  i_ppc$window <- 20
  
  # plot i_ppc
  # i_ppc_long <- melt(i_ppc, id.vars = "day", measure.vars = samplecols,
  #                 variable.name = "sample", value.name = "value")
  # ggplot(i_ppc_long) +
  #   geom_line(aes(x = day, y = value, group=sample, alpha=0.01)) +
  #   theme_minimal()
  
  if (param_num == "45181") {
    # param used for example plots
    fwrite(i_ppc, paste0(plot_dir, param_num, "_epiFilter_i_ppc.csv"))
  }
  
  print("posterior checks")
  post_checks_dt <- Reduce(function(x, y) merge(x, y, by = "window"), list(
    rt_rmse(pR0_dt, first_epi_day, evaluate_on=FALSE, colname="rt_rmse_up_to_first_epi_day"),
    avg_wasserstein2(pR0_dt, synthetic_dt, data_dt, i_ppc, dd=first_epi_day, evaluate_on=FALSE, colname="avg_w2_up_to_first_epi_day"),
    avg_kl_divergence(pR0_dt, synthetic_dt, data_dt, i_ppc, dd=first_epi_day, evaluate_on=FALSE, colname="avg_kl_up_to_first_epi_day"),
    rt_rmse(pR0_dt, last_epi_day, evaluate_on=FALSE, colname="rt_rmse_up_to_last_epi_day"),
    avg_wasserstein2(pR0_dt, synthetic_dt, data_dt, i_ppc, dd=last_epi_day, evaluate_on=FALSE, colname="avg_w2_up_to_last_epi_day"),
    avg_kl_divergence(pR0_dt, synthetic_dt, data_dt, i_ppc, dd=last_epi_day, evaluate_on=FALSE, colname="avg_kl_up_to_last_epi_day")
  ))
  # crps
  i_ppc <- merge(i_ppc, synthetic_dt[, .(day, i)], by="day", all.x=TRUE)
  i_ppc <- i_ppc[!is.na(i)]
  crps_dt <- rbindlist(lapply(unique(i_ppc$day), function(d) {
    compute_crps(i_ppc, d, colname="crps", ww=20L, evaluate_on=TRUE)
  }))
  crps_dt$day <- sort(unique(i_ppc$day))
  crps_up_to_first_epi_day <- mean(crps_dt[day <= first_epi_day, crps], na.rm=TRUE)
  crps_up_to_last_epi_day <- mean(crps_dt[day <= last_epi_day, crps], na.rm=TRUE)
  
  post_checks_dt$crps_up_to_first_epi_day <- crps_up_to_first_epi_day
  post_checks_dt$crps_up_to_last_epi_day <- crps_up_to_last_epi_day
  
  post_checks_dt$param <- param_num
  
  # fwrite(post_checks_dt, file=paste0(out_dir,"/",param_num, "_epiFilter.csv"))
  
  print(paste(format(Sys.time(), "%c"), "--", param_num, "finished"))
  
}