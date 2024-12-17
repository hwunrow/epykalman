################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim to 
################################################################################
rm(list=ls())
.libPaths("/ifs/home/jls106_gp/nhw2114/R/x86_64-pc-linux-gnu-library/4.3/")
library(argparse)
library(data.table)
library(ggplot2)
library(EpiEstim)
library(matrixStats)
library(verification)
library(parallel)
source("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/qsub.R")
source("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/posterior_checks.R")


START_DATE <- "2021-07-01"
END_DATE <- "2022-03-01"
TIME_WINDOW_SIZE <- 7
NYC_FIPS <- 36061
RAW_CASE_URL <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"


df <- fread(RAW_CASE_URL)
df <- df[FIPS == NYC_FIPS]

df_long <- melt(df, 
                id.vars = c('UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 
                            'Country_Region', 'Lat', 'Long_', 'Combined_Key'),
                variable.name = 'Date', 
                value.name = 'Cases')

# Convert Date to date type
df_long[, Date := as.Date(Date, format="%m/%d/%y")]

# Calculate daily cases and moving average
df_long[, Daily_Cases := Cases - shift(Cases, fill = 0), by = FIPS]
df_long[, `7_Day_Avg_Daily_Cases` := frollmean(Daily_Cases, n = 7)]

# Filter between START_DATE and END_DATE
df_long <- df_long[Date >= START_DATE & Date <= END_DATE]


ggplot(df_long, aes(x = Date, y = `7_Day_Avg_Daily_Cases`)) +
  geom_line() + theme_bw() + 
  labs(title = "7 Day Average Daily Cases in NYC - Delta and Omicron Waves",
       x = "Date", 
       y = "Cases")

t <- nrow(df_long)
time_window <- 8L
t_start <- seq(3, t-(time_window-1))
t_end <- t_start + time_window-1
gamma <- 1/3.5

res <- estimate_R(
  df_long[, c("7_Day_Avg_Daily_Cases")] / 0.35,
  method="parametric_si",
  config=make_config(list(
    t_start = t_start,
    t_end = t_end,
    mean_si = 1/gamma,
    std_si = 1/gamma)))
plot(res)
r_dt <- res$R

# sample from posterior
helper_sample_posterior <- function(x, R, n) {
  return(sample_posterior_R(R, n, x))
}

n_samples <- 300L
sample_lists <- lapply(1:nrow(df_long), helper_sample_posterior, R=res, n=n_samples)
R_posterior_dt <- transpose(as.data.table(sample_lists))
samplecols <- paste0("sample",1:n_samples)
names(R_posterior_dt) <- samplecols
R_posterior_dt$day <- r_dt$t_start
R_posterior_dt$window <- time_window
# divide by susceptible to get time-varying Rt
R_posterior_dt <- merge(R_posterior_dt, synthetic_dt, by=c("day"))
R_posterior_dt[ , (samplecols) := lapply(.SD, '/', prop_S), .SDcols = samplecols]




### Split into two waves
# delta
time_window <- 8L
t_start <- seq(2, 140-(time_window-1))
t_end <- t_start + time_window-1
res_before <- estimate_R(
  df_long[, c("7_Day_Avg_Daily_Cases")],
  method="parametric_si",
  config=make_config(list(
    t_start = t_start,
    t_end = t_end,
    mean_si = 1/gamma,
    std_si = 1/gamma)))

plot(res_before, "R")

# omicron
time_window <- 8L
t_start <- seq(141, t-(time_window-1))
t_end <- t_start + time_window-1
res_after <- estimate_R(
  df_long[, c("7_Day_Avg_Daily_Cases")],
  method="parametric_si",
  config=make_config(list(
    t_start = t_start,
    t_end = t_end,
    mean_si = 1/gamma,
    std_si = 1/gamma)))

plot(res_after, "R")



# Run EpiEstim
# for (file in files) {
#   
#   # read in rt, i, and prop_S synthetic data
#   synthetic_dt <- fread(file)
#   synthetic_dt$day <- 1:nrow(synthetic_dt)
#   gamma <- 1/4
#   n_samples <- 300L
#   
#   # run epiestim for varying time-windows
#   res_plot_list <- list()
#   plot_list <- list()
#   dt_list <- list()
#   for (time_window in 1:20) {
#     print(time_window)
#     T <- nrow(synthetic_dt)
#     t_start <- seq(3, T-(time_window-1))
#     t_end <- t_start + time_window-1
#     
#     res <- estimate_R(
#       synthetic_dt$i,
#       method="parametric_si",
#       config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma)))
#     r_dt <- res$R
#     r_dt$true <- synthetic_dt[min(r_dt$t_start):max(r_dt$t_start)]$rt
#     r_dt$sucs <- synthetic_dt[min(r_dt$t_start):max(r_dt$t_start)]$prop_S
#     
#     r_dt$epiestim_mean <- r_dt$`Mean(R)` / r_dt$sucs
#     r_dt$epiestim_lower <- r_dt$`Quantile.0.025(R)` / r_dt$sucs
#     r_dt$epiestim_upper <- r_dt$`Quantile.0.975(R)` / r_dt$sucs
#     
#     res_plot_list[[time_window]] <- plot(res)
#     
#     if (plot) {
#       g1 <- ggplot(r_dt) + 
#         geom_line(aes(x=t_start, y=epiestim_mean)) + 
#         geom_ribbon(aes(x=t_start, ymin=epiestim_lower, ymax=epiestim_upper), alpha=0.5) +
#         geom_line(aes(x=t_start, y=true, color="red")) + 
#         theme_bw() + labs(title = paste0("EpiEstim ", time_window, " Day Window")) + 
#         xlab("day") + ylab("R_t")
#       
#       plot_list[[time_window]] <- g1
#     }
#     
#     # sample from posterior
#     helper_sample_posterior <- function(x, R, n) {
#       return(sample_posterior_R(R, n, x))
#     }
#     
#     sample_lists <- lapply(1:nrow(r_dt), helper_sample_posterior, R=res, n=n_samples)
#     R_posterior_dt <- transpose(as.data.table(sample_lists))
#     samplecols <- paste0("sample",1:n_samples)
#     names(R_posterior_dt) <- samplecols
#     R_posterior_dt$day <- r_dt$t_start
#     R_posterior_dt$window <- time_window
#     # divide by susceptible to get time-varying Rt
#     R_posterior_dt <- merge(R_posterior_dt, synthetic_dt, by=c("day"))
#     R_posterior_dt[ , (samplecols) := lapply(.SD, '/', prop_S), .SDcols = samplecols]
#     
#     dt_list[[time_window]] <- R_posterior_dt
#   }
#   
#   R_posterior_all_dt <- rbindlist(dt_list)
#   # 95% CI
#   percentile <- 95
#   quantiles <- c((1 - percentile/100)/2, 1 - (1 - percentile/100)/2)
#   percentile95_dt <- R_posterior_all_dt[,as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(day,window)]
#   # 50% CI
#   percentile <- 50
#   quantiles <- c((1 - percentile/100)/2, 1 - (1 - percentile/100)/2)
#   percentile50_dt <- R_posterior_all_dt[,as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(day,window)]
#   # mean
#   mean_dt <- R_posterior_all_dt[, .(mean = rowMeans(.SD, na.rm=TRUE)), .SDcols=paste0("sample", 1:300), by=.(day,window)]
#   # median
#   med_dt <- R_posterior_all_dt[, .(med = rowMedians(as.matrix(.SD), na.rm=TRUE)), .SDcols=paste0("sample", 1:300), by=.(day,window)]
#   
#   merge_dt <- merge(percentile95_dt, percentile50_dt, by=c("window","day"))
#   merge_dt <- merge(merge_dt, mean_dt, by=c("window","day"))
#   merge_dt <- merge(merge_dt, med_dt, by=c("window","day"))
#   merge_dt$param <- param_num
# 
# }
