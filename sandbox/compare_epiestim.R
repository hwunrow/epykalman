library(pacman)
pacman::p_load(EpiEstim, data.table, ggplot2)

gamma <- 1/4

setwd("/Users/hwunrow/Documents/GitHub/rt-estimation/sandbox/")
dt <- fread("data_test.csv")
rt <- fread("rt.csv")$V1

eakf_dt <- fread("eakf_rt.csv")
eakf_dt$post_mean <- rowMeans(eakf_dt)

calculate_ci <- function(x, prob) {
  ci <- quantile(x, prob)
  return(ci)
}
eakf_dt[, lower := apply(.SD, 1, calculate_ci, prob=0.025), .SDcols = paste0("ens_",0:299)]
eakf_dt[, upper := apply(.SD, 1, calculate_ci, prob=0.975), .SDcols = paste0("ens_",0:299)]
eakf_dt$day <- 1:nrow(eakf_dt)

dt$day <- 1:nrow(dt)
g0 <- ggplot(dt) + geom_point(aes(x=day, y=dt$incidence)) + theme_bw() + 
  labs(title="Deterministic  Synthetic Data") + xlab("day") + ylab("daily incidence")


# daily window entire time series -----------------------------------------


T <- nrow(dt)
time_window <- 1
t_start <- seq(3, T-(time_window-1))
t_end <- t_start + time_window-1

res <- estimate_R(dt$incidence, method="parametric_si",
                  config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma^2)))
r_dt <- res$R
r_dt$true <- rt[min(r_dt$t_start):max(r_dt$t_start)]
subset_dt <- eakf_dt[min(r_dt$t_start):max(r_dt$t_start)]
r_dt$post_mean <- subset_dt$post_mean
r_dt$lower<- subset_dt$lower
r_dt$upper<- subset_dt$upper

g1 <- ggplot(r_dt) + 
  geom_line(aes(x=t_start, y=`Mean(R)`)) + 
  geom_ribbon(aes(x=t_start, ymin=`Quantile.0.025(R)`, ymax=`Quantile.0.975(R)`), alpha=0.5) +
  geom_line(aes(x=t_start, y=post_mean, color="blue")) +
  geom_ribbon(aes(x=t_start, ymin=lower, ymax=upper), fill="blue", alpha=0.5) +
  geom_line(aes(x=t_start, y=true, color="red")) + 
  scale_color_manual(values = c("blue", "red"), labels = c("eakf", "truth")) +
  theme_bw() + labs(title = "EpiEstim Daily Window - Entire Time Series") + 
  xlab("day") + ylab("R_t")


# longer day  window entire time series -----------------------------------------

plot_list <- list()
for (window in 2:20) {
  print(window)
  T <- nrow(dt)
  time_window <- window
  t_start <- seq(3, T-(time_window-1))
  t_end <- t_start + time_window-1
  
  res <- estimate_R(dt$incidence, method="parametric_si",
                    config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma^2)))
  r_dt <- res$R
  r_dt$true <- rt[min(r_dt$t_start):max(r_dt$t_start)]
  subset_dt <- eakf_dt[min(r_dt$t_start):max(r_dt$t_start)]
  r_dt$post_mean <- subset_dt$post_mean
  r_dt$lower<- subset_dt$lower
  r_dt$upper<- subset_dt$upper
  
  g2 <- ggplot(r_dt) + 
    geom_line(aes(x=t_start, y=`Mean(R)`)) + 
    geom_ribbon(aes(x=t_start, ymin=`Quantile.0.025(R)`, ymax=`Quantile.0.975(R)`), alpha=0.5) +
    geom_line(aes(x=t_start, y=post_mean, color="blue")) +
    geom_ribbon(aes(x=t_start, ymin=lower, ymax=upper), fill="blue", alpha=0.5) +
    geom_line(aes(x=t_start, y=true, color="red")) + 
    scale_color_manual(values = c("blue", "red"), labels = c("eakf", "truth")) +
    theme_bw() + labs(title = paste0("EpiEstim ", window, " Day Window - Entire Time Series")) + 
    xlab("day") + ylab("R_t")
  
  plot_list[[window]] <- g2
}


# split by epidemic -------------------------------------------------------
midpoint <- 190

# first epidemic
T <- midpoint
time_window <- 1
t_start <- seq(3, T-(time_window-1))
t_end <- t_start + time_window-1
res <- estimate_R(dt$incidence[0:190], method="parametric_si",
                  config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma^2)))
r_dt <- res$R
r_dt$true <- rt[min(r_dt$t_start):midpoint]
g3 <- ggplot(r_dt) + geom_line(aes(x=t_start, y=`Mean(R)`)) + 
  geom_ribbon(aes(x=t_start, ymin=`Quantile.0.025(R)`, ymax=`Quantile.0.975(R)`, alpha=0.1)) + 
  geom_line(aes(x=t_start, y=true, color="red")) + 
  scale_color_manual(values = c("red"), labels = c("truth")) + 
  theme_bw() + labs(title = "EpiEstim Daily Window - First Epidemic Curve") + 
  xlab("day") + ylab("R_t")

# second epidemic
T <- nrow(dt)
time_window <- 1
t_start <- seq(midpoint, T-(time_window-1))
t_end <- t_start
res <- estimate_R(dt$incidence, method="parametric_si",
                  config=make_config(list(t_start = t_start, t_end = t_end, mean_si = 1/gamma, std_si = 1/gamma^2)))
r_dt <- res$R
r_dt$true <- rt[midpoint:max(r_dt$t_start)]
g4 <- ggplot(r_dt) + geom_line(aes(x=t_start, y=`Mean(R)`)) + 
  geom_ribbon(aes(x=t_start, ymin=`Quantile.0.025(R)`, ymax=`Quantile.0.975(R)`, alpha=0.1)) + 
  geom_line(aes(x=t_start, y=true, color="red")) + 
  scale_color_manual(values = c("red"), labels = c("truth")) + 
  theme_bw() + labs(title = "EpiEstim Daily Window - Second Epidemic Curve") + 
  xlab("day") + ylab("R_t")


pdf("compare_epiEstim_detSIR.pdf", width = 8, height = 6)
print(g0)
print(g1)
for (g in plot_list) {
  print(g)
}
print(g3)
print(g4)
dev.off()



# list <- c(2,3,4,5,6,7,8,9,10)
# var_list <- c(1,2,3,4,5,6,7,8,9,10)
# rmse_list <- c()
# for (mu in list) {
#   for (sigma in var_list) {
#     res <- estimate_R(dt$incidence, method="parametric_si", config = make_config(list(mean_si = mu, std_si = sigma)))
#     r_dt <- res$R
#     rmse <- sqrt(mean((r_dt$`Mean(R)` - rt[min(r_dt$t_start):max(r_dt$t_start)])^2))
#     rmse_list <- c(rmse_list, rmse) 
#   }
# }
# 
# 
# T <- nrow(dt)
# time_window <- 1
# t_start <- seq(3, T-(time_window-1))
# t_end <- t_start + time_window-1
# 
# res_daily <- estimate_R(dt$incidence, method = "parametric_si",
#                         config = make_config((list(t_start = t_start, t_end = t_end, mean_si = 1, std_si = 0.1))))
# 
# plot(res_daily)
# plot(rt)
