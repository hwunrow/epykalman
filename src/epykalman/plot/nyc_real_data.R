################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim on empirial data
################################################################################
rm(list=ls())
library(argparse)
library(data.table)
library(ggplot2)
library(EpiEstim)
library(matrixStats)

START_DATE <- "2021-12-05"
END_DATE <- "2022-07-01"
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
alpha <- 0.28365230839069

res <- estimate_R(
  df_long[, c("7_Day_Avg_Daily_Cases")] / alpha,
  method="parametric_si",
  config=make_config(list(
    t_start = t_start,
    t_end = t_end,
    mean_si = 1/gamma,
    std_si = 1/gamma)))
plot(res)
r_dt <- res$R
fwrite(r_dt, "/Users/hwunrow/Documents/GitHub/epyfilter/src/epyfilter/plot/nyc_real_data_epiestim_posterior.csv")

# sample from posterior
helper_sample_posterior <- function(x, R, n) {
  return(sample_posterior_R(R, n, x))
}

n_samples <- 300L
sample_lists <- lapply(1:nrow(r_dt), helper_sample_posterior, R=res, n=n_samples)
R_posterior_dt <- transpose(as.data.table(sample_lists))
samplecols <- paste0("sample",1:n_samples)
names(R_posterior_dt) <- samplecols
R_posterior_dt$day <- r_dt$t_start
R_posterior_dt$window <- time_window

# plot posterior draws and mean
library(tidyr)
library(dplyr)

R_posterior_long <- R_posterior_dt %>% 
  pivot_longer(cols = starts_with("sample"),
               names_to = "sample",
               values_to = "value")
R_posterior_mean <- R_posterior_long %>% 
  group_by(day) %>% 
  summarize(mean_value = mean(value))

ggplot() +
  geom_line(data = R_posterior_long, aes(x = day, y = value, group = sample), color = "lightgray") +
  geom_line(data = R_posterior_mean, aes(x = day, y = mean_value), color = "black") +
  geom_line(data = r_dt, aes(x=t_start, y=`Mean(R)`), linetype = "dotdash", color="red")
  theme_minimal() 

free_sim <- function(rt_dt, burn_in=3) {
  m <- 300
  N <- 1.596*10^6
  S0 <- 0.33 * N
  I0 <- 2000
  
  samplecols <- paste0("sample", 1:300)
  free_sim_w <- function() {
    S <- matrix(rep(S0,m), nrow = 1, ncol = m)
    Ir <- matrix(rep(I0,m), nrow = 1, ncol = m)
    i <- matrix(rep(0, m), nrow = 1, ncol = m)
    
    for (t in 1:max(rt_dt$day)) {
      if (t < burn_in) {
        re <- as.matrix(rt_dt[day == burn_in, ..samplecols])
        dSI <- rpois(m, re * gamma * Ir[t,])
      } else {
        re <- as.matrix(rt_dt[day == t, ..samplecols])
        dSI <- rpois(m, re * gamma * Ir[t,])
      }
      dIR <- rpois(m, Ir[t,] * gamma)
      
      S_new <- pmin(pmax(S[t, ] - dSI, 0), N)
      I_new <- pmin(pmax(Ir[t, ] + dSI - dIR, 0), N)
      
      S <- rbind(S, S_new)
      Ir <- rbind(Ir, I_new)
      i <- rbind(i, dSI)
    }
    tmp_dt <- data.table(i)
    colnames(tmp_dt) <- samplecols
    tmp_dt$day <- 0:max(rt_dt$day)
    
    return(tmp_dt)
  }
  
  i_ppc <- free_sim_w()
  
  return(i_ppc)
}

# df_long$day <- 1:nrow(df_long)
# df_long[, cases_scaled := `7_Day_Avg_Daily_Cases` / alpha]

# pdf("epiestim.pdf")
# for (day in 3:30) {
#   i_ppc <- free_sim(R_posterior_dt, burn_in=day)
#   g1 <- ggplot() + 
#     geom_line(data = i_ppc, aes(x=day, y=sample1)) +
#     geom_point(data = df_long, aes(x=day, y=cases_scaled), size=2, shape = "x", color="red") + 
#     labs(title=paste("burn in days", day)) + 
#     xlab("day") + ylab("cases")
#   
#   print(g1)
# }
# dev.off()

i_ppc <- free_sim(R_posterior_dt, burn_in=11)

# save posterior
fwrite(i_ppc, "/Users/hwunrow/Documents/GitHub/epyfilter/src/epyfilter/plot/nyc_real_data_epiestim_ippc.csv")

