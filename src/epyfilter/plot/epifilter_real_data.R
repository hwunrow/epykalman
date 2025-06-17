################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiFilter on empirial data
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


# Load data ---------------------------------------------------------------


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

df_long[, Date := as.Date(Date, format="%m/%d/%y")]
df_long[, Daily_Cases := Cases - shift(Cases, fill = 0), by = FIPS]
df_long[, avg_cases := frollmean(Daily_Cases, n = 7)]
df_long <- df_long[Date >= START_DATE & Date <= END_DATE]


# EpiFilter ---------------------------------------------------------------

# EpiFilter params
Rmin = 0.03; Rmax = 3;
# Uniform prior over grid of size m
m = 300; pR0 = (1/m)*rep(1, m)
# Delimited grid defining space of R
Rgrid = seq(Rmin, Rmax, length.out = m)

D <- 3.5
alpha <- 0.28365230839069

Iday = as.integer(df_long$avg_cases / alpha) # need to convert to integer
nday <- length(Iday)
w = discr_si(seq(1, nday), D, D)
Lday = rep(0, nday) 
for(i in 2:nday){
  # Total infectiousness
  Lday[i] = sum(Iday[seq(i-1, 1, -1)]*w[1:(i-1)])    
}

a = 0.025
eta = 0.1
res_epifilter <- epiFilter(Rgrid, m, eta, pR0, nday, Lday, Iday, a)

# save Rt posterior
plot_dt <- data.table(
  day = 1:nday,
  lower = res_epifilter$Rhat[1,],
  upper = res_epifilter$Rhat[2,],
  mean = res_epifilter$Rmean
)

ggplot(plot_dt) + geom_line(aes(day, mean)) + 
  geom_ribbon(aes(x=day, ymin=lower, ymax=upper), color='red', alpha=0.025)

plot_dir <- "/burg/apam/users/nhw2114/repos/epyfilter/src/epyfilter/plot/thesis_proposal_figs/"
fwrite(plot_dt, paste0(plot_dir, "nyc_real_data_epifilter_posterior.csv"))


# Generate ippc -----------------------------------------------------------

maxIgrid <- as.integer(max(Iday)*1.2)
pred_res <- recursPredict(Rgrid, res_epifilter$pR, Lday, res_epifilter$Rmean, a, maxIgrid)

# sample realizations
Igrid <- 0:maxIgrid
realizationsI <- matrix(0, nrow = nday, ncol = m)
pI <- pred_res$pI_list
for (k in 1:m) {
  for (t in 2:nday) {
    realizationsI[t, k] <- sample(Igrid, size = 1, replace = TRUE, prob = pI[t-1, ])
  }
}
realizationsI[0,] <- as.integer(Iday[1])
i_ppc <- data.table(realizationsI)
samplecols <- paste0("sample",1:m)
names(i_ppc) <- samplecols
i_ppc$day <- 1:nday

fwrite(i_ppc,  paste0(plot_dir, "nyc_real_data_epifilter_ippc.csv"))



