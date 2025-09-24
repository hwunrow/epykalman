################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Compare ppd from SIR and renewal equation
################################################################################
rm(list=ls())

pacman::p_load(data.table, ggplot2, EpiEstim)
source("/burg/apam/users/nhw2114/repos/EpiFilter/R files/main/epiFilter.R")
source("/burg/apam/users/nhw2114/repos/EpiFilter/R files/main/recursPredict.R")

# Read in synthetic data
param_num <- 45181
data_dir <- "/burg/apam/users/nhw2114/epyfilter/20231106_synthetic_data/"
in_file <- paste0(data_dir, param_num, "_for_epiestim.csv")
synthetic_dt <- fread(in_file)
synthetic_dt$day <- 1:nrow(synthetic_dt)
nday <- nrow(synthetic_dt)

# run EpiFilter
Iday = as.integer(synthetic_dt$i) # need to convert to integer
Lday = rep(0, nday) 
w <- discr_si(seq(1, nday), 4, 4)
for(i in 2:nday){
  # Total infectiousness
  Lday[i] = sum(Iday[seq(i-1, 1, -1)]*w[1:(i-1)])    
}

Rmin = 0.04; Rmax = 8;
m = 300; pR0 = (1/m)*rep(1, m)
Rgrid = seq(Rmin, Rmax, length.out = m)
a = 0.025; eta <- 0.1

res_epifilter <- epiFilter(Rgrid, m, eta, pR0, nday, Lday, Iday, a)
pR <- res_epifilter$pR

# generate i_ppc (renewal) ------------------------------------------------
maxIgrid <- as.integer(max(synthetic_dt$i)* 1.2)
pred_res <- recursPredict(Rgrid, pR, Lday, res_epifilter$Rmean, a, maxIgrid)
renewal_i_ppc <- pred_res$pred


# generate i_ppc (SIR) ----------------------------------------------------
N <- 100000
S0 <- 99900
I0 <- 100
gamma <- 1/4

free_sim_w <- function() {
  S <- c(S0)
  Ir <- c(I0)
  i <- c(0)
  
  for (t in 1:nday) {
   
    if (t < 10) {
      re <- synthetic_dt[t]$rt * synthetic_dt[t]$prop_S
    } else {
      re <- res_epifilter$Rmean[t]
    }
    
    dSI <- re * gamma * Ir[t]
    dIR <- Ir[t] * gamma
    
    S_new <- min(max(S[t] - dSI, 0), N)
    I_new <- min(max(Ir[t] + dSI - dIR, 0), N)
    
    S <- c(S, S_new)
    Ir <- c(Ir, I_new)
    i <- c(i, dSI)
  }
  
  return(i[2:length(i)])
}

sir_i_ppc <- free_sim_w()



# plot --------------------------------------------------------------------


plot_dt <- data.table(
  day = 1:(nday-1),
  sir = sir_i_ppc[1:(length(sir_i_ppc)-1)],
  truth = synthetic_dt$i[1:(nday-1)],
  renewal = renewal_i_ppc
)

dt_long <- melt(plot_dt,
                id.vars = c("day"),
                measure.vars = c("sir", "truth", "renewal"))

g1 <- ggplot(dt_long, aes(x = day, y = value, color = variable)) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Comparison of SIR and Renewal Model Time Series",
    x = "day",
    y = "daily case counts",
    color = "method"
  ) +
  theme_bw()

ggsave("renewal_vs_sir_pub.png", plot = g1, dpi = 300, width = 7, height = 4.5, units = "in")

g2 <- ggplot(plot_dt, aes(x = renewal, y = sir)) +
  geom_point(alpha = 0.6, color = "dodgerblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") + # Add 1:1 line
  labs(
    title = "SIR Model vs. Renewal Equation Values",
    x = "Renewal Equation Values",
    y = "SIR Model Values"
  ) +
  coord_fixed() +
  theme_bw()

ggsave("scatter_sir_renewal_pub.png", plot = g2, dpi = 300, width = 5, height = 5, units = "in")


