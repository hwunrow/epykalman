# Name: Han Yong Wunrow
# Date: 2022/09/29
# Description: MCMC to learn Rt sigmoid parameters

pacman::p_load(runjags, data.table)


#' Generate logistic curve
#'
#' `logistic_curve()` Generates logistic curve with parameters to shift and
#' change steepness.
#'
#' @param rt_before numeric. Rt value before midpoint
#' @param rt_after numeric. Rt value after midpoint
#' @param midpoint numeric. x value of sigmoid's midpoint
#' @param k numeric. The logistic growth rate or steepness of the curve
#' @param n_t numeric. Integer value for number of days
#' @returns A numeric vector of length n_t
#' @examples
#' logistic_curve(2.9, 1.7, 50, 0.5, 10)
#' @export
#' 
logistic_curve <- function(rt_before, rt_after, midpoint, k, n_t) {
  x <- seq(0,n_t)
  return(rt_before + (rt_after - rt_before) / (1 + exp(-k*(x - midpoint))))
}

simulate_seir_ode_stoch <- function(
    rt, t_E, t_I,
    N, S_init, E_init, I_init,
    n_t
) {
  beta <- construct_beta(rt, t_I, n_t)
  if(t_E > 0) {
    # SEIR model
    S <- c(S_init)
    E <- c(E_init)
    I <- c(I_init)
    R <- N - S - E - I
    for(t in 1:n_t) {
      dSE <- rpois(1, beta(t)*I[t]*S[t]/N)
      dEI <- rpois(1, E[t]/t_E)
      dIR <- rpois(1, I[t]/t_I)
      
      S_new <- min(max(S[t]-dSE,0),N)
      E_new <- min(max(E[t]+dSE-dEI,0),N)
      I_new <- min(max(I[t]+dEI-dIR,0),N)
      R_new <- min(max(R[t]+dIR,0),N)
      
      S <- c(S,S_new)
      E <- c(E,E_new)
      I <- c(I,I_new)
      R <- c(R,R_new)
    }
    seir_dt <- data.table(time=0:n_t,S=S,E=E,I=I,R=R)
    return(seir_dt)
  }
  else {
    # SIR model
    S <- c(S_init)
    I <- c(I_init)
    R <- N - S - I
    dSI_list <- c(0)
    for(t in 1:n_t) {
      dSI <- rpois(1, beta(t)*I[t]*S[t]/N)
      dIR <- rpois(1, I[t]/t_I)
      
      S_new <- min(max(S[t]-dSI,0),N)
      I_new <- min(max(I[t]+dSI-dIR,0),N)
      R_new <- min(max(R[t]+dIR,0),N)
      
      S <- c(S,S_new)
      I <- c(I,I_new)
      R <- c(R,R_new)
      dSI_list <- c(dSI_list, dSI)
    }
    sir_dt <- data.table(time=0:n_t,S=S,I=I,R=R,i=dSI_list)
    return(sir_dt)
  }
}

construct_beta <- function(rt, t_I, n_t) {
  beta_t_all <- rt / t_I
  if(length(rt) == 1) {
    function(t) beta_t_all
  } else {
    approxfun(0:n_t, beta_t_all)
  }
}

# Known
n_t <- 100
t_E <-  0
t_I <- 4 # mean time infected
N <- 100000
E_init <- 0
I_init <- 2000
S_init <- 98000

# Truth
rt_before <- 3.1
rt_after <- 1.3
midpoint <- 30
k <- 0.5

rt <- logistic_curve(rt_before, rt_after, midpoint, k, n_t)

stoch_data <- simulate_seir_ode_stoch(
  rt, t_E, t_I,
  N, S_init, E_init, I_init,
  n_t
)


# deBInfer ----------------------------------------------------------------

library(deBInfer)

sir_model <- function(t, y, params) {
  beta <- construct_beta(rt, t_I, n_t)
  dS <- y['S'] * beta(t) * y['I'] / N
  dIR <- y['I'] / t_I
  # SIR model
  list(c(
    S = -dS,
    I = dS - dIR,
    R = dIR
  ), NULL)
}

obs_model <- function(data, sim.data, samp){
  epsilon <- 1e-6
  llik <- sum(dlnorm(data$i, meanlog = log(sim.data[, "i"] + epsilon),
                     sdlog = samp[["sdlog.N"]],log = TRUE))
  return(llik)
}

r <- debinfer_par(name = "r", var.type = "de", fixed = FALSE,
                  value = 0.5, prior = "norm", hypers = list(mean = 0, sd = 1),
                  prop.var = 0.005, samp.type = "rw")


# JAGS (doesn't really work) ----------------------------------------------



# Priors
set.seed(32)
rt_before_init <- rnorm(1, 3, 0.0025)
rt_after_init <- rnorm(1, 1.5, 0.0025)
midpoint_init <- runif(1, 0, n_t)
k_intit <- runif(1, 0.001, 1)

init_params <- c(rt_before_init, rt_after_init, midpoint_init, k_intit)
print(init_params)

modelString <-"
model {
## sampling
rt <- logistic_curve(rt_before, rt_after, midpoint, k, n_t)
stoch_data <- simulate_seir_ode_stoch(
                rt, t_E, t_I,
                N, S_init, E_init, I_init,
                n_t
              )
sum((stoch_data$i - 4)^2)

## priors
rt_before ~ dnorm(mu0, g0)
rt_after ~ dnorm(mu1, g1)
midpoint <- dunif(a0, b0)
k <- dunif(a1, b1)
}
"
  
data <- list("i" = stoch_data$i, "n_t" = n_t,
             "t_E" = t_E, "t_I" = t_I, "N" = N,
             "S_init" = S_init, "E_init" = E_init, "I_init" = I_init,
             "mu0" = 3, "g0" = 0.0025,
             "mu1" = 1.5, "g1" = 0.0025,
             "a0" = 0, "b0" = n_t,
             "a1" = 0.001, "b1" = 1
             )

posterior <- run.jags(modelString,
                      n.chains = 1,
                      data = data,
                      monitor = c("rt_before", "rt_after", "midpoint", "k"),
                      adapt = 1000,
                      burnin = 5000,
                      sample = 20000)
    