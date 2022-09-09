# Name: Han Yong Wunrow
# Date: 2022/08/26
# Description: Deterministic SIR and SEIR simulation code


pacman::p_load(deSolve, plotly, data.table)


# Rt scenarios ------------------------------------------------------------

logistic_curve <- function(rt_before, rt_after, n_t, midpoint_day, k) {
  x <- seq(0,n_t)
  return(rt_before + (rt_after - rt_before) / (1 + exp(-k*(x - midpoint_day))))
}

heaviside <- function(rt_before, rt_after, n_t, midpoint_day) {
  x <- seq(0,n_t)
  ifelse(x < midpoint_day, rt_before, rt_after)
}

construct_beta <- function(rt, t_I, n_t) {
  beta_t_all <- rt / t_I
  if(length(rt) == 1) {
    function(t) beta_t_all
  } else {
    approxfun(0:n_t, beta_t_all)
  }
}

simulate_seir_ode <- function(
    rt, t_E, t_I,
    N, S_init, E_init, I_init,
    n_t,
    n_steps_per_t = 1 # Ignored; included so the function signature matches stochastic version
) {
  library(deSolve)
  
  beta <- construct_beta(rt, t_I, n_t)
  d_dt <- function(t, y, params) {
    dS <- y['S'] * beta(t) * y['I'] / N
    dIR <- y['I'] / t_I
    
    if(t_E > 0) {
      # SEIR model
      dEI <- y['E'] / t_E
      list(c(
        S = -dS,
        E = dS - dEI,
        I = dEI - dIR,
        R = dIR,
        cum_dS = dS,
        cum_dEI = dEI
      ), NULL)
    }
    else {
      # SIR model
      list(c(
        S = -dS,
        E = 0,
        I = dS - dIR,
        R = dIR,
        cum_dS = dS,
        cum_dEI = dS
      ), NULL)
    }
  }
  
  y_init <- c(
    S = S_init,
    E = if(t_E > 0) E_init else 0,
    I = if(t_E > 0) I_init else E_init + I_init,
    R = 0,
    cum_dS = 0,
    cum_dEI = 0
  )
  #automatic ode solver is lsoda, an "automatic stiff/non-stiff solver"
  as.data.table(ode(y_init, 0:n_t, d_dt, NULL)) %>%
    mutate(dS = cum_dS - lag(cum_dS, 1)) %>%
    mutate(dEI = cum_dEI - lag(cum_dEI, 1)) %>%
    mutate(dIR = R - lag(R, 1))
}

t_E = 0
t_I = 4 # mean time infected
n_t = 100
N = 1000
E_init = 0
I_init = 50
S_init = 950

rt <- logistic_curve(2.3, 0.6, n_t, 40, 0.5)
rt <- heaviside(2.3, 0.6, n_t, 40)

seir_dt <- simulate_seir_ode(
  rt, t_E, t_I,
  N, S_init, E_init, I_init,
  n_t,
  n_steps_per_t = 1 # Ignored; included so the function signature matches stochastic version
)

long_seir_dt <- melt(seir_dt[,.(time,S,I,R)], id.vars = "time")

ggplot(data.table(day=seq(0,100),rt=logistic_curve(2.3, 0.6, n_t, 40, 0.5)),
       aes(x = day, y = rt)) + geom_line() + geom_point() + theme_bw()

ggplot(data.table(day=seq(0,100),rt=heaviside(2.3, 0.6, 100, 40)),
       aes(x = day, y = rt)) + geom_point() + theme_bw()

g <- ggplot(long_seir_dt, aes(x=time,y=value,color=variable)) + geom_line() + theme_bw()
