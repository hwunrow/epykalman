# Name: Han Yong Wunrow
# Date: 2022/08/26
# Description: Deterministic SIR and SEIR simulation code


pacman::p_load(deSolve, plotly, data.table)


sir_1 <- function(beta, gamma, S0, I0, R0, times) {
  require(deSolve) # for the "ode" function
  
  # the differential equations:
  sir_equations <- function(time, variables, parameters) {
    with(as.list(c(variables, parameters)), {
      dS <- -beta * I * S
      dI <-  beta * I * S - gamma * I
      dR <-  gamma * I
      return(list(c(dS, dI, dR)))
    })
  }
  
  # the parameters values:
  parameters_values <- c(beta  = beta, gamma = gamma)
  
  # the initial values of variables:
  initial_values <- c(S = S0, I = I0, R = R0)
  
  # solving
  out <- ode(initial_values, times, sir_equations, parameters_values)
  
  # returning the output:
  melt(as.data.table(out),id.vars="time")
}

sir_dt <- sir_1(beta = 0.004, gamma = 0.5, S0 = 999, I0 = 1, R0 = 0, times = seq(0, 10))

p <- ggplot(sir_dt, aes(x=time,y=value,color=variable)) +
  geom_line() +
  geom_point() +
  theme_bw()

ggplotly(p)


beta_list <- c(0.7, 0.001, 0.08, 0.004)
beta_time <- sapply(beta_list,function(beta) rep(beta,3))
beta_t_dt <- data.table(beta=as.vector(beta_time),time=seq(0,11))

ggplot(beta_t_dt, aes(x=time,y=beta)) + geom_line() + theme_bw()


# Rt scenarios ------------------------------------------------------------

logistic_curve <- function(rt_before, rt_after, num_days, midpoint_day, k) {
  x <- seq(0,num_days)
  return(rt_before + (rt_after - rt_before) / (1 + exp(-k*(x - midpoint_day))))
}

heaviside <- function(rt_before, rt_after, num_days, midpoint_day) {
  x <- seq(0,num_days)
  ifelse(x < midpoint_day, rt_before, rt_after)
}

ggplot(data.table(day=seq(0,100),rt=logistic_curve(1.3, 0.6, 100, 20, 0.5)),
       aes(x = day, y = rt)) + geom_line() + geom_point() + theme_bw()

ggplot(data.table(day=seq(0,100),rt=heaviside(1.3, 0.6, 100, 20)),
       aes(x = day, y = rt)) + geom_point() + theme_bw()

