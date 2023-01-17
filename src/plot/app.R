# Name: Han Yong Wunrow
# Date: 9/15/2022
# Description: Various plots to 

library(shiny)
source("~/GitHub/rt-estimation/src/sir_seir_simulation.R")

ui <- fluidPage(

    # Application title
    titlePanel("Rt Estimation"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          radioButtons(
            "fun",
            h3("Functional form"),
            choices = list(
              "Logistic" = 1,
              "Heaviside" = 2
            ),
            selected = 1
          ),
          sliderInput(
            "k",
            h3("k"),
            min = 0,
            max = 3,
            value = 0.5,
            step = 0.1
          ), 
          numericInput(
            "rt_before",
            h3("Rt before"),
            value = 2.8
          ),
          numericInput(
            "rt_after",
            h3("Rt after"),
            value = 1.3
          ),
          numericInput(
            "mid",
            h3("Midpoint"),
            value = 100
          ),
          numericInput(
            "n_t",
            h3("Number of days"),
            value = 300
          ),
          numericInput(
            "t_E",
            h3("Mean latent period (1/sigma)"),
            value = 0
          ),
          numericInput(
            "t_I",
            h3("Mean duration of infectiousness (1/gamma)"),
            value = 4
          ),
          numericInput(
            "N",
            h3("N"),
            value = 2e6
          ),
          numericInput(
            "S_init",
            h3("Initial S"),
            value = 1999940
          ),
          numericInput(
            "E_init",
            h3("Initial E"),
            value = 0
          ),
          numericInput(
            "I_init",
            h3("Initial I"),
            value = 60
          ),
          downloadLink('downloadSIRData', 'Download SIR'),
          downloadLink('downloadRtData', 'Download Rt')
        ),

        # Show a plot of the generated distribution
        mainPanel(
          plotlyOutput("rtPlot"),
          plotlyOutput("simPlot")
        )
    )
)

server <- function(input, output) {
    rtData <- reactive({
      if (input$fun == 1) {
        rt <- logistic_curve(input$rt_before, input$rt_after, input$n_t, input$mid, input$k)
      } else {
        rt <- heaviside(input$rt_before, input$rt_after, input$n_t, input$mid)
      }
      rt_dt <- data.table(time=seq(0,input$n_t), rt=rt)
      return(rt_dt)
    })
    
    output$rtPlot <- renderPlotly({
      g <- ggplot(rtData(), aes(x=time, y=rt)) + geom_line() + geom_point() + theme_bw()
      ggplotly(g)
    })
    
    sirData <- reactive({
      if (input$fun == 1) {
        rt <- logistic_curve(input$rt_before, input$rt_after, input$n_t,
                             input$mid, input$k)
      } else {
        rt <- heaviside(input$rt_before, input$rt_after, input$n_t, input$mid)
      }
      
      seir_dt <- simulate_seir_ode_det(
        rt, input$t_E, input$t_I,
        input$N, input$S_init, input$E_init, input$I_init,
        input$n_t
      )
      
      seir_stoch_dt <- simulate_seir_ode_stoch(
        rt, input$t_E, input$t_I,
        input$N, input$S_init, input$E_init, input$I_init,
        input$n_t
      )
      
      names(seir_stoch_dt) <- c("time", "S_stoch","I_stoch","R_stoch","i")
      merge_dt <- merge(seir_dt, seir_stoch_dt, by ="time")
      return(merge_dt)
    })
    
    output$simPlot <- renderPlotly({
      merge_dt <- sirData()
      long_merge_dt <- melt(merge_dt[,.(time,S_stoch,I_stoch,R_stoch)], id.vars = "time")
      long_merge_dt$value <- long_merge_dt$value / input$N
      g <- ggplot(long_merge_dt, aes(x=time,y=value,color=variable)) + geom_line() + theme_bw()
      ggplotly(g)
    })
    
    
    output$downloadSIRData <- downloadHandler(
      filename = function() {
        paste('sir-data-', Sys.Date(), '.csv', sep='')
      },
      content = function(con) {
        write.csv(sirData(), con)
      }
    )
    
    
    
    output$downloadRtData <- downloadHandler(
      filename = function() {
        paste('rt-data-', Sys.Date(), '.csv', sep='')
      },
      content = function(con) {
        write.csv(rtData(), con)
      }
    )
}

# Run the application 
shinyApp(ui = ui, server = server)
