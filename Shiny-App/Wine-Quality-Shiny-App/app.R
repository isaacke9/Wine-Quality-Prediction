
require(shiny)
require(caret)
require(dplyr)
require(ggplot2)
require(caret)
require(smotefamily)  # SMOTE
require(randomForest)
require(party)
require(ranger)
require(rpart)
require(rpart.plot)
require(tree)

setwd("~/GitHub/Wine-Quality-Prediction/Shiny-App/Wine-Quality-Shiny-App")
load('data/wineData.Rdata')
load('data/rf_smote_res.Rdata')

# Define UI for application 
ui <- fluidPage(

    # Application title
    titlePanel("STAT 656 Project: Random Forest & Synthetic Minority
               Oversampling Technique (SMOTE) on Imbalanced Data"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          p("Here, we illustrate how SMOTE iteratively creates
            synthetic data for the minority class. Click the buttons
            below to see how the dataset changes with each 
            round of SMOTE. 'No SMOTE' refers to the original dataset.
            Each subsequent round of SMOTE addresses the current minority
            class, so 4 rounds are required to create a balanced dataset."),  
          radioButtons(inputId     = "smoteNum",
                         label        = "Number of Rounds of SMOTE to Perform:",
                         choiceNames  = c('No SMOTE','1','2','3','4 (Fully Balanced Data)'),
                         choiceValues = c(0,1,2,3,4)),
        ),

        # Show a plot of the generated distribution
        mainPanel(
          p("(1) The bar plot displays the distribution of the class labels
            after the given number of SMOTE rounds.") ,
          p("(2) Below the bar plot, a line graph is given that illustrates how the
            cross-validation accuracy for the random forest model increases as 
            more SMOTE is performed, and the response classes become more balanced.
            Note that as the number of SMOTE rounds increases, the increase 
            in CV accuracy becomes smaller and smaller (although it continues to increase
            in general)."),
          p("(3) Below the line graph, a table with sample synthetic 
            observations created for the minority class are also given for 
            every subsequent round of SMOTE."),

          plotOutput("histPlot"),
          plotOutput("lineGraph"),
        )
    ),
    fluidRow(
      column(12,
             dataTableOutput('table')
      )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    # Example SMOTEd data from SMOTE round
    output$table <- renderDataTable({
      if(input$smoteNum==1) {
        tail(filter(smote_data1, quality=='X3'))
      }
      else if(input$smoteNum==2) {
        tail(filter(smote_data2, quality=='X8'))
      }
      else if(input$smoteNum==3) {
        tail(filter(smote_data3, quality=='X4'))
      }
      else if(input$smoteNum==4) {
        tail(filter(smote_data4, quality=='X7'))
      }
    })
    
    output$histPlot <- renderPlot({
        if(input$smoteNum ==0) {
          ggplot(data, aes(x=quality)) + 
            geom_bar(fill="darkblue") + 
            ggtitle("Distribution of Response: Wine Quality Rating") +
            labs(y="Count", x = "Wine Quality Rating") +
            theme(plot.title = element_text(hjust = 0.5))
        }
        else if(input$smoteNum ==1) {
          ggplot(smote_data1, aes(x=quality)) + 
            geom_bar(fill="darkblue") + 
            ggtitle("Distribution of Response: Wine Quality Rating") +
            labs(y="Count", x = "Wine Quality Rating") +
            theme(plot.title = element_text(hjust = 0.5))
        }
        else if(input$smoteNum ==2) {
          ggplot(smote_data2, aes(x=quality)) + 
            geom_bar(fill="darkblue") + 
            ggtitle("Distribution of Response: Wine Quality Rating") +
            labs(y="Count", x = "Wine Quality Rating") +
            theme(plot.title = element_text(hjust = 0.5))
        }
        else if(input$smoteNum ==3) {
          ggplot(smote_data3, aes(x=quality)) + 
            geom_bar(fill="darkblue") + 
            ggtitle("Distribution of Response: Wine Quality Rating") +
            labs(y="Count", x = "Wine Quality Rating") +
            theme(plot.title = element_text(hjust = 0.5))
        }
        else if(input$smoteNum ==4) {
          ggplot(smote_data4, aes(x=quality)) + 
            geom_bar(fill="darkblue") + 
            ggtitle("Distribution of Response: Wine Quality Rating") +
            labs(y="Count", x = "Wine Quality Rating") +
            theme(plot.title = element_text(hjust = 0.5))
        }
    })
    
    output$lineGraph = renderPlot({
      if(input$smoteNum ==0) {
        ggplot(data=rf_smote_res[1,], aes(x=round, y=accuracy)) +
          geom_step(aes(col="blue"), size = 1)+
          geom_point(aes(col="blue"), size = 4)+
          ggtitle("Plot of CV Accuracy for Each Round of SMOTE Sampling of Minority Class") +
          theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
          xlab("SMOTE Round") + ylab("Random Forest CV Test Accuracy Estimate") +
          xlim(0,4) +
          ylim(0.65, 0.85)
      }
      else if(input$smoteNum ==1) {
        ggplot(data=rf_smote_res[1:2,], aes(x=round, y=accuracy)) +
          geom_step(aes(col="blue"), size = 1)+
          geom_point(aes(col="blue"), size = 4)+
          ggtitle("Plot of CV Accuracy for Each Round of SMOTE Sampling of Minority Class") +
          theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
          xlab("SMOTE Round") + ylab("Random Forest CV Test Accuracy Estimate") +
          xlim(0,4) +
          ylim(0.65, 0.85)
      }
      else if(input$smoteNum ==2) {
        ggplot(data=rf_smote_res[1:3,], aes(x=round, y=accuracy)) +
          geom_step(aes(col="blue"), size = 1)+
          geom_point(aes(col="blue"), size = 4)+
          ggtitle("Plot of CV Accuracy for Each Round of SMOTE Sampling of Minority Class") +
          theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
          xlab("SMOTE Round") + ylab("Random Forest CV Test Accuracy Estimate") +
          xlim(0,4) +
          ylim(0.65, 0.85)
      }
      else if(input$smoteNum ==3) {
        ggplot(data=rf_smote_res[1:4,], aes(x=round, y=accuracy)) +
          geom_step(aes(col="blue"), size = 1)+
          geom_point(aes(col="blue"), size = 4)+
          ggtitle("Plot of CV Accuracy for Each Round of SMOTE Sampling of Minority Class") +
          theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
          xlab("SMOTE Round") + ylab("Random Forest CV Test Accuracy Estimate") +
          xlim(0,4) +
          ylim(0.65, 0.85)
      }
      else if(input$smoteNum ==4) {
        ggplot(data=rf_smote_res[1:5,], aes(x=round, y=accuracy)) +
          geom_step(aes(col="blue"), size = 1)+
          geom_point(aes(col="blue"), size = 4)+
          ggtitle("Plot of CV Accuracy for Each Round of SMOTE Sampling of Minority Class") +
          theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
          xlab("SMOTE Round") + ylab("Random Forest CV Test Accuracy Estimate") +
          xlim(0,4) +
          ylim(0.65, 0.85)
      }
    
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
