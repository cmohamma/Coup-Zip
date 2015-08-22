### Coup Predictive Analysis: Single Parties and Institutionalization of Executive Constraint ###
## Install necessary packages ##
install.packages("boot")
library(boot)
install.packages("pROC")
library(pROC)
install.packages("ROCR")
library("ROCR")
install.packages("DAAG")
library(DAAG)
library(caret)
library(randomForest)
install.packages("Hmisc")
library(Hmisc)

## Read the File 
coup<-read.csv(file="/Users/Cyrusmohammadian/coupriskdata1.csv")

#Many Thanks to Jay Ulfelder for templates for use on graphics
## Take log of regime duration and gdp ##
coup$regdurln<-sapply(coup$regdur, log)
coup$gdpln<-sapply(coup$gdp, log)

# Manually code 'slowgrowth' for countries with missing values in 2013 using previous year's CIA World Factbook Data
coup$slowgrowth[coup$predyr==2014 & coup$sftgcode=="PRK"] <- 1
coup$slowgrowth[coup$predyr==2014 & coup$sftgcode=="SOM"] <- 0
coup$slowgrowth[coup$predyr==2014 & coup$sftgcode=="SYR"] <- 1
coup$slowgrowth[coup$predyr==2014 & coup$sftgcode=="CUB"] <- 0

##Validation Procedure using p-values and predictive error scores ##
# Subset data to 1960-2012 to create validation set
valdat <- subset(coup, year >= 1960 & year <= 2012 & is.na(coup.1) == FALSE)

# Generate 10 folds based on DV
y <- valdat$coup.1
valdat$k <- createFolds(y, k = 10, list = FALSE)
table(valdat$k, valdat$coup.1)

# Create models
logit.base <- formula(coup.1 ~ postcw + gdpln + infantmort + pastcoup + slowgrowth + anocracy + regdurln + civconc)
logit.part <- formula(coup.1 ~ postcw + gdpln + infantmort + pastcoup + slowgrowth + anocracy + regdurln + civconc + party + xconst)
logit.full <- formula(coup.1 ~ postcw + gdpln + infantmort + pastcoup + slowgrowth + anocracy + regdurln + civconc +party*xconst + xconst + civconc)

full.rev<- formula(coup.1 ~ postcw + gdpln + infantmort + pastcoup + slowgrowth + anocracy + regdurln + civconc + rev_party*xconst + xconst)

# Run model and check results
base<-glm(logit.base, family = binomial, data = coup)
summary(base)
part<-glm(logit.part, family = binomial, data = coup)
summary(part)
full<-glm(logit.full, family = binomial, data = coup)
summary(full)
rev<-glm(full.rev, family = binomial, data = coup)
summary(rev)

# Create function to run models on generated training and test sets
predict.fun <- function(x) {
  train <- subset(valdat, k != x)
  test <- subset(valdat, k == x)
  test$logit.b <- predict(glm(logit.base, family = binomial, data = train, na.action = na.exclude),
                          newdata = test, type = "response")
  test$logit.p <- predict(glm(logit.part, family = binomial, data = train, na.action = na.exclude),
                          newdata = test, type = "response")
  test$logit.f <- predict(glm(logit.full, family = binomial, data = train, na.action = na.exclude),
                          newdata = test, type = "response")
  out <- subset(test, select = c(sftgcode, year, predyr, coup.1, logit.b, logit.p, logit.f, k))
  return(out)
}

test1 <- predict.fun(1)
test2 <- predict.fun(2)
test3 <- predict.fun(3)
test4 <- predict.fun(4)
test5 <- predict.fun(5)
test6 <- predict.fun(6)
test7 <- predict.fun(7)
test8 <- predict.fun(8)
test9 <- predict.fun(9)
test10 <- predict.fun(10)
out <- rbind(test1, test2, test3, test4, test5, test6, test7, test8, test9, test10)

# Create function to obtain AUC distribution for each fold 
fun.auc <- function(df, x) {
  require(verification)
  base.auc.sc <- roc.area(df$coup.1[out$k==x], df$logit.b[out$k==x])
  part.auc.sc <- roc.area(df$coup.1[out$k==x], df$logit.p[out$k==x])
  full.auc.sc <- roc.area(df$coup.1[out$k==x], df$logit.f[out$k==x])
  all <- c(x, base.auc.sc$A, part.auc.sc$A, full.auc.sc$A )
  names(all) <- c("fold", "base", "part", "full")
  return(all)
}

auc1 <- fun.auc(out, 1)
auc2 <- fun.auc(out, 2)
auc3 <- fun.auc(out, 3)
auc4 <- fun.auc(out, 4)
auc5 <- fun.auc(out, 5)
auc6 <- fun.auc(out, 6)
auc7 <- fun.auc(out, 7)
auc8 <- fun.auc(out, 8)
auc9 <- fun.auc(out, 9)
auc10 <- fun.auc(out, 10)
auctab <- as.data.frame(rbind(auc1, auc2, auc3, auc4, auc5, auc6, auc7, auc8, auc9, auc10))

# Create ROC curves for each model
base.pred <- prediction(out$logit.b, out$coup.1)
base.roc <- performance(base.pred, "tpr", "fpr")
base.auc <- performance(base.pred, measure = "auc")
part.pred <- prediction(out$logit.p, out$coup.1) 
part.roc <- performance(part.pred, "tpr", "fpr")
part.auc <- performance(part.pred, measure = "auc")
full.pred <- prediction(out$logit.f, out$coup.1)
full.roc <- performance(full.pred, "tpr", "fpr")
full.auc <- performance(full.pred, measure = "auc")

# Plot ROC Curves together
png(file = "ROCcurve.png",
    width=12, height=12, units='cm', bg='white', res=150)
plot(base.roc, col = "black", lwd=2, add = FALSE)
plot(part.roc, col = "blue", add = TRUE)
plot(full.roc, col = "red", add = TRUE)
title(main= "ROC Curves", sub = NULL)
text(x=1,y=0.10,
     labels = paste("Base AUC", substring(as.character(base.auc@y.values),1,5), sep=' = '),
     pos=2, cex=0.75, col = "black")
text(x=1,y=0.05,
     labels = paste("Partial AUC", substring(as.character(part.auc@y.values),1,5), sep=' = '),
     pos=2, cex=0.75, col = "blue")
text(x=1,y=0,
     labels = paste("Full AUC", substring(as.character(full.auc@y.values),1,5), sep=' = '),
     pos=2, cex=0.75, col = "red")
dev.off()

## Forecast Procedure using logit and random forest ##
# Read full dataset and then subset data to 2010 to run models
pred.data<-read.csv(file="/Users/Cyrusmohammadian/pred_data.csv")
pred.data <- subset(pred.data, predyr <= 2010)

# Create formula for random forest and logistic regression
rf.f <- formula(as.factor(coup.1) ~ postcw + gdpln + infantmort + pastcoup + slowgrowth + party + anocracy + regdurln + xconst*regdurln + civconc)
logit.f <- formula(coup.1 ~ postcw + log(gdp) + infantmort + pastcoup + slowgrowth + anocracy + log(regdur) + party*xconst + xconst + civconc)

# Run logit and random forest along with predictions
logit.mod <- glm(logit.f, data = pred.data, family = binomial, na.action = na.exclude)
rf.mod <- randomForest(rf.f, data = pred.data, na.action="na.exclude", ntree = 1000, mtry = 4, cutoff=c(0.3,0.7))

coup$logit.p <- predict(logit.mod, newdata = coup, type = "response")
coup$rf.p <- predict(rf.mod, newdata = coup, type = "prob", na.action = "na.exclude")[,2]
coup$mean.p <- (coup$logit.p + coup$rf.p)/2

# Plot forecasts for 2011
pred11 <- subset(coup, predyr==2011, select=c(country, coup.1, mean.p))
pred11 <- pred11[order(-pred11$mean.p),]
row.names(pred11) <- NULL
pred11$country <- as.character(pred11$country)
pred11$country[pred11$country=="Congo-Kinshasa"] <- "DRC"
pred11$country[pred11$country=="Congo-Brazzaville"] <- "Republic of Congo"
pred11$rank <- as.numeric(as.character(row.names(pred11)))
condcol <- ifelse(pred11$coup.1==1, "deeppink1", "gray")
png(file = "forecast.2011.png", width=14, height=18, units='cm', bg='white', res=150)
dotchart2(pred11$mean.p[1:40], labels=pred11$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col=condcol, cex.labels=0.75, xlim=c(0,0.45) )
title(main=list("Risk of Coup Attempts in 2011", cex=1),
      sub = list(paste("Coup attempts outside top 40:", paste(pred11$country[pred11$coup.1==1 & pred11$rank > 40]), sep=" "), cex=0.8))
dev.off()


# Plot forecasts for 2012 #occasionally these ones get clunky I am working out some of the kinks still#
pred12 <- subset(coup, predyr==2012, select=c(country, coup.1, mean.p))
pred12 <- pred12[order(-pred12$mean.p),]
row.names(pred12) <- NULL
pred12$country <- as.character(pred12$country)
pred12$country[pred12$country=="Congo-Kinshasa"] <- "DRC"
pred12$country[pred12$country=="Congo-Brazzaville"] <- "Republic of Congo"
pred12$rank <- as.numeric(as.character(row.names(pred12)))
condcol <- ifelse(pred12$coup.1==1, "deeppink1", "gray")
png(file = "forecast.2012.png", width=14, height=18, units='cm', bg='white', res=150)
dotchart2(pred12$mean.p[1:40], labels=pred12$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col=condcol, cex.labels=0.75, xlim=c(0,0.4) )
title(main=list("Risk of Coup Attempts in 2012", cex=1),
      sub = list(paste("Coup attempts outside top 40:", paste(pred11$country[pred12$coup.1==1 & pred12$rank > 40]), sep=" "), cex=0.8))
dev.off()

# Plot forecasts for 2013
pred13 <- subset(coup, predyr==2013, select=c(country, coup.1, mean.p))
pred13 <- pred13[order(-pred13$mean.p),]
row.names(pred13) <- NULL
pred13$country <- as.character(pred13$country)
pred13$country[pred13$country=="Congo-Kinshasa"] <- "DRC"
pred13$country[pred13$country=="Congo-Brazzaville"] <- "Republic of Congo"
pred13$rank <- as.numeric(as.character(row.names(pred13)))
condcol <- ifelse(pred13$coup.1==1, "deeppink1", "gray")
png(file = "forecast.2013.png", width=14, height=18, units='cm', bg='white', res=150)
dotchart2(pred13$mean.p[1:40], labels=pred13$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col=condcol, cex.labels=0.75, xlim=c(0,0.4) )
title(main=list("Risk ofCoup Attempts in 2013", cex=1),
      sub = list(paste("Coup attempts outside top 40:", paste(pred11$country[pred13$coup.1==1 & pred13$rank > 40]), sep=" "), cex=0.8))
dev.off()

# Plot forecasts for 2014
pred14 <- subset(coup, predyr==2014, select=c(country, sftgcode, mean.p, logit.p, rf.p))
pred14 <- pred14[order(-pred14$mean.p),]
pred14$country <- as.character(pred14$country)
pred14$country[pred14$country=="Congo-Kinshasa"] <- "DRC"
pred14$country[pred14$country=="Congo-Brazzaville"] <- "Republic of Congo"
png(file = "forecast.2014.png", width=14, height=18, units='cm', bg='white', res=150)
dotchart2(pred14$logit.p[1:40], labels=pred14$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col="gray", cex.labels=0.75, xlim=c(0,0.4) )
dotchart2(pred14$rf.p[1:40], labels=pred14$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col="gray", cex.labels=0.75, xlim=c(0,0.4), add = TRUE )
dotchart2(pred14$mean.p[1:40], labels=pred14$country[1:40],
          lines=TRUE, lwd=0.05, lty=3, sort=FALSE, dotsize=1.25, pch=20,
          col="firebrick", cex.labels=0.75, xlim=c(0,0.4), add = TRUE )
title(main=list("Risk of Coup Attempts in 2014", cex=1))
dev.off()

# Map 2014 forecasts 
png("heatmap.2014.png", width=800, height=450, bg="white")
par(mai=c(0,0,0.2,0),xaxs="i",yaxs="i")
map.score <- mapCountryData(map2014,
                            nameColumnToPlot="mean.p",
                            addLegend = FALSE,
                            numCats = 5, catMethod="quantiles",
                            colourPalette = "white2Black", borderCol = "gray",
                            mapTitle = "Risk of Coup Attempts in 2014")
do.call(addMapLegend, c(map.score, legendWidth=0.5, legendMar = 2))
mtext("map made using rworldmap             ", line=-4, side=1, adj=1, cex=0.8)
dev.off()
